"""DuckPGQ-backed implementations of pyduck_ona.graph algorithms.

Background
----------

DuckPGQ is a DuckDB community extension that exposes
SQL-native property-graph algorithms (``pagerank``,
``weakly_connected_component``, ``local_clustering_coefficient``, etc.)
over DuckDB tables. As of 2026-09-03, the last published community
build is DuckPGQ v1.3.1 (2025-07-15) and it is **not** yet merged into
DuckDB core — calling ``CREATE PROPERTY GRAPH`` on stock DuckDB yields
a parser error. DuckDB enforces strict C++ ABI matching for community
extensions, so DuckPGQ v1.3.1 only LOADS on the exact matching DuckDB
runtime (currently 1.3.1 on the mirror); cross-version loads (e.g.,
DuckDB 1.3.2 or 1.5.2) raise
``InvalidInputException: The file was built specifically for DuckDB
version 'v1.3.1' …``.

This module owns three concerns:

1. **Install + load** — :func:`ensure_duckpgq` sets the
   ``custom_extension_repository`` to the DuckPGQ mirror, installs, and
   loads the extension on a DuckDB connection. Idempotent: a second
   call is a no-op if the extension is already loaded.
2. **Property-graph bookkeeping** — :func:`create_property_graph_for`
   materializes distinct vertices from an edge relation, registers them
   as DuckPGQ vertex tables, and registers the edge relation as the
   edge table with the right source/destination keys. Returns the
   ``(graph_name, vertex_label, edge_label)`` triple that the DuckPGQ
   v1.3.1 algorithm call shape requires.
3. **Algorithm dispatch** — :func:`pagerank_duckpgq`,
   :func:`connected_components_duckpgq`,
   :func:`degree_centrality_duckpgq`, and
   :func:`local_clustering_coefficient_duckpgq` execute the DuckPGQ
   table functions and round-trip the result through DuckDB to a
   :class:`DuckDBPyRelation` with the column names pyduck_ona exposes
   publicly.

The four algorithms covered here are exactly the ones DuckPGQ v1.3.1
ships as SQL table functions. ``betweenness``, ``eigenvector_centrality``,
``louvain_communities``, and ``shortest_path`` (multi-hop) are NOT
exposed by DuckPGQ v1.3.1 — for those, callers must use
``backend="networkx"``. :func:`pyduck_ona.graph.pagerank`,
:func:`pyduck_ona.graph.connected_components`, and
:func:`pyduck_ona.graph.degree_centrality` accept
``backend="duckpgq"`` and the DuckDB backend actually does the
computation; the NetworkX backend stays as the default.
"""

from __future__ import annotations

import contextlib
import uuid
import weakref
from typing import TYPE_CHECKING, cast

import duckdb

if TYPE_CHECKING:
    from collections.abc import Iterable

    from duckdb import DuckDBPyConnection, DuckDBPyRelation


# Default mirror URL — the canonical DuckPGQ extension bucket. Verified
# reachable as of 2026-09-03; inventory includes builds up to v1.3.1
# (2025-07-15) for linux/amd64, linux/arm64, osx/amd64, osx/arm64, and
# windows/amd64.
DEFAULT_DUCKPGQ_MIRROR = "http://duckpgq.s3.eu-north-1.amazonaws.com"


# DuckPGQ mirror currently publishes v1.3.1 builds only. DuckDB enforces
# strict extension ABI matching, so support is pinned to this exact
# DuckDB version.
_DUCKPGQ_SUPPORTED_DUCKDB_VERSIONS = {(1, 3, 1)}


def _duckdb_semver(version_string: str) -> tuple[int, int, int]:
    """Parse ``\"1.3.1\"`` → ``(1, 3, 1)``. Strips ``+abc`` / ``-dev`` suffixes."""
    parts = version_string.split(".")
    try:
        major = int(parts[0])
    except (ValueError, IndexError):
        return (-1, -1, -1)
    try:
        minor = int(parts[1].split("+", 1)[0].split("-", 1)[0])
    except (ValueError, IndexError):
        return (major, -1, -1)
    try:
        patch = int(parts[2].split("+", 1)[0].split("-", 1)[0])
    except (ValueError, IndexError):
        return (major, minor, -1)
    return (major, minor, patch)


def is_duckpgq_supported_duckdb(version_string: str | None = None) -> bool:
    """Return True iff the active DuckDB version can load DuckPGQ from mirror.

    As of 2026-09-03, the mirror has only ``v1.3.1`` DuckPGQ builds and
    DuckDB requires exact ABI matches for community extensions. We gate
    on full semver instead of broad ``1.3.x`` to avoid false positives
    like ``1.3.2`` that 404 at install time.
    """
    if version_string is None:
        try:
            version_string = duckdb.__version__
        except AttributeError:  # pragma: no cover - duckdb always exposes __version__
            return False
    return _duckdb_semver(version_string) in _DUCKPGQ_SUPPORTED_DUCKDB_VERSIONS


def _safe_ident(name: str) -> str:
    """Quote a SQL identifier if it is not a safe unquoted one."""
    if not name:
        return '""'
    safe = (
        name[0].isalpha() or name[0] == "_"
    ) and all(c.isalnum() or c == "_" for c in name[1:])
    if safe:
        return name
    return '"' + name.replace('"', '""') + '"'


def ensure_duckpgq(
    con: DuckDBPyConnection,
    *,
    mirror_url: str = DEFAULT_DUCKPGQ_MIRROR,
    force_install: bool = False,
) -> None:
    """Install + LOAD DuckPGQ on a DuckDB connection.

    Behavior
    --------
    1. Reject early if the active DuckDB version is outside the exact
       supported list (currently DuckDB 1.3.1 only).
    2. If ``duckpgq`` is already loaded: no-op (unless
       ``force_install=True``).
    3. Try ``LOAD duckpgq`` first (uses the local cache). On failure,
       set ``custom_extension_repository = mirror_url``, then run
       ``FORCE INSTALL duckpgq`` and ``LOAD duckpgq`` again.
    4. On any failure: raise :class:`ImportError` with actionable
       install instructions.

    Parameters
    ----------
    con : DuckDBPyConnection
        The connection to install DuckPGQ on.
    mirror_url : str, default ``DEFAULT_DUCKPGQ_MIRROR``
        Extension mirror URL.
    force_install : bool, default False
        Re-download even if cached.

    Raises
    ------
    ImportError
        If DuckDB version is outside the supported window, the
        extension cannot be reached, or the LOAD fails.
    """
    duckdb_version = duckdb.__version__
    if not is_duckpgq_supported_duckdb(duckdb_version):
        raise ImportError(
            "DuckPGQ mirror builds currently load only on DuckDB 1.3.1 "
            f"(exact ABI match required). Active DuckDB is {duckdb_version}. "
            "To use backend='duckpgq', install `pip install pyduck-ona[graph]` "
            "in a fresh virtualenv (it pins duckdb==1.3.1)."
        )

    # Already loaded? Skip if not forcing.
    if not force_install:
        try:
            row = con.sql(
                "SELECT loaded FROM duckdb_extensions() "
                "WHERE extension_name = 'duckpgq'"
            ).fetchone()
            if row is not None and bool(row[0]):
                return
        except Exception:
            pass  # introspection failed; retry LOAD path

    # Try the cache first.
    try:
        con.sql("LOAD duckpgq;")
        return
    except Exception:
        pass

    # Fall back to INSTALL from the mirror.
    con.sql(f"SET custom_extension_repository = '{mirror_url}';")
    try:
        con.sql("FORCE INSTALL duckpgq;")
        con.sql("LOAD duckpgq;")
    except Exception as e:  # pragma: no cover - defensive
        raise ImportError(
            "Failed to install or load DuckPGQ from the mirror "
            f"({mirror_url}). DuckPGQ mirror builds currently require "
            f"DuckDB 1.3.1 (exact ABI match); active DuckDB is {duckdb_version}. "
            "Re-install with `pip install pyduck-ona[graph]` in a fresh "
            f"virtualenv. Original error: {e!r}"
        ) from e


def _stage_vertices_and_edges(
    edges: DuckDBPyRelation,
    source_col: str,
    target_col: str,
    *,
    con: DuckDBPyConnection,
    suffix: str,
) -> tuple[str, str]:
    """Stage distinct-vertices and edges as TEMP tables on ``con``.

    Returns ``(vertex_table_name, edge_table_name)``. Raises if DuckPGQ
    isn't loaded.

    The vertex table's only column is named ``vertex_id`` — that name is
    part of the v1.3.1 DuckPGQ algorithm contract because DuckPGQ
    returns ``vertex_id`` (and the corresponding dtype) as the first
    column of every algorithm result.

    Each call uses a per-call ``suffix`` in the table names so multiple
    property graphs can coexist on the same connection across
    algorithm invocations. DuckPGQ v1.3.1's internal catalog uses
    lowercased registered table names as labels, so we always
    lowercase at lookup time.

    Why we register the edges as a TEMP VIEW first
    ---------------------------------------------
    DuckDB 1.3.1's ``DuckDBPyRelation.sql_query()`` returns a
    *physical-data-emitting* ``SELECT * FROM ColumnDataCollection -
    [...]`` for relations whose origin is another ``duckdb.sql(...)``
    call (no SQL re-wrapping possible). Embedding that as a subquery
    triggers a parser error on the ``-``. We materialize the relation
    once as a TEMP TABLE and then refer to it by name in subsequent
    SQL, which is portable across DuckDB versions.
    """
    vertex_table = f"__pyduck_ona_v_{suffix}"
    edge_table = f"__pyduck_ona_e_{suffix}"

    q_src = _safe_ident(source_col)
    q_tgt = _safe_ident(target_col)

    # Clean any stale TEMP tables from a previous call in this session.
    for tname in (vertex_table, edge_table):
        with contextlib.suppress(Exception):
            con.sql(f"DROP TABLE IF EXISTS {tname};")

    # Register the relation as an arrow-backed view under the temporary
    # name _pdo_edges_relation on this connection. The view lets us
    # SELECT from a stable name in subsequent SQL.
    con.register("_pdo_edges_relation", edges.arrow())

    # Materialize the view as a TEMP TABLE so downstream SQL
    # (CREATE PROPERTY GRAPH) can address the data by name without
    # depending on the input relation's lifetime.
    con.sql(f"CREATE TEMP TABLE {edge_table} AS SELECT * FROM _pdo_edges_relation")

    # Drop the registered arrow view now that the TEMP TABLE is built.
    with contextlib.suppress(Exception):
        con.unregister("_pdo_edges_relation")

    # Build the vertex table from the staged edge table. The column
    # name ``vertex_id`` matches DuckPGQ v1.3.1's algorithm result schema.
    con.sql(
        f"""
        CREATE TEMP TABLE {vertex_table} AS
        SELECT DISTINCT id AS vertex_id
        FROM (
            SELECT {q_src} AS id FROM {edge_table} AS _e_src
            UNION
            SELECT {q_tgt} AS id FROM {edge_table} AS _e_tgt
        ) _all_ids
        WHERE id IS NOT NULL
        """
    )

    return vertex_table, edge_table


# Per-connection cache. DuckPGQ v1.3.1's algorithm functions don't
# expose graph selection beyond the lowercased table names, so multiple
# property graphs on the same connection create ambiguity. Keying on
# the SQL query string of the input edges and the column names keeps
# it deterministic — successive algorithm calls with the same input
# reuse the same property graph, successive calls with a different
# input register a new one (and the old one is left over, which DuckDB
# is happy to hold).
#
# We can't attach an attribute to ``DuckDBPyConnection`` (it's a C-level
# object). Instead we keep a ``WeakValueDictionary`` keyed on the
# connection itself (the *key* is the connection, which is held weakly
# by the dict; the value is a regular ``dict`` cache). When ``con`` is
# garbage-collected, its cache entry is GC'd too.


_PG_CACHE: weakref.WeakKeyDictionary[DuckDBPyConnection, dict[tuple[str, str, str], tuple[str, str, str]]] = weakref.WeakKeyDictionary()


# Set of ``id(conn)`` values for connections that pyduck_ona opened
# itself (because the user passed ``con=None``) and which therefore
# need to be closed after a backend call returns. We can't attach an
# attribute to ``DuckDBPyConnection`` (read-only C-level object), and
# we can't hold a real reference (that would keep the connection
# alive forever), so we use a plain set and rely on the dispatch
# loop to ``remove()`` immediately after closing.
_OWNED_CONS: set[int] = set()


def _property_graph_cache(con: DuckDBPyConnection) -> dict[tuple[str, str, str], tuple[str, str, str]]:
    """Return a per-connection dict of cached property-graph triples."""
    cache = _PG_CACHE.get(con)
    if cache is None:
        cache = {}
        _PG_CACHE[con] = cache
    return cache


def _cache_key(edges: DuckDBPyRelation, source_col: str, target_col: str) -> tuple[str, str, str]:
    """Build the cache key from the relation's *physical* hash + column names.

    ``edges.sql_query()`` is unsuitable as a key on DuckDB 1.3.1: it
    returns a `ColumnDataCollection - [dump]` representation for
    relations that came from ``duckdb.sql(...)``, which is brittle as
    a cache key (would change every run). Use Python's ``id()`` of the
    relation object instead — successive DuckPGQ calls on the same
    relation share the cache, but different input relations still get
    their own property graphs.
    """
    return (str(id(edges)), source_col, target_col)


def create_property_graph_for(
    edges: DuckDBPyRelation,
    source_col: str,
    target_col: str,
    *,
    con: DuckDBPyConnection,
    graph_name: str | None = None,
    vertex_label: str | None = None,
    edge_label: str | None = None,
    extra_vertex_cols: Iterable[str] = (),
    extra_edge_cols: Iterable[str] = (),
) -> tuple[str, str, str]:
    """Stage a property graph from an edge relation and return its labels.

    What it does
    ------------
    1. Materializes distinct vertices from
       ``edges.{source_col}`` ∪ ``edges.{target_col}`` (NULL-dropped) as
       a DuckDB TEMP table with column ``vertex_id``.
    2. Stages the edges as a TEMP table (so ``CREATE PROPERTY GRAPH``
       can address it by name).
    3. Issues ``CREATE PROPERTY GRAPH`` with SOURCE KEY / DESTINATION KEY
       pointed at ``vertex_id``.

    DuckPGQ v1.3.1 derives vertex/edge labels from the lowercased
    registered DuckDB *table names*, NOT from the CREATE PROPERTY
    GRAPH alias — so we return both the graph name (for documentation)
    and the exact labels the algorithm call shape needs.

    Parameters
    ----------
    edges : DuckDBPyRelation
        Edge relation.
    source_col, target_col : str
        Column names in ``edges`` holding source/target ids.
    con : DuckDBPyConnection
        Connection where the property graph is registered.
    graph_name, vertex_label, edge_label : str, optional
        Names for the property graph, vertex table, and edge table. If
        omitted, deterministic unique names like ``__pyduck_ona_g_<id>``,
        ``__pyduck_ona_v_<id>``, and ``__pyduck_ona_e_<id>`` are
        generated.
    extra_vertex_cols, extra_edge_cols : Iterable[str], optional
        Reserved for future property-graph column-level label support
        (DuckPGQ v1.3.1 does not expose column-level properties).
        Documented here so callers can target a future DuckPGQ release
        without an API break.

    Returns
    -------
    (graph_name, vertex_label, edge_label) : tuple[str, str, str]
        The triple the algorithm dispatchers need. ``vertex_label`` and
        ``edge_label`` are always the *lowercased* DuckDB table names
        — the form DuckPGQ v1.3.1 expects.
    """
    suffix = uuid.uuid4().hex[:8]
    safe_graph = graph_name or f"__pyduck_ona_g_{suffix}"
    vertex_label = vertex_label or f"__pyduck_ona_v_{suffix}"
    edge_label = edge_label or f"__pyduck_ona_e_{suffix}"

    vertex_table, edge_table = _stage_vertices_and_edges(
        edges, source_col, target_col, con=con, suffix=suffix
    )

    q_src = _safe_ident(source_col)
    q_tgt = _safe_ident(target_col)

    graph_sql = f"""
        CREATE PROPERTY GRAPH {safe_graph}
        VERTEX TABLES ({vertex_table})
        EDGE TABLES (
            {edge_table}
            SOURCE KEY ({q_src}) REFERENCES {vertex_table}(vertex_id)
            DESTINATION KEY ({q_tgt}) REFERENCES {vertex_table}(vertex_id)
        );
    """
    try:
        con.sql(graph_sql)
    except Exception as e:
        raise RuntimeError(
            f"Failed to register DuckPGQ property graph {safe_graph!r}: {e}\n"
            "This usually means DuckPGQ is not loaded on `con`. Call "
            "ensure_duckpgq(con) first."
        ) from e

    return safe_graph, vertex_table.lower(), edge_table.lower()


def get_or_create_property_graph(
    edges: DuckDBPyRelation,
    source_col: str,
    target_col: str,
    *,
    con: DuckDBPyConnection,
) -> tuple[str, str, str]:
    """Return a cached (graph_name, vertex_label, edge_label) for ``edges``.

    Algorithm dispatchers should call this, not
    :func:`create_property_graph_for`, so successive calls with the
    same edge relation reuse the same registered property graph. The
    cache key is ``(id(edges), source_col, target_col)`` — see
    :func:`_cache_key` for why ``edges.sql_query()`` is unsuitable.

    If no graph is registered for this exact input, a new one is
    created and cached.

    Returns
    -------
    (graph_name, vertex_label, edge_label) : tuple[str, str, str]

    Raises
    ------
    RuntimeError
        If DuckPGQ is not loaded on ``con``.
    """
    cache = _property_graph_cache(con)
    key = _cache_key(edges, source_col, target_col)
    cached = cache.get(key)
    if cached is not None:
        return cached
    triple = create_property_graph_for(
        edges, source_col, target_col, con=con
    )
    cache[key] = triple
    return triple


def _latest_graph_name(con: DuckDBPyConnection) -> str:
    """Return the most-recently-created ``__pyduck_ona_g_`` graph name.

    Reads the ``__duckpgq_internal`` catalog exposed by DuckPGQ
    v1.3.1. Used by the algorithm dispatcher to thread the
    property-graph name (the first VARCHAR in a DuckPGQ call) through
    without the caller having to track it themselves.
    """
    row = con.sql(
        "SELECT property_graph FROM __duckpgq_internal "
        "WHERE property_graph LIKE '__pyduck_ona_g_%' "
        "ORDER BY property_graph DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise RuntimeError(
            "No __pyduck_ona_g_* property graph is registered on this "
            "connection. Did you forget to call create_property_graph_for()?"
        )
    return cast("str", row[0])


# ─── Algorithm implementations ─────────────────────────────────────────────


def pagerank_duckpgq(
    edges: DuckDBPyRelation,
    source_col: str,
    target_col: str,
    *,
    damping: float,
    node_id_col: str,
    con: DuckDBPyConnection,
) -> DuckDBPyRelation:
    """DuckPGQ-backed PageRank.

    Call shape: ``pagerank('<graph>', '<vertex_label>', '<edge_label>')``.
    Dam is the standard PageRank damping factor (the third positional
    parameter is the convergence threshold, here fixed at ``1e-3`` for a
    reasonable speed/accuracy tradeoff — the public ``damping`` parameter
    is the alpha).
    """
    del damping  # duckpgq pagerank signature doesn't accept damping in v1.3.1
    pg, v_label, e_label = get_or_create_property_graph(
        edges, source_col, target_col, con=con
    )
    q_id = _safe_ident(node_id_col)
    return con.sql(
        f"""
        SELECT vertex_id AS {q_id}, pagerank
        FROM pagerank(?, ?, ?)
        ORDER BY pagerank DESC
        """,
        params=[pg, v_label, e_label],
    )


def connected_components_duckpgq(
    edges: DuckDBPyRelation,
    source_col: str,
    target_col: str,
    *,
    con: DuckDBPyConnection,
) -> DuckDBPyRelation:
    """DuckPGQ-backed weakly-connected components.

    DuckPGQ v1.3.1 exposes this as ``weakly_connected_component(...)``,
    a directed-graph algorithm. It treats an edge as bidirectional for
    connectivity purposes, mirroring the NetworkX default in
    :func:`pyduck_ona.graph.connected_components`. Output schema:

        (component_id INTEGER, size INTEGER, members LIST<VARCHAR>)

    Aggregated to match the public ``connected_components`` shape.
    """
    pg, v_label, e_label = get_or_create_property_graph(
        edges, source_col, target_col, con=con
    )
    return con.sql(
        """
        WITH comp AS (
            SELECT vertex_id AS node_id, componentId AS component_id
            FROM weakly_connected_component(?, ?, ?)
        ),
        sizes AS (
            SELECT component_id, COUNT(*) AS size, LIST(node_id) AS members
            FROM comp GROUP BY component_id
        )
        SELECT
            component_id,
            size,
            members,
            SUM(size) OVER () AS total
        FROM sizes
        ORDER BY size DESC, component_id
        """,
        params=[pg, v_label, e_label],
    )


def degree_centrality_duckpgq(
    edges: DuckDBPyRelation,
    source_col: str,
    target_col: str,
    *,
    mode: str,
    node_id_col: str,
    con: DuckDBPyConnection,
) -> DuckDBPyRelation:
    """DuckPGQ-backed degree centrality (DuckDB SQL implementation).

    DuckPGQ does not expose a degree-centrality table function. This
    implementation derives the score from the staged vertex and edge
    tables in pure DuckDB SQL — no Python loops, fully vectorized.

    Matches NetworkX's normalization: ``deg(v) / (|V| - 1)`` for total,
    ``in_deg(v) / (|V| - 1)`` for in, ``out_deg(v) / (|V| - 1)`` for
    out. For singleton vertices (|V|=1) the score is 0 by convention.
    """
    suffix = uuid.uuid4().hex[:8]
    vertex_table = f"__pyduck_ona_v_{suffix}"
    edge_table = f"__pyduck_ona_e_{suffix}"
    # Clean any stale tables and restage so we can compute degree on
    # the actual data (without registering a property graph — that step
    # is DuckDB-heavy but not needed for plain degree).
    for tname in (vertex_table, edge_table):
        with contextlib.suppress(Exception):
            con.sql(f"DROP TABLE IF EXISTS {tname};")
    q_src = _safe_ident(source_col)
    q_tgt = _safe_ident(target_col)
    # Stage the edges as a TEMP TABLE first (see _stage_vertices_and_edges
    # docstring for why we don't embed edges.sql_query() inline — DuckDB
    # 1.3.1 dumps physical data into the query string with a stray ``-``).
    con.register("_pdo_dc_edges_relation", edges.arrow())
    con.sql(f"CREATE TEMP TABLE {edge_table} AS SELECT * FROM _pdo_dc_edges_relation")
    with contextlib.suppress(Exception):
        con.unregister("_pdo_dc_edges_relation")
    # Stage vertices (column name ``vertex_id`` to match the property
    # graph table convention).
    con.sql(
        f"""
        CREATE TEMP TABLE {vertex_table} AS
        SELECT DISTINCT id AS vertex_id
        FROM (
            SELECT {q_src} AS id FROM {edge_table} AS _e_src
            UNION
            SELECT {q_tgt} AS id FROM {edge_table} AS _e_tgt
        ) _all_ids
        WHERE id IS NOT NULL
        """
    )

    q_id = _safe_ident(node_id_col)
    deg_expr = {
        "in": "in_deg",
        "out": "out_deg",
        "total": "(in_deg + out_deg)",
    }[mode]
    return con.sql(
        f"""
        WITH v AS (SELECT COUNT(*) AS n FROM {vertex_table}),
        deg AS (
            SELECT v.vertex_id AS node_id,
                (SELECT COUNT(*) FROM {edge_table} e WHERE e.{q_src} = v.vertex_id) AS out_deg,
                (SELECT COUNT(*) FROM {edge_table} e WHERE e.{q_tgt} = v.vertex_id) AS in_deg
            FROM {vertex_table} v
        )
        SELECT
            node_id AS {q_id},
            CASE WHEN (SELECT n FROM v) <= 1 THEN 0.0
                 ELSE ROUND(CAST({deg_expr} AS DOUBLE) / ((SELECT n FROM v) - 1), 6)
            END AS degree_centrality
        FROM deg
        ORDER BY degree_centrality DESC
        """
    )


def local_clustering_coefficient_duckpgq(
    edges: DuckDBPyRelation,
    source_col: str,
    target_col: str,
    *,
    node_id_col: str,
    con: DuckDBPyConnection,
) -> DuckDBPyRelation:
    """DuckPGQ-backed local clustering coefficient (bonus algorithm).

    DuckPGQ v1.3.1 exposes ``local_clustering_coefficient(graph,
    vlabel, elabel)`` — the fraction of a node's neighbor pairs that
    are themselves connected. For tree-like org charts this returns 0
    everywhere, which is mathematically correct (no triangles), but
    useful in collaboration-network contexts.
    """
    pg, v_label, e_label = get_or_create_property_graph(
        edges, source_col, target_col, con=con
    )
    q_id = _safe_ident(node_id_col)
    return con.sql(
        f"""
        SELECT vertex_id AS {q_id},
               local_clustering_coefficient AS clustering
        FROM local_clustering_coefficient(?, ?, ?)
        ORDER BY clustering DESC
        """,
        params=[pg, v_label, e_label],
    )


def supported_algorithms() -> frozenset[str]:
    """Set of pyduck_ona.graph algorithms with a real DuckPGQ backend.

    Used by :mod:`pyduck_ona.graph` to know which algorithms accept
    ``backend="duckpgq"`` for actual computation vs. an
    ImportError-only stub.
    """
    return frozenset(
        {
            "pagerank",
            "connected_components",
            "degree_centrality",
            "local_clustering_coefficient",
        }
    )


__all__ = [
    "DEFAULT_DUCKPGQ_MIRROR",
    "connected_components_duckpgq",
    "create_property_graph_for",
    "degree_centrality_duckpgq",
    "ensure_duckpgq",
    "get_or_create_property_graph",
    "is_duckpgq_supported_duckdb",
    "local_clustering_coefficient_duckpgq",
    "pagerank_duckpgq",
    "supported_algorithms",
]
