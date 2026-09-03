"""Graph algorithms on org-chart relations.

Three backends are available for the algorithm functions exposed by
this module (``pagerank``, ``betweenness``, ``connected_components``,
``degree_centrality``, ``shortest_path``, ``eigenvector_centrality``,
``louvain_communities``):

  - **NetworkX (default)** — pure-Python, always available because
    NetworkX is in ``[project].dependencies``. Works on any
    DuckDB-compatible arrow transfer. Use this unless you have a
    DuckPGQ extension loaded.

  - **DuckPGQ (optional)** — DuckDB-native property graph algorithms
    via the DuckPGQ community extension. Selected with
    ``backend="duckpgq"``. Requires ``pip install pyduck-ona[graph]``,
    which currently pins DuckDB to ``1.3.1`` (exact ABI match for the
    published DuckPGQ build).

  - **SQL-only (under DuckPGQ backend name)** — :func:`degree_centrality`
    uses pure DuckDB SQL for the DuckPGQ backend path (DuckPGQ does
    not expose a degree-centrality table function).

The four algorithms with a real DuckPGQ implementation as of
DuckPGQ v1.3.1 (2025-07-15):

    :func:`pagerank`              ← ``pagerank(graph, vlabel, elabel)``
    :func:`connected_components`   ← ``weakly_connected_component(...)``
    :func:`degree_centrality`     ← pure DuckDB SQL over the property graph
    (bonus, not in the public pyduck_ona catalog:)
        ``local_clustering_co``     ← ``local_clustering_coefficient(...)``

The other algorithms (``betweenness``, ``shortest_path`` multi-hop,
``eigenvector_centrality``, ``louvain_communities``) currently have
no DuckPGQ backend; selecting ``backend="duckpgq"`` for them raises
:func:`ImportError` with install instructions. The DuckPGQ backend is
shipped as a separate optional ``[graph]`` extra on the
package, and DuckPGQ's future API surface may expand this list.

For every function:

  - Pass a DuckDB relation of edges (typically the output of
    :func:`pyduck_ona.core.hierarchy_long`) plus the column names.
  - We then dispatch to the chosen backend (``networkx`` by default,
    ``duckpgq`` if requested and supported).
  - The result is a DuckDB relation so the API is uniform across
    backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import duckdb

if TYPE_CHECKING:
    import networkx as nx
    import pandas as pd
    from duckdb import DuckDBPyRelation


def _require_duckpgq(
    con: duckdb.DuckDBPyConnection | None = None,
) -> None:
    """Raise if DuckPGQ is not loadable on the active connection.

    Implementation policy
    ---------------------
    This used to be a hard-coded error path that simply raised
    ``ImportError`` with a "not installable" message. As of v0.2.1 it
    actually attempts to install + load DuckPGQ via the official
    S3 mirror. Failure modes:

    - Active DuckDB outside the currently supported version list
      (DuckDB 1.3.1) — clear ABI-mismatch error
      with instructions to ``pip install pyduck-ona[graph]`` in a fresh
      venv.
    - Extension cannot be installed — wrapped ImportError with the
      underlying cause.
    - ``con`` not provided — we use ``duckdb.connect()`` and close it
      after the attempt so callers aren't holding a connection they
      didn't ask for.
    """
    from pyduck_ona.graph import _duckpgq_backend as _dg

    if not _dg.is_duckpgq_supported_duckdb():
        raise ImportError(
            "DuckPGQ mirror builds currently load only on DuckDB 1.3.1 "
            f"(exact ABI match required). Active DuckDB is {duckdb.__version__}. "
            "To use backend='duckpgq', run `pip install "
            "pyduck-ona[graph]` in a fresh virtualenv so duckdb is "
            "pinned to ==1.3.1."
        )
    if con is None:
        ephemeral = duckdb.connect(config={"allow_unsigned_extensions": "true"})
        try:
            _dg.ensure_duckpgq(ephemeral)
        finally:
            ephemeral.close()
        return
    _dg.ensure_duckpgq(con)


# ─── Shared helpers ─────────────────────────────────────────────────────────


def _edges_arrow(
    edges: DuckDBPyRelation,
    source_col: str,
    target_col: str,
) -> tuple[list[Any], list[Any]]:
    """Materialize the edge relation to ``(source_list, target_list)``.

    DuckDB 1.3+ returns a streaming ``RecordBatchReader`` from
    ``rel.arrow()``; we materialize so we can call ``.column()``.
    """
    result = edges.arrow()
    if hasattr(result, "read_all"):
        result = result.read_all()
    return (
        result.column(source_col).to_pylist(),
        result.column(target_col).to_pylist(),
    )


def _nx_digraph(
    edges: DuckDBPyRelation,
    source_col: str,
    target_col: str,
) -> nx.DiGraph[Any]:
    """Build a NetworkX ``DiGraph`` from an edge relation."""
    import networkx as nx

    src, tgt = _edges_arrow(edges, source_col, target_col)
    pairs = [(s, t) for s, t in zip(src, tgt, strict=False) if s is not None and t is not None]
    graph: nx.DiGraph[Any] = nx.DiGraph()
    graph.add_edges_from(pairs)
    return graph


def _wrap_as_relation(
    df: pd.DataFrame,
    con: duckdb.DuckDBPyConnection | None = None,
) -> DuckDBPyRelation:
    """Round-trip a pandas DataFrame through DuckDB to get a relation.

    Used so every graph function returns the same type regardless of
    which backend produced the answer. When ``con`` is supplied the
    relation is created on that connection so callers can chain on the
    same database session.

    Implementation note
    -------------------
    ``target.sql("SELECT * FROM df")`` only works when DuckDB can find
    a Python local named ``df`` in the calling scope (DuckDB's module-
    level ``sql`` consults ``sys._getframe`` for local variables). For
    programmatic use — and for DuckDB 1.3.1 where string-column
    dataframes cannot be registered directly — we materialize via
    PyArrow into the egress connection (see
    :func:`_wrap_as_relation_safe` for the full rationale). When
    ``con`` is provided, the PyArrow buffer is registered on that
    connection so the relation lives on the caller's session.
    """
    import pyarrow as pa

    arrow_tbl = pa.Table.from_pandas(df)
    target = con if con is not None else _duckpgq_egress_connection()
    # Register under a unique-per-call name so successive pandas-only
    # graph calls don't shadow each other's relation.
    global _EGRESS_COUNTER  # noqa: PLW0603
    _EGRESS_COUNTER += 1
    name = f"__pyduck_ona_wrap_{_EGRESS_COUNTER}"
    target.register(name, arrow_tbl)
    return target.sql(f"SELECT * FROM {name}")


# Module-level singleton DuckDB connection used as the egress target
# for DuckPGQ backend results. It lives for the lifetime of the
# process — not great in long-running services, but for scripts and
# notebooks (the typical pyduck_ona usage) this is fine and lets the
# returned ``DuckDBPyRelation`` stay valid after the per-call
# ephemeral connection closes.
#
# Imported lazily via ``_duckpgq_egress_connection()`` so module import
# does not pin DuckDB on every test.
_EGRESS_PERSIST: duckdb.DuckDBPyConnection | None = None
_EGRESS_COUNTER = 0


def _duckpgq_egress_connection() -> duckdb.DuckDBPyConnection:
    """Return a process-level DuckDB connection used to egress DuckPGQ
    backend results.

    The DuckPGQ dispatcher creates a per-call ephemeral connection to
    run the property-graph registration and the DuckPGQ algorithm,
    then closes that connection in its ``finally``. The relation
    returned to the user would normally die on close. To avoid that,
    we materialize the DuckPGQ result into a pandas DataFrame, build
    a PyArrow table from it, and register it on this singleton
    connection. The returned ``DuckDBPyRelation`` lives on top of
    this connection.

    For notebook / CLI usage this is fine — the process exits and the
    connection is closed by the OS. For a long-running server the
    connection will accumulate registered tables; that is a minor
    leak (one Arrow buffer per call) and users can call
    ``reset_duckpgq_egress_connection()`` periodically to drop them.
    """
    global _EGRESS_PERSIST
    if _EGRESS_PERSIST is None:
        _EGRESS_PERSIST = duckdb.connect()
    return _EGRESS_PERSIST


def reset_duckpgq_egress_connection() -> None:
    """Drop all DuckPGQ-backed results from the process-egress connection.

    Called by callers that want to reclaim the small per-call Arrow
    buffers accumulated in :func:`_duckpgq_egress_connection`. Not
    required for normal use; intended for long-running services that
    run many graph algorithms.
    """
    global _EGRESS_PERSIST
    if _EGRESS_PERSIST is not None:
        try:
            _EGRESS_PERSIST.close()
        finally:
            _EGRESS_PERSIST = None


def _wrap_as_relation_safe(
    df: pd.DataFrame,
) -> DuckDBPyRelation:
    """DuckPGQ-egress helper: build a standalone relation from ``df``.

    The plain ``_wrap_as_relation`` path uses ``target.sql("SELECT *
    FROM df")`` which works on DuckDB >=1.4 but breaks on DuckDB 1.3.1
    when ``df`` has object/string columns. The DuckPGQ backend forces
    DuckDB onto 1.3.1, so we cannot use that pattern.

    Implementation: build a PyArrow table from ``df`` and register it
    on the process-egress DuckDB connection (see
    :func:`_duckpgq_egress_connection`). Return a ``SELECT *`` relation
    pointing at that registration. The relation stays valid as long as
    the Python process holds the connection alive.
    """
    global _EGRESS_COUNTER
    import pyarrow as pa

    arrow_tbl = pa.Table.from_pandas(df)
    conn = _duckpgq_egress_connection()
    _EGRESS_COUNTER += 1
    name = f"__pyduck_ona_duckpgq_result_{_EGRESS_COUNTER}"
    conn.register(name, arrow_tbl)
    return conn.sql(f"SELECT * FROM {name}")


def _duckpgq_dispatch_or_raise(algorithm: str, con: duckdb.DuckDBPyConnection | None) -> duckdb.DuckDBPyConnection:
    """Return a connection with DuckPGQ loaded, or raise ImportError.

    The DuckPGQ backend functions in :mod:`_duckpgq_backend` need a
    real connection (LOAD is connection-scoped). For the default
    ``con=None`` we open a fresh ephemeral connection so users can
    call ``pagerank(edges, 'src', 'dst', backend='duckpgq')`` without
    pre-wiring anything. Track ownership in a module-level set keyed
    on ``id(con)`` because we cannot attach attributes to
    ``DuckDBPyConnection``.

    The returned relation is bound to this connection and is only
    valid for as long as the connection stays open. The DuckPGQ
    algorithm dispatchers in :func:`pagerank`, :func:`connected_components`,
    and :func:`degree_centrality` handle the lifecycle by materializing
    the result eagerly when the user passes ``con=None`` — see those
    functions for the close-on-finally dance.
    """
    from pyduck_ona.graph import _duckpgq_backend as _dg

    if con is None:
        con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
        _dg._OWNED_CONS.add(id(con))
    _dg.ensure_duckpgq(con)
    return con


def _duckpgq_owned(con: duckdb.DuckDBPyConnection) -> bool:
    """True iff ``_duckpgq_dispatch_or_raise`` opened this connection itself."""
    from pyduck_ona.graph import _duckpgq_backend as _dg

    return id(con) in _dg._OWNED_CONS


# ─── Algorithms ─────────────────────────────────────────────────────────────


def shortest_path(
    edges: DuckDBPyRelation,
    source_col: str,
    target_col: str,
    source: str,
    target: str,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    backend: Literal["networkx", "duckpgq"] = "networkx",
) -> DuckDBPyRelation:
    """Shortest path between two nodes in the edge graph.

    Parameters
    ----------
    edges : DuckDBPyRelation
        Edge relation. Typically the output of
        :func:`pyduck_ona.core.hierarchy_long`.
    source_col, target_col : str
        Column names in ``edges`` holding the source and target of
        each directed edge.
    source, target : str
        The two node IDs to find a path between.
    backend : {"networkx", "duckpgq"}, default "networkx"
        Algorithm backend. DuckPGQ v1.3.1 does not expose a multi-hop
        shortest-path table function; selecting ``backend="duckpgq"``
        raises :class:`ImportError`.

    Returns
    -------
    DuckDBPyRelation
        One row with columns ``(source, target, path_length, path)``.
        ``path`` is a ``->``-delimited sequence. If no path exists,
        ``path_length`` is NULL and ``path`` is empty.

    Examples
    --------
    >>> long = hierarchy_long(rel, "emp_id", "mgr_id")
    >>> shortest_path(long, "employee_id", "supervisor_id",
    ...               "E001", "E999").df()

    Notes
    -----
    When ``source == target``, returns ``path_length=0`` and
    ``path=<source>`` (the trivial self-path). This is by design: a
    distance-to-self of zero is the standard graph-theory convention.
    If you need a different definition, filter upstream.
    """
    if backend == "duckpgq":
        raise ImportError(
            "shortest_path() does not have a DuckPGQ backend in "
            "DuckPGQ v1.3.1 — the extension exposes reachability for "
            "single pairs (5-arg scalar) but no multi-hop shortest "
            "path table function. Use backend='networkx'."
        )

    import pandas as pd

    graph = _nx_digraph(edges, source_col, target_col)
    import networkx as nx

    if source not in graph or target not in graph:
        result_df = pd.DataFrame(
            [(source, target, None, "")],
            columns=["source", "target", "path_length", "path"],
        )
        return _wrap_as_relation(result_df, con=con)

    try:
        node_path = nx.shortest_path(graph, source=source, target=target)
        path_str = "->".join(str(n) for n in node_path)
        length = len(node_path) - 1
    except nx.NetworkXNoPath:
        length = None
        path_str = ""

    result_df = pd.DataFrame(
        [(source, target, length, path_str)],
        columns=["source", "target", "path_length", "path"],
    )
    return _wrap_as_relation(result_df, con=con)


def betweenness(
    edges: DuckDBPyRelation,
    source_col: str,
    target_col: str,
    *,
    node_id_col: str = "node_id",
    con: duckdb.DuckDBPyConnection | None = None,
    backend: Literal["networkx", "duckpgq"] = "networkx",
) -> DuckDBPyRelation:
    """Betweenness centrality for every node (broker detection).

    High betweenness = the employee sits on many shortest paths
    between other pairs = information broker. Removing them would
    disconnect parts of the org.

    Parameters
    ----------
    edges, source_col, target_col
        Edge relation and column names.
    node_id_col : str, default "node_id"
        Name of the node-id column in the returned relation.
    backend : {"networkx", "duckpgq"}, default "networkx"
        Algorithm backend. DuckPGQ v1.3.1 does not expose a
        betweenness table function; selecting ``backend="duckpgq"``
        raises :class:`ImportError`.

    Returns
    -------
    DuckDBPyRelation
        Columns ``(node_id_col, betweenness)`` sorted by betweenness DESC.
    """
    if backend == "duckpgq":
        raise ImportError(
            "betweenness() does not have a DuckPGQ backend in DuckPGQ "
            "v1.3.1 — the extension does not expose a betweenness table "
            "function. Use backend='networkx'."
        )

    import pandas as pd

    graph = _nx_digraph(edges, source_col, target_col)
    import networkx as nx

    scores = nx.betweenness_centrality(graph)
    df = pd.DataFrame(
        [(node, float(score)) for node, score in scores.items()],
        columns=[node_id_col, "betweenness"],
    ).sort_values("betweenness", ascending=False, kind="mergesort").reset_index(drop=True)
    return _wrap_as_relation(df, con=con)


def pagerank(
    edges: DuckDBPyRelation,
    source_col: str,
    target_col: str,
    *,
    damping: float = 0.85,
    node_id_col: str = "node_id",
    con: duckdb.DuckDBPyConnection | None = None,
    backend: Literal["networkx", "duckpgq"] = "networkx",
) -> DuckDBPyRelation:
    """PageRank centrality (influence scoring).

    Parameters
    ----------
    edges, source_col, target_col
    damping : float, default 0.85
        Standard PageRank damping factor. **Note:** the DuckPGQ v1.3.1
        ``pagerank`` table function does not expose a damping parameter;
    the value is accepted for API compatibility but ignored on the
    DuckPGQ backend (the engine uses its default). NetworkX may require
    SciPy in clean environments.
    node_id_col : str, default "node_id"
        Name of the node-id column in the returned relation.
    backend : {"networkx", "duckpgq"}, default "networkx"
        Algorithm backend. The DuckPGQ backend runs the SQL table
        function ``pagerank(graph, vlabel, elabel)`` on a registered
        property graph; requires ``pip install pyduck-ona[graph]``.

    Returns
    -------
    DuckDBPyRelation
        Columns ``(node_id_col, pagerank)`` sorted by pagerank DESC.

    Notes
    -----
    The DuckPGQ backend installs and loads the DuckPGQ extension on
    the supplied ``con`` (or an ephemeral one if ``con`` is ``None``)
    and runs the computation inside DuckDB. Results are not
    byte-identical to NetworkX because DuckPGQ and NetworkX use
    different convergence criteria; for trend analysis on the same
    graph the relative ordering of nodes is preserved.
    """
    if backend == "duckpgq":
        from pyduck_ona.graph import _duckpgq_backend as _dg

        _require_duckpgq(con)
        conn = _duckpgq_dispatch_or_raise("pagerank", con)
        try:
            result = _dg.pagerank_duckpgq(
                edges,
                source_col,
                target_col,
                damping=damping,
                node_id_col=node_id_col,
                con=conn,
            )
            # If we own this connection, the result relation is bound
            # to it and would die when we close. Eagerly materialize.
            df = result.df() if _duckpgq_owned(conn) else None
            return _wrap_as_relation_safe(df) if df is not None else result
        finally:
            if _duckpgq_owned(conn):
                _dg._OWNED_CONS.discard(id(conn))
                conn.close()

    import pandas as pd

    graph = _nx_digraph(edges, source_col, target_col)
    import networkx as nx

    scores = nx.pagerank(graph, alpha=damping)
    df = pd.DataFrame(
        [(node, float(score)) for node, score in scores.items()],
        columns=[node_id_col, "pagerank"],
    ).sort_values("pagerank", ascending=False, kind="mergesort").reset_index(drop=True)
    return _wrap_as_relation(df, con=con)


def connected_components(
    edges: DuckDBPyRelation,
    source_col: str,
    target_col: str,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    backend: Literal["networkx", "duckpgq"] = "networkx",
) -> DuckDBPyRelation:
    """Weakly-connected components in the edge graph.

    Parameters
    ----------
    edges, source_col, target_col
    backend : {"networkx", "duckpgq"}, default "networkx"
        Algorithm backend. The DuckPGQ backend runs DuckPGQ's
        ``weakly_connected_component(graph, vlabel, elabel)`` on a
        registered property graph; requires
        ``pip install pyduck-ona[graph]``.

    Returns
    -------
    DuckDBPyRelation
        Columns ``(component_id, size, members)`` sorted by size DESC.
        ``members`` is a list of node IDs in that component.

    Notes
    -----
    "Weakly connected" treats the graph as undirected for component
    purposes — appropriate for org charts where up/down direction is
    conventional but connectivity is what matters. In a healthy org
    chart there should be exactly 1 component. More than 1 indicates
    multiple top-level hierarchies (acquired companies, business
    units, or — most often — data-quality issues).
    """
    if backend == "duckpgq":
        from pyduck_ona.graph import _duckpgq_backend as _dg

        _require_duckpgq(con)
        conn = _duckpgq_dispatch_or_raise("connected_components", con)
        try:
            result = _dg.connected_components_duckpgq(
                edges, source_col, target_col, con=conn
            )
            df = result.df() if _duckpgq_owned(conn) else None
            return _wrap_as_relation_safe(df) if df is not None else result
        finally:
            if _duckpgq_owned(conn):
                _dg._OWNED_CONS.discard(id(conn))
                conn.close()

    import pandas as pd

    graph = _nx_digraph(edges, source_col, target_col)
    import networkx as nx

    components = list(nx.weakly_connected_components(graph))
    components.sort(key=len, reverse=True)
    rows = [(int(idx), len(members), sorted(members)) for idx, members in enumerate(components)]
    df = pd.DataFrame(rows, columns=["component_id", "size", "members"])
    return _wrap_as_relation(df, con=con)


def eigenvector_centrality(
    edges: DuckDBPyRelation,
    source_col: str,
    target_col: str,
    *,
    node_id_col: str = "node_id",
    con: duckdb.DuckDBPyConnection | None = None,
    backend: Literal["networkx", "duckpgq"] = "networkx",
) -> DuckDBPyRelation:
    """Eigenvector centrality for every node.

    Parameters
    ----------
    edges, source_col, target_col
    node_id_col : str, default "node_id"
        Name of the node-id column in the returned relation.
    backend : {"networkx", "duckpgq"}, default "networkx"
        Algorithm backend. DuckPGQ v1.3.1 does not expose an
        eigenvector-centrality table function; selecting
        ``backend="duckpgq"`` raises :class:`ImportError`.

    Returns
    -------
    DuckDBPyRelation
        Columns ``(node_id_col, eigenvector)`` sorted by eigenvector DESC.
    """
    if backend == "duckpgq":
        raise ImportError(
            "eigenvector_centrality() does not have a DuckPGQ backend "
            "in DuckPGQ v1.3.1 — the extension does not expose an "
            "eigenvector table function. Use backend='networkx'."
        )

    import pandas as pd

    graph = _nx_digraph(edges, source_col, target_col)
    import networkx as nx

    try:
        scores = nx.eigenvector_centrality(graph)
    except nx.PowerIterationFailedConvergence:
        zero_nodes = {node: 0.0 for node in graph.nodes()}
        if len(graph) == 0:
            scores = zero_nodes
        else:
            in_degrees = dict(graph.in_degree())
            max_deg = max(in_degrees.values(), default=1) or 1
            scores = {node: in_degrees.get(node, 0) / max_deg for node in graph.nodes()}

    df = pd.DataFrame(
        [(node, float(score)) for node, score in scores.items()],
        columns=[node_id_col, "eigenvector"],
    ).sort_values("eigenvector", ascending=False, kind="mergesort").reset_index(drop=True)
    return _wrap_as_relation(df, con=con)


def degree_centrality(
    edges: DuckDBPyRelation,
    source_col: str,
    target_col: str,
    *,
    mode: Literal["in", "out", "total"] = "out",
    node_id_col: str = "node_id",
    con: duckdb.DuckDBPyConnection | None = None,
    backend: Literal["networkx", "duckpgq"] = "networkx",
) -> DuckDBPyRelation:
    """Degree centrality for every node.

    Normalized degree centrality is the fraction of possible nodes a
    node is connected to. ``mode="in"`` counts incoming edges (e.g.
    reports received), ``mode="out"`` counts outgoing edges (e.g.
    managers an employee reports to), and ``mode="total"`` counts both
    directions.

    Parameters
    ----------
    edges, source_col, target_col
    mode : {"in", "out", "total"}, default "out"
    node_id_col : str, default "node_id"
        Name of the node-id column in the returned relation.
    backend : {"networkx", "duckpgq"}, default "networkx"
        Algorithm backend. The DuckPGQ backend runs a pure DuckDB SQL
        aggregation over the staged vertex + edge tables; requires
        ``pip install pyduck-ona[graph]`` because the DuckPGQ extension
        must be loaded for the property-graph registration plumbing.

    Returns
    -------
    DuckDBPyRelation
        Columns ``(node_id_col, degree_centrality)`` sorted by degree DESC.
    """
    if backend == "duckpgq":
        from pyduck_ona.graph import _duckpgq_backend as _dg

        _require_duckpgq(con)
        conn = _duckpgq_dispatch_or_raise("degree_centrality", con)
        try:
            result = _dg.degree_centrality_duckpgq(
                edges,
                source_col,
                target_col,
                mode=mode,
                node_id_col=node_id_col,
                con=conn,
            )
            df = result.df() if _duckpgq_owned(conn) else None
            return _wrap_as_relation_safe(df) if df is not None else result
        finally:
            if _duckpgq_owned(conn):
                _dg._OWNED_CONS.discard(id(conn))
                conn.close()

    import pandas as pd

    graph = _nx_digraph(edges, source_col, target_col)
    import networkx as nx

    if mode == "in":
        scores = nx.in_degree_centrality(graph)
    elif mode == "out":
        scores = nx.out_degree_centrality(graph)
    elif mode == "total":
        scores = nx.degree_centrality(graph.to_undirected())
    else:
        raise ValueError(f"mode must be 'in', 'out', or 'total', got {mode!r}")

    df = pd.DataFrame(
        [(node, float(score)) for node, score in scores.items()],
        columns=[node_id_col, "degree_centrality"],
    ).sort_values("degree_centrality", ascending=False, kind="mergesort").reset_index(drop=True)
    return _wrap_as_relation(df, con=con)


def louvain_communities(
    edges: DuckDBPyRelation,
    source_col: str,
    target_col: str,
    *,
    weight_col: str | None = None,
    resolution: float = 1.0,
    node_id_col: str = "node_id",
    con: duckdb.DuckDBPyConnection | None = None,
    backend: Literal["networkx", "duckpgq"] = "networkx",
) -> DuckDBPyRelation:
    """Louvain community detection on the edge graph.

    Parameters
    ----------
    edges, source_col, target_col
    weight_col : str, optional
        Column holding edge weight. If None, all edges weight 1.
    resolution : float, default 1.0
        Louvain resolution parameter (higher = more / smaller communities).
    node_id_col : str, default "node_id"
        Name of the node-id column in the returned relation.
    backend : {"networkx", "duckpgq"}, default "networkx"
        Algorithm backend. DuckPGQ v1.3.1 does not expose a Louvain
        table function; selecting ``backend="duckpgq"`` raises
        :class:`ImportError`.

    Returns
    -------
    DuckDBPyRelation
        Columns ``(node_id_col, community_id)`` sorted by community_id,
        then node_id_col.
    """
    if backend == "duckpgq":
        raise ImportError(
            "louvain_communities() does not have a DuckPGQ backend in "
            "DuckPGQ v1.3.1 — the extension does not expose a Louvain "
            "table function. Use backend='networkx'."
        )

    import networkx as nx
    import pandas as pd

    if weight_col is None:
        graph = _nx_digraph(edges, source_col, target_col)
    else:
        result = edges.arrow()
        if hasattr(result, "read_all"):
            result = result.read_all()
        src = result.column(source_col).to_pylist()
        tgt = result.column(target_col).to_pylist()
        wgt = result.column(weight_col).to_pylist()
        graph = nx.DiGraph()
        for s, t, w in zip(src, tgt, wgt, strict=False):
            if s is None or t is None:
                continue
            if graph.has_edge(s, t):
                graph[s][t]["weight"] = graph[s][t].get("weight", 0.0) + float(w)
            else:
                graph.add_edge(s, t, weight=float(w))

    communities = nx.community.louvain_communities(
        graph.to_undirected(), resolution=resolution, seed=42
    )
    rows: list[tuple[Any, int]] = []
    for idx, members in enumerate(communities):
        for node in members:
            rows.append((node, idx))
    df = pd.DataFrame(rows, columns=[node_id_col, "community_id"]).sort_values(
        [node_id_col, "community_id"], kind="mergesort"
    ).reset_index(drop=True)
    return _wrap_as_relation(df, con=con)


# ─── Public DuckPGQ helpers ─────────────────────────────────────────────────
#
# Re-exported at the ``pyduck_ona.graph`` namespace so callers don't
# need to know about the internal ``_duckpgq_backend`` module path.
from pyduck_ona.graph._duckpgq_backend import (  # noqa: E402,F401,I001
    DEFAULT_DUCKPGQ_MIRROR as _DEFAULT_DUCKPGQ_MIRROR,
    ensure_duckpgq as _ensure_duckpgq,
    is_duckpgq_supported_duckdb as is_duckpgq_supported_duckdb,
)


def duckpgq_setup(
    con: duckdb.DuckDBPyConnection,
    *,
    mirror_url: str = _DEFAULT_DUCKPGQ_MIRROR,
    force_install: bool = False,
) -> None:
    """Install + LOAD DuckPGQ on a DuckDB connection.

    Convenience re-export of
    :func:`pyduck_ona.graph._duckpgq_backend.ensure_duckpgq`. Callers
    that want to confirm DuckPGQ is loaded before dispatching backend
    decisions can call this explicitly; otherwise
    :func:`_require_duckpgq` handles the install on demand.

    Parameters
    ----------
    con : DuckDBPyConnection
        The connection to install DuckPGQ on.
    mirror_url : str
        Extension mirror URL. Default is the official DuckPGQ S3
        bucket (``http://duckpgq.s3.eu-north-1.amazonaws.com``).
    force_install : bool, default False
        Re-download even if cached. Useful when switching between
        DuckDB versions.

    Raises
    ------
    ImportError
        If DuckDB version is outside the supported set (currently
        1.3.1), the extension cannot be reached, or the LOAD fails.
    """
    _ensure_duckpgq(con, mirror_url=mirror_url, force_install=force_install)
