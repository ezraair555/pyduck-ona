"""
Graph algorithms on org-chart relations.

Two backends are available:

  - **NetworkX (default)** — pure-Python, always available. Uses the
    edge relation built by :func:`pyduck_ona.core.hierarchy_long`. For
    org-chart-sized graphs (≤10⁴ employees) this is fast enough.

  - **DuckPGQ (optional)** — DuckDB-native property graph queries via
    the DuckPGQ community extension. Currently NOT installable from the
    DuckDB extension registry on most releases (the extension is in flux).
    The slot is reserved here so the API surface stays stable when it
    returns; calling with ``backend="duckpgq"`` raises a clear error
    pointing at the GitHub repo.

The four exported functions mirror DuckPGQ's algorithm vocabulary:

    - :func:`shortest_path`           path between two employees
    - :func:`betweenness`             broker detection
    - :func:`pagerank`                influence scoring
    - :func:`connected_components`    organizational silos

For any algo, pass a DuckDB relation of edges (typically the output of
:func:`pyduck_ona.core.hierarchy_long`) plus the column names. We then
build an in-memory NetworkX graph from the relation's Arrow buffer and
run the algorithm. Everything is wrapped as a DuckDB relation at the end
so the API is uniform across backends.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import duckdb

if TYPE_CHECKING:
    from duckdb import DuckDBPyRelation


_PGQ_NOT_INSTALLED_MSG = (
    "DuckPGQ is the optional DuckDB-native backend for "
    "pyduck_ona.graph.* algorithms. It is not currently installable "
    "from the DuckDB community-extension registry (HTTP 404). "
    "Until it ships again, use the default backend (NetworkX). "
    "See: https://github.com/duckdb/duckpgq"
)


def _require_duckpgq() -> None:
    """Raise if DuckPGQ extension is not loadable.

    The check is best-effort: it tries to LOAD the extension in a fresh
    in-memory connection. If the extension registry returns 404, we
    raise the same ``ImportError`` users would see if the package itself
    were missing, with install instructions.
    """
    try:
        con = duckdb.connect()
        con.execute("LOAD duckpgq;")
    except Exception as e:
        raise ImportError(_PGQ_NOT_INSTALLED_MSG) from e


# ─── Shared helpers ─────────────────────────────────────────────────────────

def _edges_arrow(
    edges: "DuckDBPyRelation",
    source_col: str,
    target_col: str,
) -> tuple[list, list]:
    """Materialize the edge relation to (source_list, target_list).

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


def _nx_digraph(edges: "DuckDBPyRelation", source_col: str, target_col: str):
    """Build a NetworkX DiGraph from an edge relation."""
    import networkx as nx

    src, tgt = _edges_arrow(edges, source_col, target_col)
    G = nx.DiGraph()
    G.add_edges_from(zip(src, tgt))
    return G


def _wrap_as_relation(df) -> "DuckDBPyRelation":
    """Round-trip a pandas DataFrame through DuckDB to get a relation.

    Used so every graph function returns the same type regardless of
    whether the answer came from NX or DuckPGQ.
    """
    return duckdb.sql("SELECT * FROM df")


# ─── Algorithms ─────────────────────────────────────────────────────────────

def shortest_path(
    edges: "DuckDBPyRelation",
    source_col: str,
    target_col: str,
    source: str,
    target: str,
    *,
    backend: Literal["networkx", "duckpgq"] = "networkx",
) -> "DuckDBPyRelation":
    """Shortest path between two nodes in the edge graph.

    Parameters
    ----------
    edges : DuckDBPyRelation
        Edge relation. Typically the output of
        :func:`pyduck_ona.core.hierarchy_long`.
    source_col, target_col : str
        Column names in ``edges`` holding the source and target of each
        directed edge.
    source, target : str
        The two node IDs to find a path between.
    backend : {"networkx", "duckpgq"}, default "networkx"
        Algorithm backend. DuckPGQ is not currently installable from
        the community registry; selecting it raises ``ImportError``.

    Returns
    -------
    DuckDBPyRelation
        One row with columns ``(source, target, path_length, path)``.
        ``path`` is a ``->``-delimited sequence. If no path exists,
        ``path_length`` is NULL and ``path`` is empty.

    Examples
    --------
    >>> long = hierarchy_long(rel, "emp_id", "mgr_id")
    >>> shortest_path(long, "employee_id", "supervisor_id", "E001", "E999").df()

    Notes
    -----
    When ``source == target``, returns ``path_length=0`` and
    ``path=<source>`` (the trivial self-path). This is by design: a
    distance-to-self of zero is the standard graph-theory convention.
    If you need a different definition, filter upstream.
    """
    import pandas as pd

    if backend == "duckpgq":
        _require_duckpgq()
        # Reserved slot — would dispatch to DuckPGQ here when it ships.
        raise NotImplementedError(
            "DuckPGQ backend not yet implemented; use backend='networkx'"
        )

    G = _nx_digraph(edges, source_col, target_col)
    import networkx as nx

    if source not in G or target not in G:
        # Source or target not in graph at all → no path possible.
        result_df = pd.DataFrame(
            [(source, target, None, "")],
            columns=["source", "target", "path_length", "path"],
        )
        return _wrap_as_relation(result_df)

    try:
        node_path = nx.shortest_path(G, source=source, target=target)
        path_str = "->".join(str(n) for n in node_path)
        length = len(node_path) - 1  # edges in path
    except nx.NetworkXNoPath:
        length = None
        path_str = ""

    result_df = pd.DataFrame(
        [(source, target, length, path_str)],
        columns=["source", "target", "path_length", "path"],
    )
    return _wrap_as_relation(result_df)


def betweenness(
    edges: "DuckDBPyRelation",
    source_col: str,
    target_col: str,
    *,
    backend: Literal["networkx", "duckpgq"] = "networkx",
) -> "DuckDBPyRelation":
    """Betweenness centrality for every node (broker detection).

    High betweenness = the employee sits on many shortest paths between
    other pairs = information broker. Removing them would disconnect
    parts of the org.

    Parameters
    ----------
    edges, source_col, target_col
        Edge relation and column names.
    backend : {"networkx", "duckpgq"}

    Returns
    -------
    DuckDBPyRelation
        Columns ``(node_id, betweenness)`` sorted by betweenness DESC.
        In an org chart the CEO dominates (sits on every path); in a
        collaboration network top collaborators rise even if not senior.
    """
    import pandas as pd

    if backend == "duckpgq":
        _require_duckpgq()
        raise NotImplementedError(
            "DuckPGQ backend not yet implemented; use backend='networkx'"
        )

    G = _nx_digraph(edges, source_col, target_col)
    import networkx as nx

    scores = nx.betweenness_centrality(G)
    df = pd.DataFrame(
        [(node, float(score)) for node, score in scores.items()],
        columns=["node_id", "betweenness"],
    ).sort_values("betweenness", ascending=False, kind="mergesort")
    return _wrap_as_relation(df)


def pagerank(
    edges: "DuckDBPyRelation",
    source_col: str,
    target_col: str,
    *,
    damping: float = 0.85,
    backend: Literal["networkx", "duckpgq"] = "networkx",
) -> "DuckDBPyRelation":
    """PageRank centrality (influence scoring).

    In a formal org chart the root dominates; in a collaboration network
    top collaborators rise even if they're not senior.

    Parameters
    ----------
    edges, source_col, target_col
    damping : float, default 0.85
        Standard PageRank damping factor (probability that a random walk
        follows a link vs. teleports to a random node).
    backend : {"networkx", "duckpgq"}

    Returns
    -------
    DuckDBPyRelation
        Columns ``(node_id, pagerank)`` sorted by pagerank DESC.
    """
    import pandas as pd

    if backend == "duckpgq":
        _require_duckpgq()
        raise NotImplementedError(
            "DuckPGQ backend not yet implemented; use backend='networkx'"
        )

    G = _nx_digraph(edges, source_col, target_col)
    import networkx as nx

    scores = nx.pagerank(G, alpha=damping)
    df = pd.DataFrame(
        [(node, float(score)) for node, score in scores.items()],
        columns=["node_id", "pagerank"],
    ).sort_values("pagerank", ascending=False, kind="mergesort")
    return _wrap_as_relation(df)


def connected_components(
    edges: "DuckDBPyRelation",
    source_col: str,
    target_col: str,
    *,
    backend: Literal["networkx", "duckpgq"] = "networkx",
) -> "DuckDBPyRelation":
    """Weakly-connected components in the edge graph.

    In a healthy org chart there should be exactly 1 component. More than
    1 indicates multiple top-level hierarchies (acquired companies,
    business units, or — most often — data-quality issues).

    Parameters
    ----------
    edges, source_col, target_col
    backend : {"networkx", "duckpgq"}

    Returns
    -------
    DuckDBPyRelation
        Columns ``(component_id, size, members)`` sorted by size DESC.
        ``members`` is a list of node IDs in that component.

    Notes
    -----
    "Weakly connected" treats the graph as undirected for component
    purposes — appropriate for org charts where up/down direction is
    conventional but connectivity is what matters.
    """
    import pandas as pd

    if backend == "duckpgq":
        _require_duckpgq()
        raise NotImplementedError(
            "DuckPGQ backend not yet implemented; use backend='networkx'"
        )

    G = _nx_digraph(edges, source_col, target_col)
    import networkx as nx

    components = list(nx.weakly_connected_components(G))
    # Sort by size DESC, stable so ties keep insertion order
    components.sort(key=len, reverse=True)
    rows = [
        (int(idx), len(members), sorted(members))
        for idx, members in enumerate(components)
    ]
    df = pd.DataFrame(rows, columns=["component_id", "size", "members"])
    return _wrap_as_relation(df)