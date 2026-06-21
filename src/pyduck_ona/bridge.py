"""Bridge layer: export DuckDB relations to NetworkX / igraph via Arrow.

We use zero-copy Apache Arrow transfers (`.arrow()`) for the heavy edge
transfer, then build the in-memory graph in one shot. This is 10-100x
faster than fetching row-by-row or building pandas DataFrames first.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb

if TYPE_CHECKING:
    from duckdb import DuckDBPyRelation
    import networkx as nx
    import igraph as ig


def _to_arrow_table(rel: "DuckDBPyRelation"):
    """Materialize a DuckDB relation into a pyarrow.Table.

    DuckDB >=1.3 returns a streaming ``RecordBatchReader`` from
    ``rel.arrow()`` for zero-copy transfer — but streaming objects don't
    expose ``.column()``. We materialize the full result here; for the
    edge-count workloads this package targets, it's a wash between the
    two approaches. If callers need true streaming, they can use
    ``rel.arrow()`` directly.
    """
    result = rel.arrow()
    # DuckDB 1.3+ returns a RecordBatchReader; older versions return Table.
    if hasattr(result, "read_all"):
        return result.read_all()
    return result


def to_networkx(
    edges: "DuckDBPyRelation",
    source_col: str,
    target_col: str,
    weight_col: str | None = None,
    graph_type: str = "DiGraph",
    node_attrs: "DuckDBPyRelation | None" = None,
    node_id_col: str = "node_id",
) -> "nx.Graph | nx.DiGraph":
    """Convert an edge relation into a NetworkX graph via Arrow.

    Parameters
    ----------
    edges : DuckDBPyRelation
        Relation with at minimum `source_col` and `target_col`.
    source_col, target_col : str
        Names of columns holding source and target node IDs.
    weight_col : str, optional
        Name of column holding edge weight. If None, all edges get weight 1.
    graph_type : {"Graph", "DiGraph"}, default "DiGraph"
        Whether the graph is directed. People analytics almost always
        wants "DiGraph" (manager → report).
    node_attrs : DuckDBPyRelation, optional
        Optional node-attribute relation with columns `node_id_col` plus
        arbitrary additional columns. These are merged into node data.
    node_id_col : str, default "node_id"
        Column name in `node_attrs` that matches edge source/target values.

    Returns
    -------
    networkx.Graph or networkx.DiGraph

    Examples
    --------
    >>> edges = duckdb.sql("SELECT manager_id AS src, report_id AS dst FROM chain")
    >>> G = to_networkx(edges, "src", "dst", weight_col="interaction_count")
    >>> print(G.number_of_edges(), G.number_of_nodes())

    Notes
    -----
    Duplicate edges (same source + target appearing more than once) are
    silently merged: the last row's attributes win. If you need to
    preserve duplicate edges, use ``nx.MultiDiGraph`` instead, or
    deduplicate upstream with ``SELECT DISTINCT source, target, weight``.
    Self-loops are preserved as a single edge with both endpoints equal.
    """
    import networkx as nx

    if graph_type == "DiGraph":
        G = nx.DiGraph()
    elif graph_type == "Graph":
        G = nx.Graph()
    else:
        raise ValueError(f"graph_type must be 'Graph' or 'DiGraph', got {graph_type!r}")

    # ── Edges: zero-copy Arrow → Python list ──
    arrow_table = _to_arrow_table(edges)
    src_data = arrow_table.column(source_col).to_pylist()
    tgt_data = arrow_table.column(target_col).to_pylist()
    weight_data = (
        arrow_table.column(weight_col).to_pylist() if weight_col else None
    )

    if weight_data is not None:
        for s, t, w in zip(src_data, tgt_data, weight_data):
            G.add_edge(s, t, weight=w)
    else:
        # NetworkX's add_edges_from is the fastest path for unweighted
        edge_list = list(zip(src_data, tgt_data))
        G.add_edges_from(edge_list)

    # ── Optional node attributes ──
    if node_attrs is not None:
        attr_table = _to_arrow_table(node_attrs)
        cols = attr_table.column_names
        if node_id_col not in cols:
            raise ValueError(
                f"node_id_col={node_id_col!r} not found in node_attrs. "
                f"Available: {list(cols)}"
            )
        ids = attr_table.column(node_id_col).to_pylist()
        for idx, nid in enumerate(ids):
            if nid not in G:
                G.add_node(nid)
            for col in cols:
                if col == node_id_col:
                    continue
                val = attr_table.column(col)[idx].as_py()
                G.nodes[nid][col] = val

    return G


def to_igraph(
    edges: "DuckDBPyRelation",
    source_col: str,
    target_col: str,
    weight_col: str | None = None,
    directed: bool = True,
    node_attrs: "DuckDBPyRelation | None" = None,
    node_id_col: str = "node_id",
) -> "ig.Graph":
    """Convert an edge relation into an igraph.Graph via Arrow.

    igraph is preferred over NetworkX for:
      - Large graphs (>100k edges): 5-10x faster algorithms
      - Community detection (Leiden, Louvain implementations)
      - Statistical inference (ERGM-style models)

    Parameters mirror `to_networkx`. Returns an `igraph.Graph`.
    """
    import igraph as ig

    arrow_table = _to_arrow_table(edges)
    src_data = arrow_table.column(source_col).to_pylist()
    tgt_data = arrow_table.column(target_col).to_pylist()

    # igraph wants vertices enumerated 0..n-1, then an edge list over those.
    nodes = sorted(set(src_data) | set(tgt_data))
    node_index = {n: i for i, n in enumerate(nodes)}

    edge_list = [(node_index[s], node_index[t]) for s, t in zip(src_data, tgt_data)]

    g = ig.Graph(n=len(nodes), edges=edge_list, directed=directed)
    g.vs["name"] = nodes

    if weight_col is not None:
        g.es["weight"] = arrow_table.column(weight_col).to_pylist()

    if node_attrs is not None:
        attr_table = _to_arrow_table(node_attrs)
        cols = attr_table.column_names
        if node_id_col not in cols:
            raise ValueError(
                f"node_id_col={node_id_col!r} not found in node_attrs. "
                f"Available: {list(cols)}"
            )
        ids = attr_table.column(node_id_col).to_pylist()
        # Build per-node dict of attrs
        attr_by_id: dict = {ids[i]: {} for i in range(len(ids))}
        for col in cols:
            if col == node_id_col:
                continue
            col_data = attr_table.column(col).to_pylist()
            for i, nid in enumerate(ids):
                attr_by_id[nid][col] = col_data[i]
        # Assign each attr as a vertex attribute (igraph needs parallel lists)
        if attr_by_id:
            all_attr_keys = list(next(iter(attr_by_id.values())).keys())
            for key in all_attr_keys:
                g.vs[key] = [attr_by_id.get(node, {}).get(key) for node in nodes]

    return g