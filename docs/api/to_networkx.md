# `to_networkx`

**Module:** `pyduck_ona.bridge`

## Signature

```python
to_networkx(edges'DuckDBPyRelation', source_col'str', target_col'str', weight_col'str | None'=None, graph_type'str'='DiGraph', node_attrs'DuckDBPyRelation | None'=None, node_id_col'str'='node_id')
```

## Description

Convert an edge relation into a NetworkX graph via Arrow

## Parameters

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

## Returns

-------
networkx.Graph or networkx.DiGraph

## Example

--------
>>> edges = duckdb.sql("SELECT manager_id AS src, report_id AS dst FROM chain")
>>> G = to_networkx(edges, "src", "dst", weight_col="interaction_count")
>>> print(G.number_of_edges(), G.number_of_nodes())

## Notes

-----
Duplicate edges (same source + target appearing more than once) are
silently merged: the last row's attributes win. If you need to
preserve duplicate edges, use ``nx.MultiDiGraph`` instead, or
deduplicate upstream with ``SELECT DISTINCT source, target, weight``.
Self-loops are preserved as a single edge with both endpoints equal.

---

[Back to API catalog](../README.md#api-catalog)
