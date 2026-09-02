# `louvain_communities`

**Module:** `pyduck_ona.graph`

## Signature

```python
louvain_communities(edges'DuckDBPyRelation', source_col'str', target_col'str', weight_col'str | None'=None, resolution'float'=1.0, node_id_col'str'='node_id', con'duckdb.DuckDBPyConnection | None'=None, backend"Literal['networkx', 'duckpgq']"='networkx')
```

## Description

Louvain community detection on the edge graph

## Parameters

----------
edges, source_col, target_col
weight_col : str, optional
    Column holding edge weight. If None, all edges weight 1.
resolution : float, default 1.0
    Louvain resolution parameter (higher = more / smaller communities).
node_id_col : str, default "node_id"
    Name of the node-id column in the returned relation.
backend : {"networkx", "duckpgq"}

## Returns

-------
DuckDBPyRelation
    Columns ``(node_id_col, community_id)`` sorted by community_id, then
    node_id_col.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
