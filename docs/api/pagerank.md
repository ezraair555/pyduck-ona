# `pagerank`

**Module:** `pyduck_ona.graph`

## Signature

```python
pagerank(edges'DuckDBPyRelation', source_col'str', target_col'str', damping'float'=0.85, node_id_col'str'='node_id', con'duckdb.DuckDBPyConnection | None'=None, backend"Literal['networkx', 'duckpgq']"='networkx')
```

## Description

PageRank centrality (influence scoring)

## Parameters

----------
edges, source_col, target_col
damping : float, default 0.85
    Standard PageRank damping factor (probability that a random walk
    follows a link vs. teleports to a random node).
node_id_col : str, default "node_id"
    Name of the node-id column in the returned relation.
backend : {"networkx", "duckpgq"}

## Returns

-------
DuckDBPyRelation
    Columns ``(node_id_col, pagerank)`` sorted by pagerank DESC.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
