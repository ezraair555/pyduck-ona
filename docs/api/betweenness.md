# `betweenness`

**Module:** `pyduck_ona.graph`

## Signature

```python
betweenness(edges'DuckDBPyRelation', source_col'str', target_col'str', node_id_col'str'='node_id', con'duckdb.DuckDBPyConnection | None'=None, backend"Literal['networkx', 'duckpgq']"='networkx')
```

## Description

Betweenness centrality for every node (broker detection)

## Parameters

----------
edges, source_col, target_col
    Edge relation and column names.
node_id_col : str, default "node_id"
    Name of the node-id column in the returned relation.
backend : {"networkx", "duckpgq"}

## Returns

-------
DuckDBPyRelation
    Columns ``(node_id_col, betweenness)`` sorted by betweenness DESC.
    In an org chart the CEO dominates (sits on every path); in a
    collaboration network top collaborators rise even if not senior.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
