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
backend : {"networkx", "duckpgq"}, default "networkx"
    Algorithm backend. DuckPGQ v1.3.1 does not expose a
    betweenness table function; selecting ``backend="duckpgq"``
    raises :class:`ImportError`.

## Returns

-------
DuckDBPyRelation
    Columns ``(node_id_col, betweenness)`` sorted by betweenness DESC.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
