# `eigenvector_centrality`

**Module:** `pyduck_ona.graph`

## Signature

```python
eigenvector_centrality(edges'DuckDBPyRelation', source_col'str', target_col'str', node_id_col'str'='node_id', con'duckdb.DuckDBPyConnection | None'=None, backend"Literal['networkx', 'duckpgq']"='networkx')
```

## Description

Eigenvector centrality for every node

## Parameters

----------
edges, source_col, target_col
node_id_col : str, default "node_id"
    Name of the node-id column in the returned relation.
backend : {"networkx", "duckpgq"}, default "networkx"
    Algorithm backend. DuckPGQ v1.3.1 does not expose an
    eigenvector-centrality table function; selecting
    ``backend="duckpgq"`` raises :class:`ImportError`.

## Returns

-------
DuckDBPyRelation
    Columns ``(node_id_col, eigenvector)`` sorted by eigenvector DESC.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
