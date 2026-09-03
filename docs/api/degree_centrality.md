# `degree_centrality`

**Module:** `pyduck_ona.graph`

## Signature

```python
degree_centrality(edges'DuckDBPyRelation', source_col'str', target_col'str', mode"Literal['in', 'out', 'total']"='out', node_id_col'str'='node_id', con'duckdb.DuckDBPyConnection | None'=None, backend"Literal['networkx', 'duckpgq']"='networkx')
```

## Description

Degree centrality for every node

## Parameters

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

## Returns

-------
DuckDBPyRelation
    Columns ``(node_id_col, degree_centrality)`` sorted by degree DESC.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
