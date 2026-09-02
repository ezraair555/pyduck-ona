# `connected_components`

**Module:** `pyduck_ona.graph`

## Signature

```python
connected_components(edges'DuckDBPyRelation', source_col'str', target_col'str', con'duckdb.DuckDBPyConnection | None'=None, backend"Literal['networkx', 'duckpgq']"='networkx')
```

## Description

Weakly-connected components in the edge graph

## Parameters

----------
edges, source_col, target_col
backend : {"networkx", "duckpgq"}

## Returns

-------
DuckDBPyRelation
    Columns ``(component_id, size, members)`` sorted by size DESC.
    ``members`` is a list of node IDs in that component.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

## Notes

-----
"Weakly connected" treats the graph as undirected for component
purposes — appropriate for org charts where up/down direction is
conventional but connectivity is what matters.

---

[Back to API catalog](../README.md#api-catalog)
