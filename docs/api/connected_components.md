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
backend : {"networkx", "duckpgq"}, default "networkx"
    Algorithm backend. The DuckPGQ backend runs DuckPGQ's
    ``weakly_connected_component(graph, vlabel, elabel)`` on a
    registered property graph; requires
    ``pip install pyduck-ona[graph]``.

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
conventional but connectivity is what matters. In a healthy org
chart there should be exactly 1 component. More than 1 indicates
multiple top-level hierarchies (acquired companies, business
units, or — most often — data-quality issues).

---

[Back to API catalog](../README.md#api-catalog)
