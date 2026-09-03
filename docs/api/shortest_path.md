# `shortest_path`

**Module:** `pyduck_ona.graph`

## Signature

```python
shortest_path(edges'DuckDBPyRelation', source_col'str', target_col'str', source'str', target'str', con'duckdb.DuckDBPyConnection | None'=None, backend"Literal['networkx', 'duckpgq']"='networkx')
```

## Description

Shortest path between two nodes in the edge graph

## Parameters

----------
edges : DuckDBPyRelation
    Edge relation. Typically the output of
    :func:`pyduck_ona.core.hierarchy_long`.
source_col, target_col : str
    Column names in ``edges`` holding the source and target of
    each directed edge.
source, target : str
    The two node IDs to find a path between.
backend : {"networkx", "duckpgq"}, default "networkx"
    Algorithm backend. DuckPGQ v1.3.1 does not expose a multi-hop
    shortest-path table function; selecting ``backend="duckpgq"``
    raises :class:`ImportError`.

## Returns

-------
DuckDBPyRelation
    One row with columns ``(source, target, path_length, path)``.
    ``path`` is a ``->``-delimited sequence. If no path exists,
    ``path_length`` is NULL and ``path`` is empty.

## Example

--------
>>> long = hierarchy_long(rel, "emp_id", "mgr_id")
>>> shortest_path(long, "employee_id", "supervisor_id",
...               "E001", "E999").df()

## Notes

-----
When ``source == target``, returns ``path_length=0`` and
``path=<source>`` (the trivial self-path). This is by design: a
distance-to-self of zero is the standard graph-theory convention.
If you need a different definition, filter upstream.

---

[Back to API catalog](../README.md#api-catalog)
