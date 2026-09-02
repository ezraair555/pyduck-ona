# `DuckONA.build_org_edges`

**Module:** `pyduck_ona.analysis`

## Signature

```python
DuckONA.build_org_edges(self, employee_id_col'str'='employee_id', supervisor_id_col'str'='supervisor_id', active_as_of'date | str | None'=None, table_name'str'='hris')
```

## Description

Build a directed edge relation from the HRIS hierarchy

## Parameters

----------
employee_id_col, supervisor_id_col : str
    Column names in the HRIS table.
active_as_of : date or str, optional
    If the HRIS table has a ``snapshot_date`` / ``effective_date``
    column, filter to rows active as of this date. Not required
    for single-snapshot HRIS tables.
table_name : str, default "hris"
    Source table name.

## Returns

-------
DuckDBPyRelation
    Columns ``(employee_id, supervisor_id)`` for every non-NULL
    supervisor edge. This is the correct input for graph metrics
    such as betweenness and PageRank.

## Example

--------
>>> edges = ona.build_org_edges("emp_id", "mgr_id")
>>> metrics = ona.betweenness(edges, "emp_id", "mgr_id")

---

[Back to API catalog](../README.md#api-catalog)
