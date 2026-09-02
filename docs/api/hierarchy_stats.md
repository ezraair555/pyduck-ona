# `hierarchy_stats`

**Module:** `pyduck_ona.core`

## Signature

```python
hierarchy_stats(df'duckdb.DuckDBPyRelation', employee_id'str', supervisor_id'str', max_depth'int'=50)
```

## Description

Calculate span-of-control metrics for every manager

## Parameters

----------
df : duckdb.DuckDBPyRelation
employee_id, supervisor_id : str
max_depth : int, default 50

## Returns

-------
duckdb.DuckDBPyRelation
    Columns: `(manager_id, direct_reports, indirect_reports,
    total_reports, team_size, levels_below)`.

## Example

--------
>>> stats = hierarchy_stats(rel, "emp_id", "mgr_id").df()
>>> stats.sort_values("direct_reports", ascending=False).head()
   manager_id  direct_reports  indirect_reports  total_reports  team_size  levels_below
0        E010              12                47             59         59             5

---

[Back to API catalog](../README.md#api-catalog)
