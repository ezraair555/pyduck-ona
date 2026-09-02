# `hierarchy_wide`

**Module:** `pyduck_ona.core`

## Signature

```python
hierarchy_wide(df'duckdb.DuckDBPyRelation', employee_id'str', supervisor_id'str', max_depth'int'=15, level_prefix'str'='Level_')
```

## Description

Flatten the reporting chain into a single row per employee

## Parameters

----------
df : duckdb.DuckDBPyRelation
employee_id, supervisor_id : str
max_depth : int, default 15
    Number of level-columns to produce. Raises if exceeded.
level_prefix : str, default "Level_"
    Prefix for generated column names. Output columns will be
    `{prefix}1`, `{prefix}2`, ..., `{prefix}{max_depth}`. The prefix
    is validated as a safe identifier prefix.

## Returns

-------
duckdb.DuckDBPyRelation
    Columns: `{employee_id}, {level_prefix}1, ..., {level_prefix}{max_depth}`.

## Example

--------
>>> wide = hierarchy_wide(rel, "emp_id", "mgr_id", max_depth=5).df()
>>> wide.head()
   emp_id    Level_1    Level_2    Level_3   Level_4   Level_5
0   E001        E010        E005       E002      E001      None

## Notes

-----
Implemented as PIVOT over the long-format chain. DuckDB's PIVOT
requires explicit value columns, so we generate the pivot IN-list
programmatically from `max_depth` and the validated `level_prefix`.

---

[Back to API catalog](../README.md#api-catalog)
