# `hierarchy_valid`

**Module:** `pyduck_ona.core`

## Signature

```python
hierarchy_valid(df'duckdb.DuckDBPyRelation', employee_id'str', supervisor_id'str')
```

## Description

Diagnose the integrity of an organizational reporting structure

## Parameters

----------
df : duckdb.DuckDBPyRelation
    Input relation with at minimum two columns.
employee_id : str
    Name of the column holding unique employee identifiers.
supervisor_id : str
    Name of the column holding the supervisor's employee identifier
    (NULL/empty for the top of the hierarchy).

## Returns

-------
duckdb.DuckDBPyRelation
    Relation with columns `(issue_type, employee_id, detail)`. One row
    per detected issue. Empty relation if the hierarchy is clean.

## Example

--------
>>> import duckdb
>>> rel = duckdb.from_df(some_pandas_df)
>>> issues = hierarchy_valid(rel, "employee_id", "supervisor_id")
>>> issues.fetchall()
[('multiple_roots', None, 'Found 2 employees with no supervisor'),
 ('loop', 'emp_42', 'Cycle detected at depth 4')]

---

[Back to API catalog](../README.md#api-catalog)
