# `hierarchy_long`

**Module:** `pyduck_ona.core`

## Signature

```python
hierarchy_long(df'duckdb.DuckDBPyRelation', employee_id'str', supervisor_id'str', max_depth'int'=50)
```

## Description

Unroll the org tree into long format via a recursive CTE

## Parameters

----------
df : duckdb.DuckDBPyRelation
employee_id, supervisor_id : str
max_depth : int, default 50
    Safety bound on recursion. If the org has more than `max_depth`
    levels, deeper ancestors will be silently truncated. 50 covers
    every realistic organization (Amazon has ~10).

## Returns

-------
duckdb.DuckDBPyRelation
    Columns: `(employee_id, supervisor_id, depth, path)`.
    - `depth`: 1 = direct manager, 2 = manager's manager, ...
    - `path`: Arrow-style "->" delimited ancestor chain ending at this
      supervisor (useful for debugging cycles visually).

## Example

--------
>>> long = hierarchy_long(rel, "emp_id", "mgr_id").df()
>>> long.head()
   employee_id supervisor_id  depth              path
0        E001          E010      1          E001->E010
1        E001          E005      2     E001->E010->E005

---

[Back to API catalog](../README.md#api-catalog)
