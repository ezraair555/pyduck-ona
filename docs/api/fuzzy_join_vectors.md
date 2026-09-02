# `fuzzy_join_vectors`

**Module:** `pyduck_ona.search`

## Signature

```python
fuzzy_join_vectors(left_table'str', right_table'str', left_col'str', right_col'str', k'int'=5, metric'str'='l2sq', con'DuckDBPyConnection | None'=None)
```

## Description

Approximate nearest-neighbor join between two embedding tables

## Parameters

----------
left_table, right_table
    Names of the two tables to join.
left_col, right_col
    Embedding array columns in each table.
k
    Number of neighbors per left row.
metric
    Distance metric.
con
    Existing DuckDB connection.

## Returns

-------
DuckDBPyRelation
    Relation with left id, right id, and distance.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
