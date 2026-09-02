# `DuckONA.join_hris`

**Module:** `pyduck_ona.analysis`

## Signature

```python
DuckONA.join_hris(self, metrics_rel'DuckDBPyRelation', metrics_id_col'str'='node_id', hris_id_col'str'='employee_id', hris_table'str'='hris')
```

## Description

Join a metric relation back to the HRIS demographics table

## Parameters

----------
metrics_rel : DuckDBPyRelation
    Relation containing per-employee network or model metrics,
    e.g. the output of ``betweenness()``.
metrics_id_col : str, default "node_id"
    Column in ``metrics_rel`` holding the employee identifier.
hris_id_col : str, default "employee_id"
    Column in the HRIS table holding the employee identifier.
hris_table : str, default "hris"
    Name of the HRIS table registered on ``self.con``.

## Returns

-------
DuckDBPyRelation
    A left join of HRIS onto the metric relation so every metric
    row keeps its network scores and gains demographic columns.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
