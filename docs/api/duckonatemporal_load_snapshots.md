# `DuckONATemporal.load_snapshots`

**Module:** `pyduck_ona.temporal`

## Signature

```python
DuckONATemporal.load_snapshots(self, df'pd.DataFrame', snapshot_date_col'str'='snapshot_date', employee_id_col'str'='employee_id', supervisor_id_col'str'='supervisor_id', freq'str'='Q', table_name'str'='snapshots')
```

## Description

Load HRIS snapshot data

## Parameters

----------
df : pandas.DataFrame
snapshot_date_col : str
employee_id_col : str
supervisor_id_col : str
freq : {"M", "Q", "Y"}
    Period frequency for all subsequent analysis.
table_name : str
    DuckDB table name for the registered snapshots.

## Returns

-------
list[str]
    Sorted period labels detected in the data.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
