# `DuckONA.filter_noise`

**Module:** `pyduck_ona.analysis`

## Signature

```python
DuckONA.filter_noise(df'pd.DataFrame', id_col'str'='employee_id', date_col'str | None'=None, test_ids'list[Any] | None'=None, bots'list[Any] | None'=None, min_records'int'=1)
```

## Description

Filter noise from an HR DataFrame

## Parameters

----------
df : pandas.DataFrame
id_col : str, default "employee_id"
date_col : str, optional
    If given, also drop rows where the date is missing/future.
test_ids : list, optional
    IDs to drop (e.g. test accounts).
bots : list, optional
    IDs to drop (e.g. service accounts).
min_records : int, default 1
    Minimum number of rows an ID must have to be retained.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
