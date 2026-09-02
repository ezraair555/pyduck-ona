# `DuckONA.deduplicate`

**Module:** `pyduck_ona.analysis`

## Signature

```python
DuckONA.deduplicate(df'pd.DataFrame', id_col'str'='employee_id', date_col'str | None'=None, keep"Literal['first', 'last', False]"='last')
```

## Description

Deduplicate an HR DataFrame by ``(id_col, date_col)``

## Parameters

----------
df : pandas.DataFrame
id_col : str, default "employee_id"
date_col : str, optional
    If given, deduplicate on the combination; otherwise on
    ``id_col`` alone.
keep : {"first", "last"}, default "last"
    Which duplicate row to retain.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
