# `DuckONA.validate_keys`

**Module:** `pyduck_ona.analysis`

## Signature

```python
DuckONA.validate_keys(self, table_name'str', employee_id_col'str'='employee_id', date_col'str | None'=None, date_lower'date | str | None'=None, date_upper'date | str | None'=None)
```

## Description

Validate HR table keys: non-null IDs, no duplicate snapshots, sensible dates

## Parameters

----------
table_name : str
    Name of the registered HR table to validate.
employee_id_col : str, default "employee_id"
    Column holding the employee identifier.
date_col : str, optional
    Snapshot/effective-date column. If given, ``employee_id``
    must be unique per date and dates must not be in the future.
allow_null_supervisor : bool, default True
    Reserved for future hierarchy-aware validation; currently
    has no effect because this method validates keys/dates only.
date_lower, date_upper : date or str, optional
    Inclusive bounds for ``date_col`` values. Strings are parsed
    as ISO dates.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
