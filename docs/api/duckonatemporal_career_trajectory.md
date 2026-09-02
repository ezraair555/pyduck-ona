# `DuckONATemporal.career_trajectory`

**Module:** `pyduck_ona.temporal`

## Signature

```python
DuckONATemporal.career_trajectory(self, employee_id'Any', lookback'str'='4Q')
```

## Description

Per-employee career path across periods

## Parameters

----------
employee_id : Any
lookback : str

## Returns

-------
pandas.DataFrame
    Columns: ``period, supervisor_id, job_level, department,
    promoted, transferred, manager_changed``.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
