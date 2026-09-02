# `DuckONATemporal.event_window`

**Module:** `pyduck_ona.temporal`

## Signature

```python
DuckONATemporal.event_window(self, event_date'str', pre_window'tuple[str, str] | None'=None, post_window'tuple[str, str] | None'=None, metrics'list[str] | None'=None)
```

## Description

Before/after comparison around a specific event date

## Parameters

----------
event_date : str
    ISO date. Periods before this date are "pre"; after are "post".
pre_window, post_window : tuple(str, str), optional
    (start_date, end_date) to bound the pre/post windows. If
    omitted, all periods before / after event_date are used.
metrics : list[str], optional
    ONA metrics to compute. Default: betweenness + pagerank.

## Returns

-------
pandas.DataFrame
    Columns: ``period, period_type ("pre"|"post"), employee_id,
    metric, value``.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
