# `DuckONATemporal.org_design_change_alerts`

**Module:** `pyduck_ona.temporal`

## Signature

```python
DuckONATemporal.org_design_change_alerts(self, lookback'str'='8Q', span_shift_threshold'int'=3, component_growth_threshold'float'=0.25)
```

## Description

Flag periods with potentially unhealthy organizational-design shifts

## Parameters

----------
lookback : str, default "8Q"
span_shift_threshold : int, default 3
    Trigger when >= this many managers change direct reports by 2+.
component_growth_threshold : float, default 0.25
    Trigger when weak components grow by this fraction period-over-period.

## Returns

-------
pandas.DataFrame
    One row per period transition with alert flags and severity.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
