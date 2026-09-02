# `DuckONATemporal.org_design_scorecard`

**Module:** `pyduck_ona.temporal`

## Signature

```python
DuckONATemporal.org_design_scorecard(self, lookback'str'='8Q')
```

## Description

Per-period organizational design metrics and a composite score

## Parameters

----------
lookback : str, default "8Q"

## Returns

-------
pandas.DataFrame
    Columns include span/load/layering/silo metrics and
    ``org_design_score`` (0-100, higher is healthier).

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
