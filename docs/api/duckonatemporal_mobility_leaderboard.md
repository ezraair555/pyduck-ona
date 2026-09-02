# `DuckONATemporal.mobility_leaderboard`

**Module:** `pyduck_ona.temporal`

## Signature

```python
DuckONATemporal.mobility_leaderboard(self, lookback'str'='4Q', top_n'int'=20, w_promotion'float'=1.0, w_lateral'float'=0.5, w_dept_change'float'=0.3, w_demotion'float'=-1.0)
```

## Description

Top movers by composite mobility score

## Parameters

----------
lookback : str
top_n : int
w_promotion, w_lateral, w_dept_change, w_demotion : float
    Weights for each mobility event type.

## Returns

-------
pandas.DataFrame

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
