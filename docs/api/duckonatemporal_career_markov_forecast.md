# `DuckONATemporal.career_markov_forecast`

**Module:** `pyduck_ona.temporal`

## Signature

```python
DuckONATemporal.career_markov_forecast(self, employee_id'Any', horizon'int'=2, state_col'str'='job_level', lookback'str'='8Q', by'str | None'='department')
```

## Description

Forecast future state probabilities for one employee via Markov transitions

## Parameters

----------
employee_id : Any
    Employee identifier.
horizon : int, default 2
    Number of forward periods to forecast.
state_col : str, default "job_level"
lookback : str, default "8Q"
by : str, optional
    Segment column used for segment-specific transition matrix.

## Returns

-------
pandas.DataFrame
    Columns: ``employee_id, step, state, probability, is_most_likely``.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
