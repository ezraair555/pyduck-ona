# `DuckONATemporal.change_detection`

**Module:** `pyduck_ona.temporal`

## Signature

```python
DuckONATemporal.change_detection(self, metric'str'='betweenness', top_n'int'=20, lookback'str'='4Q')
```

## Description

Top movers for a given metric over the lookback window

## Parameters

----------
metric : str
    ONA metric name (betweenness, pagerank, etc.).
top_n : int
    Number of top movers to return.
lookback : str
    e.g. "4Q", "12M", "3Y".

## Returns

-------
pandas.DataFrame
    Columns: ``employee_id, start_value, end_value, delta,
    pct_change, rolling_zscore, rank``.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
