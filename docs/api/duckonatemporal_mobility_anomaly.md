# `DuckONATemporal.mobility_anomaly`

**Module:** `pyduck_ona.temporal`

## Signature

```python
DuckONATemporal.mobility_anomaly(self, lookback'str'='4Q', peer_basis'list[str] | None'=None, stuckness_threshold'float'=1.5)
```

## Description

Peer-relative stuckness z-score per employee

## Parameters

----------
lookback : str
peer_basis : list[str], optional
    Columns defining the peer group. Default:
    ``["job_level", "department"]`` from the first period.
stuckness_threshold : float, default 1.5
    Z-score above which an employee is flagged as "stuck".

## Returns

-------
pandas.DataFrame
    Columns: ``employee_id, starting_level, starting_dept,
    mobility_events, peer_median, peer_std, stuckness_zscore,
    is_stuck, is_mobility_leader``.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
