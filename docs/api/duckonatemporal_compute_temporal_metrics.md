# `DuckONATemporal.compute_temporal_metrics`

**Module:** `pyduck_ona.temporal`

## Signature

```python
DuckONATemporal.compute_temporal_metrics(self, metrics'list[str] | None'=None, employee_id_col'str | None'=None, supervisor_id_col'str | None'=None)
```

## Description

Per-employee ONA metric time-series across all periods

## Parameters

----------
metrics : list[str], optional
    Metric names. Default: ``["betweenness", "pagerank",
    "degree_centrality", "team_size"]``.
employee_id_col, supervisor_id_col : str, optional
    Override the column names set by ``load_snapshots``.

## Returns

-------
pandas.DataFrame
    Columns: ``period, employee_id, metric, value, prev_value,
    delta, pct_change``.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
