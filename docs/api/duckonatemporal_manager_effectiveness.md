# `DuckONATemporal.manager_effectiveness`

**Module:** `pyduck_ona.temporal`

## Signature

```python
DuckONATemporal.manager_effectiveness(self, lookback'str'='4Q', w_engagement'float'=0.5, w_retention'float'=0.25, w_promotion'float'=0.15, w_span'float'=0.1, survey_table'str | None'=None, promotions_table'str | None'=None)
```

## Description

Composite manager effectiveness score

## Parameters

----------
lookback : str
w_engagement, w_retention, w_promotion, w_span : float
    Composite weights. Must sum to 1.0.
survey_table : str, optional
    Name of the registered survey table. If None, looks for
    "survey" in extra tables.
promotions_table : str, optional
    Name of the registered promotions table.

## Returns

-------
pandas.DataFrame
    Columns: ``manager_id, manager_level, n_periods_active,
    team_engagement_t1, team_engagement_tn, engagement_trend,
    retention_rate, promotion_rate, span_efficiency,
    peer_engagement_trend, peer_retention_rate,
    peer_promotion_rate, effectiveness_score, rank``.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
