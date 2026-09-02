# `ols`

**Module:** `pyduck_ona.stats`

## Signature

```python
ols(data'DuckDBPyRelation | pd.DataFrame', formula'str', cov_type'str'='nonrobust', alpha'float'=0.05)
```

## Description

Fit an OLS linear regression. Returns (tidy, glance)

## Parameters

----------
data
formula : str
    Patsy-style formula, e.g. ``"salary ~ team_size + tenure_yrs"``.
cov_type : str, default "nonrobust"
    Standard-error estimator (``"nonrobust"``, ``"HC1"``, ``"HC3"``,
    ``"cluster"``). Use ``"HC3"`` for heteroskedasticity-robust SE.
alpha : float, default 0.05
    Significance level for confidence intervals.

## Returns

-------
(tidy, glance) : tuple of pandas.DataFrame
    - tidy: per-coefficient table with estimate, std error, t-stat,
      p-value, conf.low, conf.high
    - glance: model-level summary (R², adj-R², AIC, BIC, F, df, nobs)

## Example

--------
>>> tidy, glance = ols(rel, "salary ~ team_size + tenure_yrs")
>>> tidy[tidy["p.value"] < 0.05]

---

[Back to API catalog](../README.md#api-catalog)
