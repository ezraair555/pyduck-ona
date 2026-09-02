# `logistic`

**Module:** `pyduck_ona.stats`

## Signature

```python
logistic(data'DuckDBPyRelation | pd.DataFrame', formula'str', cov_type'str'='nonrobust', alpha'float'=0.05)
```

## Description

Fit a logistic regression. Returns (tidy, glance)

## Parameters

----------
data
formula : str
    Patsy formula, e.g. ``"attrition ~ team_size + salary"``. The
    outcome must be binary (0/1) or interpretable as such.
cov_type : str, default "nonrobust"
alpha : float, default 0.05

## Returns

-------
(tidy, glance) : tuple of pandas.DataFrame
    - tidy: per-coefficient log-odds table; exp(estimate) gives the
      odds ratio for that variable
    - glance: model-level summary (deviance, AIC, BIC, df, nobs,
      pseudo-R² via log-likelihood)

## Example

--------
>>> tidy, glance = logistic(rel, "attrition ~ team_size + tenure_yrs")
>>> tidy.assign(odds_ratio=lambda d: np.exp(d["estimate"]))

---

[Back to API catalog](../README.md#api-catalog)
