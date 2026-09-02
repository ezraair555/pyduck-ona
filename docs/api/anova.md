# `anova`

**Module:** `pyduck_ona.stats`

## Signature

```python
anova(data'DuckDBPyRelation | pd.DataFrame', formula'str', anova_type'int'=2)
```

## Description

One-way ANOVA via OLS, tidy output

## Parameters

----------
data
formula : str
    Patsy-style formula, e.g. ``"salary ~ department"``.
anova_type : {1, 2, 3}, default 2
    Type of ANOVA (1 = sequential, 2 = partial SS, 3 = marginal).

## Returns

-------
pandas.DataFrame
    Columns ``(term, sum_sq, df, statistic, p.value)``.

## Example

--------
>>> anova(rel, "salary ~ department")

---

[Back to API catalog](../README.md#api-catalog)
