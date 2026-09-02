# `plot_coefficients`

**Module:** `pyduck_ona.stats`

## Signature

```python
plot_coefficients(tidy_df'pd.DataFrame', reference_line'float'=0.0, sort'bool'=True)
```

## Description

Forest plot of regression coefficients with confidence intervals

## Parameters

----------
tidy_df : pandas.DataFrame
    The ``tidy`` output from :func:`ols` or :func:`logistic`. Must
    have columns ``term``, ``estimate``, ``conf.low``, ``conf.high``.
reference_line : float, default 0.0
    Vertical reference line. Set to 1.0 for odds-ratio plots.
sort : bool, default True
    If True, sort by estimate size (largest effect at top).

## Returns

-------
(figure, axes) : matplotlib objects

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
