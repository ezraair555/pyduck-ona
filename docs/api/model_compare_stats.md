# `model_compare_stats`

**Module:** `pyduck_ona.stats`

## Signature

```python
model_compare_stats(models'dict[str, Any]')
```

## Description

Side-by-side comparison of multiple fitted models

## Parameters

----------
models : dict
    Mapping of ``name -> fitted statsmodels result``. Each name
    appears as a column in the output.

## Returns

-------
pandas.DataFrame
    Each row is a glance statistic (R², AIC, BIC, log-lik, df, etc.);
    each column is one of the input models.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
