# `chi_square`

**Module:** `pyduck_ona.stats`

## Signature

```python
chi_square(data'DuckDBPyRelation | pd.DataFrame', x'str', y'str')
```

## Description

Chi-square test of independence between two categorical variables

## Parameters

----------
data
x, y : str
    Column names of the two categorical variables.

## Returns

-------
(table, figure) : tuple
    - table: ``(chi2, p_value, dof)`` summary
    - figure: matplotlib Figure with a heatmap of the observed
      counts overlaid with the expected counts, plus the chi-square
      and p-value annotated on the title.

## Example

--------
>>> table, fig = chi_square(rel, "department", "gender")
>>> fig.savefig("dept_by_gender.png", dpi=120, bbox_inches="tight")

---

[Back to API catalog](../README.md#api-catalog)
