# `correlation`

**Module:** `pyduck_ona.stats`

## Signature

```python
correlation(data'DuckDBPyRelation | pd.DataFrame', columns'Sequence[str] | None'=None, col1'str | None'=None, col2'str | None'=None, method'str'='pearson')
```

## Description

Pairwise correlations across a set of columns

## Parameters

----------
data
    DataFrame or DuckDB relation.
columns : sequence of str, optional
    If given, return all pairwise correlations among these columns.
    Mutually exclusive with ``col1`` + ``col2``.
col1, col2 : str, optional
    Compute a single correlation between two columns. Mutually
    exclusive with ``columns``.
method : {"pearson", "spearman", "kendall"}, default "pearson"
    Correlation coefficient.

## Returns

-------
pandas.DataFrame
    Columns ``(term1, term2, correlation, p.value)``.

## Example

--------
>>> corr = correlation(rel, columns=["team_size", "salary", "tenure_yrs"])
>>> corr[corr["p.value"] < 0.05]

---

[Back to API catalog](../README.md#api-catalog)
