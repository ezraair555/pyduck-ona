# `DuckONA.mrqap`

**Module:** `pyduck_ona.analysis`

## Signature

```python
DuckONA.mrqap(Y'NDArray[np.float64]', X_matrices'list[NDArray[np.float64]]', n_permutations'int'=1000, method"Literal['pearson', 'spearman']"='pearson')
```

## Description

Small pure-Python MRQAP-style permutation test for matrix regression

## Parameters

----------
Y : (n, n) array
    Dependent square matrix (e.g. similarity / distance).
X_matrices : list of (n, n) arrays
    Independent square matrices. The first column is automatically
    an intercept.
n_permutations : int, default 1000
    Number of row/column permutations.
method : {"pearson", "spearman"}, default "pearson"
    Correlation method used for the semi-partial correlation
    shortcut diagnostics in the result dict.

## Returns

-------
dict
    ``coefficients``: estimated beta vector.
    ``p_values``: empirical two-tailed p-values per predictor.
    ``r2``: R² of the unpermuted model.
    ``permutation_betas``: (n_permutations, n_predictors) array.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

## Notes

-----
This is a minimal MRQAP approximation. It does not replace a full
QAP package for large matrices or complex dependence structures;
it is included so pyduck-ona stays R-free for simple hypothesis
tests on HR network matrices.

---

[Back to API catalog](../README.md#api-catalog)
