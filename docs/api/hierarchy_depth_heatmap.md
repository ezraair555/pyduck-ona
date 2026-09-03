# `hierarchy_depth_heatmap`

**Module:** `pyduck_ona.viz.hierarchy_viz`

## Signature

```python
hierarchy_depth_heatmap(df'pd.DataFrame', employee_col'str'='employee_id', level_prefix'str'='Level_', max_levels'int | None'=None, metadata'pd.DataFrame | None'=None, name_col'str'='name', title'str | None'=None, figsize'tuple[float, float]'=(10.0, 9.0), annotate'bool'=False)
```

## Description

Render the hierarchy-wide table as a heatmap of depth vs employee

## Parameters

----------
df
    Wide-form hierarchy (output of ``pyduck_ona.hierarchy_wide(...)``).
    Columns ``Level_1``, ``Level_2``, ... are depth columns; each cell
    contains the manager_id at that depth (or NaN).
employee_col
    Column holding the employee identifier.
level_prefix
    Prefix for level columns. Defaults to ``"Level_"``.
max_levels
    Optional cap on the number of levels rendered.
metadata
    Optional per-employee metadata, used to label rows with names.

## Returns

-------
matplotlib.figure.Figure

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
