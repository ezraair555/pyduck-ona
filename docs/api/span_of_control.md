# `span_of_control`

**Module:** `pyduck_ona.viz.span_control`

## Signature

```python
span_of_control(df'pd.DataFrame', id_col'str'='manager_id', metric_col'str'='direct_reports', label_col'str | None'=None, metadata'pd.DataFrame | None'=None, name_col'str'='name', department_col'str | None'=None, top_n'int'=20, color_by_department'bool'=False, title'str | None'=None, figsize'tuple[float, float]'=(10.0, 7.0), return_html'bool'=False)
```

## Description

Plot span of control for the top ``top_n`` managers

## Parameters

----------
df
    DataFrame with one row per manager (typically the output of
    ``pyduck_ona.hierarchy_stats(...)`` after ``.df()``).
id_col
    Manager identifier column.
metric_col
    Numeric metric column (default ``"direct_reports"``). Could also be
    ``"total_reports"`` for total team size.
label_col
    Optional explicit label column in ``df``.
metadata
    Optional per-manager metadata. ``name_col`` and ``department_col``
    are looked up here.
color_by_department
    If True, colour each bar by the manager's department.
return_html
    If True, return a Plotly Figure's HTML string (interactive) instead
    of a matplotlib Figure.

## Returns

-------
matplotlib.figure.Figure or str (HTML)

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
