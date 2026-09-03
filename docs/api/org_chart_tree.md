# `org_chart_tree`

**Module:** `pyduck_ona.viz.org_chart`

## Signature

```python
org_chart_tree(df'pd.DataFrame', id_col'str'='employee_id', supervisor_col'str'='supervisor_id', metadata'pd.DataFrame | None'=None, name_col'str'='name', title_col'str'='title', department_col'str'='department', level_col'str'='level', color_by'str | None'='department', root_id'str | None'=None, title'str | None'='Organizational Chart', height'str'='820px', width'str'='100%')
```

## Description

Return a standalone HTML string containing an interactive org chart

## Parameters

----------
df
    Long-form hierarchy (one row per manager → report edge). The output
    of ``pyduck_ona.hierarchy_long(...)`` after ``.df()`` works directly.
metadata
    Optional per-employee metadata DataFrame keyed by ``id_col``.
color_by
    One of ``"department"``, ``"level"`` or ``None``. Controls node fill.
root_id
    Optional explicit root node; defaults to the topmost supervisor.

## Returns

-------
str
    A full HTML document with an embedded D3 tree layout. Save to disk
    with ``Path("org.html").write_text(html)`` to view in a browser.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
