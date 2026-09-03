# `silo_map`

**Module:** `pyduck_ona.viz.network_viz`

## Signature

```python
silo_map(edges'pd.DataFrame', components'pd.DataFrame | None'=None, communities'pd.DataFrame | None'=None, source_col'str'='employee_id', target_col'str'='supervisor_id', node_col'str'='node_id', component_col'str'='component', community_col'str'='community', metadata'pd.DataFrame | None'=None, name_col'str'='name', department_col'str | None'=None, title'str | None'=None, figsize'tuple[float, float]'=(12.0, 9.0), return_html'bool'=False, physics'bool'=True)
```

## Description

Render an organisational silo map

## Parameters

----------
return_html
    If True, return pyvis interactive HTML. If False, return a matplotlib
    Figure with the same communities laid out via a force-directed
    spring layout.

## Returns

-------
str (HTML) or matplotlib.figure.Figure

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
