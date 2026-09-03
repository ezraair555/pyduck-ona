# `reporting_chain_walk`

**Module:** `pyduck_ona.viz.org_chart`

## Signature

```python
reporting_chain_walk(df'pd.DataFrame', employee_id'str', id_col'str'='employee_id', supervisor_col'str'='supervisor_id', metadata'pd.DataFrame | None'=None, name_col'str'='name', title_col'str'='title', level_col'str'='level', title'str | None'=None, figsize'tuple[float, float]'=(12.0, 4.5))
```

## Description

Plot the reporting chain from ``employee_id`` up to the top of the org

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
