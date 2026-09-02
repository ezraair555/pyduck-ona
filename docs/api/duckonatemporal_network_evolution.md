# `DuckONATemporal.network_evolution`

**Module:** `pyduck_ona.temporal`

## Signature

```python
DuckONATemporal.network_evolution(self, employee_id_col'str | None'=None, supervisor_id_col'str | None'=None)
```

## Description

Aggregate network-shape metrics per period

## Returns

-------
pandas.DataFrame
    Columns: ``period, n_employees, n_edges, density,
    centralization, n_components, avg_path_length``.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
