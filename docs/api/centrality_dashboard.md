# `centrality_dashboard`

**Module:** `pyduck_ona.viz.network_viz`

## Signature

```python
centrality_dashboard(betweenness'pd.DataFrame', pagerank'pd.DataFrame', eigenvector'pd.DataFrame', degree'pd.DataFrame', id_col'str'='node_id', betweenness_col'str'='betweenness', pagerank_col'str'='pagerank', eigenvector_col'str'='eigenvector', degree_col'str'='degree', metadata'pd.DataFrame | None'=None, name_col'str'='name', department_col'str | None'='department', top_n'int'=12, title'str | None'=None, figsize'tuple[float, float]'=(13.0, 9.0))
```

## Description

Plot a 2×2 grid comparing four centrality measures

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
