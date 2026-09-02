# `DuckONA.profile_clusters`

**Module:** `pyduck_ona.analysis`

## Signature

```python
DuckONA.profile_clusters(self, features'list[str]', n_clusters'int'=6, method"Literal['kmeans', 'gmm']"='kmeans', include_network'bool'=True, employee_id_col'str'='employee_id', supervisor_id_col'str'='supervisor_id', hris_table'str'='hris', random_state'int'=42)
```

## Description

Cluster employee profiles from HR attributes and optional network features

## Parameters

----------
features : list[str]
    Base HRIS feature columns to cluster on.
n_clusters : int, default 6
    Number of profile clusters.
method : {"kmeans", "gmm"}, default "kmeans"
    Clustering algorithm.
include_network : bool, default True
    If True, append pagerank / betweenness / degree / louvain labels.
employee_id_col, supervisor_id_col : str
    HRIS key columns.
hris_table : str, default "hris"
    Name of registered HRIS table.
random_state : int, default 42
    Seed for deterministic clustering.

## Returns

-------
pandas.DataFrame
    Columns include employee id, ``cluster_id``, optional
    ``cluster_confidence`` (GMM), and the feature columns used.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
