# `build_vector_index`

**Module:** `pyduck_ona.search`

## Signature

```python
build_vector_index(table_name'str', vector_col'str', metric'str'='l2sq', con'DuckDBPyConnection | None'=None, ef_construction'int | None'=None, ef_search'int | None'=None, M'int | None'=None, overwrite'bool'=False)
```

## Description

Create an HNSW index on a fixed-size ``ARRAY`` embedding column

## Parameters

----------
table_name
    Table that holds the embeddings.
vector_col
    ``FLOAT[N]`` / ``DOUBLE[N]`` array column to index.
metric
    Distance metric: ``"l2sq"`` / ``"l2"`` / ``"euclidean"``,
    ``"cosine"``, or ``"ip"`` / ``"inner_product"``.
con
    Existing DuckDB connection.
ef_construction
    HNSW build-time accuracy/speed trade-off.
ef_search
    HNSW query-time accuracy/speed trade-off.
M
    Maximum neighbors per vertex.
overwrite
    Drop an existing index before creating.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
