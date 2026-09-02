# `vector_search`

**Module:** `pyduck_ona.search`

## Signature

```python
vector_search(table_name'str', query_vector'Sequence[float]', vector_col'str'='embedding', id_col'str'='employee_id', k'int'=10, metric'str'='l2sq', con'DuckDBPyConnection | None'=None, build_index'bool'=True, index_kwargs'Any')
```

## Description

Approximate nearest-neighbor search over an embedding column

## Parameters

----------
table_name
    Table that holds the embeddings.
query_vector
    Target vector as a sequence of floats.
vector_col
    ``FLOAT[N]`` column storing the embeddings.
id_col
    Id column to return alongside distance.
k
    Number of nearest neighbors.
metric
    Distance metric (see :func:`build_vector_index`).
con
    Existing DuckDB connection.
build_index
    If True, build the HNSW index before searching.
**index_kwargs
    Forwarded to :func:`build_vector_index`.

## Returns

-------
DuckDBPyRelation
    Relation with ``id_col`` and ``distance`` ordered ascending.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
