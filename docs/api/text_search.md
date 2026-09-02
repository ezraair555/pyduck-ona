# `text_search`

**Module:** `pyduck_ona.search`

## Signature

```python
text_search(table_name'str', query'str', id_col'str'='employee_id', text_cols'str | Sequence[str] | None'=None, k'int'=10, con'DuckDBPyConnection | None'=None, build_index'bool'=True, index_options'Any')
```

## Description

Full-text search an HR table and return the top-k matches

## Parameters

----------
table_name
    Table to search.
query
    Free-text query.
id_col
    Document id column.
text_cols
    Columns that were / should be indexed. If None, the index must
    already exist.
k
    Number of rows to return.
con
    Existing DuckDB connection.
build_index
    If True and ``text_cols`` is supplied, build the index before
    searching (idempotent if schema already exists).
**index_options
    Forwarded to :func:`build_fts_index` (stemmer, stopwords, ...).

## Returns

-------
DuckDBPyRelation
    Relation with all original columns plus ``score`` ordered by BM25.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
