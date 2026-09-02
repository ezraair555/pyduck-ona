# `build_fts_index`

**Module:** `pyduck_ona.search`

## Signature

```python
build_fts_index(table_name'str', id_col'str', text_cols'str | Sequence[str]', con'DuckDBPyConnection | None'=None, stemmer'str'='porter', stopwords'str'='english', overwrite'bool'=False)
```

## Description

Create a DuckDB full-text search index on an HR text table

## Parameters

----------
table_name
    Name of the table to index.
id_col
    Document identifier column.
text_cols
    Column(s) to index. Pass ``"*"`` to index all VARCHAR columns.
con
    Existing DuckDB connection. If None, an in-memory connection is used.
stemmer
    Stemmer to use (e.g. ``"porter"``, ``"english"``, ``"none"``).
stopwords
    Stopword table name or ``"none"`` / ``"english"``.
overwrite
    Re-create an existing index.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
