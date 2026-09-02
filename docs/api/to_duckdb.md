# `to_duckdb`

**Module:** `pyduck_ona.stats`

## Signature

```python
to_duckdb(data'DuckDBPyRelation | pd.DataFrame', table_name'str', con'duckdb.DuckDBPyConnection | None'=None)
```

## Description

Register a DataFrame or relation as a DuckDB table

## Parameters

----------
data
table_name : str
con : DuckDBPyConnection, optional
    Existing connection. If None, a new in-memory connection is
    created. If ``data`` is a ``DuckDBPyRelation`` created on a
    different connection, you MUST pass the same connection here
    — DuckDB relations are not portable across connections.

## Returns

-------
(relation, con) : tuple
    A queryable relation on the table, and the connection that
    owns it.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
