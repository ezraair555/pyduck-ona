# `DuckONA`

**Module:** `pyduck_ona.analysis`

## Signature

```python
DuckONA(db_path'str'=':memory:')
```

## Description

A DuckDB-backed workspace for HR analytics

## Parameters

----------
db_path : str, default ":memory:"
    DuckDB database path. ``:memory:`` (the default) is fine for
    in-memory analyses; pass a file path to persist tables.

Attributes
----------
con : duckdb.DuckDBPyConnection
    The underlying DuckDB connection. Exposed so callers can run
    arbitrary SQL on the same connection.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
