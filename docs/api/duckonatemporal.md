# `DuckONATemporal`

**Module:** `pyduck_ona.temporal`

## Signature

```python
DuckONATemporal(db_path'str'=':memory:')
```

## Description

A DuckDB-backed temporal ONA workspace

## Parameters

----------
db_path : str, default ":memory:"
    DuckDB path. Use a file path to persist across sessions.

Attributes
----------
con : duckdb.DuckDBPyConnection
    The underlying DuckDB connection.
freq : str
    Period frequency (M / Q / Y). Set by ``load_snapshots``.
periods : list[str]
    Sorted period labels.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
