# `DuckONAFrame`

**Module:** `pyduck_ona.frame`

## Signature

```python
DuckONAFrame(con'DuckDBPyConnection | None'=None, source'str | None'=None)
```

## Description

A relation-first, uniform-verb façade over pyduck-ona analytics

## Parameters

----------
con : duckdb.DuckDBPyConnection, optional
    Connection that owns the source table. Defaults to a fresh
    in-memory connection.
source : str, optional
    Name of the registered source table. If omitted, the frame
    starts empty and callers must load data via ``prep_load_*``.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
