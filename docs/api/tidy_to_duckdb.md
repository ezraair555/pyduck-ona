# `tidy_to_duckdb`

**Module:** `pyduck_ona.stats`

## Signature

```python
tidy_to_duckdb(tidy_df'pd.DataFrame', con'duckdb.DuckDBPyConnection | None'=None, table_name'str'='model_tidy')
```

## Description

Write a tidy model result into a DuckDB table

## Parameters

----------
tidy_df : pandas.DataFrame
    Output of :func:`ols`, :func:`logistic`, or any other broom
    tidy result.
con : DuckDBPyConnection, optional
    Existing connection. If None, a new in-memory connection is
    created. **Important:** DuckDB tables live on a single
    connection, so if you want to query the table afterwards, use
    the same ``con`` returned here (not the default ``duckdb.sql``,
    which uses a separate connection).
table_name : str, default "model_tidy"
    Destination table name.

## Returns

-------
(table_name, con) : tuple
    The table name and the connection that owns it. Pass ``con``
    to subsequent ``con.sql(...)`` calls.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

## Notes

-----
``broom-sm`` tidy DataFrames use R-style dotted column names
(``p.value``, ``conf.low``) that DuckDB parses as struct field
access unless quoted. To save users from quoting every reference,
this function rewrites dotted column names to underscore equivalents
on write (``p.value`` → ``p_value``, ``conf.low`` → ``conf_low``).
The returned table is therefore queryable with unquoted identifiers
like ``SELECT term, p_value FROM model_tidy``.

---

[Back to API catalog](../README.md#api-catalog)
