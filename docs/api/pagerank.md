# `pagerank`

**Module:** `pyduck_ona.graph`

## Signature

```python
pagerank(edges'DuckDBPyRelation', source_col'str', target_col'str', damping'float'=0.85, node_id_col'str'='node_id', con'duckdb.DuckDBPyConnection | None'=None, backend"Literal['networkx', 'duckpgq']"='networkx')
```

## Description

PageRank centrality (influence scoring)

## Parameters

----------
edges, source_col, target_col
damping : float, default 0.85
    Standard PageRank damping factor. **Note:** the DuckPGQ v1.3.1
    ``pagerank`` table function does not expose a damping parameter;
the value is accepted for API compatibility but ignored on the
DuckPGQ backend (the engine uses its default). NetworkX may require
SciPy in clean environments.
node_id_col : str, default "node_id"
    Name of the node-id column in the returned relation.
backend : {"networkx", "duckpgq"}, default "networkx"
    Algorithm backend. The DuckPGQ backend runs the SQL table
    function ``pagerank(graph, vlabel, elabel)`` on a registered
    property graph; requires ``pip install pyduck-ona[graph]``.

## Returns

-------
DuckDBPyRelation
    Columns ``(node_id_col, pagerank)`` sorted by pagerank DESC.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

## Notes

-----
The DuckPGQ backend installs and loads the DuckPGQ extension on
the supplied ``con`` (or an ephemeral one if ``con`` is ``None``)
and runs the computation inside DuckDB. Results are not
byte-identical to NetworkX because DuckPGQ and NetworkX use
different convergence criteria; for trend analysis on the same
graph the relative ordering of nodes is preserved.

---

[Back to API catalog](../README.md#api-catalog)
