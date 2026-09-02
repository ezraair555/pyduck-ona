"""Full-text and vector-similarity search helpers for pyduck-ona.

This module wraps the DuckDB ``fts`` and ``vss`` extensions so they can be
used against HR tables without leaving the relational API. Typical use
cases:

* ``text_search`` — find employees/jobs/skills by free-text query over
  HR text columns.
* ``vector_search`` — nearest-neighbor search over embedding columns
  (skills, survey open-ends, job descriptions, manager feedback).
* ``fuzzy_join_vectors`` — align two tables by approximate vector match,
  useful for skills-to-role mapping or resume-to-requisition matching.

Both extensions are installed and loaded on first use, so they remain
optional at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import duckdb

if TYPE_CHECKING:
    from collections.abc import Sequence

    from duckdb import DuckDBPyConnection, DuckDBPyRelation

from pyduck_ona.sql_builder import quote_identifier

# Distance-function mapping for VSS queries.  Each metric maps to the DuckDB
# expression that computes distance (smaller = more similar).
_VSS_DISTANCE_FUNCS: dict[str, str] = {
    "l2sq": "array_distance",
    "l2": "array_distance",
    "euclidean": "array_distance",
    "cosine": "array_cosine_distance",
    "ip": "array_negative_inner_product",
    "inner_product": "array_negative_inner_product",
}


def _canonical_metric(metric: str) -> str:
    m = metric.lower().strip()
    if m in ("l2sq", "l2", "euclidean"):
        return "l2sq"
    if m in ("cosine",):
        return "cosine"
    if m in ("ip", "inner_product"):
        return "ip"
    raise ValueError(f"unsupported VSS metric: {metric!r}")


def _require_extension(ext: str, con: DuckDBPyConnection) -> None:
    """Install and load a DuckDB extension if it is not already loaded."""
    try:
        con.execute(f"LOAD {ext};")
    except Exception:
        con.execute(f"INSTALL {ext};")
        con.execute(f"LOAD {ext};")


def _ensure_con(con: DuckDBPyConnection | None) -> DuckDBPyConnection:
    return con if con is not None else duckdb.connect()


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ", ".join(str(float(v)) for v in vector) + "]"


# ─── FTS helpers ───────────────────────────────────────────────────────────


def build_fts_index(
    table_name: str,
    id_col: str,
    text_cols: str | Sequence[str],
    *,
    con: DuckDBPyConnection | None = None,
    stemmer: str = "porter",
    stopwords: str = "english",
    overwrite: bool = False,
) -> None:
    """Create a DuckDB full-text search index on an HR text table.

    Parameters
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
    """
    con = _ensure_con(con)
    _require_extension("fts", con)

    cols = [text_cols] if isinstance(text_cols, str) else list(text_cols)
    cols_repr = ", ".join(repr(c) for c in cols)

    pragma = (
        f"PRAGMA create_fts_index({table_name!r}, {id_col!r}, {cols_repr}, "
        f"stemmer={stemmer!r}, stopwords={stopwords!r}, "
        f"overwrite={overwrite!r});"
    )
    con.execute(pragma)


def drop_fts_index(table_name: str, *, con: DuckDBPyConnection | None = None) -> None:
    """Drop a DuckDB FTS index for ``table_name``."""
    con = _ensure_con(con)
    _require_extension("fts", con)
    con.execute(f"PRAGMA drop_fts_index({table_name!r});")


def text_search(
    table_name: str,
    query: str,
    *,
    id_col: str = "employee_id",
    text_cols: str | Sequence[str] | None = None,
    k: int = 10,
    con: DuckDBPyConnection | None = None,
    build_index: bool = True,
    **index_options: Any,
) -> DuckDBPyRelation:
    """Full-text search an HR table and return the top-k matches.

    Parameters
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

    Returns
    -------
    DuckDBPyRelation
        Relation with all original columns plus ``score`` ordered by BM25.
    """
    con = _ensure_con(con)
    _require_extension("fts", con)

    if text_cols is not None and build_index:
        build_fts_index(
            table_name,
            id_col,
            text_cols,
            con=con,
            **index_options,
        )

    q_table = quote_identifier(table_name)
    q_id = quote_identifier(id_col)
    schema_name = f"fts_main_{table_name}"
    # DuckDB FTS creates a schema named fts_main_<table> for indexes in main.
    sql = (
        f"SELECT *, {schema_name}.match_bm25({q_id}, ?) AS score "
        f"FROM {q_table} "
        f"WHERE score IS NOT NULL "
        f"ORDER BY score DESC "
        f"LIMIT ?"
    )
    return con.sql(sql, params=[query, k])


# ─── VSS helpers ───────────────────────────────────────────────────────────


def build_vector_index(
    table_name: str,
    vector_col: str,
    *,
    metric: str = "l2sq",
    con: DuckDBPyConnection | None = None,
    ef_construction: int | None = None,
    ef_search: int | None = None,
    M: int | None = None,  # noqa: N803
    overwrite: bool = False,
) -> None:
    """Create an HNSW index on a fixed-size ``ARRAY`` embedding column.

    Parameters
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
    """
    con = _ensure_con(con)
    _require_extension("vss", con)

    index_name = f"{table_name}_{vector_col}_hnsw_idx"
    metric = _canonical_metric(metric)

    if overwrite:
        con.execute(
            f"DROP INDEX IF EXISTS {quote_identifier(index_name)};"
        )

    options_parts = [f"metric = '{metric}'"]
    if ef_construction is not None:
        options_parts.append(f"ef_construction = {ef_construction}")
    if ef_search is not None:
        options_parts.append(f"ef_search = {ef_search}")
    if M is not None:
        options_parts.append(f"M = {M}")
    options = ", ".join(options_parts)

    sql = (
        f"CREATE INDEX {quote_identifier(index_name)} "
        f"ON {quote_identifier(table_name)} "
        f"USING HNSW ({quote_identifier(vector_col)}) "
        f"WITH ({options});"
    )
    con.execute(sql)


def drop_vector_index(
    table_name: str,
    vector_col: str,
    *,
    con: DuckDBPyConnection | None = None,
) -> None:
    """Drop an HNSW index created by :func:`build_vector_index`."""
    con = _ensure_con(con)
    _require_extension("vss", con)
    index_name = f"{table_name}_{vector_col}_hnsw_idx"
    con.execute(f"DROP INDEX IF EXISTS {quote_identifier(index_name)};")


def vector_search(
    table_name: str,
    query_vector: Sequence[float],
    *,
    vector_col: str = "embedding",
    id_col: str = "employee_id",
    k: int = 10,
    metric: str = "l2sq",
    con: DuckDBPyConnection | None = None,
    build_index: bool = True,
    **index_kwargs: Any,
) -> DuckDBPyRelation:
    """Approximate nearest-neighbor search over an embedding column.

    Parameters
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

    Returns
    -------
    DuckDBPyRelation
        Relation with ``id_col`` and ``distance`` ordered ascending.
    """
    con = _ensure_con(con)
    _require_extension("vss", con)

    if build_index:
        build_vector_index(table_name, vector_col, metric=metric, con=con, **index_kwargs)

    distance_fn = _VSS_DISTANCE_FUNCS.get(metric, _VSS_DISTANCE_FUNCS[_canonical_metric(metric)])
    metric = _canonical_metric(metric)
    q_table = quote_identifier(table_name)
    q_id = quote_identifier(id_col)
    q_vec = quote_identifier(vector_col)
    vec_literal = _vector_literal(query_vector)

    # DuckDB requires the array literal to be cast to the same type/dims
    # as the column. We infer by selecting the first row's dimension.
    dim_sql = f"SELECT len({q_vec}) FROM {q_table} LIMIT 1"
    dim_row = con.sql(dim_sql).fetchone()
    if dim_row is None:
        raise ValueError(f"table {table_name!r} has no rows to infer embedding dimension")
    dim = int(dim_row[0])
    cast = f"{vec_literal}::FLOAT[{dim}]"

    sql = (
        f"SELECT {q_id}, {distance_fn}({q_vec}, {cast}) AS distance "
        f"FROM {q_table} "
        f"ORDER BY distance "
        f"LIMIT ?"
    )
    return con.sql(sql, params=[k])


def fuzzy_join_vectors(
    left_table: str,
    right_table: str,
    left_col: str,
    right_col: str,
    *,
    k: int = 5,
    metric: str = "l2sq",
    con: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation:
    """Approximate nearest-neighbor join between two embedding tables.

    For every row in ``left_table``, returns the ``k`` closest rows from
    ``right_table`` based on vector distance. Useful for matching
    employee skill embeddings to role requirement embeddings.

    Parameters
    ----------
    left_table, right_table
        Names of the two tables to join.
    left_col, right_col
        Embedding array columns in each table.
    k
        Number of neighbors per left row.
    metric
        Distance metric.
    con
        Existing DuckDB connection.

    Returns
    -------
    DuckDBPyRelation
        Relation with left id, right id, and distance.
    """
    con = _ensure_con(con)
    _require_extension("vss", con)

    distance_fn = _VSS_DISTANCE_FUNCS.get(metric, _VSS_DISTANCE_FUNCS[_canonical_metric(metric)])
    metric = _canonical_metric(metric)

    l_tbl = quote_identifier(left_table)
    r_tbl = quote_identifier(right_table)
    l_vec = quote_identifier(left_col)
    r_vec = quote_identifier(right_col)

    # Infer dimension from the right table and cast the left vector to match.
    dim_row = con.sql(f"SELECT len({r_vec}) FROM {r_tbl} LIMIT 1").fetchone()
    if dim_row is None:
        raise ValueError(f"right_table {right_table!r} has no rows to infer embedding dimension")
    dim = int(dim_row[0])

    sql = (
        f"SELECT l.*, r.*, {distance_fn}(l.{l_vec}, r.{r_vec}) AS distance "
        f"FROM {l_tbl} AS l, LATERAL ("
        f"  SELECT * FROM {r_tbl} "
        f"  ORDER BY {distance_fn}({r_vec}, l.{l_vec}::FLOAT[{dim}]) "
        f"  LIMIT ?"
        f") AS r "
        f"ORDER BY distance"
    )
    return con.sql(sql, params=[k])
