"""Fluent, contract-first analytics façade for pyduck-ona.

This module implements the v0.3 API contract draft:

- Relation-first runtime: methods return ``duckdb.DuckDBPyRelation`` by default.
- ``as_pandas=True`` is the explicit materialization switch.
- ``output=<name>`` registers the result and returns ``self`` for fluent chaining.
- ``entity_id`` is the canonical employee key in all employee-level outputs.
- Verbs are grouped into five families: ``prep_*, graph_*, temporal_*,
  model_*, report_*``.
- ``pipeline()`` composes a sequence of frame verbs into one workflow.

The façade is additive: it does not replace ``DuckONA`` or
``DuckONATemporal``. Existing code keeps working unchanged.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import duckdb

from pyduck_ona import graph as _graph
from pyduck_ona import search as _search
from pyduck_ona import stats as _stats
from pyduck_ona import temporal as _temporal
from pyduck_ona.core import hierarchy_long, hierarchy_valid, hierarchy_wide
from pyduck_ona.sql_builder import quote_identifier

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    from duckdb import DuckDBPyConnection, DuckDBPyRelation


class DuckONAFrame:
    """A relation-first, uniform-verb façade over pyduck-ona analytics.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection, optional
        Connection that owns the source table. Defaults to a fresh
        in-memory connection.
    source : str, optional
        Name of the registered source table. If omitted, the frame
        starts empty and callers must load data via ``prep_load_*``.
    """

    def __init__(
        self,
        con: DuckDBPyConnection | None = None,
        source: str | None = None,
    ) -> None:
        self.con = con if con is not None else duckdb.connect(":memory:")
        self.source = source

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, table_name: str = "hris") -> DuckONAFrame:
        """Create a frame from a pandas DataFrame."""
        con = duckdb.connect(":memory:")
        con.register(table_name, df)
        return cls(con, table_name)

    @classmethod
    def from_janitor(cls, janitor_obj: Any, table_name: str = "hris") -> DuckONAFrame:
        """Create a frame from a pyduck-janitor ``DuckJanitor`` instance."""
        relation = getattr(janitor_obj, "_relation", None)
        connection = getattr(janitor_obj, "_connection", None)
        if relation is None or connection is None:
            raise TypeError(
                "from_janitor expects a DuckJanitor-like object with "
                "'_relation' and '_connection' attributes"
            )
        connection.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM relation"
        )
        return cls(connection, table_name)

    # ─── Internal helpers ─────────────────────────────────────────────────

    def relation(self) -> DuckDBPyRelation:
        """Return the current source relation."""
        if self.source is None:
            raise RuntimeError(
                "No source table is set. Use a prep_load_* method first."
            )
        return self.con.sql(f"SELECT * FROM {self.source}")

    def _emit(
        self,
        rel: DuckDBPyRelation,
        *,
        as_pandas: bool,
        output: str | None,
    ) -> DuckDBPyRelation | pd.DataFrame | DuckONAFrame:
        """Handle the v0.3 return contract."""
        if as_pandas:
            return rel.df()
        if output:
            rel.create_view(output)
            self.source = output
            return self
        return rel

    @staticmethod
    def _canonical(
        rel: DuckDBPyRelation,
        id_col: str,
        key: str = "entity_id",
    ) -> DuckDBPyRelation:
        """Rename ``id_col`` to the canonical ``entity_id`` key."""
        cols = [c for c in rel.columns if c != id_col]
        select_list = f'"{id_col}" AS {key}' + (
            ", " + ", ".join(f'"{c}"' for c in cols) if cols else ""
        )
        return rel.project(select_list)

    def _snapshot_to_table(self) -> str:
        """Materialize the current source relation as a temp table."""
        if self.source is None:
            raise RuntimeError(
                "No source table is set. Use a prep_load_* method first."
            )
        tmp = f"__frame_{uuid.uuid4().hex}"
        self.con.execute(f"CREATE OR REPLACE TABLE {tmp} AS SELECT * FROM {self.source}")
        return tmp

    # ─── 1. prep_* — data preparation and validation ──────────────────────

    def prep_validate(
        self,
        employee_id_col: str = "employee_id",
        supervisor_id_col: str = "supervisor_id",
        *,
        key: str = "entity_id",
        output: str | None = None,
        as_pandas: bool = False,
    ) -> DuckDBPyRelation | pd.DataFrame | DuckONAFrame:
        """Validate hierarchy integrity.

        Output schema: ``entity_id, issue_type, detail``.
        """
        rel = hierarchy_valid(self.relation(), employee_id_col, supervisor_id_col)
        rel = self._canonical(rel, "employee_id", key)
        return self._emit(rel, as_pandas=as_pandas, output=output)

    def prep_long(
        self,
        employee_id_col: str = "employee_id",
        supervisor_id_col: str = "supervisor_id",
        *,
        output: str | None = None,
        as_pandas: bool = False,
    ) -> DuckDBPyRelation | pd.DataFrame | DuckONAFrame:
        """Return a long-form transitive closure of the reporting chain."""
        rel = hierarchy_long(self.relation(), employee_id_col, supervisor_id_col)
        return self._emit(rel, as_pandas=as_pandas, output=output)

    def prep_wide(
        self,
        employee_id_col: str = "employee_id",
        supervisor_id_col: str = "supervisor_id",
        *,
        max_depth: int = 8,
        output: str | None = None,
        as_pandas: bool = False,
    ) -> DuckDBPyRelation | pd.DataFrame | DuckONAFrame:
        """Return a wide-form ancestor table (Level_1, Level_2, ...)."""
        rel = hierarchy_wide(
            self.relation(),
            employee_id_col,
            supervisor_id_col,
            max_depth=max_depth,
        )
        return self._emit(rel, as_pandas=as_pandas, output=output)

    def prep_load_snapshots(
        self,
        df: pd.DataFrame,
        snapshot_date_col: str = "snapshot_date",
        employee_id_col: str = "employee_id",
        supervisor_id_col: str = "supervisor_id",
        freq: str = "Q",
        table_name: str = "snapshots",
    ) -> DuckONAFrame:
        """Load snapshot data and wire up a temporal engine on this frame."""
        dt = _temporal.DuckONATemporal._from_connection(self.con)
        dt.load_snapshots(
            df,
            snapshot_date_col=snapshot_date_col,
            employee_id_col=employee_id_col,
            supervisor_id_col=supervisor_id_col,
            freq=freq,
            table_name=table_name,
        )
        self.source = table_name
        return self

    # ─── 2. graph_* — network metrics ───────────────────────────────────────

    def graph_pagerank(
        self,
        source_col: str = "employee_id",
        target_col: str = "supervisor_id",
        *,
        key: str = "entity_id",
        output: str | None = None,
        as_pandas: bool = False,
    ) -> DuckDBPyRelation | pd.DataFrame | DuckONAFrame:
        """Compute PageRank on the direct-edge relation."""
        rel = _graph.pagerank(self.relation(), source_col, target_col, con=self.con)
        rel = self._canonical(rel, "node_id", key)
        return self._emit(rel, as_pandas=as_pandas, output=output)

    def graph_betweenness(
        self,
        source_col: str = "employee_id",
        target_col: str = "supervisor_id",
        *,
        key: str = "entity_id",
        output: str | None = None,
        as_pandas: bool = False,
    ) -> DuckDBPyRelation | pd.DataFrame | DuckONAFrame:
        """Compute betweenness centrality on the direct-edge relation."""
        rel = _graph.betweenness(self.relation(), source_col, target_col, con=self.con)
        rel = self._canonical(rel, "node_id", key)
        return self._emit(rel, as_pandas=as_pandas, output=output)

    # ─── 3. temporal_* — time-aware analytics ─────────────────────────────

    def temporal_metrics(
        self,
        metrics: list[str] | None = None,
        *,
        employee_id_col: str = "employee_id",
        supervisor_id_col: str = "supervisor_id",
    ) -> pd.DataFrame:
        """Compute temporal ONA metrics across loaded snapshots."""
        dt = _temporal.DuckONATemporal._from_connection(self.con)
        dt._table_name = self.source or "snapshots"
        dt._emp_col = employee_id_col
        dt._sup_col = supervisor_id_col
        dt._date_col = "snapshot_date"
        dt._freq = "Q"
        dt._loaded = True
        dt._periods = []
        return dt.compute_temporal_metrics(metrics=metrics)

    # ─── 4. model_* — statistical models ──────────────────────────────────

    def model_ols(
        self,
        formula: str,
        *,
        as_pandas: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | DuckDBPyRelation:
        """Fit an OLS model via broom_sm.

        Note: broom_sm consumes a pandas DataFrame, so the input is
        materialized internally. Returns the tidy/glance pair.
        """
        return _stats.ols(self.relation(), formula=formula)

    # ─── 5. report_* — packaging outputs ──────────────────────────────────

    def report_export(
        self,
        table_name: str,
        *,
        rel: DuckDBPyRelation | None = None,
    ) -> DuckONAFrame:
        """Register the current (or supplied) relation as a named table."""
        src = rel if rel is not None else self.relation()
        return self._emit(src, as_pandas=False, output=table_name)  # type: ignore[return-value]

    # ─── 6. search_* — text and vector search ────────────────────────────

    def search_text(
        self,
        query: str,
        text_col: str = "bio",
        *,
        id_col: str = "employee_id",
        k: int = 10,
        output: str | None = None,
        as_pandas: bool = False,
    ) -> DuckDBPyRelation | pd.DataFrame | DuckONAFrame:
        """Full-text search over a text column in the current relation."""
        tmp = self._snapshot_to_table()
        rel = _search.text_search(
            tmp,
            query,
            id_col=id_col,
            text_cols=[text_col],
            con=self.con,
            k=k,
        )
        rel = self._canonical(rel, id_col)
        return self._emit(rel, as_pandas=as_pandas, output=output)

    def search_similar(
        self,
        query_vector: list[float],
        vector_col: str = "embedding",
        *,
        id_col: str = "employee_id",
        k: int = 10,
        metric: str = "l2sq",
        output: str | None = None,
        as_pandas: bool = False,
    ) -> DuckDBPyRelation | pd.DataFrame | DuckONAFrame:
        """Approximate nearest-neighbor search over an embedding column."""
        tmp = self._snapshot_to_table()
        q_vec = quote_identifier(vector_col)
        # DuckDB HNSW indexes require FLOAT[N]; pandas lists often become DOUBLE[N].
        type_row = self.con.sql(
            f"SELECT typeof({q_vec}) FROM {tmp} LIMIT 1"
        ).fetchone()
        if type_row is not None and "DOUBLE" in str(type_row[0]):
            dim_row = self.con.sql(f"SELECT len({q_vec}) FROM {tmp} LIMIT 1").fetchone()
            dim = int(dim_row[0]) if dim_row else 0
            self.con.execute(
                f"ALTER TABLE {tmp} ALTER {q_vec} SET DATA TYPE FLOAT[{dim}];"
            )
        rel = _search.vector_search(
            tmp,
            query_vector,
            id_col=id_col,
            vector_col=vector_col,
            con=self.con,
            k=k,
            metric=metric,
        )
        rel = self._canonical(rel, id_col)
        return self._emit(rel, as_pandas=as_pandas, output=output)

    # ─── Pipeline combinator ──────────────────────────────────────────────

    def pipeline(
        self,
        steps: list[Callable[[DuckONAFrame], DuckONAFrame]],
    ) -> DuckONAFrame:
        """Compose a sequence of frame verbs into a single workflow.

        Example
        -------
        >>> frame.pipeline([
        ...     lambda f: f.graph_pagerank(output="pr"),
        ...     lambda f: f.prep_validate(output="validated"),
        ... ])
        """
        result = self
        for step in steps:
            result = step(result)
            if not isinstance(result, DuckONAFrame):
                raise TypeError(
                    "pipeline steps must return a DuckONAFrame "
                    "(use output=<table_name> to frame a relation)"
                )
        return result
