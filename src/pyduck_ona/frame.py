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
from pyduck_ona import stats as _stats
from pyduck_ona import temporal as _temporal
from pyduck_ona.core import hierarchy_long, hierarchy_valid, hierarchy_wide

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
            # Materialize on the frame's own connection so downstream
            # pipeline steps can reference it by name regardless of which
            # connection produced the relation.
            tmp = f"_emit_{uuid.uuid4().hex[:8]}"
            self.con.register(tmp, rel.df())
            self.con.execute(f"CREATE OR REPLACE TABLE {output} AS SELECT * FROM {tmp}")
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
        dt = _temporal.DuckONATemporal(self.con)
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
        rel = _graph.pagerank(self.relation(), source_col, target_col)
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
        rel = _graph.betweenness(self.relation(), source_col, target_col)
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
        dt = _temporal.DuckONATemporal(self.con)
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
        return self._emit(src, as_pandas=False, output=table_name)

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
