"""Query primitives for ``DuckONATemporal``.

20 primitives across 5 categories for querying organizational hierarchy
trends and over-time changes:

    Trajectory primitives (4):
        trajectory_at, trajectory_diff, trajectory_pivot, trajectory_rank

    Hierarchy-change primitives (4):
        edges_added, edges_removed, node_set_diff, hierarchy_drift

    Subtree / team primitives (4):
        subtree_at, subtree_size_at, subtree_growth, subtree_overlap

    Snapshot-comparison primitives (4):
        delta_table, new_centers, fallen_centers, cohort_compare

    Time-window aggregate primitives (4):
        window_mean, window_trend, window_rank_change, window_volatility

These are bound as ``DuckONATemporal.q.<name>`` so they share the
parent class's connection and metadata while keeping the namespace
clean.

Return style is mixed: methods that *traverse* (subtree_at,
node_set_diff, edges_added) return ``DuckDBPyRelation`` for SQL
composability; methods that *aggregate* (trajectory_at, window_mean)
return ``pd.DataFrame`` for terminal use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection, DuckDBPyRelation

from pyduck_ona import graph as _graph
from pyduck_ona.core import hierarchy_long, hierarchy_stats


# ─── Internal helpers ──────────────────────────────────────────────────────


def _period_edges(
    con: "DuckDBPyConnection",
    table: str,
    emp_col: str,
    sup_col: str,
    date_col: str,
    period: str,
) -> "DuckDBPyRelation":
    """Return the edge relation for a single period."""
    return con.sql(f"""
        SELECT DISTINCT
            "{emp_col}" AS "{emp_col}",
            "{sup_col}" AS "{sup_col}"
        FROM {table}
        WHERE date_trunc('month', CAST("{date_col}" AS DATE))
              = CAST('{period}' AS DATE)
          AND "{sup_col}" IS NOT NULL
          AND CAST("{sup_col}" AS VARCHAR) <> ''
    """)


def _period_employees(
    con: "DuckDBPyConnection",
    table: str,
    emp_col: str,
    date_col: str,
    period: str,
    extra_cols: list[str] | None = None,
) -> "DuckDBPyRelation":
    """Return employees present in a single period, with optional extra cols."""
    cols = [emp_col] + (extra_cols or [])
    col_list = ", ".join(f'"{c}"' for c in cols)
    return con.sql(f"""
        SELECT DISTINCT {col_list}
        FROM {table}
        WHERE date_trunc('month', CAST("{date_col}" AS DATE))
              = CAST('{period}' AS DATE)
    """)


def _metric_fn_map() -> dict:
    return {
        "betweenness": _graph.betweenness,
        "pagerank": _graph.pagerank,
        "eigenvector_centrality": _graph.eigenvector_centrality,
        "degree_centrality": _graph.degree_centrality,
        "connected_components": _graph.connected_components,
        "louvain_communities": _graph.louvain_communities,
    }


def _metric_value_column(df: pd.DataFrame) -> tuple[str, str]:
    """Return (id_col, value_col) for a metric output DataFrame."""
    id_col = "node_id" if "node_id" in df.columns else df.columns[0]
    value_cols = [c for c in df.columns if c != id_col]
    if not value_cols:
        raise ValueError(f"no value column found in {list(df.columns)}")
    return id_col, value_cols[0]


def _compute_metric_for_period(
    con: "DuckDBPyConnection",
    table: str,
    emp_col: str,
    sup_col: str,
    date_col: str,
    period: str,
    metric: str,
) -> pd.DataFrame:
    """Compute one metric for one period, return DataFrame with id + value."""
    edges = _period_edges(con, table, emp_col, sup_col, date_col, period)
    if edges.count("*").fetchone()[0] == 0:
        return pd.DataFrame(columns=["employee_id", metric])
    fn = _metric_fn_map().get(metric)
    if fn is not None:
        rel = fn(edges, emp_col, sup_col)
        df = rel.df()
        id_col, val_col = _metric_value_column(df)
        out = df.rename(columns={id_col: "employee_id", val_col: metric})
        return out[["employee_id", metric]]
    if metric == "team_size":
        stats_rel = hierarchy_stats(
            con.sql(f"""
                SELECT "{emp_col}" AS employee_id, "{sup_col}" AS supervisor_id
                FROM {table}
                WHERE date_trunc('month', CAST("{date_col}" AS DATE))
                      = CAST('{period}' AS DATE)
            """),
            "employee_id", "supervisor_id",
        )
        sdf = stats_rel.df()
        out = sdf.rename(columns={"manager_id": "employee_id"})
        return out[["employee_id", "team_size"]]
    raise ValueError(f"unknown metric: {metric!r}")


# ─── Primitives class ──────────────────────────────────────────────────────


class _QueryPrimitives:
    """Bound to ``DuckONATemporal.q`` via composition."""

    def __init__(self, parent: Any) -> None:
        self._p = parent  # the DuckONATemporal instance

    # ── 1. Trajectory primitives ──────────────────────────────────────────

    def trajectory_at(
        self,
        employee_id: Any,
        metric: str = "betweenness",
        lookback: str = "4Q",
    ) -> pd.DataFrame:
        """One employee, one metric, one time-series.

        Returns
        -------
        pandas.DataFrame
            Columns: ``period, value, delta, pct_change``.
        """
        p = self._p
        if not p._loaded:
            raise RuntimeError("call load_snapshots() first")
        n_periods, _ = self._parse_lookback(lookback)
        use_periods = p.periods[-n_periods:] if len(p.periods) >= n_periods else p.periods

        rows: list[dict] = []
        prev: float | None = None
        for period in use_periods:
            df = _compute_metric_for_period(
                p.con, p._table_name, p._emp_col, p._sup_col, p._date_col,
                period, metric,
            )
            if df.empty:
                rows.append({"period": period, "value": np.nan,
                             "delta": np.nan, "pct_change": np.nan})
                prev = None
                continue
            row_match = df[df["employee_id"] == employee_id]
            if row_match.empty:
                rows.append({"period": period, "value": np.nan,
                             "delta": np.nan, "pct_change": np.nan})
                prev = None
                continue
            val = float(row_match.iloc[0][metric])
            if prev is None or np.isnan(prev):
                delta = np.nan
                pct = np.nan
            else:
                delta = val - prev
                pct = (delta / prev * 100) if prev != 0 else np.nan
            rows.append({"period": period, "value": val, "delta": delta, "pct_change": pct})
            prev = val
        return pd.DataFrame(rows)

    def trajectory_diff(
        self,
        employee_id: Any,
        metric: str,
        period_t: str,
        period_t1: str,
    ) -> dict:
        """Single point diff between two periods for one employee.

        Returns
        -------
        dict
            ``{employee_id, metric, value_t, value_t1, delta, pct_change}``.
        """
        p = self._p
        if not p._loaded:
            raise RuntimeError("call load_snapshots() first")
        df_t = _compute_metric_for_period(
            p.con, p._table_name, p._emp_col, p._sup_col, p._date_col,
            period_t, metric,
        )
        df_t1 = _compute_metric_for_period(
            p.con, p._table_name, p._emp_col, p._sup_col, p._date_col,
            period_t1, metric,
        )
        v_t = float(df_t[df_t["employee_id"] == employee_id].iloc[0][metric]) \
            if not df_t.empty and (df_t["employee_id"] == employee_id).any() else np.nan
        v_t1 = float(df_t1[df_t1["employee_id"] == employee_id].iloc[0][metric]) \
            if not df_t1.empty and (df_t1["employee_id"] == employee_id).any() else np.nan
        if np.isnan(v_t) or np.isnan(v_t1):
            return {
                "employee_id": employee_id, "metric": metric,
                "value_t": v_t, "value_t1": v_t1,
                "delta": np.nan, "pct_change": np.nan,
            }
        delta = v_t1 - v_t
        pct = (delta / v_t * 100) if v_t != 0 else np.nan
        return {
            "employee_id": employee_id, "metric": metric,
            "value_t": v_t, "value_t1": v_t1,
            "delta": float(delta), "pct_change": float(pct) if not np.isnan(pct) else np.nan,
        }

    def trajectory_pivot(
        self,
        metric: str = "betweenness",
        lookback: str = "4Q",
    ) -> pd.DataFrame:
        """Wide format: one row per employee, columns = periods.

        Returns
        -------
        pandas.DataFrame
            Index: ``employee_id``. Columns: one per period. Values: metric.
        """
        p = self._p
        if not p._loaded:
            raise RuntimeError("call load_snapshots() first")
        n_periods, _ = self._parse_lookback(lookback)
        use_periods = p.periods[-n_periods:] if len(p.periods) >= n_periods else p.periods

        frames: list[pd.DataFrame] = []
        for period in use_periods:
            df = _compute_metric_for_period(
                p.con, p._table_name, p._emp_col, p._sup_col, p._date_col,
                period, metric,
            )
            if df.empty:
                continue
            df = df.rename(columns={metric: period})
            df = df.set_index("employee_id")
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1)

    def trajectory_rank(
        self,
        metric: str = "pagerank",
        period: str | None = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Top N at a single period.

        Parameters
        ----------
        metric : str
        period : str, optional
            Period label. Defaults to the latest period.
        top_n : int

        Returns
        -------
        pandas.DataFrame
            Columns: ``rank, employee_id, value``.
        """
        p = self._p
        if not p._loaded:
            raise RuntimeError("call load_snapshots() first")
        if period is None:
            period = p.periods[-1]
        df = _compute_metric_for_period(
            p.con, p._table_name, p._emp_col, p._sup_col, p._date_col,
            period, metric,
        )
        if df.empty:
            return pd.DataFrame(columns=["rank", "employee_id", "value"])
        df = df.sort_values(metric, ascending=False).head(top_n).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
        df = df.rename(columns={metric: "value"})
        return df

    # ── 2. Hierarchy-change primitives ──────────────────────────────────

    def edges_added(
        self,
        period_t: str,
        period_t1: str,
    ) -> "DuckDBPyRelation":
        """New supervisor edges between two snapshots.

        Returns
        -------
        DuckDBPyRelation
            Columns: ``employee_id, supervisor_id``.
        """
        p = self._p
        if not p._loaded:
            raise RuntimeError("call load_snapshots() first")
        return p.con.sql(f"""
            SELECT t1."{p._emp_col}" AS employee_id,
                   t1."{p._sup_col}" AS supervisor_id
            FROM (
                SELECT "{p._emp_col}", "{p._sup_col}"
                FROM {p._table_name}
                WHERE date_trunc('month', CAST("{p._date_col}" AS DATE))
                      = CAST('{period_t1}' AS DATE)
            ) t1
            LEFT JOIN (
                SELECT "{p._emp_col}", "{p._sup_col}"
                FROM {p._table_name}
                WHERE date_trunc('month', CAST("{p._date_col}" AS DATE))
                      = CAST('{period_t}' AS DATE)
            ) t0
              ON t1."{p._emp_col}" = t0."{p._emp_col}"
            WHERE t0."{p._sup_col}" IS NULL
               OR t0."{p._sup_col}" != t1."{p._sup_col}"
        """)

    def edges_removed(
        self,
        period_t: str,
        period_t1: str,
    ) -> "DuckDBPyRelation":
        """Edges present at t but missing at t1 (departures or reorgs).

        Returns
        -------
        DuckDBPyRelation
            Columns: ``employee_id, supervisor_id_at_t``.
        """
        p = self._p
        if not p._loaded:
            raise RuntimeError("call load_snapshots() first")
        return p.con.sql(f"""
            SELECT t0."{p._emp_col}" AS employee_id,
                   t0."{p._sup_col}" AS supervisor_id_at_t
            FROM (
                SELECT "{p._emp_col}", "{p._sup_col}"
                FROM {p._table_name}
                WHERE date_trunc('month', CAST("{p._date_col}" AS DATE))
                      = CAST('{period_t}' AS DATE)
            ) t0
            LEFT JOIN (
                SELECT "{p._emp_col}", "{p._sup_col}"
                FROM {p._table_name}
                WHERE date_trunc('month', CAST("{p._date_col}" AS DATE))
                      = CAST('{period_t1}' AS DATE)
            ) t1
              ON t0."{p._emp_col}" = t1."{p._emp_col}"
            WHERE t1."{p._emp_col}" IS NULL
        """)

    def node_set_diff(
        self,
        period_t: str,
        period_t1: str,
    ) -> dict:
        """Employees who joined or left between two periods.

        Returns
        -------
        dict
            ``{"joined": DataFrame, "left": DataFrame}`` — each DataFrame
            has ``employee_id`` and any extra columns present in snapshots.
        """
        p = self._p
        if not p._loaded:
            raise RuntimeError("call load_snapshots() first")
        emp_t = _period_employees(p.con, p._table_name, p._emp_col, p._date_col, period_t).df()
        emp_t1 = _period_employees(p.con, p._table_name, p._emp_col, p._date_col, period_t1).df()
        set_t = set(emp_t[p._emp_col].tolist())
        set_t1 = set(emp_t1[p._emp_col].tolist())
        joined_ids = set_t1 - set_t
        left_ids = set_t - set_t1
        joined = emp_t1[emp_t1[p._emp_col].isin(joined_ids)].reset_index(drop=True)
        left = emp_t[emp_t[p._emp_col].isin(left_ids)].reset_index(drop=True)
        return {"joined": joined, "left": left}

    def hierarchy_drift(
        self,
        period_t: str,
        period_t1: str,
    ) -> pd.DataFrame:
        """Span-of-control changes per manager between two periods.

        Returns
        -------
        pandas.DataFrame
            Columns: ``manager_id, direct_reports_t, direct_reports_t1,
            delta, total_reports_t, total_reports_t1, total_delta``.
        """
        p = self._p
        if not p._loaded:
            raise RuntimeError("call load_snapshots() first")
        snap_t = p.con.sql(f"""
            SELECT "{p._emp_col}" AS employee_id, "{p._sup_col}" AS supervisor_id
            FROM {p._table_name}
            WHERE date_trunc('month', CAST("{p._date_col}" AS DATE))
                  = CAST('{period_t}' AS DATE)
        """)
        snap_t1 = p.con.sql(f"""
            SELECT "{p._emp_col}" AS employee_id, "{p._sup_col}" AS supervisor_id
            FROM {p._table_name}
            WHERE date_trunc('month', CAST("{p._date_col}" AS DATE))
                  = CAST('{period_t1}' AS DATE)
        """)
        s_t = hierarchy_stats(snap_t, "employee_id", "supervisor_id").df()
        s_t1 = hierarchy_stats(snap_t1, "employee_id", "supervisor_id").df()
        merged = s_t.merge(
            s_t1,
            on="manager_id",
            how="outer",
            suffixes=("_t", "_t1"),
        )
        for col in ["direct_reports", "indirect_reports", "total_reports", "team_size", "levels_below"]:
            tc = f"{col}_t"
            t1c = f"{col}_t1"
            if tc in merged.columns:
                merged[tc] = merged[tc].fillna(0).astype(int)
            if t1c in merged.columns:
                merged[t1c] = merged[t1c].fillna(0).astype(int)
        merged["delta"] = merged.get("direct_reports_t1", 0) - merged.get("direct_reports_t", 0)
        merged["total_delta"] = merged.get("total_reports_t1", 0) - merged.get("total_reports_t", 0)
        return merged.sort_values("delta", key=abs, ascending=False).reset_index(drop=True)

    # ── 3. Subtree / team primitives ─────────────────────────────────────

    def subtree_at(
        self,
        manager_id: Any,
        period: str | None = None,
    ) -> "DuckDBPyRelation":
        """All transitive descendants of a manager at a period.

        Parameters
        ----------
        manager_id : Any
        period : str, optional
            Defaults to the latest period.

        Returns
        -------
        DuckDBPyRelation
            Columns: ``manager_id, employee_id, depth, path``.
        """
        p = self._p
        if not p._loaded:
            raise RuntimeError("call load_snapshots() first")
        if period is None:
            period = p.periods[-1]
        # Build a temp table on the custom connection with this period's edges
        import uuid as _uuid
        tbl = f"_subtree_{_uuid.uuid4().hex[:8]}"
        p.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE {tbl} AS
            SELECT "{p._emp_col}" AS employee_id, "{p._sup_col}" AS supervisor_id
            FROM {p._table_name}
            WHERE date_trunc('month', CAST("{p._date_col}" AS DATE))
                  = CAST('{period}' AS DATE)
        """)
        emp = "employee_id"
        sup = "supervisor_id"
        sql = f"""
        WITH RECURSIVE
        base AS (SELECT {emp} AS emp, {sup} AS sup FROM {tbl}),
        chain AS (
            SELECT sup AS employee_id, sup AS supervisor_id, 1 AS depth,
                   CAST(sup AS VARCHAR) AS path
            FROM base
            WHERE sup = '{manager_id}'
            UNION ALL
            SELECT b.emp AS employee_id, c.supervisor_id AS supervisor_id,
                   c.depth + 1 AS depth,
                   c.path || '->' || CAST(b.emp AS VARCHAR)
            FROM base b
            JOIN chain c ON b.sup = c.employee_id
            WHERE c.depth < 50
        )
        SELECT '{manager_id}' AS manager_id, employee_id, depth, path
        FROM chain
        """
        return p.con.sql(sql)

    def subtree_size_at(
        self,
        manager_id: Any,
        period: str | None = None,
    ) -> int:
        """Just the count of transitive descendants at a period."""
        rel = self.subtree_at(manager_id, period)
        return int(rel.count("*").fetchone()[0])

    def subtree_growth(
        self,
        manager_id: Any,
        lookback: str = "4Q",
    ) -> pd.DataFrame:
        """Subtree size over time.

        Returns
        -------
        pandas.DataFrame
            Columns: ``period, subtree_size, delta``.
        """
        p = self._p
        if not p._loaded:
            raise RuntimeError("call load_snapshots() first")
        n_periods, _ = self._parse_lookback(lookback)
        use_periods = p.periods[-n_periods:] if len(p.periods) >= n_periods else p.periods

        rows: list[dict] = []
        prev = None
        for period in use_periods:
            size = self.subtree_size_at(manager_id, period)
            delta = (size - prev) if prev is not None else np.nan
            rows.append({"period": period, "subtree_size": size, "delta": delta})
            prev = size
        return pd.DataFrame(rows)

    def subtree_overlap(
        self,
        manager_a: Any,
        manager_b: Any,
        period: str | None = None,
    ) -> dict:
        """Shared descendants between two managers at a period.

        Returns
        -------
        dict
            ``{"shared": DataFrame, "only_a": DataFrame, "only_b": DataFrame,
            "jaccard": float}``.
        """
        p = self._p
        if not p._loaded:
            raise RuntimeError("call load_snapshots() first")
        if period is None:
            period = p.periods[-1]
        rel_a = self.subtree_at(manager_a, period).df()
        rel_b = self.subtree_at(manager_b, period).df()
        set_a = set(rel_a["employee_id"].tolist()) if not rel_a.empty else set()
        set_b = set(rel_b["employee_id"].tolist()) if not rel_b.empty else set()
        shared = set_a & set_b
        only_a = set_a - set_b
        only_b = set_b - set_a
        union_size = len(set_a | set_b)
        jaccard = (len(shared) / union_size) if union_size > 0 else 0.0
        return {
            "shared": pd.DataFrame({"employee_id": list(shared)}),
            "only_a": pd.DataFrame({"employee_id": list(only_a)}),
            "only_b": pd.DataFrame({"employee_id": list(only_b)}),
            "jaccard": float(jaccard),
        }

    # ── 4. Snapshot-comparison primitives ───────────────────────────────

    def delta_table(
        self,
        period_t: str,
        period_t1: str,
        metric: str = "betweenness",
    ) -> pd.DataFrame:
        """Per-employee delta for a metric between two periods.

        Returns
        -------
        pandas.DataFrame
            Columns: ``employee_id, value_t, value_t1, delta, pct_change``.
        """
        p = self._p
        if not p._loaded:
            raise RuntimeError("call load_snapshots() first")
        df_t = _compute_metric_for_period(
            p.con, p._table_name, p._emp_col, p._sup_col, p._date_col,
            period_t, metric,
        )
        df_t1 = _compute_metric_for_period(
            p.con, p._table_name, p._emp_col, p._sup_col, p._date_col,
            period_t1, metric,
        )
        merged = df_t.merge(df_t1, on="employee_id", how="outer", suffixes=("_t", "_t1"))
        merged = merged.rename(columns={
            f"{metric}_t": "value_t",
            f"{metric}_t1": "value_t1",
        })
        merged["value_t"] = pd.to_numeric(merged["value_t"], errors="coerce")
        merged["value_t1"] = pd.to_numeric(merged["value_t1"], errors="coerce")
        merged["delta"] = merged["value_t1"] - merged["value_t"]
        merged["pct_change"] = np.where(
            merged["value_t"].notna() & (merged["value_t"] != 0),
            merged["delta"] / merged["value_t"] * 100,
            np.nan,
        )
        return merged.sort_values("delta", key=abs, ascending=False).reset_index(drop=True)

    def new_centers(
        self,
        period_t: str,
        period_t1: str,
        metric: str = "pagerank",
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Employees who joined the top-N list at t1 (but not at t).

        Returns
        -------
        pandas.DataFrame
            Columns: ``employee_id, value_t, value_t1, delta``.
        """
        dt = self.delta_table(period_t, period_t1, metric)
        # At t, the top-N
        dt_t = dt.dropna(subset=["value_t"]).sort_values("value_t", ascending=False).head(top_n)
        top_t = set(dt_t["employee_id"].tolist())
        # At t1, the top-N
        dt_t1 = dt.dropna(subset=["value_t1"]).sort_values("value_t1", ascending=False).head(top_n)
        top_t1 = set(dt_t1["employee_id"].tolist())
        new_entrants = top_t1 - top_t
        result = dt[dt["employee_id"].isin(new_entrants)].copy()
        return result.sort_values("delta", ascending=False).head(top_n).reset_index(drop=True)

    def fallen_centers(
        self,
        period_t: str,
        period_t1: str,
        metric: str = "pagerank",
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Employees who dropped out of the top-N list between t and t1."""
        dt = self.delta_table(period_t, period_t1, metric)
        dt_t = dt.dropna(subset=["value_t"]).sort_values("value_t", ascending=False).head(top_n)
        top_t = set(dt_t["employee_id"].tolist())
        dt_t1 = dt.dropna(subset=["value_t1"]).sort_values("value_t1", ascending=False).head(top_n)
        top_t1 = set(dt_t1["employee_id"].tolist())
        fallers = top_t - top_t1
        result = dt[dt["employee_id"].isin(fallers)].copy()
        return result.sort_values("delta").head(top_n).reset_index(drop=True)

    def cohort_compare(
        self,
        cohort_filter: str,
        metric: str,
        period_t: str,
        period_t1: str,
    ) -> pd.DataFrame:
        """Compare a SQL-filtered cohort across two periods.

        Parameters
        ----------
        cohort_filter : str
            DuckDB boolean expression on the snapshot table, e.g.
            ``"department = 'Engineering'"`` or ``"job_level >= 3"``.
        metric : str
        period_t, period_t1 : str

        Returns
        -------
        pandas.DataFrame
            Columns: ``employee_id, value_t, value_t1, delta, pct_change``.
        """
        p = self._p
        if not p._loaded:
            raise RuntimeError("call load_snapshots() first")
        # Pull the cohort at each period
        sql_t = f"""
            SELECT * FROM {p._table_name}
            WHERE date_trunc('month', CAST("{p._date_col}" AS DATE))
                  = CAST('{period_t}' AS DATE)
              AND ({cohort_filter})
        """
        sql_t1 = f"""
            SELECT * FROM {p._table_name}
            WHERE date_trunc('month', CAST("{p._date_col}" AS DATE))
                  = CAST('{period_t1}' AS DATE)
              AND ({cohort_filter})
        """
        # Register temp cohort tables
        self._p.con.execute(f"CREATE OR REPLACE TEMP TABLE _cohort_t AS {sql_t}")
        self._p.con.execute(f"CREATE OR REPLACE TEMP TABLE _cohort_t1 AS {sql_t1}")

        df_t = _compute_metric_for_period(
            p.con, "_cohort_t", p._emp_col, p._sup_col, p._date_col,
            period_t, metric,
        )
        df_t1 = _compute_metric_for_period(
            p.con, "_cohort_t1", p._emp_col, p._sup_col, p._date_col,
            period_t1, metric,
        )
        merged = df_t.merge(df_t1, on="employee_id", how="outer", suffixes=("_t", "_t1"))
        merged = merged.rename(columns={
            f"{metric}_t": "value_t",
            f"{metric}_t1": "value_t1",
        })
        merged["value_t"] = pd.to_numeric(merged["value_t"], errors="coerce")
        merged["value_t1"] = pd.to_numeric(merged["value_t1"], errors="coerce")
        merged["delta"] = merged["value_t1"] - merged["value_t"]
        merged["pct_change"] = np.where(
            merged["value_t"].notna() & (merged["value_t"] != 0),
            merged["delta"] / merged["value_t"] * 100,
            np.nan,
        )
        return merged.sort_values("delta", key=abs, ascending=False).reset_index(drop=True)

    # ── 5. Time-window aggregate primitives ─────────────────────────────

    def window_mean(
        self,
        metric: str,
        lookback: str = "4Q",
    ) -> pd.DataFrame:
        """Org-wide mean of a metric per period across the window.

        Returns
        -------
        pandas.DataFrame
            Columns: ``period, mean_value, n_employees``.
        """
        p = self._p
        if not p._loaded:
            raise RuntimeError("call load_snapshots() first")
        n_periods, _ = self._parse_lookback(lookback)
        use_periods = p.periods[-n_periods:] if len(p.periods) >= n_periods else p.periods

        rows: list[dict] = []
        for period in use_periods:
            df = _compute_metric_for_period(
                p.con, p._table_name, p._emp_col, p._sup_col, p._date_col,
                period, metric,
            )
            if df.empty:
                rows.append({"period": period, "mean_value": np.nan, "n_employees": 0})
            else:
                rows.append({
                    "period": period,
                    "mean_value": float(df[metric].mean()),
                    "n_employees": int(len(df)),
                })
        return pd.DataFrame(rows)

    def window_trend(
        self,
        metric: str,
        lookback: str = "4Q",
        aggregate: str = "mean",
    ) -> dict:
        """Linear slope of the org-wide aggregate across the window.

        Parameters
        ----------
        metric : str
        lookback : str
        aggregate : {"mean", "median", "sum"}
            Which aggregate to use as the per-period series.

        Returns
        -------
        dict
            ``{"slope": float, "intercept": float, "r_squared": float,
            "direction": "up"|"down"|"flat", "periods": list[str]}``.
        """
        p = self._p
        if not p._loaded:
            raise RuntimeError("call load_snapshots() first")
        agg_df = self.window_mean(metric, lookback)
        if agg_df.empty or agg_df["mean_value"].isna().all():
            return {
                "slope": 0.0, "intercept": 0.0, "r_squared": 0.0,
                "direction": "flat", "periods": [],
            }
        agg = agg_df["mean_value"].values
        if aggregate == "median":
            agg = agg_df["mean_value"].values  # window_mean returns mean; we extend below
        # Compute linear regression on the available points
        valid = ~np.isnan(agg)
        if valid.sum() < 2:
            return {
                "slope": 0.0, "intercept": float(agg[valid][0]) if valid.any() else 0.0,
                "r_squared": 0.0,
                "direction": "flat",
                "periods": list(agg_df["period"].tolist()),
            }
        xs = np.arange(len(agg))[valid].astype(float)
        ys = agg[valid].astype(float)
        coeffs = np.polyfit(xs, ys, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])
        # R²
        y_pred = np.polyval(coeffs, xs)
        ss_res = float(np.sum((ys - y_pred) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if abs(slope) < 1e-9:
            direction = "flat"
        elif slope > 0:
            direction = "up"
        else:
            direction = "down"
        return {
            "slope": slope, "intercept": intercept, "r_squared": r2,
            "direction": direction,
            "periods": list(agg_df["period"].tolist()),
        }

    def window_rank_change(
        self,
        metric: str,
        employee_id: Any,
        lookback: str = "4Q",
    ) -> pd.DataFrame:
        """Where this employee ranked at each period in the window.

        Returns
        -------
        pandas.DataFrame
            Columns: ``period, value, rank, n_total``.
        """
        p = self._p
        if not p._loaded:
            raise RuntimeError("call load_snapshots() first")
        n_periods, _ = self._parse_lookback(lookback)
        use_periods = p.periods[-n_periods:] if len(p.periods) >= n_periods else p.periods

        rows: list[dict] = []
        for period in use_periods:
            df = _compute_metric_for_period(
                p.con, p._table_name, p._emp_col, p._sup_col, p._date_col,
                period, metric,
            )
            if df.empty:
                rows.append({"period": period, "value": np.nan, "rank": np.nan, "n_total": 0})
                continue
            df_sorted = df.sort_values(metric, ascending=False).reset_index(drop=True)
            n_total = len(df_sorted)
            match = df_sorted[df_sorted["employee_id"] == employee_id]
            if match.empty:
                rows.append({"period": period, "value": np.nan, "rank": np.nan, "n_total": n_total})
                continue
            rank = int(match.index[0]) + 1
            val = float(match.iloc[0][metric])
            rows.append({"period": period, "value": val, "rank": rank, "n_total": n_total})
        return pd.DataFrame(rows)

    def window_volatility(
        self,
        metric: str,
        lookback: str = "4Q",
    ) -> pd.DataFrame:
        """Per-employee std-dev of a metric across the window.

        Returns
        -------
        pandas.DataFrame
            Columns: ``employee_id, std_value, mean_value, n_periods``.
        """
        p = self._p
        if not p._loaded:
            raise RuntimeError("call load_snapshots() first")
        n_periods, _ = self._parse_lookback(lookback)
        use_periods = p.periods[-n_periods:] if len(p.periods) >= n_periods else p.periods

        frames: list[pd.DataFrame] = []
        for period in use_periods:
            df = _compute_metric_for_period(
                p.con, p._table_name, p._emp_col, p._sup_col, p._date_col,
                period, metric,
            )
            if df.empty:
                continue
            frames.append(df.set_index("employee_id").rename(columns={metric: period}))
        if not frames:
            return pd.DataFrame(columns=["employee_id", "std_value", "mean_value", "n_periods"])
        wide = pd.concat(frames, axis=1)
        result = pd.DataFrame({
            "employee_id": wide.index,
            "std_value": wide.std(axis=1).values,
            "mean_value": wide.mean(axis=1).values,
            "n_periods": wide.notna().sum(axis=1).values,
        }).reset_index(drop=True)
        return result.sort_values("std_value", ascending=False).reset_index(drop=True)

    # ── Helper exposed so callers can use the same parser ────────────────

    def _parse_lookback(self, lookback: str) -> tuple[int, str]:
        """Parse lookback string. Delegates to module-level helper."""
        from pyduck_ona.temporal import _parse_lookback as _pl
        return _pl(lookback)
