"""Temporal ONA: time-series analysis of organizational networks.

``DuckONATemporal`` extends pyduck-ona's point-in-time analytics to
multi-period HRIS snapshots.  It computes per-employee ONA metrics
across time, detects mobility and career trajectories, scores manager
effectiveness, and flags structural drift in the org chart.

The class accepts a single HRIS DataFrame with a ``snapshot_date``
column (the realistic shape for Workday / SAP / SuccessFactors
delta extracts).  Internal period alignment uses DuckDB's
``date_trunc``, so monthly / quarterly / yearly slicing is exact.

Eight public methods cover the four buckets of temporal ONA
questions:

    Node-level trajectory:
        1. ``compute_temporal_metrics`` — per-employee ONA time-series
        2. ``change_detection``              — top movers / anomalies

    Structural drift:
        3. ``network_evolution`` — aggregate network-shape over time

    Event study:
        4. ``event_window`` — before/after a specific date

    Mobility & career path:
        5. ``mobility_leaderboard`` — top movers by composite score
        6. ``career_trajectory`` + ``manager_chain`` — per-employee path
        7. ``mobility_anomaly`` — peer-relative stuckness z-score

    Manager effectiveness:
        8. ``manager_effectiveness`` — composite score (engagement-dominant)

Example
-------
>>> from pyduck_ona import DuckONATemporal
>>> import pandas as pd
>>> dt = DuckONATemporal()
>>> dt.load_snapshots(hris_df, snapshot_date_col="snapshot_date", freq="Q")
>>> ts = dt.compute_temporal_metrics(metrics=["betweenness", "pagerank"])
>>> movers = dt.mobility_leaderboard(lookback="4Q", top_n=20)
>>> eff = dt.manager_effectiveness(lookback="4Q")
"""

from __future__ import annotations

import re
import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Literal

import duckdb
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from duckdb import DuckDBPyRelation

from pyduck_ona import graph as _graph
from pyduck_ona import stats as _stats
from pyduck_ona.core import hierarchy_long, hierarchy_stats
from pyduck_ona.temporal_primitives import _QueryPrimitives

# ─── Constants ─────────────────────────────────────────────────────────────

_IDENT_SAFE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_FREQ_MAP: dict[str, str] = {
    "D": "day",
    "W": "week",
    "M": "month",
    "Q": "quarter",
    "Y": "year",
}

_LOOKBACK_MAP: dict[str, int] = {
    "1M": 1, "2M": 2, "3M": 3, "4M": 4, "6M": 6, "12M": 12,
    "1Q": 1, "2Q": 2, "3Q": 3, "4Q": 4, "6Q": 6, "8Q": 8,
    "1Y": 1, "2Y": 2, "3Y": 3, "4Y": 4, "5Y": 5,
}


# ─── Helpers ──────────────────────────────────────────────────────────────

def _quote(name: str) -> str:
    """Safely quote a DuckDB identifier."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"identifier must be non-empty string, got {name!r}")
    if _IDENT_SAFE_RE.match(name):
        return f'"{name}"'
    return f'"{name.replace(chr(34), chr(34)*2)}"'


def _validate_table(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("table name must be non-empty string")
    if not _IDENT_SAFE_RE.match(name):
        raise ValueError(f"unsafe table name: {name!r}")


def _parse_lookback(lookback: str) -> tuple[int, str]:
    """Parse '4Q' → (4, 'quarter'), '12M' → (12, 'month'), '3Y' → (3, 'year')."""
    lookback = lookback.strip()
    if lookback not in _LOOKBACK_MAP:
        raise ValueError(
            f"lookback must be like '4Q', '12M', '3Y'; got {lookback!r}"
        )
    n = _LOOKBACK_MAP[lookback]
    freq_char = lookback[-1]
    freq_word = _FREQ_MAP[freq_char]
    return n, freq_word


def _periods_from_snapshots(
    con: duckdb.DuckDBPyConnection,
    table: str,
    date_col: str,
    freq: str,
) -> list[str]:
    """Return sorted list of period labels from the snapshot table."""
    freq_word = _FREQ_MAP.get(freq.upper())
    if freq_word is None:
        raise ValueError(f"freq must be one of {list(_FREQ_MAP)}, got {freq!r}")
    df = con.sql(f"""
        SELECT DISTINCT date_trunc('{freq_word}', CAST({_quote(date_col)} AS DATE)) AS period
        FROM {table}
        WHERE {_quote(date_col)} IS NOT NULL
        ORDER BY period
    """).df()
    if df.empty:
        return []
    return [pd.Timestamp(p).strftime("%Y-%m-%d") for p in df["period"]]


def _edges_for_period(
    con: duckdb.DuckDBPyConnection,
    table: str,
    emp_col: str,
    sup_col: str,
    date_col: str,
    period_label: str,
    freq: str,
) -> "DuckDBPyRelation":
    """Return edge relation (emp, sup) for a single period snapshot."""
    freq_word = _FREQ_MAP.get(freq.upper())
    if freq_word is None:
        raise ValueError(f"freq must be one of {list(_FREQ_MAP)}, got {freq!r}")
    sql = f"""
        SELECT DISTINCT
            {_quote(emp_col)} AS {_quote(emp_col)},
            {_quote(sup_col)} AS {_quote(sup_col)}
        FROM {table}
        WHERE date_trunc('{freq_word}', CAST({_quote(date_col)} AS DATE))
              = CAST('{period_label}' AS DATE)
          AND {_quote(sup_col)} IS NOT NULL
          AND CAST({_quote(sup_col)} AS VARCHAR) <> ''
    """
    return con.sql(sql)


def _subtree_ids(
    con: duckdb.DuckDBPyConnection,
    table: str,
    emp_col: str,
    sup_col: str,
    date_col: str,
    manager_id: Any,
    period_label: str,
    freq: str,
) -> list[Any]:
    """Return all employee_ids in manager's transitive subtree for a period."""
    freq_word = _FREQ_MAP.get(freq.upper())
    if freq_word is None:
        raise ValueError(f"freq must be one of {list(_FREQ_MAP)}, got {freq!r}")
    # Get the snapshot rows for this period
    rel = con.sql(f"""
        SELECT {_quote(emp_col)} AS emp, {_quote(sup_col)} AS sup
        FROM {table}
        WHERE date_trunc('{freq_word}', CAST({_quote(date_col)} AS DATE))
              = CAST('{period_label}' AS DATE)
    """)
    df = rel.df()
    if df.empty:
        return []
    # Build adjacency: supervisor → list of direct reports
    children: dict[Any, list[Any]] = {}
    for _, row in df.iterrows():
        sup = row["sup"]
        emp = row["emp"]
        if pd.isna(sup) or sup is None or str(sup).strip() == "":
            continue
        children.setdefault(sup, []).append(emp)
    # BFS from manager_id
    subtree: list[Any] = []
    queue = [manager_id]
    seen: set = set()
    while queue:
        node = queue.pop(0)
        if node in seen:
            continue
        seen.add(node)
        for child in children.get(node, []):
            if child not in seen:
                subtree.append(child)
                queue.append(child)
    return subtree


def _safe_zscore(values: np.ndarray, peer_median: float, peer_std: float) -> float:
    """Compute z-score; return 0.0 if peer_std is 0 or NaN."""
    if peer_std is None or peer_std == 0 or np.isnan(peer_std):
        return 0.0
    return float((peer_median - values) / peer_std)


# ─── Main class ─────────────────────────────────────────────────────────────


class DuckONATemporal:
    """A DuckDB-backed temporal ONA workspace.

    Parameters
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
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.con = duckdb.connect(db_path)
        self._table_name: str = "snapshots"
        self._emp_col: str = "employee_id"
        self._sup_col: str = "supervisor_id"
        self._date_col: str = "snapshot_date"
        self._freq: str = "Q"
        self._periods: list[str] = []
        self._loaded: bool = False
        self._extra_tables: dict[str, str] = {}  # name → registered table
        # Query primitives namespace (20 tools across 5 categories)
        self.q = _QueryPrimitives(self)

    # ── Loading ─────────────────────────────────────────────────────────────

    def load_snapshots(
        self,
        df: pd.DataFrame,
        snapshot_date_col: str = "snapshot_date",
        employee_id_col: str = "employee_id",
        supervisor_id_col: str = "supervisor_id",
        freq: str = "Q",
        table_name: str = "snapshots",
    ) -> list[str]:
        """Load HRIS snapshot data.

        The DataFrame must contain at least:
            - ``employee_id_col`` — unique employee identifier
            - ``supervisor_id_col`` — manager identifier (NULL for root)
            - ``snapshot_date_col`` — date of the HRIS extract

        Additional columns (job_level, department, salary, engagement, etc.)
        are preserved for downstream analysis.

        Parameters
        ----------
        df : pandas.DataFrame
        snapshot_date_col : str
        employee_id_col : str
        supervisor_id_col : str
        freq : {"M", "Q", "Y"}
            Period frequency for all subsequent analysis.
        table_name : str
            DuckDB table name for the registered snapshots.

        Returns
        -------
        list[str]
            Sorted period labels detected in the data.
        """
        _validate_table(table_name)
        self._table_name = table_name
        self._emp_col = employee_id_col
        self._sup_col = supervisor_id_col
        self._date_col = snapshot_date_col
        self._freq = freq.upper()

        if snapshot_date_col not in df.columns:
            raise ValueError(f"{snapshot_date_col!r} not in DataFrame columns: {list(df.columns)}")
        if employee_id_col not in df.columns:
            raise ValueError(f"{employee_id_col!r} not in DataFrame columns")
        if supervisor_id_col not in df.columns:
            raise ValueError(f"{supervisor_id_col!r} not in DataFrame columns")

        self.con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
        self._loaded = True
        self._periods = _periods_from_snapshots(
            self.con, table_name, snapshot_date_col, self._freq
        )
        return self._periods

    def load_survey(self, df: pd.DataFrame, table_name: str = "survey") -> None:
        """Load a survey / engagement table with employee_id + snapshot_date."""
        _validate_table(table_name)
        self.con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
        self._extra_tables["survey"] = table_name

    def load_promotions(self, df: pd.DataFrame, table_name: str = "promotions") -> None:
        """Load a promotions / internal-mobility event table."""
        _validate_table(table_name)
        self.con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
        self._extra_tables["promotions"] = table_name

    @property
    def periods(self) -> list[str]:
        return self._periods

    @property
    def freq(self) -> str:
        return self._freq

    # ── 1. compute_temporal_metrics ────────────────────────────────────────

    def compute_temporal_metrics(
        self,
        metrics: list[str] | None = None,
        employee_id_col: str | None = None,
        supervisor_id_col: str | None = None,
    ) -> pd.DataFrame:
        """Per-employee ONA metric time-series across all periods.

        For each period, builds the org-chart edge relation, computes
        the requested graph metrics, and stacks them into a long-format
        DataFrame with delta and pct_change vs. the previous period.

        Parameters
        ----------
        metrics : list[str], optional
            Metric names. Default: ``["betweenness", "pagerank",
            "degree_centrality", "team_size"]``.
        employee_id_col, supervisor_id_col : str, optional
            Override the column names set by ``load_snapshots``.

        Returns
        -------
        pandas.DataFrame
            Columns: ``period, employee_id, metric, value, prev_value,
            delta, pct_change``.
        """
        if not self._loaded:
            raise RuntimeError("call load_snapshots() first")
        emp = employee_id_col or self._emp_col
        sup = supervisor_id_col or self._sup_col
        freq_word = _FREQ_MAP[self._freq]
        if metrics is None:
            metrics = ["betweenness", "pagerank", "degree_centrality", "team_size"]

        metric_fn_map = {
            "betweenness": _graph.betweenness,
            "pagerank": _graph.pagerank,
            "eigenvector_centrality": _graph.eigenvector_centrality,
            "degree_centrality": _graph.degree_centrality,
            "connected_components": _graph.connected_components,
            "louvain_communities": _graph.louvain_communities,
        }

        all_rows: list[dict] = []
        prev_values: dict[str, dict[Any, float]] = {}  # metric → {emp_id: value}

        for period in self._periods:
            edges = _edges_for_period(
                self.con, self._table_name, emp, sup, self._date_col, period, self._freq
            )
            n_edges = edges.count("*").fetchone()[0]
            if n_edges == 0:
                continue

            for metric_name in metrics:
                if metric_name == "team_size":
                    # team_size via hierarchy_stats
                    stats_rel = hierarchy_stats(
                        self.con.sql(
                            f"SELECT {emp} AS employee_id, {sup} AS supervisor_id "
                            f"FROM {self._table_name} "
                            f"WHERE date_trunc('{freq_word}', CAST({_quote(self._date_col)} AS DATE))"
                            f"      = CAST('{period}' AS DATE)"
                        ),
                        "employee_id",
                        "supervisor_id",
                    )
                    stats_df = stats_rel.df()
                    for _, row in stats_df.iterrows():
                        eid = row["manager_id"]
                        val = float(row["team_size"])
                        prev = prev_values.get(metric_name, {}).get(eid, np.nan)
                        delta = val - prev if not np.isnan(prev) else np.nan
                        pct = (delta / prev * 100) if (not np.isnan(prev) and prev != 0) else np.nan
                        all_rows.append({
                            "period": period, "employee_id": eid,
                            "metric": metric_name, "value": val,
                            "prev_value": prev, "delta": delta, "pct_change": pct,
                        })
                        prev_values.setdefault(metric_name, {})[eid] = val
                elif metric_name in metric_fn_map:
                    fn = metric_fn_map[metric_name]
                    rel = fn(edges, emp, sup)
                    df = rel.df()
                    id_col = "node_id" if "node_id" in df.columns else df.columns[0]
                    val_col = [c for c in df.columns if c != id_col][0]
                    for _, row in df.iterrows():
                        eid = row[id_col]
                        val = float(row[val_col])
                        prev = prev_values.get(metric_name, {}).get(eid, np.nan)
                        delta = val - prev if not np.isnan(prev) else np.nan
                        pct = (delta / prev * 100) if (not np.isnan(prev) and prev != 0) else np.nan
                        all_rows.append({
                            "period": period, "employee_id": eid,
                            "metric": metric_name, "value": val,
                            "prev_value": prev, "delta": delta, "pct_change": pct,
                        })
                        prev_values.setdefault(metric_name, {})[eid] = val
                else:
                    raise ValueError(
                        f"unknown metric {metric_name!r}; "
                        f"choose from {list(metric_fn_map.keys()) + ['team_size']}"
                    )

        return pd.DataFrame(all_rows)

    # ── 2. network_evolution ────────────────────────────────────────────────

    def network_evolution(
        self,
        employee_id_col: str | None = None,
        supervisor_id_col: str | None = None,
    ) -> pd.DataFrame:
        """Aggregate network-shape metrics per period.

        Returns
        -------
        pandas.DataFrame
            Columns: ``period, n_employees, n_edges, density,
            centralization, n_components, avg_path_length``.
        """
        import networkx as nx

        if not self._loaded:
            raise RuntimeError("call load_snapshots() first")
        emp = employee_id_col or self._emp_col
        sup = supervisor_id_col or self._sup_col
        freq_word = _FREQ_MAP[self._freq]

        rows: list[dict] = []
        for period in self._periods:
            edges = _edges_for_period(
                self.con, self._table_name, emp, sup, self._date_col, period, self._freq
            )
            df = edges.df()
            n_emp = df[emp].nunique() + df[sup].nunique()  # nodes appearing in edges
            # Also count from the snapshot to get total employees
            snap_df = self.con.sql(f"""
                SELECT DISTINCT {_quote(emp)} AS e FROM {self._table_name}
                WHERE date_trunc('{freq_word}', CAST({_quote(self._date_col)} AS DATE))
                      = CAST('{period}' AS DATE)
            """).df()
            n_total = len(snap_df)
            n_edges = len(df)

            if n_edges == 0:
                rows.append({
                    "period": period, "n_employees": n_total,
                    "n_edges": 0, "density": 0.0, "centralization": 0.0,
                    "n_components": 0, "avg_path_length": np.nan,
                })
                continue

            # Build NetworkX graph
            pairs = [(s, t) for s, t in zip(df[emp], df[sup]) if pd.notna(s) and pd.notna(t)]
            G = nx.DiGraph()
            G.add_edges_from(pairs)

            n_nodes = G.number_of_nodes()
            max_possible = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
            density = nx.density(G) if max_possible > 0 else 0.0

            # Freeman centralization (out-degree)
            degrees = dict(G.out_degree())
            max_deg = max(degrees.values()) if degrees else 0
            if max_deg > 0 and n_nodes > 1:
                centralization = sum(max_deg - d for d in degrees.values()) / (
                    (n_nodes - 1) * max_deg
                )
            else:
                centralization = 0.0

            n_comp = nx.number_weakly_connected_components(G)

            # Avg shortest path (on undirected, largest component)
            try:
                largest_cc = max(nx.weakly_connected_components(G), key=len)
                sub = G.subgraph(largest_cc).to_undirected()
                avg_pl = nx.average_shortest_path_length(sub) if sub.number_of_nodes() > 1 else 0.0
            except Exception:
                avg_pl = np.nan

            rows.append({
                "period": period, "n_employees": n_total, "n_edges": n_edges,
                "density": float(density), "centralization": float(centralization),
                "n_components": int(n_comp), "avg_path_length": float(avg_pl) if not np.isnan(avg_pl) else np.nan,
            })

        return pd.DataFrame(rows)

    # ── 3. event_window ─────────────────────────────────────────────────────

    def event_window(
        self,
        event_date: str,
        pre_window: tuple[str, str] | None = None,
        post_window: tuple[str, str] | None = None,
        metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        """Before/after comparison around a specific event date.

        Parameters
        ----------
        event_date : str
            ISO date. Periods before this date are "pre"; after are "post".
        pre_window, post_window : tuple(str, str), optional
            (start_date, end_date) to bound the pre/post windows. If
            omitted, all periods before / after event_date are used.
        metrics : list[str], optional
            ONA metrics to compute. Default: betweenness + pagerank.

        Returns
        -------
        pandas.DataFrame
            Columns: ``period, period_type ("pre"|"post"), employee_id,
            metric, value``.
        """
        if not self._loaded:
            raise RuntimeError("call load_snapshots() first")
        if metrics is None:
            metrics = ["betweenness", "pagerank"]

        event_ts = pd.Timestamp(event_date)
        pre_periods = [p for p in self._periods if pd.Timestamp(p) < event_ts]
        post_periods = [p for p in self._periods if pd.Timestamp(p) >= event_ts]

        if pre_window:
            pre_periods = [p for p in pre_periods if pre_window[0] <= p <= pre_window[1]]
        if post_window:
            post_periods = [p for p in post_periods if post_window[0] <= p <= post_window[1]]

        rows: list[dict] = []
        metric_fn_map = {
            "betweenness": _graph.betweenness,
            "pagerank": _graph.pagerank,
            "eigenvector_centrality": _graph.eigenvector_centrality,
            "degree_centrality": _graph.degree_centrality,
        }

        for ptype, periods in [("pre", pre_periods), ("post", post_periods)]:
            for period in periods:
                edges = _edges_for_period(
                    self.con, self._table_name, self._emp_col, self._sup_col,
                    self._date_col, period, self._freq,
                )
                if edges.count("*").fetchone()[0] == 0:
                    continue
                for mname in metrics:
                    if mname not in metric_fn_map:
                        continue
                    rel = metric_fn_map[mname](edges, self._emp_col, self._sup_col)
                    df = rel.df()
                    id_col = "node_id" if "node_id" in df.columns else df.columns[0]
                    val_col = [c for c in df.columns if c != id_col][0]
                    for _, row in df.iterrows():
                        rows.append({
                            "period": period, "period_type": ptype,
                            "employee_id": row[id_col],
                            "metric": mname, "value": float(row[val_col]),
                        })

        return pd.DataFrame(rows)

    # ── 4. change_detection ────────────────────────────────────────────────

    def change_detection(
        self,
        metric: str = "betweenness",
        top_n: int = 20,
        lookback: str = "4Q",
    ) -> pd.DataFrame:
        """Top movers for a given metric over the lookback window.

        Parameters
        ----------
        metric : str
            ONA metric name (betweenness, pagerank, etc.).
        top_n : int
            Number of top movers to return.
        lookback : str
            e.g. "4Q", "12M", "3Y".

        Returns
        -------
        pandas.DataFrame
            Columns: ``employee_id, start_value, end_value, delta,
            pct_change, rolling_zscore, rank``.
        """
        n_periods, _ = _parse_lookback(lookback)
        all_periods = self._periods
        if len(all_periods) < 2:
            raise ValueError(f"need ≥2 periods for change_detection; have {len(all_periods)}")
        use_periods = all_periods[-n_periods:] if len(all_periods) >= n_periods else all_periods

        ts = self.compute_temporal_metrics(metrics=[metric])
        ts = ts[ts["metric"] == metric].copy()
        ts = ts[ts["period"].isin(use_periods)]

        # Pivot: employee_id × period → value
        pivot = ts.pivot_table(index="employee_id", columns="period", values="value")
        # Only keep employees with data in both first and last period
        first_col = use_periods[0]
        last_col = use_periods[-1]
        pivot = pivot.dropna(subset=[first_col, last_col])

        pivot["start_value"] = pivot[first_col]
        pivot["end_value"] = pivot[last_col]
        pivot["delta"] = pivot["end_value"] - pivot["start_value"]
        pivot["pct_change"] = np.where(
            pivot["start_value"] != 0,
            pivot["delta"] / pivot["start_value"] * 100,
            np.nan,
        )
        # Rolling z-score of delta
        mean_delta = pivot["delta"].mean()
        std_delta = pivot["delta"].std()
        pivot["rolling_zscore"] = (
            (pivot["delta"] - mean_delta) / std_delta if std_delta > 0 else 0.0
        )
        pivot = pivot.sort_values("delta", key=abs, ascending=False)
        pivot["rank"] = range(1, len(pivot) + 1)
        result = pivot[["start_value", "end_value", "delta", "pct_change", "rolling_zscore", "rank"]].head(top_n)
        result = result.reset_index()
        return result

    # ── 5. mobility_leaderboard ─────────────────────────────────────────────

    def mobility_leaderboard(
        self,
        lookback: str = "4Q",
        top_n: int = 20,
        w_promotion: float = 1.0,
        w_lateral: float = 0.5,
        w_dept_change: float = 0.3,
        w_demotion: float = -1.0,
    ) -> pd.DataFrame:
        """Top movers by composite mobility score.

        Mobility score = w_promotion * n_promotions
                       + w_lateral   * n_lateral_moves
                       + w_dept_change * n_dept_changes
                       + w_demotion  * n_demotions

        A *promotion* is a period-over-period increase in job_level with a
        supervisor change. A *lateral move* is a supervisor change with
        level unchanged. A *dept change* is a department change with
        supervisor unchanged. A *demotion* is a level decrease.

        Parameters
        ----------
        lookback : str
        top_n : int
        w_promotion, w_lateral, w_dept_change, w_demotion : float
            Weights for each mobility event type.

        Returns
        -------
        pandas.DataFrame
        """
        if not self._loaded:
            raise RuntimeError("call load_snapshots() first")
        n_periods, _ = _parse_lookback(lookback)
        use_periods = self._periods[-n_periods:] if len(self._periods) >= n_periods else self._periods
        freq_word = _FREQ_MAP[self._freq]

        emp = self._emp_col
        sup = self._sup_col

        # Pull per-period snapshots with key columns
        snapshots: list[pd.DataFrame] = []
        for p in use_periods:
            df = self.con.sql(f"""
                SELECT * FROM {self._table_name}
                WHERE date_trunc('{freq_word}', CAST({_quote(self._date_col)} AS DATE))
                      = CAST('{p}' AS DATE)
            """).df()
            df["_period"] = p
            snapshots.append(df)

        if not snapshots or all(s.empty for s in snapshots):
            raise ValueError("no snapshot data found for the lookback window")

        # Build per-employee transition history
        all_emp_ids: set = set()
        for s in snapshots:
            if not s.empty:
                all_emp_ids.update(s[emp].tolist())

        rows: list[dict] = []
        for eid in all_emp_ids:
            periods_data: list[dict] = []
            for s in snapshots:
                if s.empty:
                    continue
                match = s[s[emp] == eid]
                if match.empty:
                    periods_data.append({"period": s["_period"].iloc[0] if not s.empty else None})
                    continue
                row = match.iloc[0]
                periods_data.append({
                    "period": row["_period"],
                    "supervisor_id": row.get(sup),
                    "job_level": row.get("job_level"),
                    "department": row.get("department"),
                })

            # Count transitions
            n_promotions = 0
            n_lateral = 0
            n_dept = 0
            n_demotions = 0
            n_manager_changes = 0
            for i in range(1, len(periods_data)):
                prev = periods_data[i - 1]
                curr = periods_data[i]
                if not prev or not curr:
                    continue
                if prev.get("period") is None or curr.get("period") is None:
                    continue

                prev_sup = prev.get("supervisor_id")
                curr_sup = curr.get("supervisor_id")
                prev_lvl = prev.get("job_level")
                curr_lvl = curr.get("job_level")
                prev_dept = prev.get("department")
                curr_dept = curr.get("department")

                # Coerce NaN→None for safe comparison (avoids pd.NA ambiguity)
                def _clean(v):
                    if v is None:
                        return None
                    try:
                        if pd.isna(v):
                            return None
                    except (TypeError, ValueError):
                        pass
                    return v
                p_sup = _clean(prev_sup)
                c_sup = _clean(curr_sup)
                p_lvl = _clean(prev_lvl)
                c_lvl = _clean(curr_lvl)
                p_dept = _clean(prev_dept)
                c_dept = _clean(curr_dept)

                mgr_changed = p_sup != c_sup
                lvl_changed = p_lvl != c_lvl
                dept_changed = p_dept != c_dept

                if mgr_changed:
                    n_manager_changes += 1
                if lvl_changed and c_lvl is not None and p_lvl is not None:
                    if c_lvl > p_lvl:
                        n_promotions += 1
                    else:
                        n_demotions += 1
                elif mgr_changed and not lvl_changed:
                    n_lateral += 1
                if dept_changed and not mgr_changed:
                    n_dept += 1

            score = (
                w_promotion * n_promotions
                + w_lateral * n_lateral
                + w_dept_change * n_dept
                + w_demotion * n_demotions
            )

            # First/last for reporting
            valid = [pd for pd in periods_data if pd.get("period") is not None]
            first = valid[0] if valid else {}
            last = valid[-1] if valid else {}

            rows.append({
                "employee_id": eid,
                "mobility_score": float(score),
                "n_promotions": n_promotions,
                "n_lateral_moves": n_lateral,
                "n_dept_changes": n_dept,
                "n_demotions": n_demotions,
                "n_manager_changes": n_manager_changes,
                "first_level": first.get("job_level"),
                "last_level": last.get("job_level"),
                "first_dept": first.get("department"),
                "last_dept": last.get("department"),
            })

        result = pd.DataFrame(rows)
        if result.empty:
            return result
        result = result.sort_values("mobility_score", ascending=False).head(top_n)
        result["rank"] = range(1, len(result) + 1)
        return result.reset_index(drop=True)

    # ── 6. career_trajectory + manager_chain ───────────────────────────────

    def career_trajectory(
        self,
        employee_id: Any,
        lookback: str = "4Q",
    ) -> pd.DataFrame:
        """Per-employee career path across periods.

        Parameters
        ----------
        employee_id : Any
        lookback : str

        Returns
        -------
        pandas.DataFrame
            Columns: ``period, supervisor_id, job_level, department,
            promoted, transferred, manager_changed``.
        """
        if not self._loaded:
            raise RuntimeError("call load_snapshots() first")
        n_periods, _ = _parse_lookback(lookback)
        use_periods = self._periods[-n_periods:] if len(self._periods) >= n_periods else self._periods
        emp = self._emp_col
        sup = self._sup_col
        freq_word = _FREQ_MAP[self._freq]
        freq_word = _FREQ_MAP[self._freq]
        freq_word = _FREQ_MAP[self._freq]

        rows: list[dict] = []
        prev_sup = None
        prev_lvl = None
        prev_dept = None

        for p in use_periods:
            df = self.con.sql(f"""
                SELECT * FROM {self._table_name}
                WHERE date_trunc('{freq_word}', CAST({_quote(self._date_col)} AS DATE))
                      = CAST('{p}' AS DATE)
                  AND {_quote(emp)} = '{employee_id}'
            """).df()
            if df.empty:
                rows.append({
                    "period": p, "supervisor_id": None,
                    "job_level": None, "department": None,
                    "promoted": False, "transferred": False,
                    "manager_changed": False,
                })
                prev_sup = prev_lvl = prev_dept = None
                continue
            row = df.iloc[0]
            curr_sup = row.get(sup)
            curr_lvl = row.get("job_level")
            curr_dept = row.get("department")

            promoted = (
                prev_lvl is not None and curr_lvl is not None and curr_lvl > prev_lvl
            )
            transferred = (
                prev_dept is not None and curr_dept is not None and curr_dept != prev_dept
            )
            manager_changed = (
                prev_sup is not None and curr_sup is not None and curr_sup != prev_sup
            )

            rows.append({
                "period": p, "supervisor_id": curr_sup,
                "job_level": curr_lvl, "department": curr_dept,
                "promoted": promoted, "transferred": transferred,
                "manager_changed": manager_changed,
            })
            prev_sup = curr_sup
            prev_lvl = curr_lvl
            prev_dept = curr_dept

        return pd.DataFrame(rows)

    def manager_chain(
        self,
        employee_id: Any,
        lookback: str = "4Q",
    ) -> pd.DataFrame:
        """Managers along the way for a given employee.

        Returns
        -------
        pandas.DataFrame
            Columns: ``period, supervisor_id, supervisor_name,
            supervisor_level, supervisor_path_to_ceo``.
        """
        if not self._loaded:
            raise RuntimeError("call load_snapshots() first")
        n_periods, _ = _parse_lookback(lookback)
        use_periods = self._periods[-n_periods:] if len(self._periods) >= n_periods else self._periods
        emp = self._emp_col
        sup = self._sup_col
        freq_word = _FREQ_MAP[self._freq]

        rows: list[dict] = []
        for p in use_periods:
            df = self.con.sql(f"""
                SELECT * FROM {self._table_name}
                WHERE date_trunc('{freq_word}', CAST({_quote(self._date_col)} AS DATE))
                      = CAST('{p}' AS DATE)
                  AND {_quote(emp)} = '{employee_id}'
            """).df()
            if df.empty:
                rows.append({
                    "period": p, "supervisor_id": None,
                    "supervisor_name": None, "supervisor_level": None,
                    "supervisor_path_to_ceo": [],
                })
                continue
            row = df.iloc[0]
            mgr_id = row.get(sup)
            # Look up manager's info in the same snapshot
            mgr_name = None
            mgr_level = None
            path: list = []
            if mgr_id is not None:
                mgr_df = self.con.sql(f"""
                    SELECT * FROM {self._table_name}
                    WHERE date_trunc('{freq_word}', CAST({_quote(self._date_col)} AS DATE))
                          = CAST('{p}' AS DATE)
                      AND {_quote(emp)} = '{mgr_id}'
                """).df()
                if not mgr_df.empty:
                    mgr_row = mgr_df.iloc[0]
                    mgr_name = mgr_row.get("name", mgr_row.get("employee_id"))
                    mgr_level = mgr_row.get("job_level")
                # Build path to CEO via hierarchy_long
                snap_rel = self.con.sql(f"""
                    SELECT {_quote(emp)} AS employee_id, {_quote(sup)} AS supervisor_id
                    FROM {self._table_name}
                    WHERE date_trunc('{freq_word}', CAST({_quote(self._date_col)} AS DATE))
                          = CAST('{p}' AS DATE)
                """)
                try:
                    long_rel = hierarchy_long(snap_rel, "employee_id", "supervisor_id")
                    long_df = long_rel.df()
                    # Walk up from employee
                    current = employee_id
                    visited: set = set()
                    while current is not None and current not in visited:
                        visited.add(current)
                        match = long_df[long_df["employee_id"] == current]
                        if match.empty:
                            break
                        ancestor = match.iloc[0]["supervisor_id"]
                        if ancestor is not None and not pd.isna(ancestor):
                            path.append(ancestor)
                            current = ancestor
                        else:
                            break
                except Exception:
                    pass

            rows.append({
                "period": p, "supervisor_id": mgr_id,
                "supervisor_name": mgr_name, "supervisor_level": mgr_level,
                "supervisor_path_to_ceo": path,
            })

        return pd.DataFrame(rows)

    # ── 7. mobility_anomaly ────────────────────────────────────────────────

    def mobility_anomaly(
        self,
        lookback: str = "4Q",
        peer_basis: list[str] | None = None,
        stuckness_threshold: float = 1.5,
    ) -> pd.DataFrame:
        """Peer-relative stuckness z-score per employee.

        Parameters
        ----------
        lookback : str
        peer_basis : list[str], optional
            Columns defining the peer group. Default:
            ``["job_level", "department"]`` from the first period.
        stuckness_threshold : float, default 1.5
            Z-score above which an employee is flagged as "stuck".

        Returns
        -------
        pandas.DataFrame
            Columns: ``employee_id, starting_level, starting_dept,
            mobility_events, peer_median, peer_std, stuckness_zscore,
            is_stuck, is_mobility_leader``.
        """
        if not self._loaded:
            raise RuntimeError("call load_snapshots() first")
        if peer_basis is None:
            peer_basis = ["job_level", "department"]

        n_periods, _ = _parse_lookback(lookback)
        use_periods = self._periods[-n_periods:] if len(self._periods) >= n_periods else self._periods
        emp = self._emp_col
        sup = self._sup_col
        freq_word = _FREQ_MAP[self._freq]

        # Gather per-employee mobility events
        snapshots: list[pd.DataFrame] = []
        for p in use_periods:
            df = self.con.sql(f"""
                SELECT * FROM {self._table_name}
                WHERE date_trunc('{freq_word}', CAST({_quote(self._date_col)} AS DATE))
                      = CAST('{p}' AS DATE)
            """).df()
            df["_period"] = p
            snapshots.append(df)

        all_emp_ids: set = set()
        for s in snapshots:
            if not s.empty:
                all_emp_ids.update(s[emp].tolist())

        emp_data: list[dict] = []
        for eid in all_emp_ids:
            periods_data: list[dict] = []
            for s in snapshots:
                if s.empty:
                    continue
                match = s[s[emp] == eid]
                if match.empty:
                    periods_data.append({"period": s["_period"].iloc[0]})
                    continue
                row = match.iloc[0]
                periods_data.append({
                    "period": row["_period"],
                    "supervisor_id": row.get(sup),
                    "job_level": row.get("job_level"),
                    "department": row.get("department"),
                })

            # Count mobility events
            events = 0
            valid = [pd for pd in periods_data if pd.get("period") is not None]
            for i in range(1, len(valid)):
                prev = valid[i - 1]
                curr = valid[i]
                if prev.get("supervisor_id") != curr.get("supervisor_id"):
                    events += 1
                if prev.get("job_level") != curr.get("job_level"):
                    events += 1
                if prev.get("department") != curr.get("department"):
                    events += 1

            first = valid[0] if valid else {}
            emp_data.append({
                "employee_id": eid,
                "starting_level": first.get("job_level"),
                "starting_dept": first.get("department"),
                "mobility_events": events,
            })

        result = pd.DataFrame(emp_data)
        if result.empty:
            return result

        # Compute peer groups
        result["peer_key"] = result.apply(
            lambda r: tuple(str(r.get(c, "")) for c in peer_basis), axis=1
        )

        # Per-peer-group stats
        group_stats = result.groupby("peer_key")["mobility_events"].agg(["median", "std"]).reset_index()
        group_stats.columns = ["peer_key", "peer_median", "peer_std"]
        result = result.merge(group_stats, on="peer_key", how="left")
        result["peer_std"] = result["peer_std"].fillna(0)

        # Z-score: (peer_median - events) / peer_std
        # Positive = less mobile than peers (stuck), negative = more mobile (leader)
        result["stuckness_zscore"] = result.apply(
            lambda r: _safe_zscore(r["mobility_events"], r["peer_median"], r["peer_std"]),
            axis=1,
        )
        result["is_stuck"] = result["stuckness_zscore"] > stuckness_threshold
        result["is_mobility_leader"] = result["stuckness_zscore"] < -stuckness_threshold
        result = result.drop(columns=["peer_key"])
        return result.sort_values("stuckness_zscore", ascending=False).reset_index(drop=True)

    # ── 8. manager_effectiveness ─────────────────────────────────────────────

    def manager_effectiveness(
        self,
        lookback: str = "4Q",
        w_engagement: float = 0.50,
        w_retention: float = 0.25,
        w_promotion: float = 0.15,
        w_span: float = 0.10,
        survey_table: str | None = None,
        promotions_table: str | None = None,
    ) -> pd.DataFrame:
        """Composite manager effectiveness score.

        For each manager M over the lookback window:

            1. Enumerate M's transitive subtree at each period via
               ``hierarchy_long``.
            2. Aggregate team engagement (from survey table) across
               indirect reports.
            3. Compute the engagement trend (slope of team engagement
               over time).
            4. Compute retention rate (1 - attrition rate in subtree).
            5. Compute promotion rate (fraction of subtree promoted).
            6. Compute span efficiency (1 / total_reports, normalized).
            7. Peer-normalize each metric against same-level managers.
            8. Composite score = weighted sum of z-scores.

        Default weights are engagement-dominant:
            0.50 * z(engagement_trend)
          + 0.25 * z(retention_rate)
          + 0.15 * z(promotion_rate)
          + 0.10 * z(span_efficiency)

        All weights are exposed as kwargs for auditability.

        Parameters
        ----------
        lookback : str
        w_engagement, w_retention, w_promotion, w_span : float
            Composite weights. Must sum to 1.0.
        survey_table : str, optional
            Name of the registered survey table. If None, looks for
            "survey" in extra tables.
        promotions_table : str, optional
            Name of the registered promotions table.

        Returns
        -------
        pandas.DataFrame
            Columns: ``manager_id, manager_level, n_periods_active,
            team_engagement_t1, team_engagement_tn, engagement_trend,
            retention_rate, promotion_rate, span_efficiency,
            peer_engagement_trend, peer_retention_rate,
            peer_promotion_rate, effectiveness_score, rank``.
        """
        if not self._loaded:
            raise RuntimeError("call load_snapshots() first")

        # Validate weights sum
        total_w = w_engagement + w_retention + w_promotion + w_span
        if abs(total_w - 1.0) > 0.01:
            raise ValueError(
                f"weights must sum to 1.0; got {total_w:.4f}"
            )

        # Resolve survey table
        survey_tbl = survey_table or self._extra_tables.get("survey", "survey")
        prom_tbl = promotions_table or self._extra_tables.get("promotions", "promotions")

        n_periods, _ = _parse_lookback(lookback)
        use_periods = self._periods[-n_periods:] if len(self._periods) >= n_periods else self._periods
        emp = self._emp_col
        sup = self._sup_col
        freq_word = _FREQ_MAP[self._freq]

        # Gather all managers (anyone who is a supervisor in any period)
        managers: set = set()
        for p in use_periods:
            df = self.con.sql(f"""
                SELECT DISTINCT {_quote(sup)} AS mgr
                FROM {self._table_name}
                WHERE date_trunc('{freq_word}', CAST({_quote(self._date_col)} AS DATE))
                      = CAST('{p}' AS DATE)
                  AND {_quote(sup)} IS NOT NULL
                  AND CAST({_quote(sup)} AS VARCHAR) <> ''
            """).df()
            managers.update(df["mgr"].tolist())

        if not managers:
            return pd.DataFrame()

        # Check if survey table exists
        has_survey = survey_tbl in self._extra_tables or self._con_table_exists(survey_tbl)
        has_promotions = prom_tbl in self._extra_tables or self._con_table_exists(prom_tbl)

        manager_data: list[dict] = []
        for mgr_id in managers:
            # For each period, get subtree and aggregate
            period_engagements: list[float] = []
            period_retention: list[float] = []
            period_promotions: list[float] = []
            period_spans: list[int] = []
            mgr_level = None

            for p in use_periods:
                subtree = _subtree_ids(
                    self.con, self._table_name, emp, sup,
                    self._date_col, mgr_id, p, self._freq,
                )
                if not subtree:
                    period_engagements.append(np.nan)
                    period_retention.append(np.nan)
                    period_promotions.append(0.0)
                    period_spans.append(0)
                    continue

                period_spans.append(len(subtree))

                # Manager's own level
                mgr_row = self.con.sql(f"""
                    SELECT * FROM {self._table_name}
                    WHERE date_trunc('{freq_word}', CAST({_quote(self._date_col)} AS DATE))
                          = CAST('{p}' AS DATE)
                      AND {_quote(emp)} = '{mgr_id}'
                """).df()
                if not mgr_row.empty:
                    mgr_level = mgr_row.iloc[0].get("job_level", mgr_level)

                # Team engagement from survey
                if has_survey:
                    placeholders = ",".join([f"'{s}'" for s in subtree])
                    eng_df = self.con.sql(f"""
                        SELECT AVG(engagement) AS avg_eng
                        FROM {survey_tbl}
                        WHERE employee_id IN ({placeholders})
                          AND date_trunc('{freq_word}', CAST(snapshot_date AS DATE))
                              = CAST('{p}' AS DATE)
                    """).df()
                    avg_eng = eng_df["avg_eng"].iloc[0] if not eng_df.empty else np.nan
                    period_engagements.append(float(avg_eng) if not pd.isna(avg_eng) else np.nan)
                else:
                    period_engagements.append(np.nan)

                # Retention: fraction of subtree still present next period
                period_retention.append(np.nan)  # placeholder, computed below

                # Promotions
                if has_promotions:
                    prom_df = self.con.sql(f"""
                        SELECT COUNT(DISTINCT employee_id) AS n_promoted
                        FROM {prom_tbl}
                        WHERE employee_id IN ({placeholders})
                    """).df()
                    n_promoted = int(prom_df["n_promoted"].iloc[0]) if not prom_df.empty else 0
                    period_promotions.append(n_promoted / len(subtree) if subtree else 0.0)
                else:
                    period_promotions.append(0.0)

            # Retention: compare subtree at period t with subtree at t+1
            for i in range(len(use_periods) - 1):
                sub_t = set(_subtree_ids(
                    self.con, self._table_name, emp, sup,
                    self._date_col, mgr_id, use_periods[i], self._freq,
                ))
                sub_t1 = set(_subtree_ids(
                    self.con, self._table_name, emp, sup,
                    self._date_col, mgr_id, use_periods[i + 1], self._freq,
                ))
                if sub_t:
                    retained = len(sub_t & sub_t1) / len(sub_t)
                    period_retention[i] = retained
                else:
                    period_retention[i] = np.nan
            if period_retention:
                period_retention[-1] = period_retention[-2] if len(period_retention) >= 2 else np.nan

            # Engagement trend: linear regression slope
            valid_eng = [(i, e) for i, e in enumerate(period_engagements) if not np.isnan(e)]
            if len(valid_eng) >= 2:
                xs = np.array([v[0] for v in valid_eng], dtype=float)
                ys = np.array([v[1] for v in valid_eng], dtype=float)
                if xs.std() > 0:
                    engagement_trend = float(np.polyfit(xs, ys, 1)[0])
                else:
                    engagement_trend = 0.0
            elif len(valid_eng) == 1:
                engagement_trend = 0.0
            else:
                engagement_trend = np.nan

            avg_retention = float(np.nanmean(period_retention)) if period_retention else np.nan
            avg_promotion = float(np.nanmean(period_promotions)) if period_promotions else 0.0
            avg_span = float(np.mean(period_spans)) if period_spans else 1

            manager_data.append({
                "manager_id": mgr_id,
                "manager_level": mgr_level,
                "n_periods_active": len([s for s in period_spans if s > 0]),
                "team_engagement_t1": period_engagements[0] if period_engagements else np.nan,
                "team_engagement_tn": period_engagements[-1] if period_engagements else np.nan,
                "engagement_trend": engagement_trend,
                "retention_rate": avg_retention,
                "promotion_rate": avg_promotion,
                "span_efficiency": 1.0 / avg_span if avg_span > 0 else 0.0,
            })

        result = pd.DataFrame(manager_data)
        if result.empty:
            return result

        # Peer normalization by manager_level
        def _peer_z(group: pd.DataFrame, col: str) -> pd.Series:
            vals = group[col]
            median = vals.median()
            std = vals.std()
            if std == 0 or pd.isna(std):
                return pd.Series([0.0] * len(group), index=group.index)
            return (vals - median) / std

        # Compute peer z-scores within each manager_level
        for col in ["engagement_trend", "retention_rate", "promotion_rate", "span_efficiency"]:
            z_col = f"peer_{col}"
            result[z_col] = result.groupby("manager_level")[col].transform(
                lambda x: (x - x.median()) / x.std() if x.std() and x.std() > 0 else 0.0
            )

        # Composite score
        result["effectiveness_score"] = (
            w_engagement * result["peer_engagement_trend"]
          + w_retention * result["peer_retention_rate"]
          + w_promotion * result["peer_promotion_rate"]
          + w_span * result["peer_span_efficiency"]
        )

        result = result.sort_values("effectiveness_score", ascending=False)
        result["rank"] = range(1, len(result) + 1)
        return result.reset_index(drop=True)

    def career_markov_matrix(
        self,
        state_col: str = "job_level",
        lookback: str = "8Q",
        by: str | None = "department",
    ) -> pd.DataFrame:
        """Estimate career-transition Markov probabilities from snapshot history.

        Parameters
        ----------
        state_col : str, default "job_level"
            Employee state used for transitions (e.g., job_level, role_band).
        lookback : str, default "8Q"
            Number of periods to include.
        by : str, optional
            Segment column for separate transition matrices (e.g., department).
            Set to ``None`` for one global matrix.

        Returns
        -------
        pandas.DataFrame
            Columns: ``segment, from_state, to_state, transitions, probability``.
            If ``by is None``, ``segment`` is ``"all"``.
        """
        if not self._loaded:
            raise RuntimeError("call load_snapshots() first")
        n_periods, _ = _parse_lookback(lookback)
        use_periods = self._periods[-n_periods:] if len(self._periods) >= n_periods else self._periods
        if len(use_periods) < 2:
            return pd.DataFrame(
                columns=["segment", "from_state", "to_state", "transitions", "probability"]
            )

        freq_word = _FREQ_MAP[self._freq]
        emp = self._emp_col
        seg_select = f", {_quote(by)} AS segment" if by else ", 'all' AS segment"
        hist = self.con.sql(f"""
            SELECT
                {_quote(emp)} AS employee_id,
                date_trunc('{freq_word}', CAST({_quote(self._date_col)} AS DATE)) AS period,
                {_quote(state_col)} AS state
                {seg_select}
            FROM {self._table_name}
            WHERE date_trunc('{freq_word}', CAST({_quote(self._date_col)} AS DATE))
                  BETWEEN CAST('{use_periods[0]}' AS DATE) AND CAST('{use_periods[-1]}' AS DATE)
        """).df()
        if hist.empty:
            return pd.DataFrame(
                columns=["segment", "from_state", "to_state", "transitions", "probability"]
            )

        hist["period"] = pd.to_datetime(hist["period"])
        hist = hist.sort_values(["employee_id", "period"])
        transitions: list[dict[str, Any]] = []
        for _, g in hist.groupby("employee_id", sort=False):
            g = g.reset_index(drop=True)
            for i in range(len(g) - 1):
                frm = g.iloc[i]["state"]
                to = g.iloc[i + 1]["state"]
                seg = g.iloc[i]["segment"] if by else "all"
                if pd.isna(frm) or pd.isna(to):
                    continue
                transitions.append(
                    {
                        "segment": str(seg),
                        "from_state": str(frm),
                        "to_state": str(to),
                    }
                )

        if not transitions:
            return pd.DataFrame(
                columns=["segment", "from_state", "to_state", "transitions", "probability"]
            )

        trans_df = pd.DataFrame(transitions)
        out = (
            trans_df.groupby(["segment", "from_state", "to_state"], as_index=False)
            .size()
            .rename(columns={"size": "transitions"})
        )
        totals = out.groupby(["segment", "from_state"])["transitions"].transform("sum")
        out["probability"] = out["transitions"] / totals
        return out.sort_values(
            ["segment", "from_state", "probability"], ascending=[True, True, False]
        ).reset_index(drop=True)

    def career_markov_forecast(
        self,
        employee_id: Any,
        horizon: int = 2,
        state_col: str = "job_level",
        lookback: str = "8Q",
        by: str | None = "department",
    ) -> pd.DataFrame:
        """Forecast future state probabilities for one employee via Markov transitions.

        Parameters
        ----------
        employee_id : Any
            Employee identifier.
        horizon : int, default 2
            Number of forward periods to forecast.
        state_col : str, default "job_level"
        lookback : str, default "8Q"
        by : str, optional
            Segment column used for segment-specific transition matrix.

        Returns
        -------
        pandas.DataFrame
            Columns: ``employee_id, step, state, probability, is_most_likely``.
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        matrix = self.career_markov_matrix(state_col=state_col, lookback=lookback, by=by)
        if matrix.empty:
            return pd.DataFrame(
                columns=["employee_id", "step", "state", "probability", "is_most_likely"]
            )

        freq_word = _FREQ_MAP[self._freq]
        emp = self._emp_col
        seg_select = f", {_quote(by)} AS segment" if by else ", 'all' AS segment"
        latest = self.con.sql(f"""
            SELECT
                {_quote(state_col)} AS state
                {seg_select}
            FROM {self._table_name}
            WHERE {_quote(emp)} = '{employee_id}'
            ORDER BY date_trunc('{freq_word}', CAST({_quote(self._date_col)} AS DATE)) DESC
            LIMIT 1
        """).df()
        if latest.empty or pd.isna(latest.iloc[0]["state"]):
            return pd.DataFrame(
                columns=["employee_id", "step", "state", "probability", "is_most_likely"]
            )

        current_state = str(latest.iloc[0]["state"])
        segment = str(latest.iloc[0]["segment"]) if by else "all"
        matrix = matrix[matrix["segment"] == segment].copy()
        if matrix.empty:
            return pd.DataFrame(
                columns=["employee_id", "step", "state", "probability", "is_most_likely"]
            )

        states = sorted(set(matrix["from_state"]).union(set(matrix["to_state"])))
        state_to_idx = {s: i for i, s in enumerate(states)}
        P = np.zeros((len(states), len(states)), dtype=float)
        for _, row in matrix.iterrows():
            P[state_to_idx[row["from_state"]], state_to_idx[row["to_state"]]] = float(
                row["probability"]
            )
        for i in range(P.shape[0]):
            row_sum = P[i].sum()
            if row_sum == 0:
                P[i, i] = 1.0

        if current_state not in state_to_idx:
            return pd.DataFrame(
                columns=["employee_id", "step", "state", "probability", "is_most_likely"]
            )

        dist = np.zeros(len(states), dtype=float)
        dist[state_to_idx[current_state]] = 1.0

        rows: list[dict[str, Any]] = []
        for step in range(1, horizon + 1):
            dist = dist @ P
            best_idx = int(np.argmax(dist))
            for idx, state in enumerate(states):
                rows.append(
                    {
                        "employee_id": employee_id,
                        "step": step,
                        "state": state,
                        "probability": float(dist[idx]),
                        "is_most_likely": idx == best_idx,
                    }
                )

        return pd.DataFrame(rows).sort_values(
            ["step", "probability"], ascending=[True, False]
        ).reset_index(drop=True)

    def org_design_scorecard(
        self,
        lookback: str = "8Q",
    ) -> pd.DataFrame:
        """Per-period organizational design metrics and a composite score.

        Parameters
        ----------
        lookback : str, default "8Q"

        Returns
        -------
        pandas.DataFrame
            Columns include span/load/layering/silo metrics and
            ``org_design_score`` (0-100, higher is healthier).
        """
        if not self._loaded:
            raise RuntimeError("call load_snapshots() first")
        n_periods, _ = _parse_lookback(lookback)
        use_periods = self._periods[-n_periods:] if len(self._periods) >= n_periods else self._periods
        if not use_periods:
            return pd.DataFrame()

        freq_word = _FREQ_MAP[self._freq]
        emp = self._emp_col
        sup = self._sup_col
        evolution = self.network_evolution()
        evolution = evolution[evolution["period"].isin(use_periods)].copy()
        rows: list[dict[str, Any]] = []
        for period in use_periods:
            snap = self.con.sql(f"""
                SELECT {_quote(emp)} AS employee_id, {_quote(sup)} AS supervisor_id
                FROM {self._table_name}
                WHERE date_trunc('{freq_word}', CAST({_quote(self._date_col)} AS DATE))
                      = CAST('{period}' AS DATE)
            """)
            stats = hierarchy_stats(snap, "employee_id", "supervisor_id").df()
            stats = stats[stats["manager_id"].notna()].copy()
            mgr_stats = stats[stats["direct_reports"] > 0]
            avg_span = float(mgr_stats["direct_reports"].mean()) if not mgr_stats.empty else 0.0
            span_cv = (
                float(mgr_stats["direct_reports"].std() / mgr_stats["direct_reports"].mean())
                if not mgr_stats.empty and mgr_stats["direct_reports"].mean() > 0
                else 0.0
            )
            max_layers = int(stats["levels_below"].max() + 1) if not stats.empty else 1

            ev = evolution[evolution["period"] == period]
            if ev.empty:
                continue
            n_components = int(ev.iloc[0]["n_components"])
            n_employees = int(ev.iloc[0]["n_employees"])
            centralization = float(ev.iloc[0]["centralization"])
            silo_index = n_components / max(n_employees, 1)

            span_score = 1.0 / (1.0 + abs(avg_span - 7.0) / 7.0)
            layering_score = 1.0 / (1.0 + max(0, max_layers - 5))
            silo_score = 1.0 - min(1.0, silo_index * 8.0)
            centralization_score = 1.0 - min(1.0, centralization)
            org_design_score = 100.0 * (
                0.35 * span_score
                + 0.25 * layering_score
                + 0.20 * silo_score
                + 0.20 * centralization_score
            )

            rows.append(
                {
                    "period": period,
                    "n_employees": n_employees,
                    "n_components": n_components,
                    "avg_span": avg_span,
                    "span_cv": span_cv,
                    "max_layers": max_layers,
                    "centralization": centralization,
                    "silo_index": silo_index,
                    "org_design_score": org_design_score,
                }
            )

        return pd.DataFrame(rows)

    def org_design_change_alerts(
        self,
        lookback: str = "8Q",
        span_shift_threshold: int = 3,
        component_growth_threshold: float = 0.25,
    ) -> pd.DataFrame:
        """Flag periods with potentially unhealthy organizational-design shifts.

        Parameters
        ----------
        lookback : str, default "8Q"
        span_shift_threshold : int, default 3
            Trigger when >= this many managers change direct reports by 2+.
        component_growth_threshold : float, default 0.25
            Trigger when weak components grow by this fraction period-over-period.

        Returns
        -------
        pandas.DataFrame
            One row per period transition with alert flags and severity.
        """
        scorecard = self.org_design_scorecard(lookback=lookback)
        if scorecard.empty or len(scorecard) < 2:
            return pd.DataFrame()

        freq_word = _FREQ_MAP[self._freq]
        emp = self._emp_col
        sup = self._sup_col
        alerts: list[dict[str, Any]] = []
        for i in range(1, len(scorecard)):
            p0 = scorecard.iloc[i - 1]["period"]
            p1 = scorecard.iloc[i]["period"]
            snap_0 = self.con.sql(f"""
                SELECT {_quote(emp)} AS employee_id, {_quote(sup)} AS supervisor_id
                FROM {self._table_name}
                WHERE date_trunc('{freq_word}', CAST({_quote(self._date_col)} AS DATE))
                      = CAST('{p0}' AS DATE)
            """)
            snap_1 = self.con.sql(f"""
                SELECT {_quote(emp)} AS employee_id, {_quote(sup)} AS supervisor_id
                FROM {self._table_name}
                WHERE date_trunc('{freq_word}', CAST({_quote(self._date_col)} AS DATE))
                      = CAST('{p1}' AS DATE)
            """)
            s0 = hierarchy_stats(snap_0, "employee_id", "supervisor_id").df()
            s1 = hierarchy_stats(snap_1, "employee_id", "supervisor_id").df()
            drift = s0[["manager_id", "direct_reports"]].merge(
                s1[["manager_id", "direct_reports"]],
                on="manager_id",
                how="outer",
                suffixes=("_t", "_t1"),
            ).fillna(0)
            drift["delta"] = drift["direct_reports_t1"] - drift["direct_reports_t"]
            n_span_shifts = int((drift["delta"].abs() >= 2).sum())
            max_span_shift = float(drift["delta"].abs().max()) if not drift.empty else 0.0

            comp_t = float(scorecard.iloc[i - 1]["n_components"])
            comp_t1 = float(scorecard.iloc[i]["n_components"])
            component_growth = ((comp_t1 - comp_t) / comp_t) if comp_t > 0 else 0.0

            score_delta = float(scorecard.iloc[i]["org_design_score"] - scorecard.iloc[i - 1]["org_design_score"])

            reasons: list[str] = []
            if n_span_shifts >= span_shift_threshold:
                reasons.append("span_shift")
            if component_growth >= component_growth_threshold:
                reasons.append("component_growth")
            if score_delta <= -8.0:
                reasons.append("design_score_drop")

            if len(reasons) >= 2:
                severity = "high"
            elif len(reasons) == 1:
                severity = "medium"
            else:
                severity = "low"

            alerts.append(
                {
                    "period_from": p0,
                    "period_to": p1,
                    "n_span_shifts": n_span_shifts,
                    "max_span_shift": max_span_shift,
                    "component_growth": component_growth,
                    "design_score_delta": score_delta,
                    "severity": severity,
                    "reasons": ",".join(reasons),
                }
            )

        return pd.DataFrame(alerts)

    # ── Utility ─────────────────────────────────────────────────────────────

    def sql(self, query: str) -> "DuckDBPyRelation":
        """Run arbitrary SQL on the owned connection."""
        return self.con.sql(query)

    def _con_table_exists(self, table_name: str) -> bool:
        """Check if a table exists on the connection."""
        try:
            self.con.sql(f"SELECT 1 FROM {table_name} LIMIT 1")
            return True
        except Exception:
            return False
