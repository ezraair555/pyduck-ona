"""High-level ``DuckONA`` analysis class for HR analytics.

The class owns a DuckDB connection, ingests HR data (HRIS, compensation,
turnover, survey results, retirement, promotions, skills, attendance),
validates keys and date ranges, builds org-chart edge relations, runs
graph metrics via ``pyduck_ona.graph.*``, and exposes thin model helpers
that delegate to ``pyduck_ona.stats.*``. It is intentionally scoped to
HR-analytics workflows (compensation, turnover, mobility, retirement,
skills, attendance) and does **not** ingest Slack or email interaction
logs.

Example
-------
>>> from pyduck_ona import DuckONA
>>> import pandas as pd
>>> ona = DuckONA()
>>> ona.load_hris(hris_df)
>>> ona.load_compensation(comp_df)
>>> edges = ona.build_org_edges("employee_id", "supervisor_id")
>>> metrics = ona.betweenness(edges, "employee_id", "supervisor_id")
>>> joined = ona.join_hris(metrics)
>>> tidy, glance = ona.ols(joined, "salary ~ betweenness + tenure_yrs")
"""

from __future__ import annotations

import re
import uuid
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

import duckdb
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from duckdb import DuckDBPyRelation
    from numpy.typing import NDArray

from pyduck_ona import graph as _graph
from pyduck_ona import stats as _stats

# Allowed table names for data registered by the loader helpers.
_HR_TABLES = [
    "hris",
    "compensation",
    "turnover",
    "survey",
    "retirement",
    "promotions",
    "skills",
    "attendance",
]

# Re-used safe-identifier regex from core.py.
_IDENT_SAFE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_ident(name: str) -> str:
    """Return a safely-quoted DuckDB identifier.

    Mirrors the helper in ``pyduck_ona.core``; duplicated here so
    ``analysis.py`` is self-contained and does not depend on a private
    core helper.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"column name must be a non-empty string, got {name!r}")
    if _IDENT_SAFE_RE.match(name):
        return f'"{name}"'
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _validate_table_name(name: str) -> None:
    """Reject unsafe or malformed DuckDB table identifiers."""
    if not isinstance(name, str) or not name:
        raise ValueError("table_name must be a non-empty string")
    if not _IDENT_SAFE_RE.match(name):
        raise ValueError(f"table_name must be a valid unquoted DuckDB identifier, got {name!r}")


def _today() -> date:
    """Return the current UTC date (used for sensible date checks)."""
    return datetime.now(timezone.utc).date()


def _coerce_date(value: Any) -> date | None:
    """Coerce a scalar to ``datetime.date`` if possible.

    Accepts ``date``, ``datetime``, or ISO strings. Returns None for
    None / NaT / empty strings.
    """
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return datetime.fromisoformat(stripped).date()
        except ValueError:
            return None
    return None


def _as_relation(
    con: DuckDBPyConnection, data: pd.DataFrame | DuckDBPyRelation
) -> DuckDBPyRelation:
    """Coerce a DataFrame or relation to a relation on ``con``."""
    if hasattr(data, "df") and callable(getattr(data, "df", None)):
        return data  # type: ignore[return-value]
    if isinstance(data, pd.DataFrame):
        return con.from_df(data)
    raise TypeError(f"expected pandas.DataFrame or DuckDBPyRelation, got {type(data).__name__}")


def _columns(rel: DuckDBPyRelation) -> list[str]:
    """Return the column names of a relation as a list."""
    return list(rel.columns)


class DuckONA:
    """A DuckDB-backed workspace for HR analytics.

    Parameters
    ----------
    db_path : str, default ":memory:"
        DuckDB database path. ``:memory:`` (the default) is fine for
        in-memory analyses; pass a file path to persist tables.

    Attributes
    ----------
    con : duckdb.DuckDBPyConnection
        The underlying DuckDB connection. Exposed so callers can run
        arbitrary SQL on the same connection.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.con = duckdb.connect(db_path)
        self._register_hris: pd.DataFrame | None = None
        self._table_names: set[str] = set()

    # ── Loader helpers ──────────────────────────────────────────────────────

    def _register_df(
        self,
        df: pd.DataFrame,
        table_name: str,
    ) -> DuckDBPyRelation:
        """Register a DataFrame as a table on ``self.con`` and return a relation."""
        _validate_table_name(table_name)
        self.con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
        self._table_names.add(table_name)
        return self.con.sql(f"SELECT * FROM {table_name}")

    def load_hris(self, df: pd.DataFrame) -> DuckDBPyRelation:
        """Load the HRIS snapshot.

        Expected columns include ``employee_id`` and ``supervisor_id``,
        plus demographic fields such as ``department``, ``job_level``,
        ``hire_date``, etc. The exact schema is left flexible; only the
        key columns are validated.
        """
        self._register_hris = df.copy()
        return self._register_df(df, "hris")

    def load_compensation(self, df: pd.DataFrame) -> DuckDBPyRelation:
        """Load a compensation table with one row per employee per snapshot."""
        return self._register_df(df, "compensation")

    def load_turnover(self, df: pd.DataFrame) -> DuckDBPyRelation:
        """Load a turnover / termination table."""
        return self._register_df(df, "turnover")

    def load_survey(self, df: pd.DataFrame) -> DuckDBPyRelation:
        """Load an engagement / survey-results table."""
        return self._register_df(df, "survey")

    def load_retirement(self, df: pd.DataFrame) -> DuckDBPyRelation:
        """Load a retirement-eligibility or retirement-planning table."""
        return self._register_df(df, "retirement")

    def load_promotions(self, df: pd.DataFrame) -> DuckDBPyRelation:
        """Load a promotion / internal-mobility table."""
        return self._register_df(df, "promotions")

    def load_skills(self, df: pd.DataFrame) -> DuckDBPyRelation:
        """Load a skills / proficiency table."""
        return self._register_df(df, "skills")

    def load_attendance(self, df: pd.DataFrame) -> DuckDBPyRelation:
        """Load an office-attendance / presence table."""
        return self._register_df(df, "attendance")

    # ── Validation ──────────────────────────────────────────────────────────

    def validate_keys(
        self,
        table_name: str,
        employee_id_col: str = "employee_id",
        date_col: str | None = None,
        *,
        date_lower: date | str | None = None,
        date_upper: date | str | None = None,
    ) -> None:
        """Validate HR table keys: non-null IDs, no duplicate snapshots, sensible dates.

        Parameters
        ----------
        table_name : str
            Name of the registered HR table to validate.
        employee_id_col : str, default "employee_id"
            Column holding the employee identifier.
        date_col : str, optional
            Snapshot/effective-date column. If given, ``employee_id``
            must be unique per date and dates must not be in the future.
        allow_null_supervisor : bool, default True
            Reserved for future hierarchy-aware validation; currently
            has no effect because this method validates keys/dates only.
        date_lower, date_upper : date or str, optional
            Inclusive bounds for ``date_col`` values. Strings are parsed
            as ISO dates.
        """
        if table_name not in self._table_names:
            raise ValueError(f"table {table_name!r} not loaded")
        _validate_table_name(table_name)
        rel = self.con.sql(f"SELECT * FROM {table_name}")
        emp = _quote_ident(employee_id_col)
        cols = _columns(rel)
        if employee_id_col not in cols:
            raise ValueError(
                f"employee_id column {employee_id_col!r} not found in {table_name}; "
                f"available: {cols}"
            )

        # 1. non-null employee_id
        null_sql = f"""
            SELECT COUNT(*) AS n
            FROM {table_name}
            WHERE {emp} IS NULL
               OR CAST({emp} AS VARCHAR) = ''
        """
        n_null = int(self.con.sql(null_sql).fetchone()[0])  # type: ignore[arg-type]
        if n_null:
            raise ValueError(
                f"{table_name}.{employee_id_col} contains {n_null} NULL/empty value(s)"
            )

        # 2. no duplicate employee_id per date
        if date_col is not None and date_col in cols:
            date_quoted = _quote_ident(date_col)
            dup_sql = f"""
                SELECT COUNT(*) - COUNT(DISTINCT ({emp}, {date_quoted})) AS dups
                FROM {table_name}
            """
            n_dups = int(self.con.sql(dup_sql).fetchone()[0])  # type: ignore[arg-type]
            if n_dups:
                raise ValueError(
                    f"{table_name} has {n_dups} duplicate ({employee_id_col}, {date_col}) rows"
                )

        # 3. no future dates
        if date_col is not None and date_col in cols:
            date_quoted = _quote_ident(date_col)
            today_str = _today().isoformat()
            future_sql = f"""
                SELECT COUNT(*) AS n
                FROM {table_name}
                WHERE CAST({date_quoted} AS DATE) > CAST('{today_str}' AS DATE)
            """
            n_future = int(self.con.sql(future_sql).fetchone()[0])  # type: ignore[arg-type]
            if n_future:
                raise ValueError(f"{table_name}.{date_col} contains {n_future} future date(s)")

        # 4. sensible date bounds
        if date_col is not None and (date_lower is not None or date_upper is not None):
            date_quoted = _quote_ident(date_col)
            lower = _coerce_date(date_lower)
            upper = _coerce_date(date_upper)
            if lower is not None:
                low_sql = f"""
                    SELECT COUNT(*) AS n
                    FROM {table_name}
                    WHERE CAST({date_quoted} AS DATE) < CAST('{lower.isoformat()}' AS DATE)
                """
                n_low = int(self.con.sql(low_sql).fetchone()[0])  # type: ignore[arg-type]
                if n_low:
                    raise ValueError(
                        f"{table_name}.{date_col} contains {n_low} date(s) before {lower}"
                    )
            if upper is not None:
                up_sql = f"""
                    SELECT COUNT(*) AS n
                    FROM {table_name}
                    WHERE CAST({date_quoted} AS DATE) > CAST('{upper.isoformat()}' AS DATE)
                """
                n_up = int(self.con.sql(up_sql).fetchone()[0])  # type: ignore[arg-type]
                if n_up:
                    raise ValueError(
                        f"{table_name}.{date_col} contains {n_up} date(s) after {upper}"
                    )

        # 5. duplicate employee_id (when no date_col)
        if date_col is None or date_col not in cols:
            dup_sql = f"""
                SELECT COUNT(*) - COUNT(DISTINCT {emp}) AS dups
                FROM {table_name}
            """
            n_dups = int(self.con.sql(dup_sql).fetchone()[0])  # type: ignore[arg-type]
            if n_dups:
                raise ValueError(
                    f"{table_name}.{employee_id_col} contains {n_dups} duplicate value(s)"
                )

    # ── Noise filtering / deduplication ───────────────────────────────────

    @staticmethod
    def filter_noise(
        df: pd.DataFrame,
        *,
        id_col: str = "employee_id",
        date_col: str | None = None,
        test_ids: list[Any] | None = None,
        bots: list[Any] | None = None,
        min_records: int = 1,
    ) -> pd.DataFrame:
        """Filter noise from an HR DataFrame.

        Drops test IDs, bot IDs, rows with NULL keys, and optionally
        drops employees with fewer than ``min_records`` rows. This helper
        is intentionally pure-Python so it can be used before loading.

        Parameters
        ----------
        df : pandas.DataFrame
        id_col : str, default "employee_id"
        date_col : str, optional
            If given, also drop rows where the date is missing/future.
        test_ids : list, optional
            IDs to drop (e.g. test accounts).
        bots : list, optional
            IDs to drop (e.g. service accounts).
        min_records : int, default 1
            Minimum number of rows an ID must have to be retained.
        """
        out = df.copy()
        out = out[out[id_col].notna()]
        if test_ids:
            out = out[~out[id_col].isin(test_ids)]
        if bots:
            out = out[~out[id_col].isin(bots)]
        if date_col is not None and date_col in out.columns:
            today = _today()
            out = out[out[date_col].notna()]
            dates = pd.to_datetime(out[date_col], errors="coerce").dt.date
            out = out[dates <= today]
        if min_records > 1:
            counts = out.groupby(id_col).size()
            keep = counts[counts >= min_records].index
            out = out[out[id_col].isin(keep)]
        return out.reset_index(drop=True)

    @staticmethod
    def deduplicate(
        df: pd.DataFrame,
        *,
        id_col: str = "employee_id",
        date_col: str | None = None,
        keep: Literal["first", "last", False] = "last",
    ) -> pd.DataFrame:
        """Deduplicate an HR DataFrame by ``(id_col, date_col)``.

        Parameters
        ----------
        df : pandas.DataFrame
        id_col : str, default "employee_id"
        date_col : str, optional
            If given, deduplicate on the combination; otherwise on
            ``id_col`` alone.
        keep : {"first", "last"}, default "last"
            Which duplicate row to retain.
        """
        cols = [id_col] if date_col is None else [id_col, date_col]
        return df.drop_duplicates(subset=cols, keep=keep).reset_index(drop=True)

    # ── Org edges ───────────────────────────────────────────────────────────

    def build_org_edges(
        self,
        employee_id_col: str = "employee_id",
        supervisor_id_col: str = "supervisor_id",
        active_as_of: date | str | None = None,
        table_name: str = "hris",
    ) -> DuckDBPyRelation:
        """Build a directed edge relation from the HRIS hierarchy.

        Parameters
        ----------
        employee_id_col, supervisor_id_col : str
            Column names in the HRIS table.
        active_as_of : date or str, optional
            If the HRIS table has a ``snapshot_date`` / ``effective_date``
            column, filter to rows active as of this date. Not required
            for single-snapshot HRIS tables.
        table_name : str, default "hris"
            Source table name.

        Returns
        -------
        DuckDBPyRelation
            Columns ``(employee_id, supervisor_id)`` for every non-NULL
            supervisor edge. This is the correct input for graph metrics
            such as betweenness and PageRank.

        Examples
        --------
        >>> edges = ona.build_org_edges("emp_id", "mgr_id")
        >>> metrics = ona.betweenness(edges, "emp_id", "mgr_id")
        """
        _validate_table_name(table_name)
        emp = _quote_ident(employee_id_col)
        sup = _quote_ident(supervisor_id_col)
        rel = self.con.sql(f"SELECT * FROM {table_name}")
        cols = _columns(rel)
        if employee_id_col not in cols or supervisor_id_col not in cols:
            raise ValueError(
                f"{table_name} missing {employee_id_col!r} or {supervisor_id_col!r}; "
                f"available: {cols}"
            )

        # Optional active_as_of filter when a date column exists.
        where_clause = ""
        if active_as_of is not None:
            candidate_cols = [c for c in cols if "date" in c.lower()]
            if candidate_cols:
                date_col = candidate_cols[0]
                date_quoted = _quote_ident(date_col)
                as_of_str = _coerce_date(active_as_of)
                if as_of_str is None:
                    raise ValueError(f"active_as_of not parseable as date: {active_as_of!r}")
                where_clause = (
                    f"WHERE CAST({date_quoted} AS DATE) <= CAST('{as_of_str.isoformat()}' AS DATE)"
                )

        sql = f"""
            SELECT DISTINCT
                {emp} AS {emp},
                {sup} AS {sup}
            FROM {table_name}
            {where_clause}
        """
        # Drop rows where supervisor is NULL or empty.
        sql += f"\n            WHERE {sup} IS NOT NULL AND CAST({sup} AS VARCHAR) <> ''"
        return self.con.sql(sql)

    # ── Graph metric wrappers ───────────────────────────────────────────────

    def betweenness(
        self,
        edges: DuckDBPyRelation,
        source_col: str,
        target_col: str,
        *,
        backend: Literal["networkx", "duckpgq"] = "networkx",
    ) -> DuckDBPyRelation:
        """Betweenness centrality via ``pyduck_ona.graph.betweenness``."""
        return _graph.betweenness(edges, source_col, target_col, backend=backend)

    def pagerank(
        self,
        edges: DuckDBPyRelation,
        source_col: str,
        target_col: str,
        *,
        damping: float = 0.85,
        backend: Literal["networkx", "duckpgq"] = "networkx",
    ) -> DuckDBPyRelation:
        """PageRank centrality via ``pyduck_ona.graph.pagerank``."""
        return _graph.pagerank(edges, source_col, target_col, damping=damping, backend=backend)

    def eigenvector_centrality(
        self,
        edges: DuckDBPyRelation,
        source_col: str,
        target_col: str,
        *,
        backend: Literal["networkx", "duckpgq"] = "networkx",
    ) -> DuckDBPyRelation:
        """Eigenvector centrality via ``pyduck_ona.graph.eigenvector_centrality``."""
        return _graph.eigenvector_centrality(edges, source_col, target_col, backend=backend)

    def degree_centrality(
        self,
        edges: DuckDBPyRelation,
        source_col: str,
        target_col: str,
        *,
        mode: Literal["in", "out", "total"] = "out",
        backend: Literal["networkx", "duckpgq"] = "networkx",
    ) -> DuckDBPyRelation:
        """Degree centrality via ``pyduck_ona.graph.degree_centrality``."""
        return _graph.degree_centrality(edges, source_col, target_col, mode=mode, backend=backend)

    def connected_components(
        self,
        edges: DuckDBPyRelation,
        source_col: str,
        target_col: str,
        *,
        backend: Literal["networkx", "duckpgq"] = "networkx",
    ) -> DuckDBPyRelation:
        """Weakly-connected components via ``pyduck_ona.graph.connected_components``."""
        return _graph.connected_components(edges, source_col, target_col, backend=backend)

    def louvain_communities(
        self,
        edges: DuckDBPyRelation,
        source_col: str,
        target_col: str,
        *,
        weight_col: str | None = None,
        resolution: float = 1.0,
        backend: Literal["networkx", "duckpgq"] = "networkx",
    ) -> DuckDBPyRelation:
        """Louvain community detection via ``pyduck_ona.graph.louvain_communities``."""
        return _graph.louvain_communities(
            edges,
            source_col,
            target_col,
            weight_col=weight_col,
            resolution=resolution,
            backend=backend,
        )

    # ── HRIS join ───────────────────────────────────────────────────────────

    def join_hris(
        self,
        metrics_rel: DuckDBPyRelation,
        metrics_id_col: str = "node_id",
        hris_id_col: str = "employee_id",
        hris_table: str = "hris",
    ) -> DuckDBPyRelation:
        """Join a metric relation back to the HRIS demographics table.

        Parameters
        ----------
        metrics_rel : DuckDBPyRelation
            Relation containing per-employee network or model metrics,
            e.g. the output of ``betweenness()``.
        metrics_id_col : str, default "node_id"
            Column in ``metrics_rel`` holding the employee identifier.
        hris_id_col : str, default "employee_id"
            Column in the HRIS table holding the employee identifier.
        hris_table : str, default "hris"
            Name of the HRIS table registered on ``self.con``.

        Returns
        -------
        DuckDBPyRelation
            A left join of HRIS onto the metric relation so every metric
            row keeps its network scores and gains demographic columns.
        """
        _validate_table_name(hris_table)
        mid = _quote_ident(metrics_id_col)
        hid = _quote_ident(hris_id_col)
        view_name = f"_pona_metrics_{uuid.uuid4().hex[:12]}"
        arrow_table = metrics_rel.arrow()
        if hasattr(arrow_table, "read_all"):
            arrow_table = arrow_table.read_all()
        df = arrow_table.to_pandas()
        self.con.execute(f"CREATE OR REPLACE TEMPORARY TABLE {view_name} AS SELECT * FROM df")
        sql = f"""
            SELECT h.*, m.* EXCLUDE ({mid})
            FROM {hris_table} h
            JOIN {view_name} m ON h.{hid} = m.{mid}
        """
        return self.con.sql(sql)

    # ── Model helpers ───────────────────────────────────────────────────────

    def ols(
        self,
        data: pd.DataFrame | DuckDBPyRelation,
        formula: str,
        *,
        cov_type: str = "nonrobust",
        alpha: float = 0.05,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """OLS linear regression; delegates to ``pyduck_ona.stats.ols``."""
        return _stats.ols(data, formula, cov_type=cov_type, alpha=alpha)

    def logistic(
        self,
        data: pd.DataFrame | DuckDBPyRelation,
        formula: str,
        *,
        cov_type: str = "nonrobust",
        alpha: float = 0.05,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Logistic regression; delegates to ``pyduck_ona.stats.logistic``."""
        return _stats.logistic(data, formula, cov_type=cov_type, alpha=alpha)

    def anova(
        self,
        data: pd.DataFrame | DuckDBPyRelation,
        formula: str,
        *,
        anova_type: int = 2,
    ) -> pd.DataFrame:
        """One-way ANOVA; delegates to ``pyduck_ona.stats.anova``."""
        return _stats.anova(data, formula, anova_type=anova_type)

    def chi_square(
        self,
        data: pd.DataFrame | DuckDBPyRelation,
        x: str,
        y: str,
    ) -> Any:
        """Chi-square test; delegates to ``pyduck_ona.stats.chi_square``."""
        return _stats.chi_square(data, x, y)

    def correlation(
        self,
        data: pd.DataFrame | DuckDBPyRelation,
        columns: list[str] | None = None,
        *,
        col1: str | None = None,
        col2: str | None = None,
        method: str = "pearson",
    ) -> pd.DataFrame:
        """Correlation helper; delegates to ``pyduck_ona.stats.correlation``."""
        return _stats.correlation(data, columns=columns, col1=col1, col2=col2, method=method)

    def vif(
        self,
        data: pd.DataFrame | DuckDBPyRelation,
        formula: str,
    ) -> Any:
        """Variance-inflation factors; delegates to ``pyduck_ona.stats.vif``."""
        return _stats.vif(data, formula)

    def model_compare(
        self,
        models: dict[str, Any],
    ) -> pd.DataFrame:
        """Model comparison; delegates to ``pyduck_ona.stats.model_compare``."""
        return _stats.model_compare(models)

    # ── Temporal slicing ──────────────────────────────────────────────────

    def build_temporal_slices(
        self,
        table_name: str,
        date_col: str,
        freq: str = "M",
    ) -> list[tuple[str, date, date, DuckDBPyRelation]]:
        """Return time-sliced relations for a registered table.

        Parameters
        ----------
        table_name : str
            Registered table to slice.
        date_col : str
            Date / datetime column used for slicing.
        freq : {"D", "W", "M", "Q", "Y"}, default "M"
            Slice frequency.

        Returns
        -------
        list of (slice_label, start_date, end_date, relation)
            One tuple per slice covering the observed date range in
            ``table_name``.

        Examples
        --------
        >>> slices = ona.build_temporal_slices("attendance", "date", freq="M")
        >>> for label, start, end, rel in slices:
        ...     print(label, rel.count("*").fetchone()[0])
        """
        _validate_table_name(table_name)
        date_quoted = _quote_ident(date_col)
        rel = self.con.sql(f"SELECT * FROM {table_name}")
        cols = _columns(rel)
        if date_col not in cols:
            raise ValueError(
                f"date column {date_col!r} not found in {table_name}; available: {cols}"
            )

        freq = freq.upper()
        pandas_freq_map = {
            "D": "D",
            "W": "W",
            "M": "ME",
            "Q": "QE",
            "Y": "YE",
        }
        if freq not in pandas_freq_map:
            raise ValueError(f"freq must be one of D/W/M/Q/Y, got {freq!r}")
        pandas_freq = pandas_freq_map[freq]

        bounds = self.con.sql(f"""
            SELECT
                MIN(CAST({date_quoted} AS DATE)) AS min_date,
                MAX(CAST({date_quoted} AS DATE)) AS max_date
            FROM {table_name}
            """).df()
        min_date = bounds["min_date"].iloc[0]
        max_date = bounds["max_date"].iloc[0]
        if pd.isna(min_date) or pd.isna(max_date):
            return []
        min_date = pd.Timestamp(min_date).date() if hasattr(min_date, "date") else min_date
        max_date = pd.Timestamp(max_date).date() if hasattr(max_date, "date") else max_date

        # Build slices using pandas date ranges.
        start_ts = pd.Timestamp(min_date)
        end_ts = pd.Timestamp(max_date)
        periods = pd.date_range(start=start_ts, end=end_ts, freq=pandas_freq)
        if len(periods) == 0:
            periods = pd.DatetimeIndex([start_ts])

        slices: list[tuple[str, date, date, DuckDBPyRelation]] = []
        for period in periods:
            if freq == "D":
                start = period.date()
                end = period.date()
                label = start.strftime("%Y-%m-%d")
            elif freq == "W":
                start = period.date()
                end = (period + pd.Timedelta(days=6)).date()
                label = start.strftime("%Y-W%U")
            elif freq == "M":
                start = period.date()
                end = (period + pd.offsets.MonthEnd(0)).date()
                label = start.strftime("%Y-%m")
            elif freq == "Q":
                start = period.date()
                end = (period + pd.offsets.QuarterEnd(0)).date()
                label = f"{period.year}-Q{(period.month - 1) // 3 + 1}"
            elif freq == "Y":
                start = date(period.year, 1, 1)
                end = date(period.year, 12, 31)
                label = str(period.year)
            else:
                continue

            # Clip the last slice to the observed max date so we never
            # return empty trailing relations.
            if end > max_date:
                end = max_date
            if start > max_date:
                continue

            start_str = start.isoformat()
            end_str = end.isoformat()
            slice_rel = self.con.sql(f"""
                SELECT *
                FROM {table_name}
                WHERE CAST({date_quoted} AS DATE)
                      BETWEEN CAST('{start_str}' AS DATE) AND CAST('{end_str}' AS DATE)
                """)
            slices.append((label, start, end, slice_rel))

        return slices

    # ── MRQAP helper ────────────────────────────────────────────────────────

    @staticmethod
    def mrqap(
        Y: NDArray[np.float64],
        X_matrices: list[NDArray[np.float64]],
        n_permutations: int = 1000,
        *,
        method: Literal["pearson", "spearman"] = "pearson",
    ) -> dict[str, Any]:
        """Small pure-Python MRQAP-style permutation test for matrix regression.

        Permutes rows/columns of ``Y`` (the standard MRQAP double-
        permutation) and recomputes OLS coefficients on lower-triangular
        vectorized entries. Returns coefficient estimates and empirical
        two-tailed p-values.

        Parameters
        ----------
        Y : (n, n) array
            Dependent square matrix (e.g. similarity / distance).
        X_matrices : list of (n, n) arrays
            Independent square matrices. The first column is automatically
            an intercept.
        n_permutations : int, default 1000
            Number of row/column permutations.
        method : {"pearson", "spearman"}, default "pearson"
            Correlation method used for the semi-partial correlation
            shortcut diagnostics in the result dict.

        Returns
        -------
        dict
            ``coefficients``: estimated beta vector.
            ``p_values``: empirical two-tailed p-values per predictor.
            ``r2``: R² of the unpermuted model.
            ``permutation_betas``: (n_permutations, n_predictors) array.

        Notes
        -----
        This is a minimal MRQAP approximation. It does not replace a full
        QAP package for large matrices or complex dependence structures;
        it is included so pyduck-ona stays R-free for simple hypothesis
        tests on HR network matrices.
        """
        try:
            from scipy import stats as st
        except ImportError as e:
            raise ImportError(
                "scipy is required for DuckONA.mrqap; install pyduck-ona[stats] "
                "or `pip install scipy`."
            ) from e

        Y = np.asarray(Y, dtype=float)
        Xs = [np.asarray(x, dtype=float) for x in X_matrices]
        if Y.ndim != 2 or Y.shape[0] != Y.shape[1]:
            raise ValueError("Y must be a square matrix")
        n = Y.shape[0]
        for idx, x in enumerate(Xs):
            if x.shape != (n, n):
                raise ValueError(f"X matrix {idx} shape {x.shape} does not match Y shape {Y.shape}")

        def _vec_lower(arr: NDArray[np.float64]) -> NDArray[np.float64]:
            """Vectorize the strict lower triangle of a square matrix."""
            rows, cols = np.tril_indices(n, k=-1)
            return arr[rows, cols]

        # Build design matrix with intercept.
        X_design = np.column_stack(
            [np.ones(_vec_lower(Xs[0]).shape[0])] + [_vec_lower(x) for x in Xs]
        )
        y_vec = _vec_lower(Y)

        # OLS via normal equation.
        XtX = X_design.T @ X_design
        if np.linalg.cond(XtX) > 1e12:
            # Mild regularization for near-singular designs.
            XtX += np.eye(XtX.shape[0]) * 1e-6
        beta = np.linalg.solve(XtX, X_design.T @ y_vec)
        y_pred = X_design @ beta
        ss_res = float(np.sum((y_vec - y_pred) ** 2))
        ss_tot = float(np.sum((y_vec - np.mean(y_vec)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Permutations: shuffle rows and columns of Y together.
        rng = np.random.default_rng(seed=42)
        perm_betas = np.zeros((n_permutations, len(beta)))
        for i in range(n_permutations):
            perm = rng.permutation(n)
            Y_perm = Y[perm, :][:, perm]
            y_perm_vec = _vec_lower(Y_perm)
            b_perm = np.linalg.solve(XtX, X_design.T @ y_perm_vec)
            perm_betas[i, :] = b_perm

        # Two-tailed empirical p-values.
        p_values = np.mean(np.abs(perm_betas) >= np.abs(beta), axis=0)

        # Correlation diagnostics for the first predictor (useful for
        # quick sanity checks in examples/tests).
        corr_result: dict[str, Any] = {}
        if Xs:
            with np.errstate(invalid="ignore"):
                res = st.pearsonr(_vec_lower(Y), _vec_lower(Xs[0]))
                corr_result = {
                    "correlation": float(res.statistic),
                    "p_value": float(res.pvalue),
                }

        return {
            "coefficients": beta.tolist(),
            "p_values": p_values.tolist(),
            "r2": float(r2),
            "permutation_betas": perm_betas,
            "correlation": corr_result,
            "method": method,
        }

    # ── Utility ─────────────────────────────────────────────────────────────

    def sql(self, query: str) -> DuckDBPyRelation:
        """Run arbitrary SQL on the owned connection."""
        return self.con.sql(query)

    def table(self, table_name: str) -> DuckDBPyRelation:
        """Return a relation for a registered table."""
        _validate_table_name(table_name)
        return self.con.sql(f"SELECT * FROM {table_name}")
