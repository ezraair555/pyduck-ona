"""
broom-sm integration: tidy-statistics workflow for ONA models.

`broom-sm` (https://github.com/jcvall/broom-sm) is John C. Vallier's
Python port of R's broom package. It converts statistical model output
into tidy DataFrames and provides built-in publication-quality plots
(regression scatterplots, residual diagnostics, chi-square heatmaps,
coefficient forest plots).

This module wraps the bits an HR analyst actually needs:

  - :func:`correlation`    pairwise correlations with p-values
  - :func:`anova`          one-way ANOVA with tidy output
  - :func:`ols`            OLS linear regression (tidy + glance)
  - :func:`logistic`       logistic regression (tidy + glance)
  - :func:`chi_square`     chi-square test of independence
  - :func:`plot_*`         professional matplotlib figures
  - :func:`tidy_to_duckdb` write a tidy model result into a DuckDB table

Design philosophy: ``broom_sm`` already handles the statistics; we
handle the DuckDB-side I/O and keep a thin, predictable signature. The
two compose by convention.
"""
from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, Sequence

import duckdb
import pandas as pd

if TYPE_CHECKING:
    from duckdb import DuckDBPyRelation
    import matplotlib.figure

# Same safe-identifier rule used by core.py; keep local to avoid a
# cross-module dependency for this thin validation helper.
_IDENT_SAFE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# All `broom_sm.*` functions raise ImportError at call time if the
# package isn't installed (it's an optional extra, ``pip install
# pyduck-ona[broom]``). We import lazily so the rest of pyduck_ona
# remains usable without broom-sm.
_BROOM_IMPORT_ERROR = (
    "broom-sm is required for pyduck_ona.stats.* functions. "
    "Install with: pip install pyduck-ona[broom]  (or  pip install broom-sm)"
)


def _require_broom() -> None:
    try:
        import broom_sm  # noqa: F401
    except ImportError as e:
        raise ImportError(_BROOM_IMPORT_ERROR) from e


# ─── Lazy imports (defer until call time) ──────────────────────────────────

def _tidy():
    from broom_sm import stats_tidy
    return stats_tidy


def _glance():
    from broom_sm import stats_glance
    return stats_glance


def _augment():
    from broom_sm import stats_augment
    return stats_augment


def _anova_tidy():
    from broom_sm import stats_anova_tidy
    return stats_anova_tidy


def _corr_tidy():
    from broom_sm import stats_correlation_tidy
    return stats_correlation_tidy


def _ols_plot():
    from broom_sm import stats_ols_plot
    return stats_ols_plot


def _chisq_plot():
    from broom_sm import stats_chisquare_plot
    return stats_chisquare_plot


def _residual_plot():
    from broom_sm import stats_residual_plot
    return stats_residual_plot


def _coef_forest():
    from broom_sm import stats_coef_forest
    return stats_coef_forest


def _vif():
    from broom_sm import stats_vif
    return stats_vif


def _compare():
    from broom_sm import stats_compare
    return stats_compare


# ─── DataFrame extraction ──────────────────────────────────────────────────

def _as_df(data: "DuckDBPyRelation | pd.DataFrame") -> pd.DataFrame:
    """Coerce input to a pandas DataFrame.

    DuckDB relations and pandas DataFrames are both accepted; the broom
    family wants a DataFrame. If a relation is given, materializes it.
    """
    if isinstance(data, pd.DataFrame):
        return data
    if hasattr(data, "df"):  # DuckDBPyRelation duck-type
        return data.df()
    raise TypeError(
        f"expected DuckDBPyRelation or pandas.DataFrame, got {type(data).__name__}"
    )


# ─── Correlation ───────────────────────────────────────────────────────────

def correlation(
    data: "DuckDBPyRelation | pd.DataFrame",
    columns: Sequence[str] | None = None,
    *,
    col1: str | None = None,
    col2: str | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Pairwise correlations across a set of columns.

    Parameters
    ----------
    data
        DataFrame or DuckDB relation.
    columns : sequence of str, optional
        If given, return all pairwise correlations among these columns.
        Mutually exclusive with ``col1`` + ``col2``.
    col1, col2 : str, optional
        Compute a single correlation between two columns. Mutually
        exclusive with ``columns``.
    method : {"pearson", "spearman", "kendall"}, default "pearson"
        Correlation coefficient.

    Returns
    -------
    pandas.DataFrame
        Columns ``(term1, term2, correlation, p.value)``.

    Examples
    --------
    >>> corr = correlation(rel, columns=["team_size", "salary", "tenure_yrs"])
    >>> corr[corr["p.value"] < 0.05]
    """
    _require_broom()
    if (columns is None) == (col1 is None or col2 is None):
        raise ValueError(
            "pass either `columns=[...]` (pairwise) or `col1=..., col2=...` (single)"
        )
    df = _as_df(data)
    return _corr_tidy()(df, col1=col1, col2=col2, method=method, columns=columns)


# ─── ANOVA ─────────────────────────────────────────────────────────────────

def anova(
    data: "DuckDBPyRelation | pd.DataFrame",
    formula: str,
    *,
    anova_type: int = 2,
) -> pd.DataFrame:
    """One-way ANOVA via OLS, tidy output.

    Parameters
    ----------
    data
    formula : str
        Patsy-style formula, e.g. ``"salary ~ department"``.
    anova_type : {1, 2, 3}, default 2
        Type of ANOVA (1 = sequential, 2 = partial SS, 3 = marginal).

    Returns
    -------
    pandas.DataFrame
        Columns ``(term, sum_sq, df, statistic, p.value)``.

    Examples
    --------
    >>> anova(rel, "salary ~ department")
    """
    _require_broom()
    df = _as_df(data)
    return _anova_tidy()(df, formula, anova_type=anova_type)


# ─── Linear regression (OLS) ───────────────────────────────────────────────

def ols(
    data: "DuckDBPyRelation | pd.DataFrame",
    formula: str,
    *,
    cov_type: str = "nonrobust",
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit an OLS linear regression. Returns (tidy, glance).

    Parameters
    ----------
    data
    formula : str
        Patsy-style formula, e.g. ``"salary ~ team_size + tenure_yrs"``.
    cov_type : str, default "nonrobust"
        Standard-error estimator (``"nonrobust"``, ``"HC1"``, ``"HC3"``,
        ``"cluster"``). Use ``"HC3"`` for heteroskedasticity-robust SE.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    (tidy, glance) : tuple of pandas.DataFrame
        - tidy: per-coefficient table with estimate, std error, t-stat,
          p-value, conf.low, conf.high
        - glance: model-level summary (R², adj-R², AIC, BIC, F, df, nobs)

    Examples
    --------
    >>> tidy, glance = ols(rel, "salary ~ team_size + tenure_yrs")
    >>> tidy[tidy["p.value"] < 0.05]
    """
    _require_broom()
    df = _as_df(data)
    tidy_df = _tidy()(
        df, formula, stat_type="ols",
        cov_type=cov_type, alpha=alpha,
    )
    glance_df = _glance()(
        df, formula, stat_type="ols",
        cov_type=cov_type, alpha=alpha,
    )
    return tidy_df, glance_df


# ─── Logistic regression ───────────────────────────────────────────────────

def logistic(
    data: "DuckDBPyRelation | pd.DataFrame",
    formula: str,
    *,
    cov_type: str = "nonrobust",
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a logistic regression. Returns (tidy, glance).

    Parameters
    ----------
    data
    formula : str
        Patsy formula, e.g. ``"attrition ~ team_size + salary"``. The
        outcome must be binary (0/1) or interpretable as such.
    cov_type : str, default "nonrobust"
    alpha : float, default 0.05

    Returns
    -------
    (tidy, glance) : tuple of pandas.DataFrame
        - tidy: per-coefficient log-odds table; exp(estimate) gives the
          odds ratio for that variable
        - glance: model-level summary (deviance, AIC, BIC, df, nobs,
          pseudo-R² via log-likelihood)

    Examples
    --------
    >>> tidy, glance = logistic(rel, "attrition ~ team_size + tenure_yrs")
    >>> tidy.assign(odds_ratio=lambda d: np.exp(d["estimate"]))
    """
    _require_broom()
    df = _as_df(data)
    tidy_df = _tidy()(
        df, formula, stat_type="logit",
        cov_type=cov_type, alpha=alpha,
    )
    glance_df = _glance()(
        df, formula, stat_type="logit",
        cov_type=cov_type, alpha=alpha,
    )
    return tidy_df, glance_df


# ─── Chi-square test of independence ───────────────────────────────────────

def chi_square(
    data: "DuckDBPyRelation | pd.DataFrame",
    x: str,
    y: str,
) -> tuple[pd.DataFrame, "matplotlib.figure.Figure"]:
    """Chi-square test of independence between two categorical variables.

    Parameters
    ----------
    data
    x, y : str
        Column names of the two categorical variables.

    Returns
    -------
    (table, figure) : tuple
        - table: ``(chi2, p_value, dof)`` summary
        - figure: matplotlib Figure with a heatmap of the observed
          counts overlaid with the expected counts, plus the chi-square
          and p-value annotated on the title.

    Examples
    --------
    >>> table, fig = chi_square(rel, "department", "gender")
    >>> fig.savefig("dept_by_gender.png", dpi=120, bbox_inches="tight")
    """
    _require_broom()
    df = _as_df(data)
    return _chisq_plot()(df, x, y)


# ─── Plots ─────────────────────────────────────────────────────────────────

def plot_ols(
    data: "DuckDBPyRelation | pd.DataFrame",
    x: Sequence[str],
    y: str,
) -> list[tuple[str, "matplotlib.figure.Figure"]]:
    """Per-predictor OLS scatterplots with fitted regression line.

    Returns a list of ``(x_label, figure)`` tuples — one figure per
    predictor. Each figure shows the data + fitted line + 95% CI band
    and a small annotation panel with slope, R², and p-value.

    For multivariate OLS diagnostics (residuals vs fitted, QQ, scale-
    location, leverage), use :func:`plot_residuals` instead.
    """
    _require_broom()
    df = _as_df(data)
    return _ols_plot()(df, x, y)


def plot_residuals(
    data: "DuckDBPyRelation | pd.DataFrame",
    x: Sequence[str],
    y: str,
) -> list[tuple[str, "matplotlib.figure.Figure"]]:
    """Residual-diagnostic plots for each predictor.

    One figure per predictor showing residuals against the predictor
    value. Useful for spotting nonlinearity and heteroskedasticity.
    """
    _require_broom()
    df = _as_df(data)
    return _residual_plot()(df, x, y)


def plot_coefficients(
    tidy_df: pd.DataFrame,
    *,
    reference_line: float = 0.0,
    sort: bool = True,
) -> tuple["matplotlib.figure.Figure", Any]:
    """Forest plot of regression coefficients with confidence intervals.

    Parameters
    ----------
    tidy_df : pandas.DataFrame
        The ``tidy`` output from :func:`ols` or :func:`logistic`. Must
        have columns ``term``, ``estimate``, ``conf.low``, ``conf.high``.
    reference_line : float, default 0.0
        Vertical reference line. Set to 1.0 for odds-ratio plots.
    sort : bool, default True
        If True, sort by estimate size (largest effect at top).

    Returns
    -------
    (figure, axes) : matplotlib objects
    """
    _require_broom()
    fig, ax = _coef_forest()(tidy_df, reference_line=reference_line, sort=sort)
    return fig, ax


# ─── Diagnostic helpers ────────────────────────────────────────────────────

def vif(
    data: "DuckDBPyRelation | pd.DataFrame",
    formula: str,
) -> pd.Series:
    """Variance Inflation Factors for the predictors in a formula.

    Use to check multicollinearity. VIF > 5 is a yellow flag, > 10 is a
    red flag — predictors with high VIF are linear combinations of other
    predictors and may inflate coefficient standard errors.
    """
    _require_broom()
    df = _as_df(data)
    return _vif()(df, formula)


def model_compare(
    models: dict[str, Any],
) -> pd.DataFrame:
    """Side-by-side comparison of multiple fitted models.

    Parameters
    ----------
    models : dict
        Mapping of ``name -> fitted statsmodels result``. Each name
        appears as a column in the output.

    Returns
    -------
    pandas.DataFrame
        Each row is a glance statistic (R², AIC, BIC, log-lik, df, etc.);
        each column is one of the input models.
    """
    _require_broom()
    return _compare()(models)


def _validate_table_name(name: str) -> None:
    """Reject unsafe or malformed DuckDB table identifiers.

    We intentionally restrict table names to unquoted identifiers to
    avoid SQL injection via values like ``foo; DROP TABLE bar; --``.
    DuckDB supports quoted identifiers, but allowing them here would
    re-open the injection surface; callers can use safe names.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("table_name must be a non-empty string")
    if not _IDENT_SAFE_RE.match(name):
        raise ValueError(
            f"table_name must be a valid unquoted DuckDB identifier, "
            f"got {name!r}"
        )


# ─── DuckDB I/O ────────────────────────────────────────────────────────────

def tidy_to_duckdb(
    tidy_df: pd.DataFrame,
    con: "duckdb.DuckDBPyConnection | None" = None,
    table_name: str = "model_tidy",
) -> tuple[str, "duckdb.DuckDBPyConnection"]:
    """Write a tidy model result into a DuckDB table.

    Parameters
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

    Returns
    -------
    (table_name, con) : tuple
        The table name and the connection that owns it. Pass ``con``
        to subsequent ``con.sql(...)`` calls.

    Notes
    -----
    ``broom-sm`` tidy DataFrames use R-style dotted column names
    (``p.value``, ``conf.low``) that DuckDB parses as struct field
    access unless quoted. To save users from quoting every reference,
    this function rewrites dotted column names to underscore equivalents
    on write (``p.value`` → ``p_value``, ``conf.low`` → ``conf_low``).
    The returned table is therefore queryable with unquoted identifiers
    like ``SELECT term, p_value FROM model_tidy``.
    """
    if con is None:
        con = duckdb.connect()
    _validate_table_name(table_name)
    # Rewrite dotted column names to underscored equivalents so the
    # resulting DuckDB table is queryable without manual quoting. This
    # is purely a write-time rewrite; the in-memory ``tidy_df`` argument
    # is not mutated.
    safe_df = tidy_df.rename(columns=lambda c: c.replace(".", "_"))
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM safe_df")
    return table_name, con


def to_duckdb(
    data: "DuckDBPyRelation | pd.DataFrame",
    table_name: str,
    con: "duckdb.DuckDBPyConnection | None" = None,
) -> tuple["DuckDBPyRelation", "duckdb.DuckDBPyConnection"]:
    """Register a DataFrame or relation as a DuckDB table.

    Parameters
    ----------
    data
    table_name : str
    con : DuckDBPyConnection, optional
        Existing connection. If None, a new in-memory connection is
        created. If ``data`` is a ``DuckDBPyRelation`` created on a
        different connection, you MUST pass the same connection here
        — DuckDB relations are not portable across connections.

    Returns
    -------
    (relation, con) : tuple
        A queryable relation on the table, and the connection that
        owns it.
    """
    if con is None:
        con = duckdb.connect()
    _validate_table_name(table_name)
    if isinstance(data, pd.DataFrame):
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM data")
    else:
        # DuckDBPyRelation; only works if data was created on the same
        # connection. If not, this raises — caller should re-create the
        # relation from the source data on this connection.
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM data")
    return con.sql(f"SELECT * FROM {table_name}"), con


# ─── Save figures (small UX nicety) ────────────────────────────────────────

def save_figure(
    fig: "matplotlib.figure.Figure",
    path: str,
    *,
    dpi: int = 120,
) -> str:
    """Save a matplotlib figure and return the path.

    Tiny wrapper that ensures the parent directory exists and uses a
    tight bounding box. Returns the path so you can chain.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path