"""Integration tests for broom-sm statistics integration.

Each test:
  1. Generates realistic HR-shaped data
  2. Runs one statistical test
  3. Asserts the test output is shaped correctly
  4. Saves a publication-quality plot to a tmpdir
  5. Verifies the plot file was created and is non-empty

Plot files are written under ``tests/_artifacts/`` (gitignored) and
kept around for visual inspection during development. The CI matrix
does NOT inspect the visual content — only that the file exists and is
a real PNG with reasonable byte size.

Run with:
    python -m pytest tests/integration/test_stats.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

import pyduck_ona as pona


ARTIFACTS = Path(__file__).parent.parent / "_artifacts"
ARTIFACTS.mkdir(exist_ok=True)


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def hr_data() -> pd.DataFrame:
    """Generate a realistic HR-shaped dataset.

    200 employees with:
      - team_size: integer count of direct + indirect reports
      - tenure_yrs: years at the company
      - salary: annual salary (positively correlated with team_size)
      - department: categorical (Eng / Ops / Sales / HR)
      - gender: binary (M / F)
      - attrition: binary outcome (1 = left, 0 = stayed); correlated
        with low salary and short tenure

    The known structure lets us assert that:
      - correlation(team_size, salary) is positive and significant
      - anova(salary ~ department) detects Eng-vs-Ops mean differences
      - chi_square(department, gender) shows the synthetic relationship
      - logistic(attrition ~ salary + tenure) recovers negative betas
    """
    rng = np.random.default_rng(20260620)
    n = 200

    team_size = rng.poisson(5, n).clip(0, 25)
    tenure_yrs = rng.gamma(2.0, 2.0, n).clip(0, 25)
    department = rng.choice(
        ["Eng", "Ops", "Sales", "HR"], n, p=[0.5, 0.2, 0.2, 0.1]
    )
    gender = rng.choice(["M", "F"], n, p=[0.55, 0.45])

    # Eng gets paid more; Sales is bonus-driven (higher variance)
    dept_base = {"Eng": 95000, "Ops": 78000, "Sales": 85000, "HR": 72000}
    salary = np.array([
        dept_base[d] + team_size[i] * 2500 + tenure_yrs[i] * 800
        + rng.normal(0, 12000)
        for i, d in enumerate(department)
    ])

    # Attrition: lower for high salary + long tenure
    logit = -2.0 - 0.00003 * (salary - 80000) - 0.3 * tenure_yrs + 0.05 * team_size
    p_attrition = 1 / (1 + np.exp(-logit))
    attrition = rng.binomial(1, p_attrition)

    return pd.DataFrame({
        "team_size": team_size,
        "tenure_yrs": tenure_yrs,
        "salary": salary.round(0),
        "department": department,
        "gender": gender,
        "attrition": attrition,
    })


@pytest.fixture(scope="module")
def hr_rel(hr_data) -> duckdb.DuckDBPyRelation:
    """HR data as a DuckDB relation — same fixture, different type."""
    return duckdb.from_df(hr_data)


# ─── 1. Correlation ─────────────────────────────────────────────────────────

class TestCorrelation:
    def test_pairwise_returns_full_matrix(self, hr_data):
        cols = ["team_size", "tenure_yrs", "salary"]
        result = pona.correlation(hr_data, columns=cols)
        # n choose 2 pairs
        assert len(result) == len(cols) * (len(cols) - 1) // 2
        assert list(result.columns) == ["term1", "term2", "correlation", "p.value"]

    def test_single_pair(self, hr_data):
        result = pona.correlation(hr_data, col1="team_size", col2="salary")
        assert len(result) == 1
        # The synthetic DGP: salary = base + 2500*team_size + noise(σ=12k).
        # With noise of that magnitude, the Pearson r is around 0.37
        # for a 200-sample draw — positive and significant, but not as
        # strong as the 2500/12000 ratio suggests (because noise is
        # constant, not proportional). 0.3 is a realistic lower bound.
        row = result.iloc[0]
        assert row["correlation"] > 0.3
        assert row["p.value"] < 0.001

    def test_works_with_duckdb_relation(self, hr_rel):
        # Same call, different input type — should produce same result
        result = pona.correlation(hr_rel, col1="team_size", col2="salary")
        assert result.iloc[0]["correlation"] > 0.3

    def test_correlation_plot_heatmap(self, hr_data):
        # broom-sm doesn't have a built-in corr heatmap, so we build one
        # using the corr tidy output + matplotlib. This is a small
        # convenience function worth having.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cols = ["team_size", "tenure_yrs", "salary"]
        corr_df = pona.correlation(hr_data, columns=cols)
        # Build the square correlation matrix directly. pivot+combine_first
        # returns a read-only view; assigning to np.fill_diagonal fails.
        mat = pd.DataFrame(
            np.eye(len(cols)),
            index=cols, columns=cols,
            dtype=float,
        )
        for _, row in corr_df.iterrows():
            mat.at[row["term1"], row["term2"]] = row["correlation"]
            mat.at[row["term2"], row["term1"]] = row["correlation"]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticklabels(cols)
        for i in range(len(cols)):
            for j in range(len(cols)):
                val = mat.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color="white" if abs(val) > 0.5 else "black", fontsize=9)
        ax.set_title("Pairwise Correlations (Pearson)\n$n$=200 synthetic HR sample",
                      fontsize=11)
        plt.colorbar(im, ax=ax, label="r")
        fig.tight_layout()

        path = pona.save_figure(fig, ARTIFACTS / "correlation_heatmap.png")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 5000  # ~5KB minimum for a real plot


# ─── 2. ANOVA ───────────────────────────────────────────────────────────────

class TestANOVA:
    def test_detects_department_difference(self, hr_data):
        result = pona.anova(hr_data, "salary ~ department")
        assert list(result.columns) == ["term", "sum_sq", "df", "statistic", "p.value"]
        # Eng has 95k base vs HR 72k → F-statistic should be large
        dept_row = result[result["term"] == "department"].iloc[0]
        assert dept_row["statistic"] > 5.0
        assert dept_row["p.value"] < 0.01

    def test_returns_one_row_per_term(self, hr_data):
        result = pona.anova(hr_data, "salary ~ department")
        # ANOVA on salary ~ department: 1 effect + 1 residual
        assert len(result) == 2

    def test_anova_plot_group_means(self, hr_data):
        """Custom boxplot-by-group — broom-sm doesn't ship one, but it's
        the canonical ANOVA visualization."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        groups = [hr_data[hr_data["department"] == d]["salary"].values
                  for d in ["Eng", "Ops", "Sales", "HR"]]
        fig, ax = plt.subplots(figsize=(7, 5))
        # ``tick_labels`` is the modern matplotlib API (>=3.9); ``labels``
        # is deprecated and was removed in matplotlib 3.11. Support both
        # so the test passes on the pinned 3.9+ and on older 3.8.
        try:
            bp = ax.boxplot(
                groups,
                tick_labels=["Eng", "Ops", "Sales", "HR"],
                patch_artist=True,
                showmeans=True,
            )
        except TypeError:
            # matplotlib < 3.9: fall back to the deprecated kwarg.
            bp = ax.boxplot(
                groups,
                labels=["Eng", "Ops", "Sales", "HR"],
                patch_artist=True,
                showmeans=True,
            )
        for patch, color in zip(bp["boxes"],
                                 ["#4C72B0", "#DD8452", "#55A467", "#C44E52"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("Salary ($)")
        ax.set_title("Salary by Department — One-Way ANOVA\n"
                      "F=?.??, p=?.???  (n=200)", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        path = pona.save_figure(fig, ARTIFACTS / "anova_boxplot.png")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 5000


# ─── 3. Chi-square ─────────────────────────────────────────────────────────

class TestChiSquare:
    def test_returns_table_and_figure(self, hr_data):
        table, fig = pona.chi_square(hr_data, "department", "gender")
        assert list(table.columns) == ["chi2", "p_value", "dof"]
        # dof = (4 depts - 1) * (2 genders - 1) = 3
        assert table.iloc[0]["dof"] == 3
        assert table.iloc[0]["chi2"] >= 0
        assert 0.0 <= table.iloc[0]["p_value"] <= 1.0

    def test_saves_professional_figure(self, hr_data):
        import matplotlib
        matplotlib.use("Agg")
        # The figure returned by chi_square is already styled by broom-sm.
        # We just need to save it.
        table, fig = pona.chi_square(hr_data, "department", "gender")
        path = pona.save_figure(fig, ARTIFACTS / "chi_square_heatmap.png")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 5000
        plt = sys.modules["matplotlib.pyplot"]
        plt.close(fig)

    def test_independent_variables_show_nonsignificant(self, hr_data):
        # Independent random draws of department and gender → no assoc
        table, _ = pona.chi_square(hr_data, "department", "gender")
        # With n=200 the test is well-powered; independence should not
        # produce a significant result by chance too often. We allow p
        # > 0.01 as a generous threshold; rerun the test if it ever
        # fails (rng is fixed so this is deterministic).
        assert table.iloc[0]["p_value"] > 0.01


# ─── 4. Linear regression (OLS) ────────────────────────────────────────────

class TestOLS:
    def test_returns_tidy_and_glance(self, hr_data):
        tidy, glance = pona.ols(hr_data, "salary ~ team_size + tenure_yrs")
        # tidy has one row per coefficient
        assert "team_size" in tidy["term"].values
        assert "tenure_yrs" in tidy["term"].values
        assert "Intercept" in tidy["term"].values
        # glance has model-level summary
        assert "r.squared" in glance.columns or "rsquared" in glance.columns
        # Both slopes should be positive (data-generating process)
        ts = tidy[tidy["term"] == "team_size"].iloc[0]
        tn = tidy[tidy["term"] == "tenure_yrs"].iloc[0]
        assert ts["estimate"] > 0
        assert tn["estimate"] > 0

    def test_works_with_duckdb_relation(self, hr_rel):
        tidy, _ = pona.ols(hr_rel, "salary ~ team_size + tenure_yrs")
        assert len(tidy) == 3

    def test_significance_levels_recover_data(self, hr_data):
        tidy, glance = pona.ols(hr_data, "salary ~ team_size + tenure_yrs")
        # team_size was constructed with a 2500 coefficient → highly
        # significant. tenure_yrs had 800 — should also be significant
        # at n=200 but the noise is higher.
        ts = tidy[tidy["term"] == "team_size"].iloc[0]
        assert ts["p.value"] < 0.001
        assert ts["estimate"] > 1000  # recovers order of magnitude

    def test_ols_scatterplot(self, hr_data):
        plots = pona.plot_ols(hr_data, x=["team_size", "tenure_yrs"], y="salary")
        assert len(plots) == 2
        for label, fig in plots:
            assert label in ("team_size", "tenure_yrs")
            path = pona.save_figure(fig, ARTIFACTS / f"ols_{label}.png")
            assert os.path.exists(path)
            assert os.path.getsize(path) > 5000

    def test_residual_plot(self, hr_data):
        plots = pona.plot_residuals(hr_data, x=["team_size"], y="salary")
        assert len(plots) == 1
        _, fig = plots[0]
        path = pona.save_figure(fig, ARTIFACTS / "residual_team_size.png")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 5000

    def test_coefficient_forest_plot(self, hr_data):
        tidy, _ = pona.ols(hr_data, "salary ~ team_size + tenure_yrs")
        fig, ax = pona.plot_coefficients(tidy)
        path = pona.save_figure(fig, ARTIFACTS / "coef_forest_ols.png")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 5000

    def test_vif_under_threshold(self, hr_data):
        vifs = pona.vif(hr_data, "salary ~ team_size + tenure_yrs")
        # team_size and tenure_yrs are independent in the DGP
        assert all(v < 5 for v in vifs)


# ─── 5. Logistic regression ────────────────────────────────────────────────

class TestLogistic:
    def test_recovers_negative_salary_effect(self, hr_data):
        # Higher salary → lower attrition in the DGP
        tidy, glance = pona.logistic(hr_data, "attrition ~ salary + tenure_yrs + team_size")
        salary_row = tidy[tidy["term"] == "salary"].iloc[0]
        assert salary_row["estimate"] < 0  # negative log-odds
        assert salary_row["p.value"] < 0.05

    def test_returns_tidy_with_odds_ratios_computable(self, hr_data):
        tidy, _ = pona.logistic(hr_data, "attrition ~ salary + tenure_yrs")
        # exp(estimate) should be in the table or trivially computable
        assert "estimate" in tidy.columns
        odds_ratios = np.exp(tidy["estimate"])
        # All odds ratios should be positive
        assert (odds_ratios > 0).all()

    def test_glance_has_deviance_and_aic(self, hr_data):
        _, glance = pona.logistic(hr_data, "attrition ~ salary + tenure_yrs + team_size")
        # Glance output is a 1-row DataFrame; AIC and deviance should
        # be present
        cols = list(glance.columns)
        assert any("aic" in c.lower() for c in cols)
        assert any("deviance" in c.lower() for c in cols)

    def test_works_with_duckdb_relation(self, hr_rel):
        tidy, _ = pona.logistic(hr_rel, "attrition ~ salary")
        assert len(tidy) == 2  # intercept + salary

    def test_logistic_coef_plot_with_or_reference(self, hr_data):
        tidy, _ = pona.logistic(hr_data, "attrition ~ salary + tenure_yrs + team_size")
        # For odds ratios, the reference line is 1.0 (neutral effect)
        fig, ax = pona.plot_coefficients(tidy, reference_line=0.0)
        path = pona.save_figure(fig, ARTIFACTS / "coef_forest_logistic.png")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 5000


# ─── Cross-cutting: DuckDB I/O ──────────────────────────────────────────────

class TestDuckDBIO:
    def test_tidy_to_duckdb_returns_connection(self, hr_data):
        tidy, _ = pona.ols(hr_data, "salary ~ team_size + tenure_yrs")
        table_name, con = pona.tidy_to_duckdb(tidy, table_name="ols_results")
        assert table_name == "ols_results"
        # tidy_to_duckdb rewrites dotted column names to underscored
        # equivalents on write (p.value → p_value), so the resulting
        # table is queryable with unquoted identifiers.
        result = con.sql(
            f'SELECT term, estimate, p_value FROM "{table_name}" '
            f'WHERE p_value < 0.05 ORDER BY p_value'
        )
        rows = result.fetchall()
        assert len(rows) >= 1
        # The significant term should include team_size
        assert any("team_size" in r[0] for r in rows)

    def test_to_duckdb_from_dataframe(self, hr_data):
        rel, con = pona.to_duckdb(hr_data, "hr_data")
        assert rel.count("*").fetchone()[0] == 200
        # Verify by querying on the same connection
        again = con.sql('SELECT COUNT(*) FROM "hr_data"').fetchone()
        assert again[0] == 200

    def test_to_duckdb_reuses_existing_connection(self):
        """A Relation created on connection X cannot be used on
        connection Y — that's a DuckDB constraint, not a pyduck-ona
        bug. Verify the happy path: same connection, same Relation,
        table gets created."""
        con = duckdb.connect()
        rel = con.sql("SELECT 1 AS a, 2 AS b UNION ALL SELECT 3, 4")
        out, _ = pona.to_duckdb(rel, "test_table", con=con)
        assert out.count("*").fetchone()[0] == 2

    def test_to_duckdb_rejects_cross_connection_relation(self, hr_rel):
        """Document the limitation: a Relation from one connection
        cannot be written into a table on a different connection. The
        error should be a clear DuckDB InvalidInputException, not a
        cryptic pyduck-ona traceback."""
        con = duckdb.connect()
        with pytest.raises(Exception) as excinfo:
            pona.to_duckdb(hr_rel, "test_table", con=con)
        # The underlying DuckDB error is the right thing to surface
        assert "not suitable for replacement scan" in str(excinfo.value).lower() or \
               "another connection" in str(excinfo.value).lower()

    def test_full_pipeline_relationship(self, hr_data):
        """End-to-end: DataFrame → OLS → tidy → DuckDB → query, all on
        one connection. tidy_to_duckdb renames p.value → p_value on
        write so the column is queryable without manual quoting."""
        tidy, _ = pona.ols(hr_data, "salary ~ team_size + tenure_yrs")
        _, con = pona.tidy_to_duckdb(tidy, table_name="salary_model")
        significant = con.sql(
            'SELECT term FROM salary_model WHERE p_value < 0.05 ORDER BY estimate'
        ).fetchall()
        assert any("team_size" in t[0] for t in significant)


# ─── Cross-cutting: model_compare ──────────────────────────────────────────

class TestModelCompare:
    def test_compare_nested_models(self, hr_data):
        import statsmodels.formula.api as smf
        m1 = smf.ols("salary ~ team_size", hr_data).fit()
        m2 = smf.ols("salary ~ team_size + tenure_yrs", hr_data).fit()
        result = pona.model_compare_stats({"baseline": m1, "+ tenure": m2})
        # broom-sm's stats_compare returns long format: one row per
        # model, columns are (model, aic, bic, llf, nobs, ...). We test
        # that both model names appear and the metric columns are present.
        assert "model" in result.columns
        models = set(result["model"])
        assert "baseline" in models
        assert "+ tenure" in models
        for metric in ("aic", "bic", "llf", "nobs"):
            assert metric in result.columns

# ─── Tidy-to-DuckDB dotted column rewrite (P2-4 regression) ─────────────

class TestTidyToDuckDBDottedNames:
    """Regression tests for the dotted-name rewrite in tidy_to_duckdb.

    Before the fix: tidy DataFrames use ``broom-sm``'s R-style column
    names (``p.value``, ``conf.low``) which DuckDB parses as struct
    field access unless quoted. Users who wrote
    ``SELECT * FROM table WHERE p.value < 0.05`` got a BinderException.

    After the fix: ``tidy_to_duckdb`` rewrites dotted names to
    underscore equivalents on write (``p.value`` → ``p_value``), so the
    table is queryable with unquoted identifiers.
    """

    def test_renames_p_value_to_p_value(self, hr_data):
        tidy, _ = pona.ols(hr_data, "salary ~ team_size + tenure_yrs")
        _, con = pona.tidy_to_duckdb(tidy, table_name="ols_renamed")
        cols = [row[0] for row in con.sql('DESCRIBE ols_renamed').fetchall()]
        assert "p_value" in cols
        assert "p.value" not in cols

    def test_renames_conf_low_and_conf_high(self, hr_data):
        tidy, _ = pona.ols(hr_data, "salary ~ team_size + tenure_yrs")
        _, con = pona.tidy_to_duckdb(tidy, table_name="ols_ci")
        cols = [row[0] for row in con.sql('DESCRIBE ols_ci').fetchall()]
        assert "conf_low" in cols
        assert "conf_high" in cols

    def test_unquoted_query_works(self, hr_data):
        """The motivating use case: write a tidy model, then query
        without quoting the dotted columns."""
        tidy, _ = pona.ols(hr_data, "salary ~ team_size + tenure_yrs")
        _, con = pona.tidy_to_duckdb(tidy, table_name="ols_q")
        # This would have raised BinderException before the fix.
        rows = con.sql(
            'SELECT term, p_value FROM ols_q WHERE p_value < 0.05'
        ).fetchall()
        # team_size should be a significant predictor
        assert any("team_size" in r[0] for r in rows)

    def test_does_not_mutate_input_dataframe(self, hr_data):
        """The rewrite happens on a copy, not on the caller's DataFrame."""
        tidy, _ = pona.ols(hr_data, "salary ~ team_size + tenure_yrs")
        original_cols = list(tidy.columns)
        pona.tidy_to_duckdb(tidy, table_name="ols_nm")
        # tidy_df is unchanged
        assert list(tidy.columns) == original_cols
        assert "p.value" in original_cols  # still dotted in source
