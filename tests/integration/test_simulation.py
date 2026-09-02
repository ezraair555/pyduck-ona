"""Simulation-to-validate tests for pyduck-ona (Principle #9).

Per the project's standing modeling principles (see ``MEMORY.md`` /
``memory/2026-08-10``): *simulate to validate — recover known params
from fake data before trusting.* These tests fill the simulation gap
in pyduck-ona's own test suite.

Each test follows the same recipe:

    1. Specify a data-generating process (DGP) with known coefficients.
    2. Generate a synthetic HR dataset from the DGP.
    3. Run the pyduck-ona analysis pipeline (load → validate → graph
       metrics → join → model fit) on the synthetic data.
    4. Assert that the recovered coefficients are within a tolerance of
       the true coefficients.

The tolerances are intentionally generous (relative error ~25-40% for
logistic, ~5-15% for OLS on n=1000) because:

    - We run a single seed by default. The user is expected to bump
      ``n_seeds`` and ``n`` for tighter inference.
    - The ONA predictor (betweenness) is endogenous: it is computed
      from the same simulated graph, so coefficient recovery has more
      variance than a textbook i.i.d. regression.
    - We want the test to fail loudly on real bugs (sign flips,
      swapped predictors, broken pipelines), not on Monte Carlo noise.

Run with:
    python -m pytest tests/integration/test_simulation.py -v

Why this file exists as a separate module:
    - test_analysis.py exercises the API contract.
    - test_stats.py exercises broom-sm integration on canned data.
    - test_simulation.py exercises the *statistical validity* of the
      pipeline end-to-end. A passing test here means pyduck-ona can
      recover the truth from a known world. A failure means either
      (a) the package has a bug, or (b) the package is being used
      wrong. Either is worth catching before a board-deck model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pyduck_ona as pona

ARTIFACTS = Path(__file__).parent.parent / "_artifacts"
ARTIFACTS.mkdir(exist_ok=True)


# ─── Helpers ───────────────────────────────────────────────────────────────


def _assert_within(
    recovered: float,
    truth: float,
    *,
    rel_tol: float = 0.30,
    abs_floor: float = 0.20,
    label: str = "coefficient",
) -> None:
    """Assert ``recovered`` is within ``rel_tol`` (or ``abs_floor``) of ``truth``.

    We use a max-style tolerance: ``|recovered - truth| <= max(rel_tol*|truth|, abs_floor)``.
    The absolute floor protects against false-pass when the true
    coefficient is near zero (where relative tolerance is meaningless).
    """
    tol = abs_floor if truth == 0 else max(rel_tol * abs(truth), abs_floor)
    diff = abs(recovered - truth)
    assert diff <= tol, (
        f"{label}: recovered={recovered:.4f}, truth={truth:.4f}, "
        f"diff={diff:.4f}, tol={tol:.4f} (rel_tol={rel_tol}, abs_floor={abs_floor})"
    )


def _build_synthetic_org(
    n_per_level: tuple[int, int, int, int] = (1, 5, 25, 200),
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build a 4-level synthetic org (CEO → VP → Director → IC).

    Returns a DataFrame with columns ``employee_id``, ``supervisor_id``.
    Counts default to a 1/5/25/200 pyramid = 231 employees.
    """
    n_ceo, n_vp, n_dir, n_ic = n_per_level
    rows: list[tuple[str, str | None]] = []

    # CEO
    rows.append(("E_CEO", None))

    # VPs report to CEO
    for i in range(n_vp):
        rows.append((f"E_VP{i:02d}", "E_CEO"))

    # Directors report to VPs (round-robin)
    for i in range(n_dir):
        mgr = f"E_VP{i % n_vp:02d}"
        rows.append((f"E_DIR{i:03d}", mgr))

    # ICs report to Directors (round-robin)
    for i in range(n_ic):
        mgr = f"E_DIR{i % n_dir:03d}"
        rows.append((f"E_IC{i:04d}", mgr))

    return pd.DataFrame(rows, columns=["employee_id", "supervisor_id"])


# ─── Test 1: OLS salary model recovers true coefficients ──────────────────


def test_ols_salary_recovers_true_coefficients() -> None:
    """OLS on a synthetic salary DGP recovers job_level + tenure betas.

    DGP:
        salary = β0 + β_level * job_level + β_tenure * tenure_yrs
                 + β_gender * gender_M + ε
        β0     = 50_000
        β_level = 25_000  (each level adds 25k)
        β_tenure = 1_500  (each year adds 1.5k)
        β_gender = 5_000  (men earn 5k more, synthetic effect)

    Run pyduck-ona end-to-end: load HRIS, validate, build edges,
    join HRIS, fit OLS, check coefficients.
    """
    rng = np.random.default_rng(seed=20260827)
    org = _build_synthetic_org(n_per_level=(1, 5, 25, 200), rng=rng)

    # True coefficients
    b0, b_level, b_tenure, b_gender = 50_000.0, 25_000.0, 1_500.0, 5_000.0

    df = org.copy()
    # Employee-level covariates
    df["job_level"] = rng.integers(1, 6, size=len(df)).astype(int)  # 1..5
    df["tenure_yrs"] = np.clip(rng.gamma(2.0, 2.0, size=len(df)), 0, 20)
    df["gender"] = rng.choice(["M", "F"], size=len(df), p=[0.55, 0.45])

    # Salary from DGP + noise
    noise = rng.normal(0, 10_000, size=len(df))
    df["salary"] = (
        b0
        + b_level * df["job_level"]
        + b_tenure * df["tenure_yrs"]
        + b_gender * (df["gender"] == "M").astype(int)
        + noise
    )

    # Pipeline
    ona = pona.DuckONA()
    ona.load_hris(df)
    # The salary model does not strictly need org edges, but we run
    # the full pipeline anyway to confirm load + join work on this
    # synthetic extract.
    edges = ona.build_org_edges("employee_id", "supervisor_id")
    metrics = ona.betweenness(edges, "employee_id", "supervisor_id")
    enriched = ona.join_hris(metrics)
    # `betweenness` for all 231 nodes in this pyramid graph is non-zero
    # only for VPs/Directors (ICs have 0). We do not use it as a
    # predictor here; we use it to prove the join works.
    assert enriched.count("*").fetchone()[0] >= len(df) * 0.9, (
        "join_hris returned suspiciously few rows; pipeline is broken"
    )

    tidy, glance = ona.ols(enriched, "salary ~ job_level + tenure_yrs + C(gender)")

    # Extract recovered coefficients (R-style C(gender) → gender[T.M])
    def _coef(term: str) -> float:
        row = tidy[tidy["term"] == term]
        assert not row.empty, f"term {term!r} not in tidy output: {list(tidy['term'])}"
        return float(row["estimate"].iloc[0])

    rec_intercept = _coef("Intercept")
    rec_level = _coef("job_level")
    rec_tenure = _coef("tenure_yrs")
    rec_gender = _coef("C(gender)[T.M]")

    # Asserts — generous tolerances for n=231 with ONA endogeneity in
    # the join (not in the model itself, but the join changes row counts).
    _assert_within(rec_level, b_level, rel_tol=0.10, abs_floor=1_500, label="beta_level")
    _assert_within(rec_tenure, b_tenure, rel_tol=0.35, abs_floor=300, label="beta_tenure")
    _assert_within(rec_gender, b_gender, rel_tol=0.40, abs_floor=2_500, label="beta_gender")
    # Intercept is harder because the centering of job_level/tenure shifts it.
    _assert_within(rec_intercept, b0, rel_tol=0.50, abs_floor=20_000, label="beta_intercept")

    # Glance should report reasonable R² for a clean DGP.
    assert "rsquared" in glance.columns, (
        f"unexpected glance columns: {list(glance.columns)}; broom-sm version drift?"
    )
    r2 = float(glance["rsquared"].iloc[0])
    assert r2 > 0.80, f"R²={r2:.3f} too low for clean DGP; pipeline or model is broken"


# ─── Test 2: Logistic attrition model recovers true coefficients ───────────


def test_logistic_attrition_recovers_true_coefficients() -> None:
    """Logistic on a synthetic attrition DGP recovers OR/coefficients.

    DGP:
        logit(P[attrition=1]) = α + β_tenure * tenure_yrs
                               + β_eng * engagement
                               + β_team * team_size
        α        = -1.0
        β_tenure = -0.30    (longer tenure → lower attrition)
        β_eng    = -0.45    (higher engagement → lower attrition)
        β_team   = +0.04    (larger teams → higher attrition, mild)

    team_size is built from the org chart (direct + indirect reports).
    This is the test that exercises the full ONA pipeline: build_org_edges
    → hierarchy_stats → join_hris → logistic.
    """
    rng = np.random.default_rng(seed=20260828)
    org = _build_synthetic_org(n_per_level=(1, 5, 25, 250), rng=rng)

    a, b_tenure, b_eng, b_team = 1.5, -0.30, -0.45, 0.04

    df = org.copy()
    df["tenure_yrs"] = np.clip(rng.gamma(2.0, 2.0, size=len(df)), 0, 20)
    df["engagement"] = np.clip(rng.normal(7.0, 1.5, size=len(df)), 1, 10)
    df["department"] = rng.choice(["Eng", "Ops", "Sales", "HR"], size=len(df))

    ona = pona.DuckONA()
    ona.load_hris(df)
    edges = ona.build_org_edges("employee_id", "supervisor_id")
    # hierarchy_stats returns manager_id, direct_reports, indirect_reports,
    # total_reports, team_size, levels_below. `team_size` here includes
    # the manager themselves in pyduck-ona's convention — we'll inspect.
    stats_rel = pona.hierarchy_stats(
        ona.con.sql("SELECT employee_id, supervisor_id FROM hris"),
        "employee_id", "supervisor_id",
    )
    stats_df = stats_rel.df()
    # team_size includes the manager; total_reports is the number of reports.
    # Use total_reports as the team_size predictor (i.e. manager team load).
    team_size_map = dict(zip(stats_df["manager_id"], stats_df["total_reports"], strict=False))
    df["team_size"] = df["employee_id"].map(team_size_map).fillna(0).astype(int)

    # Generate attrition from DGP
    logit = a + b_tenure * df["tenure_yrs"] + b_eng * df["engagement"] + b_team * df["team_size"]
    p_attrition = 1.0 / (1.0 + np.exp(-logit))
    df["attrition"] = (rng.random(len(df)) < p_attrition).astype(int)

    # Re-load HRIS now that team_size and attrition are present.
    ona.load_hris(df)

    # Pre-attrition-rate check: DGP should yield ~10-30% attrition, not 0% or 100%
    base_rate = float(df["attrition"].mean())
    assert 0.05 < base_rate < 0.50, (
        f"DGP produced base_rate={base_rate:.2%}; check DGP specification"
    )

    # Re-build edges against the now-complete HRIS table.
    edges = ona.build_org_edges("employee_id", "supervisor_id")

    # Pipeline: edges → betweenness → join → logistic
    metrics = ona.betweenness(edges, "employee_id", "supervisor_id")
    enriched = ona.join_hris(metrics)
    # The HRIS `hris` table already has attrition, tenure_yrs, engagement.
    # join_hris left-joined HRIS onto metrics, so enriched already has
    # those columns. No further join needed.
    tidy, glance = ona.logistic(
        enriched,
        "attrition ~ tenure_yrs + engagement + team_size",
    )

    def _coef(term: str) -> float:
        row = tidy[tidy["term"] == term]
        assert not row.empty, f"term {term!r} not in tidy output: {list(tidy['term'])}"
        return float(row["estimate"].iloc[0])

    rec_tenure = _coef("tenure_yrs")
    rec_eng = _coef("engagement")
    rec_team = _coef("team_size")

    # Logistic coefficient recovery has more variance than OLS, especially
    # when team_size has a long-tailed distribution. Generous tolerances.
    _assert_within(rec_tenure, b_tenure, rel_tol=0.35, abs_floor=0.15, label="beta_tenure")
    _assert_within(rec_eng, b_eng, rel_tol=0.35, abs_floor=0.15, label="beta_engagement")
    _assert_within(rec_team, b_team, rel_tol=0.50, abs_floor=0.05, label="beta_team_size")

    # Sign check — the qualitative finding should hold even if magnitude is off
    assert rec_tenure < 0, f"expected negative tenure effect, got {rec_tenure:.3f}"
    assert rec_eng < 0, f"expected negative engagement effect, got {rec_eng:.3f}"
    # team_size effect is small (+0.04); sign may not be recoverable reliably
    # in a single seed at n~281, so we don't assert sign here.


# ─── Test 3: MRQAP recovers known matrix relationship ──────────────────────


def test_mrqap_recovers_matrix_relationship() -> None:
    """MRQAP recovers a known matrix-regression coefficient.

    DGP:
        Build a collaboration matrix C (n=30 employees) and a
        department-similarity matrix D where D[i,j] = 1 if employees
        i and j are in the same department, else 0.
        Generate C = α * D + noise where noise is from a base
        similarity matrix N (small random).
        Run mrqap(Y=C, X=[D]) and assert recovered α is close to true α.

    True α = 0.6. Recovered α should be in [0.3, 0.9] with reasonable
    confidence (p < 0.05).
    """
    rng = np.random.default_rng(seed=20260829)
    n = 30
    true_alpha = 0.6

    # Department assignment: 5 departments, 6 employees each
    depts = np.repeat(np.arange(5), 6)
    d_mat = (depts[:, None] == depts[None, :]).astype(float)  # n x n same-department indicator
    np.fill_diagonal(d_mat, 0.0)

    # Noise matrix: small random similarities
    n_mat = rng.uniform(0, 0.3, size=(n, n))
    n_mat = (n_mat + n_mat.T) / 2  # symmetrize
    np.fill_diagonal(n_mat, 0.0)

    c_mat = true_alpha * d_mat + n_mat
    c_mat = (c_mat + c_mat.T) / 2  # symmetrize
    np.fill_diagonal(c_mat, 0.0)

    result = pona.DuckONA.mrqap(c_mat, [d_mat], n_permutations=500)

    # Coefficients layout: [intercept, β_D]
    coefs = result["coefficients"]
    assert len(coefs) == 2, f"expected 2 coefficients (intercept + β_D), got {len(coefs)}"

    recovered_beta = float(coefs[1])
    p_value_beta = float(result["p_values"][1])

    # MRQAP coefficient recovery on a clean synthetic is tight.
    _assert_within(recovered_beta, true_alpha, rel_tol=0.25, abs_floor=0.15, label="MRQAP beta_D")
    assert p_value_beta < 0.05, (
        f"MRQAP p_value={p_value_beta:.4f} not significant at α=0.05; "
        f"the matrix-regression may not be detecting the planted effect"
    )


# ─── Test 4: End-to-end pipeline stress test ───────────────────────────────


@pytest.mark.parametrize("seed", [20260827, 20260828, 20260829])
def test_pipeline_runs_clean_across_seeds(seed: int) -> None:
    """Smoke test: the full DuckONA pipeline runs without errors on 3 seeds.

    This is the lower bar to test_ols_salary_recovers_true_coefficients.
    If this fails, the pipeline is broken; if it passes but the
    coefficient test fails, the pipeline works but the DGP is wrong.
    """
    rng = np.random.default_rng(seed=seed)
    org = _build_synthetic_org(n_per_level=(1, 3, 10, 50), rng=rng)
    df = org.copy()
    df["job_level"] = rng.integers(1, 6, size=len(df)).astype(int)
    df["tenure_yrs"] = np.clip(rng.gamma(2.0, 2.0, size=len(df)), 0, 20)
    df["salary"] = 50_000 + 20_000 * df["job_level"] + 1_000 * df["tenure_yrs"] + rng.normal(0, 5_000, len(df))
    df["engagement"] = np.clip(rng.normal(7.0, 1.0, size=len(df)), 1, 10)
    df["attrition"] = (rng.random(len(df)) < 0.15).astype(int)

    ona = pona.DuckONA()
    ona.load_hris(df)
    edges = ona.build_org_edges("employee_id", "supervisor_id")

    # Validate all the expected graph metrics work
    for metric_fn, args in [
        (pona.betweenness, (edges, "employee_id", "supervisor_id")),
        (pona.pagerank, (edges, "employee_id", "supervisor_id")),
        (pona.degree_centrality, (edges, "employee_id", "supervisor_id")),
        (pona.eigenvector_centrality, (edges, "employee_id", "supervisor_id")),
        (pona.connected_components, (edges, "employee_id", "supervisor_id")),
        (pona.louvain_communities, (edges, "employee_id", "supervisor_id")),
    ]:
        rel = metric_fn(*args)
        n_rows = rel.count("*").fetchone()[0]
        assert n_rows > 0, f"{metric_fn.__name__} returned 0 rows; pipeline broken"

    # Model fits
    metrics = ona.betweenness(edges, "employee_id", "supervisor_id")
    enriched = ona.join_hris(metrics)
    tidy, glance = ona.ols(enriched, "salary ~ job_level + tenure_yrs")
    assert "term" in tidy.columns
    assert len(tidy) >= 2

    tidy2, glance2 = ona.logistic(enriched, "attrition ~ tenure_yrs")
    assert "term" in tidy2.columns
    assert len(tidy2) >= 2
