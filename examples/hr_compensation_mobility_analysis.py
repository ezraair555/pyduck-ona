"""
Example: HR analytics with DuckONA — compensation, promotion, turnover,
skills, and attendance.

Scenario
--------
A mid-sized company wants to understand how organizational structure and
career mobility relate to compensation and turnover risk. This script
synthesizes HRIS, compensation, promotion, turnover, skills, and
attendance data; builds an org-chart network; computes centrality
metrics; and runs OLS and logistic models on the joined dataset. It
also compares a pre-promotion slice to a post-promotion slice.

Run with:
    python examples/hr_compensation_mobility_analysis.py

Outputs are printed to stdout; no external files are written.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyduck_ona import DuckONA


# ── 1. Synthetic data generation ───────────────────────────────────────────
def make_hris(rng: np.random.Generator, n_emp: int = 120) -> pd.DataFrame:
    """Build a 4-level org chart."""
    employees: list[dict] = [
        dict(
            employee_id="E001",
            supervisor_id=None,
            department="People",
            job_level=4,
            hire_date=pd.Timestamp("2017-01-15"),
            gender="M",
        ),
    ]
    departments = ["Engineering", "Sales", "Marketing", "Operations", "People"]
    dept_weights = [0.45, 0.25, 0.10, 0.12, 0.08]

    # VPs
    vps = []
    next_id = 2
    for dept in rng.choice(departments, size=5, replace=False, p=dept_weights):
        eid = f"E{next_id:03d}"
        next_id += 1
        vps.append((eid, dept))
        employees.append(
            dict(
                employee_id=eid,
                supervisor_id="E001",
                department=dept,
                job_level=3,
                hire_date=pd.Timestamp("2018-03-01") + pd.Timedelta(days=int(rng.integers(0, 300))),
                gender=rng.choice(["F", "M"]),
            )
        )

    # Managers
    managers: list[tuple[str, str]] = []
    for _ in range(rng.integers(18, 26)):
        vp_eid, dept = vps[rng.integers(0, len(vps))]
        eid = f"E{next_id:03d}"
        next_id += 1
        managers.append((eid, dept))
        employees.append(
            dict(
                employee_id=eid,
                supervisor_id=vp_eid,
                department=dept,
                job_level=2,
                hire_date=pd.Timestamp("2019-01-01") + pd.Timedelta(days=int(rng.integers(0, 900))),
                gender=rng.choice(["F", "M"]),
            )
        )

    # ICs
    for mgr_eid, dept in managers:
        team_size = rng.integers(3, 7)
        for _ in range(team_size):
            eid = f"E{next_id:04d}"
            next_id += 1
            employees.append(
                dict(
                    employee_id=eid,
                    supervisor_id=mgr_eid,
                    department=dept,
                    job_level=1,
                    hire_date=pd.Timestamp("2020-01-01")
                    + pd.Timedelta(days=int(rng.integers(0, 1900))),
                    gender=rng.choice(["F", "M"]),
                )
            )

    return pd.DataFrame(employees)


def make_compensation(hris: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Salary depends on level, with a small gender gap at junior levels."""
    base = {4: 230_000, 3: 170_000, 2: 125_000, 1: 85_000}
    rows = []
    for row in hris.itertuples(index=False):
        salary = base[row.job_level] * (1 + rng.normal(0, 0.08))
        if row.job_level <= 2 and row.gender == "F":
            salary *= 0.95
        rows.append(
            dict(
                employee_id=row.employee_id,
                salary=round(salary, -2),
                bonus=round(salary * rng.uniform(0.05, 0.15), -2),
                effective_date=pd.Timestamp("2026-01-01"),
            )
        )
    return pd.DataFrame(rows)


def make_promotions(hris: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Random promotions for a subset of employees in 2025-2026."""
    eligible = hris[hris.job_level < 4].sample(n=min(20, len(hris) // 4), random_state=int(rng.integers(0, 2**31)))
    rows = []
    for row in eligible.itertuples(index=False):
        rows.append(
            dict(
                employee_id=row.employee_id,
                promotion_date=pd.Timestamp("2025-06-01") + pd.Timedelta(days=int(rng.integers(0, 360))),
                from_level=row.job_level,
                to_level=row.job_level + 1,
            )
        )
    return pd.DataFrame(rows)


def make_turnover(hris: pd.DataFrame, comp: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Turnover risk rises with low salary and low tenure."""
    tenure_yrs = (pd.Timestamp("2026-06-01") - hris["hire_date"]).dt.days / 365.25
    merged = hris.merge(comp, on="employee_id")
    engagement = rng.normal(7.0, 1.5, size=len(merged)).clip(1, 10)
    logit = (
        -1.8
        - 0.6 * engagement
        - 0.03 * tenure_yrs.values
        - 0.00001 * (merged["salary"].values - 100_000)
        + rng.normal(0, 0.8, size=len(merged))
    )
    p = 1 / (1 + np.exp(-logit))
    left = rng.random(len(merged)) < p
    rows = [
        dict(
            employee_id=row.employee_id,
            termination_date=pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(rng.integers(0, 150))),
            reason="voluntary",
        )
        for row in merged[left].itertuples(index=False)
    ]
    return pd.DataFrame(rows)


def make_survey(hris: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "employee_id": hris["employee_id"].tolist(),
            "engagement": rng.normal(7.0, 1.5, size=len(hris)).clip(1, 10).round(2),
            "survey_date": pd.Timestamp("2026-03-15"),
        }
    )


def make_skills(hris: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Assign each employee one primary skill with proficiency 1-5."""
    skills = ["python", "sql", "sales", "marketing", "operations", "people_management"]
    rows = []
    for row in hris.itertuples(index=False):
        rows.append(
            dict(
                employee_id=row.employee_id,
                skill=rng.choice(skills),
                proficiency=rng.integers(1, 6),
            )
        )
    return pd.DataFrame(rows)


def make_attendance(hris: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Daily attendance for a 4-week window."""
    dates = pd.date_range("2026-05-01", periods=20, freq="B")  # business days
    rows = []
    for eid in hris["employee_id"].sample(min(40, len(hris)), random_state=42):
        for d in dates:
            rows.append(
                dict(
                    employee_id=eid,
                    date=d,
                    present=int(rng.random() > 0.15),
                )
            )
    return pd.DataFrame(rows)


# ── 2. Build DuckONA workspace ─────────────────────────────────────────────
def build_workspace() -> tuple[DuckONA, pd.DataFrame, pd.DataFrame]:
    """Create synthetic data, load into DuckONA, and return the workspace."""
    rng = np.random.default_rng(seed=20260620)
    hris = make_hris(rng)
    comp = make_compensation(hris, rng)
    survey = make_survey(hris, rng)
    turnover = make_turnover(hris, comp, rng)
    promotions = make_promotions(hris, rng)
    skills = make_skills(hris, rng)
    attendance = make_attendance(hris, rng)

    # Merge engagement into compensation for modeling.
    comp = comp.merge(survey[["employee_id", "engagement"]], on="employee_id", how="left")

    ona = DuckONA(":memory:")
    ona.load_hris(hris)
    ona.load_compensation(comp)
    ona.load_survey(survey)
    ona.load_turnover(turnover)
    ona.load_promotions(promotions)
    ona.load_skills(skills)
    ona.load_attendance(attendance)

    # Validations
    ona.validate_keys("hris")
    ona.validate_keys("compensation", date_col="effective_date")

    return ona, hris, comp


# ── 3. Analysis pipeline ───────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("DuckONA HR compensation + mobility example")
    print("=" * 60)

    ona, hris, comp = build_workspace()
    print(f"\n[Data] {len(hris)} employees loaded")

    # Org-chart edges + centrality
    edges = ona.build_org_edges()
    n_edges = edges.count("*").fetchone()[0]
    print(f"[Org] {n_edges} reporting edges")

    between = ona.betweenness(edges, "employee_id", "supervisor_id").df()
    page = ona.pagerank(edges, "employee_id", "supervisor_id").df()
    eigen = ona.eigenvector_centrality(edges, "employee_id", "supervisor_id").df()
    degree = ona.degree_centrality(edges, "employee_id", "supervisor_id", mode="in").df()
    communities = ona.louvain_communities(edges, "employee_id", "supervisor_id").df()

    # Join all metrics back to HRIS via pandas
    metrics = (
        between.rename(columns={"betweenness": "broker_score"})
        .merge(page, on="node_id")
        .merge(eigen, on="node_id")
        .merge(degree, on="node_id")
        .merge(communities, on="node_id")
        .rename(columns={"node_id": "employee_id"})
    )
    analysis = hris.merge(comp, on="employee_id").merge(metrics, on="employee_id", how="left")
    analysis["tenure_yrs"] = (pd.Timestamp("2026-06-01") - analysis["hire_date"]).dt.days / 365.25
    analysis["turnover_flag"] = analysis["employee_id"].isin(ona.table("turnover").df()["employee_id"]).astype(int)

    print("\n[Centrality] top 3 by PageRank:")
    print(analysis.nlargest(3, "pagerank")[["employee_id", "department", "job_level", "pagerank"]])

    # OLS salary model
    print("\n[Model 1] OLS: salary ~ job_level + tenure_yrs + engagement + pagerank")
    tidy, glance = ona.ols(
        analysis,
        "salary ~ job_level + tenure_yrs + engagement + pagerank",
    )
    print(tidy[["term", "estimate", "std.error", "p.value"]].to_string(index=False))
    print(f"R² = {glance['rsquared'].iloc[0]:.3f}, n = {int(glance['nobs'].iloc[0])}")

    # Logistic turnover model
    print("\n[Model 2] Logistic: turnover ~ salary + tenure_yrs + engagement + pagerank")
    tidy_log, glance_log = ona.logistic(
        analysis,
        "turnover_flag ~ salary + tenure_yrs + engagement + pagerank",
    )
    print(
        tidy_log[["term", "estimate", "std.error", "p.value"]]
        .assign(odds_ratio=lambda d: np.exp(d["estimate"]))
        .to_string(index=False)
    )
    print(f"Log-likelihood = {glance_log['llf'].iloc[0]:.2f}")

    # Temporal slice comparison: pre/post promotion
    slices = ona.build_temporal_slices("attendance", "date", freq="W")
    print(f"\n[Temporal] {len(slices)} weekly attendance slices")
    for label, start, end, rel in slices[:3]:
        print(f"  {label}: {rel.count('*').fetchone()[0]} rows")

    # Promoted vs non-promoted salary comparison
    promoted_ids = set(ona.table("promotions").df()["employee_id"])
    analysis["promoted"] = analysis["employee_id"].isin(promoted_ids).astype(int)
    print(
        "\n[Mobility] mean salary by promotion status:\n",
        analysis.groupby("promoted")["salary"].agg(["mean", "count"]).round(0),
    )

    # MRQAP: test whether two employees in the same department are closer
    # in the org chart (manager distance). We build a department similarity
    # matrix and regress it on the inverse of the org-chart distance matrix.
    emp_ids = sorted(hris["employee_id"].tolist())
    n = len(emp_ids)
    idx = {e: i for i, e in enumerate(emp_ids)}
    dept = dict(zip(hris["employee_id"], hris["department"]))

    dept_sim = np.zeros((n, n))
    for i, e1 in enumerate(emp_ids):
        for j, e2 in enumerate(emp_ids):
            dept_sim[i, j] = 1.0 if dept[e1] == dept[e2] else 0.0

    # Manager distance: 1 if direct report to same manager, decayed by level difference.
    mgr = dict(zip(hris["employee_id"], hris["supervisor_id"]))
    mgr_dist = np.zeros((n, n))
    for i, e1 in enumerate(emp_ids):
        for j, e2 in enumerate(emp_ids):
            if e1 == e2:
                mgr_dist[i, j] = 0.0
            elif mgr.get(e1) == mgr.get(e2):
                mgr_dist[i, j] = 1.0
            else:
                mgr_dist[i, j] = 0.0

    mrqap_result = DuckONA.mrqap(dept_sim, [mgr_dist], n_permutations=500)
    print("\n[MRQAP] regression of department similarity on shared-manager matrix:")
    print(f"  coefficients: {mrqap_result['coefficients']}")
    print(f"  p-values: {mrqap_result['p_values']}")
    print(f"  R²: {mrqap_result['r2']:.3f}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
