"""
Example: End-to-end HR analytics with pyduck-ona.

Scenario
--------
A mid-sized tech company (~250 employees) is concerned about attrition
risk. The CHRO has three questions:

    1. Is our hierarchy structurally sound, and where are the bottlenecks?
    2. What factors predict attrition? (compensation, tenure, span of control)
    3. Are there gender pay gaps by department?

This script answers all three using one DuckDB relation and the
pyduck-ona + broom-sm stack. It is structured as five stages:

    Stage 1  — Load HR data (one relation, four views)
    Stage 2  — Diagnose hierarchy integrity
    Stage 3  — Span-of-control + ONA graph metrics
    Stage 4  — Attrition risk model (logistic regression)
    Stage 5  — Salary model + pay-equity audit

Run with:
    python examples/hr_attrition_analysis.py

Outputs are written to ./hr_outputs/ as CSVs and PNGs.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import pyduck_ona as pona


# ── Paths ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
OUT = HERE / "hr_outputs"
OUT.mkdir(exist_ok=True)


# ── Stage 1: Load HR data ──────────────────────────────────────────────────
def stage1_load() -> tuple[duckdb.DuckDBPyRelation, duckdb.DuckDBPyConnection]:
    """Build a realistic HR dataset and load it into DuckDB.

    Returns the joined HR relation AND the underlying connection. Both
    are needed downstream: pyduck-ona functions that call ``duckdb.sql``
    internally create a fresh in-memory connection, so any table or
    relation we want to share across stages must be materialized on the
    connection we hand around.

    See MEMORY.md → "DuckDB Connection Isolation".
    """
    rng = np.random.default_rng(seed=20260620)

    # ── Org structure: 250 employees across 5 departments, 4 levels ──
    n_emp = 250
    departments = ["Engineering", "Sales", "Marketing", "Operations", "People"]
    dept_weights = [0.45, 0.25, 0.10, 0.12, 0.08]

    # Level 0: CEO
    employees: list[dict] = [
        dict(emp_id="E001", supervisor_id=None, department="People",
             job_level=4, hire_date=pd.Timestamp("2018-01-15")),
    ]
    # Level 1: 5 VPs (one per department)
    vps = []
    for i, dept in enumerate(departments, start=1):
        eid = f"E{1+i:03d}"
        vps.append(eid)
        employees.append(dict(
            emp_id=eid, supervisor_id="E001",
            department=dept, job_level=3,
            hire_date=pd.Timestamp("2018-03-01") + pd.Timedelta(days=i * 30),
        ))
    # Level 2: directors/managers (~30)
    manager_ids = []
    next_id = 100
    for vp in vps:
        for _ in range(rng.integers(4, 8)):
            mid = f"E{next_id:04d}"
            next_id += 1
            manager_ids.append(mid)
            employees.append(dict(
                emp_id=mid, supervisor_id=vp,
                department=next((d for d, v in zip(departments, vps) if v == vp), "Operations"),
                job_level=2,
                hire_date=pd.Timestamp("2019-01-01") + pd.Timedelta(days=int(rng.integers(0, 1500))),
            ))
    # Level 3: individual contributors
    for mgr in manager_ids:
        mgr_dept = next((e["department"] for e in employees if e["emp_id"] == mgr), "Operations")
        for _ in range(rng.integers(4, 10)):
            eid = f"E{next_id:04d}"
            next_id += 1
            # Inject a structural issue: 2 broken-chain employees (bad supervisor ID)
            sup = mgr if rng.random() > 0.02 else "E9999"
            employees.append(dict(
                emp_id=eid, supervisor_id=sup,
                department=mgr_dept, job_level=1,
                hire_date=pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(rng.integers(0, 1800))),
            ))

    org_df = pd.DataFrame(employees)

    # ── Compensation table ─────────────────────────────────────────────
    base_salary = {4: 220_000, 3: 165_000, 2: 125_000, 1: 85_000}
    gender = rng.choice(["F", "M"], size=len(org_df), p=[0.42, 0.58])
    # Inject a 6% gender pay gap at level 1-2 (for the pay-equity audit)
    salary = np.array([
        base_salary[row.job_level]
        * (1 + rng.normal(0, 0.10))
        * (0.94 if (row.job_level <= 2 and g == "F") else 1.0)
        for row, g in zip(org_df.itertuples(index=False), gender)
    ], dtype=float)
    comp_df = org_df[["emp_id"]].assign(
        gender=gender,
        salary=np.round(salary, -2),
    )

    # ── Engagement survey + attrition ──────────────────────────────────
    engagement = rng.normal(7.2, 1.4, size=len(org_df)).clip(1, 10).round(2)
    # Attrition depends on engagement (negatively), tenure (positively — newer = more flight risk),
    # and compensation (negatively — underpaid = more likely to leave).
    tenure_yrs = (pd.Timestamp("2026-06-01") - org_df["hire_date"]).dt.days / 365.25
    logit = (
        -1.6
        - 0.55 * (engagement - 5)        # engagement protects
        - 0.40 * (tenure_yrs - 2)         # newer employees leave more
        - 0.000008 * (salary - 100_000)   # higher pay protects
        + rng.normal(0, 0.7, size=len(org_df))
    )
    p_attrition = 1 / (1 + np.exp(-logit))
    attrition = (rng.random(len(org_df)) < p_attrition).astype(int)

    survey_df = org_df[["emp_id"]].assign(
        engagement=engagement,
        attrition=attrition,
        tenure_yrs=tenure_yrs.round(2),
    )

    # ── Load all four into DuckDB ─────────────────────────────────────
    # We register the source tables on the DEFAULT in-memory connection,
    # not a custom one. pyduck-ona's core functions issue ``duckdb.sql``
    # against the default connection internally and reference the input
    # relation by its Python object (via replacement scan). For that scan
    # to find our tables, they must be registered on the same connection.
    # See MEMORY.md → "DuckDB Connection Isolation".
    duckdb.sql("SET threads TO 4")
    duckdb.register("org", org_df)
    duckdb.register("comp", comp_df)
    duckdb.register("survey", survey_df)

    # Materialize the joined view as a table so ONA graph functions
    # (which also use duckdb.sql under the hood) can query it by name.
    duckdb.sql("""
        CREATE OR REPLACE TABLE hr_employees AS
        SELECT
            o.emp_id           AS employee_id,
            o.supervisor_id    AS supervisor_id,
            o.department,
            o.job_level,
            o.hire_date,
            c.gender,
            c.salary,
            s.engagement,
            s.attrition,
            s.tenure_yrs
        FROM org o
        JOIN comp   c USING (emp_id)
        JOIN survey s USING (emp_id)
    """)

    rel = duckdb.sql("SELECT * FROM hr_employees")

    n_emp = rel.count("*").fetchone()[0]
    print(f"[Stage 1] Loaded {n_emp} employees across {len(departments)} departments")
    print(f"           Attrition rate: {survey_df['attrition'].mean():.1%}")
    return rel


# ── Stage 2: Diagnose hierarchy ────────────────────────────────────────────
def stage2_diagnose(rel: duckdb.DuckDBPyRelation) -> None:
    """Detect loops, broken chains, multiple roots, self-references.

    Uses ``rel`` directly. Because pyduck-ona's core functions wrap
    ``duckdb.sql`` internally, we must pass the relation object (which
    carries a reference to its own connection) rather than its name.
    """
    print("\n[Stage 2] Hierarchy diagnostics")
    issues = pona.hierarchy_valid(rel, "employee_id", "supervisor_id").df()
    if len(issues) == 0:
        print("           ✓ Hierarchy is clean")
        return
    print(f"           ✗ {len(issues)} issue(s) found:")
    print(issues.to_string(index=False))
    issues.to_csv(OUT / "hierarchy_issues.csv", index=False)


# ── Stage 3: Span-of-control + ONA graph metrics ───────────────────────────
def stage3_structure(rel: duckdb.DuckDBPyRelation) -> duckdb.DuckDBPyRelation:
    """Compute span-of-control stats and ONA centrality, then materialize
    them onto the employee table for downstream modeling."""
    print("\n[Stage 3] Span-of-control + ONA graph metrics")

    stats = pona.hierarchy_stats(rel, "employee_id", "supervisor_id").df()
    print(f"           {len(stats)} managers; "
          f"median team size = {stats['direct_reports'].median():.1f}")

    # ONA: betweenness, pagerank — pass the DIRECT edge relation
    # (long-format transitive closure star-flattens the graph).
    direct_rel = duckdb.sql("""
        SELECT employee_id, supervisor_id
        FROM hr_employees
        WHERE supervisor_id IS NOT NULL
    """)
    brokers = pona.betweenness(direct_rel, "employee_id", "supervisor_id").df()
    influence = pona.pagerank(direct_rel, "employee_id", "supervisor_id").df()

    # Join centrality back onto the master relation. Register the
    # intermediate DataFrames on the default connection so the SQL
    # below (also on the default connection) can see them.
    duckdb.register("stats_df", stats)
    duckdb.register("brokers_df", brokers)
    duckdb.register("influence_df", influence)
    enriched = duckdb.sql("""
        SELECT
            r.*,
            COALESCE(s.direct_reports, 0) AS direct_reports,
            COALESCE(s.total_reports, 0)  AS total_reports,
            b.betweenness                 AS betweenness,
            p.pagerank                    AS pagerank
        FROM hr_employees r
        LEFT JOIN stats_df      s ON r.employee_id = s.manager_id
        LEFT JOIN brokers_df    b ON r.employee_id = b.node_id
        LEFT JOIN influence_df  p ON r.employee_id = p.node_id
    """)
    enriched.df().to_csv(OUT / "employees_enriched.csv", index=False)
    print(f"           Wrote {OUT / 'employees_enriched.csv'}")
    return enriched


# ── Stage 4: Attrition risk model ───────────────────────────────────────────
def stage4_attrition(rel: duckdb.DuckDBPyRelation) -> None:
    """Logistic regression: what predicts attrition?

    Hypothesis: engagement is the dominant protective factor; tenure and
    team size are risk amplifiers. We want the odds ratios.
    """
    print("\n[Stage 4] Attrition risk model (logistic regression)")

    tidy, glance = pona.logistic(
        rel,
        "attrition ~ engagement + tenure_yrs + direct_reports + salary",
    )

    tidy = tidy.assign(odds_ratio=lambda d: np.exp(d["estimate"]))
    print("\n           Coefficients (sorted by |z|):")
    print(tidy[["term", "estimate", "std.error", "p.value", "odds_ratio"]]
          .sort_values("p.value")
          .to_string(index=False))
    print(f"\n           Log-likelihood: {glance['llf'].iloc[0]:.2f}")
    print(f"           AIC: {glance['aic'].iloc[0]:.1f}, n = {int(glance['nobs'].iloc[0])}")

    # Forest plot of odds ratios (reference = 1.0)
    fig, _ = pona.plot_coefficients(tidy, reference_line=1.0)
    pona.save_figure(fig, str(OUT / "attrition_forest.png"))
    print(f"           Wrote {OUT / 'attrition_forest.png'}")


# ── Stage 5: Salary model + pay-equity audit ───────────────────────────────
def stage5_pay_equity(rel: duckdb.DuckDBPyRelation) -> None:
    """Salary model + chi-square audit of department × gender distribution.

    Two-part question:
        (a) What predicts salary? (job_level, department, tenure)
        (b) Is there a residual gender gap after controlling for (a)?
    """
    print("\n[Stage 5] Salary model + pay-equity audit")

    # (a) OLS salary model
    tidy, glance = pona.ols(
        rel,
        "salary ~ job_level + tenure_yrs + direct_reports + C(department)",
    )
    print("\n           Salary model (OLS):")
    print(tidy[["term", "estimate", "std.error", "p.value"]]
          .sort_values("p.value")
          .to_string(index=False))
    print(f"\n           Adj-R² = {glance['rsquared_adj'].iloc[0]:.3f}, "
          f"F = {glance['f_statistic'].iloc[0]:.1f}, "
          f"p = {glance['f_pvalue'].iloc[0]:.2e}")

    # (b) Audit: residual = actual − predicted. If women have lower residuals
    #     than men after controlling for legitimate factors, that's a flag.
    #     We compute predicted salary by applying the model coefficients to
    #     each row in pandas — avoiding cross-connection joins.
    df = rel.df()
    # Re-create the design matrix columns and apply coefficients.
    coef = dict(zip(tidy["term"], tidy["estimate"]))
    intercept = coef.get("Intercept", 0.0)
    dept_terms = {k.removeprefix("C(department)[T.").removesuffix("]"): v
                  for k, v in coef.items() if k.startswith("C(department)[T.")}
    df["pred_salary"] = (
        intercept
        + df["job_level"] * coef.get("job_level", 0.0)
        + df["tenure_yrs"] * coef.get("tenure_yrs", 0.0)
        + df["direct_reports"] * coef.get("direct_reports", 0.0)
        + df["department"].map(dept_terms).fillna(0.0)
    )
    df["residual"] = df["salary"] - df["pred_salary"]

    # Per-gender mean residual
    resid_by_gender = df.groupby("gender")["residual"].agg(["mean", "count"])
    print("\n           Mean residual by gender (actual − predicted):")
    print(resid_by_gender.round(0).to_string())
    gap = (df.loc[df.gender == "F", "residual"].mean()
           - df.loc[df.gender == "M", "residual"].mean())
    print(f"\n           Mean residual gap (F − M): ${gap:,.0f}")

    # (c) Chi-square: are departments gender-balanced?
    chi_tbl, chi_fig = pona.chi_square(rel, "department", "gender")
    print("\n           Chi-square: department × gender")
    print(chi_tbl.to_string(index=False))
    pona.save_figure(chi_fig, str(OUT / "dept_by_gender.png"))
    print(f"           Wrote {OUT / 'dept_by_gender.png'}")


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("pyduck-ona — End-to-end HR analytics example")
    print("=" * 60)

    rel = stage1_load()
    stage2_diagnose(rel)
    enriched = stage3_structure(rel)
    stage4_attrition(enriched)
    stage5_pay_equity(enriched)

    print("\n" + "=" * 60)
    print(f"Done. Outputs in: {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
