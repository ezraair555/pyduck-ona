# End-to-End HR Analytics with pyduck-ona

A complete, runnable example that takes a synthetic HR dataset (196
employees across 5 departments) and walks through the five questions a
People Analytics team is typically asked:

1. **Is the org chart structurally sound?** → `hierarchy_valid`
2. **Where are the bottleneck managers?** → `hierarchy_stats` + ONA centrality
3. **What predicts attrition?** → logistic regression on engagement / tenure / pay
4. **Are we paying equitably across gender?** → OLS salary model + residual audit
5. **Are departments gender-balanced?** → chi-square test of independence

The full script lives at
[`examples/hr_attrition_analysis.py`](../examples/hr_attrition_analysis.py).
This page documents the design, the output, and the gotchas the example
exercises.

---

## Run it

```bash
cd projects/pyduck-ona
pip install -e .[broom,viz]
python examples/hr_attrition_analysis.py
```

Outputs land in `examples/hr_outputs/`:

| File | What it is |
|---|---|
| `hierarchy_issues.csv` | Rows from `hierarchy_valid` (loops, broken chains, multiple roots) |
| `employees_enriched.csv` | Master relation with `direct_reports`, `total_reports`, `betweenness`, `pagerank` joined on |
| `attrition_forest.png` | Forest plot of attrition model odds ratios (reference line = 1.0) |
| `dept_by_gender.png` | Chi-square heatmap: department × gender observed vs. expected counts |

---

## What the script does

### Stage 1 — Build and load the HR dataset

Three pandas DataFrames (`org`, `comp`, `survey`) are joined into a
single DuckDB table `hr_employees`. Each row is one employee with
columns for org position, compensation, gender, engagement, tenure,
and an `attrition` flag.

```python
duckdb.register("org",    org_df)
duckdb.register("comp",   comp_df)
duckdb.register("survey", survey_df)
duckdb.sql("""
    CREATE OR REPLACE TABLE hr_employees AS
    SELECT o.emp_id AS employee_id, o.supervisor_id, o.department,
           o.job_level, o.hire_date, c.gender, c.salary,
           s.engagement, s.attrition, s.tenure_yrs
    FROM org o JOIN comp c USING (emp_id) JOIN survey s USING (emp_id)
""")
```

The example **deliberately injects structural issues** so the
diagnostic stage has something to find:

- 3 employees whose `supervisor_id` references a non-existent
  employee (broken chains).
- A 6% female pay gap at levels 1–2 (so the pay-equity audit flags it).

### Stage 2 — Diagnose the hierarchy

```python
issues = pona.hierarchy_valid(rel, "employee_id", "supervisor_id").df()
```

`hierarchy_valid` runs a single recursive CTE that checks for:

| Issue | How it's detected |
|---|---|
| **Self-reference** | `employee_id = supervisor_id` on the same row |
| **Broken chain** | `supervisor_id` not in the `employee_id` set |
| **Multiple roots** | More than one row with `supervisor_id IS NULL` |
| **Loop** | Recursive walk exceeds the row count (cycle) |

Sample output:

```
  issue_type     employee_id                                              detail
broken_chain       E0133    Supervisor ID E9999 does not appear as any employee
broken_chain       E0217    Supervisor ID E9999 does not appear as any employee
broken_chain       E0263    Supervisor ID E9999 does not appear as any employee
```

### Stage 3 — Span-of-control + ONA graph metrics

```python
stats    = pona.hierarchy_stats(rel, "employee_id", "supervisor_id").df()
direct   = duckdb.sql("SELECT employee_id, supervisor_id FROM hr_employees "
                      "WHERE supervisor_id IS NOT NULL")
brokers  = pona.betweenness(direct, "employee_id", "supervisor_id").df()
infl     = pona.pagerank  (direct, "employee_id", "supervisor_id").df()
```

`hierarchy_stats` returns one row per manager with `direct_reports`,
`indirect_reports`, `total_reports`, `team_size`, and `levels_below`.

The ONA centrality metrics are joined back onto the master table via
`manager_id` (from `hierarchy_stats`) and `node_id` (from
`betweenness` / `pagerank`). The result is exported as
`employees_enriched.csv`.

> **Why the DIRECT edge relation?** The long-format transitive closure
> from `hierarchy_long()` star-flattens the org graph — every CEO
> becomes adjacent to every IC, which makes `betweenness` and
> `pagerank` degenerate. The direct (one-row-per-report) relation
> preserves the actual hierarchy topology. This is documented in
> MEMORY.md and called out in the README.

### Stage 4 — Attrition risk model

```python
tidy, glance = pona.logistic(
    rel,
    "attrition ~ engagement + tenure_yrs + direct_reports + salary",
)
tidy = tidy.assign(odds_ratio=lambda d: np.exp(d["estimate"]))
fig, _ = pona.plot_coefficients(tidy, reference_line=1.0)
pona.save_figure(fig, "attrition_forest.png")
```

The output is a tidy DataFrame with one row per coefficient, plus a
glance DataFrame with model-level fit statistics. The forest plot uses
`reference_line=1.0` because we're plotting odds ratios (1.0 = no
effect).

Sample output:

```
          term  estimate   std.error  p.value  odds_ratio
    engagement -0.422718    0.223085 0.058109    0.655263
     Intercept  5.839087    3.539467 0.099003  343.465732
        salary -0.000063    0.000040 0.121935    0.999937
    tenure_yrs -0.145991    0.256968 0.569948    0.864166
direct_reports -4.331433 3786.349149 0.999087    0.013149

   Log-likelihood: -36.79, AIC: 83.6, n = 196
```

### Stage 5 — Salary model + pay-equity audit

```python
tidy, glance = pona.ols(
    rel,
    "salary ~ job_level + tenure_yrs + direct_reports + C(department)",
)
```

The OLS model explains ~82% of salary variance. Job level is
overwhelmingly the dominant predictor (~$36k per level). Tenure adds
~1.3k/yr. Department effects are not significant once level is in the
model.

For the **pay-equity audit**, we compute the residual (actual −
predicted) for each employee, then group by gender. A non-zero mean
gap is the dollar cost of the residual disparity:

```
          mean  count
gender
F      -4710.0     72
M       2735.0    124

Mean residual gap (F − M): -$7,445
```

This matches the 6% gap injected at generation time — about $7.5k on
the average level-1/2 salary.

Finally, the **chi-square test** confirms departments are
gender-balanced (p = 0.40, no rejection):

```
    chi2  p_value  dof
4.038473 0.400824    4
```

---

## Gotchas the example exercises

This script trips every foot-gun in the pyduck-ona / DuckDB / broom-sm
stack. Documenting them here so future contributors don't re-learn them:

| # | Gotcha | Where in the script | Fix |
|---|---|---|---|
| 1 | `duckdb.sql()` and `duckdb.connect()` create **separate in-memory connections** | Stage 1 | Register all source tables on the **default** connection so pyduck-ona's internal `duckdb.sql(...)` calls can see them. |
| 2 | pyduck-ona core functions reference the input relation as `df` in their generated SQL | Stage 2 | Pass the `DuckDBPyRelation` object (not its name) to pyduck-ona — DuckDB's replacement scan then binds the right one. |
| 3 | `hierarchy_stats` returns `manager_id`, not `employee_id` | Stage 3 | Join on `manager_id` when materializing centrality back onto the master table. |
| 4 | `betweenness` and `pagerank` return `node_id`, not `node` | Stage 3 | Same — join on `node_id`. |
| 5 | Long-format org relations flatten the graph into a star | Stage 3 | For ONA centrality, pass the **direct** edge relation (one row per manager → report). |
| 6 | broom-sm column names use **underscores** in `glance` (`rsquared_adj`, `f_statistic`, `llf`, `aic`) | Stage 4 & 5 | Use `glance['rsquared_adj']` not `glance['adj.r.squared']`; `glance['llf']` not `glance['pseudo.r.squared']`. |
| 7 | broom-sm `tidy` columns use **dots** (`p.value`, `conf.low`) | Stage 4 | Quote them in SQL: `WHERE "p.value" < 0.05`. |
| 8 | Logistic regression warns about `alpha` kwarg passing through to statsmodels | Stage 4 | Cosmetic; harmless and ignorable. |

---

## Adapting to real HR data

To run this against your own HRIS export:

1. Replace the three synthetic DataFrames in `stage1_load()` with
   `duckdb.read_csv(...)` or `duckdb.read_parquet(...)` calls. Keep
   the column names (`employee_id`, `supervisor_id`, `department`,
   `job_level`, `hire_date`, `gender`, `salary`, `engagement`,
   `attrition`, `tenure_yrs`).

2. If your hire dates are strings, cast them:
   ```python
   rel = rel.assign(hire_date=lambda d: d["hire_date"].astype("datetime64[ns]"))
   ```

3. If you have multiple sources of compensation (base, bonus, equity),
   compute `total_comp = base + bonus + equity / 4` and use that as
   `salary` for the regression.

4. For very large orgs (>100k employees), the
   `hierarchy_long()` recursive CTE may approach DuckDB's recursion
   limits. Increase `max_depth` only if your hierarchy actually goes
   that deep.

5. The pay-equity audit as written is a **starting point**, not a
   legal-grade analysis. For a real audit:
   - Add controls for performance rating, location, and years-in-role.
   - Use HC3 robust standard errors if your residual distribution is heavy-tailed.
   - Run the residual regression as a formal model (`residual ~ gender + controls`)
     rather than just reporting the mean gap.

---

## Related

- [`examples/full_workflow.py`](../examples/full_workflow.py) — minimal
  end-to-end demo on a 13-employee toy org (no statistics, no pay audit).
- [README](../README.md) — package overview and API reference.
