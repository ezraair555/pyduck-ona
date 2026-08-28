# When to Reach for pyduck-ona vs. statsmodels

A one-page decision guide for the People Analytics team.

## TL;DR

**statsmodels** fits the model. **pyduck-ona** gets you to the model faster when
your data is HR-shaped (org-chart hierarchy, network relationships, snapshot
tables) and you want every step validated and re-joinable in DuckDB. If your
input is a clean CSV and your output is a coefficient table, statsmodels is
enough. If your input is an HRIS extract and your output is an audited model
that joins back to demographics for segmentation, pyduck-ona earns its keep.

## The 30-second decision tree

```
Does your analysis need ANY of these?
├── An org chart (employee → supervisor edges)
├── A network metric on employees (betweenness, PageRank, communities)
├── HR-domain validation (NULL keys, duplicate snapshots, future dates)
├── Per-cohort model fits over time (build_temporal_slices)
├── A square-matrix regression (MRQAP-style)
├── Model results stored back in DuckDB for downstream SQL
└── Plotting bundled with the model fit
        │
        ├── YES → pyduck-ona (and consider statsmodels underneath it
        │         for anything pyduck-ona doesn't cover)
        │
        └── NO  → statsmodels directly. pyduck-ona is overhead.
```

## What pyduck-ona adds (in plain language)

| You want to… | statsmodels | pyduck-ona |
|---|---|---|
| Diagnose an org chart for loops, broken chains, multiple roots | Write the recursion yourself | `hierarchy_valid(rel, "emp", "mgr")` |
| Compute span of control (direct + indirect + levels_below) | `groupby(...).size()` (loses transitive) | `hierarchy_stats(rel, ...)` |
| Compute betweenness / PageRank on the org | Build NetworkX, hand-merge to df | `ona.betweenness(edges, ...)` → relation |
| Join network metric back to demographics | `metrics.merge(hris, ...)` (off-by-one risk) | `ona.join_hris(metrics)` |
| Fit logistic with tidy output | `sm.logit` + `broom.sm.tidy` | `ona.logistic(rel, formula)` → (tidy, glance) |
| Audit pay equity by group | Statsmodels + manual residual split + seaborn | OLS residual audit + χ² heatmap, one call each |
| Recover coefficients on known fake data before trusting | Build your own harness | Use the simulation tests as a template |
| Test if collaboration similarity tracks dept similarity | **Impossible** in statsmodels | `DuckONA.mrqap(Y, X_matrices)` |

## Three-tier adoption ladder

### Tier 1 — Use today (low risk, clear win)

These have the lowest cost of adoption and the highest defensive value:

1. **`hierarchy_valid`** on every new HRIS extract. Catches data-quality issues
   before they pollute a model.
2. **`hierarchy_stats`** to find bottleneck managers (large indirect reports
   with shallow depth).
3. **`chi_square(rel, x, y)`** for categorical-independence audits
   (dept × gender, dept × race, level × tenure cohort). Returns a
   publication-ready heatmap.
4. **`DuckONA.ols` / `DuckONA.logistic`** with bundled plots when you need
   a board-deck-ready output, not a Jupyter cell.

### Tier 2 — Use for specific workstreams (medium risk, high value)

1. **Pay-equity audit.** Pattern locked in `examples/hr_attrition_analysis.py`
   Stage 5. Fit OLS `salary ~ job_level + tenure + gender`, split residuals by
   group, run χ² on the protected-attribute distribution. ~30 lines of code.
2. **ONA-augmented attrition model.** Add `betweenness`, `pagerank`,
   `eigenvector` to your existing logistic regression. `ona.join_hris(metrics)`
   does the glue.
3. **Retirement-eligibility / tenure analysis.** Use `build_temporal_slices`
   for per-month or per-quarter slices when you need period-specific models.

### Tier 3 — Hold until you have the data

1. **MRQAP (`DuckONA.mrqap`).** Powerful, but only useful when you have a
   pre-built square matrix (collaboration, communication, skills overlap).
   If MassMutual doesn't have that data, skip it.
2. **Survival analysis (time-to-attrition with censoring).** Not in pyduck-ona.
   Use `lifelines` or statsmodels `PHReg` directly.
3. **Mixed-effects / hierarchical models.** Not in pyduck-ona. Use
   `statsmodels.regression.mixed_linear_model.MixedLM`.
4. **Causal inference (DiD, synthetic control, IV).** Not in pyduck-ona.
   Use `dowhy` or `causalml`.

## Don't use pyduck-ona for

- Real-time / streaming data (DuckDB is batch, not stream)
- Graphs >10⁵ nodes (NetworkX backend can't handle it; wait for DuckPGQ)
- Anything that needs interaction data (Slack, email) — DuckONA explicitly
  does not ingest these
- Anything outside HR-analytics scope — the package is HR-domain-tuned

## The validation surface (the unsung win)

`DuckONA.load_*` + `DuckONA.validate_keys` + `DuckONA.filter_noise` is the
reason to reach for pyduck-ona even on a Tier-1 task. statsmodels will
silently fit a model on a DataFrame with 12 duplicate employee_ids and 30
NULL compensation rows. pyduck-ona refuses the load. For People Analytics
feeding a board deck, that refusal is the difference between a defensible
audit trail and a re-aggregation 3 weeks later.

## Five-command cheat sheet

```python
import pyduck_ona as pona
from pyduck_ona import DuckONA

# 1. Load (with validation)
ona = DuckONA()
ona.load_hris(hris_df)
ona.validate_keys("hris", "employee_id", date_col="snapshot_date")

# 2. Diagnose hierarchy
issues = pona.hierarchy_valid(ona.con.sql("SELECT * FROM hris"),
                             "employee_id", "supervisor_id").df()

# 3. Compute ONA metrics
edges = ona.build_org_edges("employee_id", "supervisor_id")
metrics = ona.betweenness(edges, "employee_id", "supervisor_id")

# 4. Join back to demographics, fit model
enriched = ona.join_hris(metrics)
tidy, glance = ona.logistic(enriched, "attrition ~ betweenness + tenure_yrs")

# 5. Audit a categorical relationship
table, fig = pona.chi_square(enriched, "department", "gender")
```

## When in doubt

Open `examples/hr_attrition_analysis.py`. It's a complete, runnable pipeline
on 196 synthetic employees that demonstrates the five patterns above. Copy
it, swap in your HRIS extract, run it. If you get stuck, the simulation tests
in `tests/integration/test_simulation.py` show what "good" output looks like
for a known data-generating process — useful as a sanity check on any new
workstream.

---

*Last reviewed: 2026-08-27. Companion to `pyduck-ona` v0.1.5.*
