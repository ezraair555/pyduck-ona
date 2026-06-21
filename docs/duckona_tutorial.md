# DuckONA Tutorial

`DuckONA` is the high-level class for end-to-end HR analytics with `pyduck-ona`. It owns a DuckDB connection, ingests HR tables, computes org-chart network metrics, and runs statistical models — all without leaving Python.

This tutorial walks through a synthetic but realistic workflow:

1. Load HRIS, compensation, turnover, promotions, skills, and attendance.
2. Validate keys and filter noise.
3. Build org-chart edges and compute centrality.
4. Join metrics back to HRIS and run models.
5. Slice data temporally for before/after comparisons.
6. Run a pure-Python MRQAP test on a similarity matrix.

---

## Setup

```python
import pandas as pd
import pyduck_ona as pona

# In-memory database is fine for analysis; use a file path to persist.
ona = pona.DuckONA(":memory:")
```

## Load HR tables

Each loader accepts a pandas DataFrame and registers it as a DuckDB table inside the `DuckONA` connection. The minimum required table is `hris`, which must contain `employee_id` and `supervisor_id`.

```python
hris = pd.DataFrame({
    "employee_id": ["E01", "E02", "E03", "E04", "E05"],
    "supervisor_id": [None, "E01", "E01", "E02", "E02"],
    "department": ["Exec", "Eng", "Eng", "Sales", "Sales"],
    "job_level": [5, 4, 3, 3, 2],
    "hire_date": pd.to_datetime(["2020-01-15", "2021-03-10", "2022-06-01",
                                  "2021-08-20", "2023-02-28"]),
})

comp = pd.DataFrame({
    "employee_id": ["E01", "E02", "E03", "E04", "E05"],
    "salary": [250000, 180000, 140000, 135000, 110000],
    "bonus": [50000, 30000, 20000, 20000, 15000],
    "effective_date": pd.to_datetime(["2026-01-01"] * 5),
})

turnover = pd.DataFrame({
    "employee_id": ["E05"],
    "termination_date": pd.to_datetime(["2026-04-01"]),
    "reason": ["voluntary"],
})

ona.load_hris(hris)
ona.load_compensation(comp)
ona.load_turnover(turnover)
```

Other loaders follow the same pattern:

```python
ona.load_survey(survey_df)       # columns: employee_id, survey_date, engagement, ...
ona.load_promotions(promos_df)   # columns: employee_id, promotion_date, from_level, to_level
ona.load_skills(skills_df)      # columns: employee_id, skill_name, proficiency
ona.load_attendance(attend_df)  # columns: employee_id, date, office, wfh, ...
ona.load_retirement(retire_df)   # columns: employee_id, retirement_date, expected_retirement
```

## Validation and noise filtering

`DuckONA` validates keys when you call `validate_keys`. It rejects NULL keys, duplicate `employee_id` rows, future dates, and out-of-range dates.

```python
ona.validate_keys("hris", key_col="employee_id", date_col="hire_date")
```

To drop exact duplicates:

```python
ona.deduplicate("hris", subset=["employee_id"])
```

## Org-chart edges and centrality

Build the direct reporting graph from HRIS and run any graph metric:

```python
edges = ona.build_org_edges()

between = ona.betweenness(edges, "employee_id", "supervisor_id",
                           node_id_col="employee_id")
page = ona.pagerank(edges, "employee_id", "supervisor_id",
                     node_id_col="employee_id")
eigen = ona.eigenvector_centrality(edges, "employee_id", "supervisor_id",
                                     node_id_col="employee_id")
degree = ona.degree_centrality(edges, "employee_id", "supervisor_id",
                                mode="in", node_id_col="employee_id")
communities = ona.louvain_communities(edges, "employee_id", "supervisor_id",
                                       node_id_col="employee_id")
```

The `node_id_col` parameter renames the output column from the default `node_id` to `employee_id`, which makes joining back to HRIS easier.

## Join metrics and run models

```python
metrics = (
    between.df().rename(columns={"betweenness": "broker_score"})
    .merge(page.df(), on="employee_id")
    .merge(eigen.df(), on="employee_id")
    .merge(degree.df(), on="employee_id")
    .merge(communities.df(), on="employee_id")
)

analysis = hris.merge(comp, on="employee_id").merge(metrics, on="employee_id", how="left")
analysis["tenure_yrs"] = (pd.Timestamp("2026-06-01") - analysis["hire_date"]).dt.days / 365.25
analysis["turnover_flag"] = analysis["employee_id"].isin(turnover["employee_id"]).astype(int)

# OLS salary model
tidy, glance = ona.ols(
    analysis,
    "salary ~ job_level + tenure_yrs + pagerank",
)
print(tidy[["term", "estimate", "p.value"]])

# Logistic turnover model
tidy_log, glance_log = ona.logistic(
    analysis,
    "turnover_flag ~ salary + tenure_yrs + pagerank",
)
tidy_log["odds_ratio"] = np.exp(tidy_log["estimate"])
print(tidy_log[["term", "estimate", "odds_ratio", "p.value"]])
```

## Temporal slicing

Compare network or attendance patterns across time windows:

```python
slices = ona.build_temporal_slices("attendance", "date", freq="M")
for label, start, end, relation in slices:
    count = relation.count("*").fetchone()[0]
    print(f"{label}: {count} rows")
```

`freq` accepts any pandas date-offset string: `"W"`, `"M"`, `"Q"`, `"Y"`.

## MRQAP: matrix regression under network dependence

Standard regression assumes independent rows, which is false in networks. MRQAP uses permutation to compute correct p-values for matrix relationships.

Example: is similarity in **department** related to similarity in **shared manager**?

```python
import numpy as np

employees = hris["employee_id"].tolist()
n = len(employees)

# Dependent matrix: 1 if two employees share a direct manager
manager_of = hris.set_index("employee_id")["supervisor_id"].to_dict()
Y = np.array([[1 if manager_of[a] == manager_of[b] and pd.notna(manager_of[a]) else 0
               for b in employees] for a in employees], dtype=float)

# Predictor matrix: 1 if same department
dept_of = hris.set_index("employee_id")["department"].to_dict()
X_dept = np.array([[1 if dept_of[a] == dept_of[b] else 0
                    for b in employees] for a in employees], dtype=float)

result = ona.mrqap(Y, [X_dept], n_permutations=1000)
print(result["p_values"])  # empirical p-values for each predictor
```

## Where ERGMs fit

Exponential Random Graph Models (ERGMs) model the network itself as the outcome — for example, testing whether employees are more likely to collaborate when they share a department, controlling for reciprocity. There is no production-grade Python ERGM library today; the gold-standard tooling is R's `statnet`/`ergm`. `pyduck-ona` stays Python-only, so ERGM is intentionally deferred until a mature Python implementation exists or until you choose to bridge to R separately.

## API reference

Key `DuckONA` methods:

| Method | Purpose |
|---|---|
| `load_hris` / `load_compensation` / `load_turnover` / `load_survey` / `load_promotions` / `load_skills` / `load_attendance` / `load_retirement` | Register HR tables |
| `validate_keys` | Fail loudly on NULL/duplicate/future-date keys |
| `filter_noise` / `deduplicate` | Clean input tables |
| `build_org_edges` | Direct reporting edge relation from HRIS |
| `betweenness` / `pagerank` / `eigenvector_centrality` / `degree_centrality` / `connected_components` / `louvain_communities` | Network metrics |
| `join_hris` | Join a metric relation back to HRIS |
| `ols` / `logistic` / `anova` / `chi_square` / `correlation` / `vif` / `model_compare` | Statistical models via `broom-sm` |
| `build_temporal_slices` | Time-windowed edge/table relations |
| `mrqap` | Pure-Python MRQAP helper |
| `sql` / `table` | Escape hatches to the underlying DuckDB connection |

See [`examples/hr_compensation_mobility_analysis.py`](../examples/hr_compensation_mobility_analysis.py) for the complete runnable version.
