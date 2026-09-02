# pyduck-ona

**DuckDB-native People Analytics and Organizational Network Analysis.**

`pyduck-ona` brings HR analytics to DuckDB's vectorized engine. Instead of
slow Python loops over org-chart DataFrames, it runs recursive CTEs, property
graphs (DuckPGQ), and zero-copy Arrow transfers against DuckDB relations.

It is the `hR` R-package philosophy ported to DuckDB, designed to compose
with [`pyduck-janitor`](https://github.com/ezraair555/pyduck-janitor) for
method-chaining data-cleaning workflows.

## Examples

| Script | What it shows |
|---|---|
| `examples/full_workflow.py` | Minimal end-to-end ONA workflow on a 13-employee toy org (hierarchy diagnostics, span-of-control, betweenness, pagerank, shortest path). |
| `examples/hr_attrition_analysis.py` | Full **People Analytics** pipeline on 196 synthetic employees: org diagnostics + span-of-control + ONA centrality + **logistic attrition model** + **OLS salary model with pay-equity audit** + chi-square test of department × gender. Outputs CSVs and PNGs. See [`docs/hr_analysis_tutorial.md`](docs/hr_analysis_tutorial.md) for a walkthrough. |
| `examples/hr_compensation_mobility_analysis.py` | **DuckONA class** demo with synthetic HRIS: load compensation, turnover, promotions, skills, attendance; run org-chart centrality + OLS salary + logistic turnover + temporal attendance slices + MRQAP. |

## Why this exists

The R package [`hR`](https://github.com/eehh-stanford/hR) is the
gold-standard library for org-chart analytics in R. Python has had no
equivalent that:

1. Uses DuckDB's recursive CTEs (orders of magnitude faster than Python loops)
2. Plays nicely with `pyduck-janitor` chains
3. Bridges cleanly into NetworkX / igraph for inferential ONA
4. Integrates with [`broom`](https://github.com/ezraair555/broom-sm) for
   statistical-model workflows

`pyduck-ona` fills that gap.

## Install

```bash
pip install pyduck-ona              # core only
pip install pyduck-ona[viz]         # + matplotlib + pyvis for plotting
pip install pyduck-ona[graph]       # (placeholder; DuckPGQ not currently available)
pip install pyduck-ona[broom]       # + broom-sm for statistical modeling
pip install pyduck-ona[dev]         # + testing + linting
```

## Quick start

```python
import duckdb
import pyduck_ona as pona

# Load your HR data (any DuckDB-loadable format)
rel = duckdb.read_csv("hr_data.csv")

# Diagnose the hierarchy
issues = pona.hierarchy_valid(rel, "employee_id", "supervisor_id")
print(issues.df())

# Long format: every (employee, supervisor) ancestor pair
long = pona.hierarchy_long(rel, "employee_id", "supervisor_id")
print(long.df().head())

# Wide format: one row per employee with supervisor levels as columns
wide = pona.hierarchy_wide(rel, "employee_id", "supervisor_id", max_depth=8)
print(wide.df().head())

# Span-of-control metrics for every manager
stats = pona.hierarchy_stats(rel, "employee_id", "supervisor_id")
print(stats.df().sort_values("direct_reports", ascending=False).head(10))
```

## Method-chaining (with pyduck-janitor)

```python
import pyduck_ona as pona
from pyduck_janitor import DuckJanitor

dj = DuckJanitor.from_pandas(raw_hris).clean_names()
ona = pona.DuckONA.from_janitor(dj, hris_table="hris")
issues = pona.hierarchy_valid(ona.table("hris"), "employee_id", "supervisor_id")
```

## Golden flow: from raw HRIS to org-design insights (classic API)

```python
import pandas as pd
import pyduck_ona as pona

# 1. Load snapshots with quarterly history
snapshots = pd.DataFrame([
    {"employee_id": "CEO", "supervisor_id": None, "department": "Exec",
     "job_level": 7, "snapshot_date": "2025-01-31"},
    {"employee_id": "VP1", "supervisor_id": "CEO", "department": "Sales",
     "job_level": 6, "snapshot_date": "2025-01-31"},
    {"employee_id": "M1", "supervisor_id": "VP1", "department": "Sales",
     "job_level": 5, "snapshot_date": "2025-01-31"},
    {"employee_id": "IC1", "supervisor_id": "M1", "department": "Sales",
     "job_level": 3, "snapshot_date": "2025-01-31"},
    # ... same people, moved / promoted in Q2
    {"employee_id": "CEO", "supervisor_id": None, "department": "Exec",
     "job_level": 7, "snapshot_date": "2025-04-30"},
    {"employee_id": "VP1", "supervisor_id": "CEO", "department": "Sales",
     "job_level": 6, "snapshot_date": "2025-04-30"},
    {"employee_id": "M1", "supervisor_id": "VP1", "department": "Sales",
     "job_level": 5, "snapshot_date": "2025-04-30"},
    {"employee_id": "IC1", "supervisor_id": "M1", "department": "Sales",
     "job_level": 4, "snapshot_date": "2025-04-30"},  # promotion
])

# 2. Hierarchy integrity check (run on a single snapshot)
rel, _ = pona.to_duckdb(
    snapshots.query("snapshot_date == '2025-04-30'"), "hris"
)
issues = pona.hierarchy_valid(rel, "employee_id", "supervisor_id")
print(issues.df())

# 3. Static org-design + ONA on the latest snapshot
edges, _ = pona.to_duckdb(
    snapshots.query("snapshot_date == '2025-04-30'"), "latest"
)
brokers = pona.betweenness(edges, "employee_id", "supervisor_id")
influencers = pona.pagerank(edges, "employee_id", "supervisor_id")
spans = pona.hierarchy_stats(edges, "employee_id", "supervisor_id")

# 4. Temporal analytics
dt = pona.DuckONATemporal(":memory:")
dt.load_snapshots(snapshots, snapshot_date_col="snapshot_date", freq="Q")
trends = dt.compute_temporal_metrics()

# 5. Career Markov + org-design scorecard
transitions = dt.career_markov_matrix(state_col="job_level", lookback="2Q")
forecast = dt.career_markov_forecast("IC1", horizon=2, state_col="job_level")
scorecard = dt.org_design_scorecard(lookback="2Q")
alerts = dt.org_design_change_alerts(lookback="2Q")
```


## DuckONAFrame (v0.3 contract)

For a unified, chainable API that hides the return-type differences
between functions, use the new `DuckONAFrame` façade:

```python
import pandas as pd
import pyduck_ona as pona

hris = pd.DataFrame({
    "employee_id": ["CEO", "VP1", "M1", "IC1"],
    "supervisor_id": [None, "CEO", "VP1", "M1"],
    "department": ["Exec", "Sales", "Sales", "Sales"],
    "job_level": [7, 6, 5, 3],
})

result = (
    pona.DuckONAFrame.from_pandas(hris, "hris")
    .pipeline([
        lambda f: f.graph_pagerank(output="pr"),
        lambda f: f.graph_betweenness(output="bc"),
        lambda f: f.report_export("scores"),
    ])
)

# result.relation() is a DuckDBPyRelation with canonical entity_id
```

`DuckONAFrame` returns `DuckDBPyRelation` by default, renames employee/node
keys to `entity_id`, and exposes the five verb families (`prep_*`, `graph_*`,
`temporal_*`, `model_*`, `report_*`).

## API catalog

Every public function and class has a dedicated page in `docs/api/` with full
parameters, return types, and runnable examples. Click any link below to jump
to its reference page. Pages are regenerated from source docstrings by
`python scripts/generate_api_docs.py`.

For agents and contributors: see [`docs/ai_contributor_guide.md`](docs/ai_contributor_guide.md)
and [`AGENTS.md`](AGENTS.md) for the project style, CI runbook, and extension-integration
pattern.

### DuckONAFrame (v0.3)

| Function / Class | Description |
|---|---|
| [`DuckONAFrame`](docs/api/duckonaframe.md) | A relation-first, uniform-verb façade over pyduck-ona analytics |
| [`DuckONAFrame.graph_betweenness`](docs/api/duckonaframe_graph_betweenness.md) | Compute betweenness centrality on the direct-edge relation |
| [`DuckONAFrame.graph_pagerank`](docs/api/duckonaframe_graph_pagerank.md) | Compute PageRank on the direct-edge relation |
| [`DuckONAFrame.model_ols`](docs/api/duckonaframe_model_ols.md) | Fit an OLS model via broom_sm |
| [`DuckONAFrame.pipeline`](docs/api/duckonaframe_pipeline.md) | Compose a sequence of frame verbs into a single workflow |
| [`DuckONAFrame.prep_load_snapshots`](docs/api/duckonaframe_prep_load_snapshots.md) | Load snapshot data and wire up a temporal engine on this frame |
| [`DuckONAFrame.prep_long`](docs/api/duckonaframe_prep_long.md) | Return a long-form transitive closure of the reporting chain |
| [`DuckONAFrame.prep_validate`](docs/api/duckonaframe_prep_validate.md) | Validate hierarchy integrity |
| [`DuckONAFrame.prep_wide`](docs/api/duckonaframe_prep_wide.md) | Return a wide-form ancestor table (Level_1, Level_2, ...) |
| [`DuckONAFrame.relation`](docs/api/duckonaframe_relation.md) | Return the current source relation |
| [`DuckONAFrame.report_export`](docs/api/duckonaframe_report_export.md) | Register the current (or supplied) relation as a named table |
| [`DuckONAFrame.search_similar`](docs/api/duckonaframe_search_similar.md) | Approximate nearest-neighbor search over an embedding column |
| [`DuckONAFrame.search_text`](docs/api/duckonaframe_search_text.md) | Full-text search over a text column in the current relation |
| [`DuckONAFrame.temporal_metrics`](docs/api/duckonaframe_temporal_metrics.md) | Compute temporal ONA metrics across loaded snapshots |

### Search

| Function / Class | Description |
|---|---|
| [`build_fts_index`](docs/api/build_fts_index.md) | Create a DuckDB full-text search index on an HR text table |
| [`drop_fts_index`](docs/api/drop_fts_index.md) | Drop a DuckDB FTS index for ``table_name`` |
| [`text_search`](docs/api/text_search.md) | Full-text search an HR table and return the top-k matches |
| [`build_vector_index`](docs/api/build_vector_index.md) | Create an HNSW index on a fixed-size ``ARRAY`` embedding column |
| [`drop_vector_index`](docs/api/drop_vector_index.md) | Drop an HNSW index created by :func:`build_vector_index` |
| [`vector_search`](docs/api/vector_search.md) | Approximate nearest-neighbor search over an embedding column |
| [`fuzzy_join_vectors`](docs/api/fuzzy_join_vectors.md) | Approximate nearest-neighbor join between two embedding tables |

### DuckONA class

| Function / Class | Description |
|---|---|
| [`DuckONA`](docs/api/duckona.md) | A DuckDB-backed workspace for HR analytics |
| [`DuckONA.anova`](docs/api/duckona_anova.md) | One-way ANOVA; delegates to `pyduck_ona.stats.anova` |
| [`DuckONA.betweenness`](docs/api/duckona_betweenness.md) | Betweenness centrality via `pyduck_ona.graph.betweenness` |
| [`DuckONA.build_org_edges`](docs/api/duckona_build_org_edges.md) | Build a directed edge relation from the HRIS hierarchy |
| [`DuckONA.build_temporal_slices`](docs/api/duckona_build_temporal_slices.md) | Return time-sliced relations for a registered table |
| [`DuckONA.chi_square`](docs/api/duckona_chi_square.md) | Chi-square test; delegates to `pyduck_ona.stats.chi_square` |
| [`DuckONA.connected_components`](docs/api/duckona_connected_components.md) | Weakly-connected components via `pyduck_ona.graph.connected_components` |
| [`DuckONA.correlation`](docs/api/duckona_correlation.md) | Correlation helper; delegates to `pyduck_ona.stats.correlation` |
| [`DuckONA.deduplicate`](docs/api/duckona_deduplicate.md) | Deduplicate an HR DataFrame by `(id_col, date_col)` |
| [`DuckONA.degree_centrality`](docs/api/duckona_degree_centrality.md) | Degree centrality via `pyduck_ona.graph.degree_centrality` |
| [`DuckONA.eigenvector_centrality`](docs/api/duckona_eigenvector_centrality.md) | Eigenvector centrality via `pyduck_ona.graph.eigenvector_centrality` |
| [`DuckONA.filter_noise`](docs/api/duckona_filter_noise.md) | Filter noise from an HR DataFrame |
| [`DuckONA.join_hris`](docs/api/duckona_join_hris.md) | Join a metric relation back to the HRIS demographics table |
| [`DuckONA.load_attendance`](docs/api/duckona_load_attendance.md) | Load an office-attendance / presence table |
| [`DuckONA.load_compensation`](docs/api/duckona_load_compensation.md) | Load a compensation table with one row per employee per snapshot |
| [`DuckONA.load_hris`](docs/api/duckona_load_hris.md) | Load the HRIS snapshot |
| [`DuckONA.load_promotions`](docs/api/duckona_load_promotions.md) | Load a promotion / internal-mobility table |
| [`DuckONA.load_retirement`](docs/api/duckona_load_retirement.md) | Load a retirement-eligibility or retirement-planning table |
| [`DuckONA.load_skills`](docs/api/duckona_load_skills.md) | Load a skills / proficiency table |
| [`DuckONA.load_survey`](docs/api/duckona_load_survey.md) | Load an engagement / survey-results table |
| [`DuckONA.load_turnover`](docs/api/duckona_load_turnover.md) | Load a turnover / termination table |
| [`DuckONA.logistic`](docs/api/duckona_logistic.md) | Logistic regression; delegates to `pyduck_ona.stats.logistic` |
| [`DuckONA.louvain_communities`](docs/api/duckona_louvain_communities.md) | Louvain community detection via `pyduck_ona.graph.louvain_communities` |
| [`DuckONA.model_compare`](docs/api/duckona_model_compare.md) | Model comparison; delegates to `pyduck_ona.stats.model_compare` |
| [`DuckONA.mrqap`](docs/api/duckona_mrqap.md) | Small pure-Python MRQAP-style permutation test for matrix regression |
| [`DuckONA.ols`](docs/api/duckona_ols.md) | OLS linear regression; delegates to `pyduck_ona.stats.ols` |
| [`DuckONA.pagerank`](docs/api/duckona_pagerank.md) | PageRank centrality via `pyduck_ona.graph.pagerank` |
| [`DuckONA.predict_engagement`](docs/api/duckona_predict_engagement.md) | Predict engagement scores based on demographics and ONA metrics |
| [`DuckONA.profile_clusters`](docs/api/duckona_profile_clusters.md) | Cluster employee profiles from HR attributes and optional network features |
| [`DuckONA.sql`](docs/api/duckona_sql.md) | Run arbitrary SQL on the owned connection |
| [`DuckONA.table`](docs/api/duckona_table.md) | Return a relation for a registered table |
| [`DuckONA.validate_keys`](docs/api/duckona_validate_keys.md) | Validate HR table keys: non-null IDs, no duplicate snapshots, sensible dates |
| [`DuckONA.vif`](docs/api/duckona_vif.md) | Variance-inflation factors; delegates to `pyduck_ona.stats.vif` |

### DuckONATemporal class

| Function / Class | Description |
|---|---|
| [`DuckONATemporal`](docs/api/duckonatemporal.md) | A DuckDB-backed temporal ONA workspace |
| [`DuckONATemporal.career_markov_forecast`](docs/api/duckonatemporal_career_markov_forecast.md) | Forecast future state probabilities for one employee via Markov transitions |
| [`DuckONATemporal.career_markov_matrix`](docs/api/duckonatemporal_career_markov_matrix.md) | Estimate career-transition Markov probabilities from snapshot history |
| [`DuckONATemporal.career_trajectory`](docs/api/duckonatemporal_career_trajectory.md) | Per-employee career path across periods |
| [`DuckONATemporal.change_detection`](docs/api/duckonatemporal_change_detection.md) | Top movers for a given metric over the lookback window |
| [`DuckONATemporal.compute_temporal_metrics`](docs/api/duckonatemporal_compute_temporal_metrics.md) | Per-employee ONA metric time-series across all periods |
| [`DuckONATemporal.event_window`](docs/api/duckonatemporal_event_window.md) | Before/after comparison around a specific event date |
| [`DuckONATemporal.load_promotions`](docs/api/duckonatemporal_load_promotions.md) | Load a promotions / internal-mobility event table |
| [`DuckONATemporal.load_snapshots`](docs/api/duckonatemporal_load_snapshots.md) | Load HRIS snapshot data |
| [`DuckONATemporal.load_survey`](docs/api/duckonatemporal_load_survey.md) | Load a survey / engagement table with employee_id + snapshot_date |
| [`DuckONATemporal.manager_chain`](docs/api/duckonatemporal_manager_chain.md) | Managers along the way for a given employee |
| [`DuckONATemporal.manager_effectiveness`](docs/api/duckonatemporal_manager_effectiveness.md) | Composite manager effectiveness score |
| [`DuckONATemporal.mobility_anomaly`](docs/api/duckonatemporal_mobility_anomaly.md) | Peer-relative stuckness z-score per employee |
| [`DuckONATemporal.mobility_leaderboard`](docs/api/duckonatemporal_mobility_leaderboard.md) | Top movers by composite mobility score |
| [`DuckONATemporal.network_evolution`](docs/api/duckonatemporal_network_evolution.md) | Aggregate network-shape metrics per period |
| [`DuckONATemporal.org_design_change_alerts`](docs/api/duckonatemporal_org_design_change_alerts.md) | Flag periods with potentially unhealthy organizational-design shifts |
| [`DuckONATemporal.org_design_scorecard`](docs/api/duckonatemporal_org_design_scorecard.md) | Per-period organizational design metrics and a composite score |
| [`DuckONATemporal.sql`](docs/api/duckonatemporal_sql.md) | Run arbitrary SQL on the owned connection |

### Hierarchy primitives

| Function / Class | Description |
|---|---|
| [`hierarchy_long`](docs/api/hierarchy_long.md) | Unroll the org tree into long format via a recursive CTE |
| [`hierarchy_stats`](docs/api/hierarchy_stats.md) | Calculate span-of-control metrics for every manager |
| [`hierarchy_valid`](docs/api/hierarchy_valid.md) | Diagnose the integrity of an organizational reporting structure |
| [`hierarchy_wide`](docs/api/hierarchy_wide.md) | Flatten the reporting chain into a single row per employee |

### Graph metrics

| Function / Class | Description |
|---|---|
| [`betweenness`](docs/api/betweenness.md) | Betweenness centrality for every node (broker detection) |
| [`connected_components`](docs/api/connected_components.md) | Weakly-connected components in the edge graph |
| [`degree_centrality`](docs/api/degree_centrality.md) | Degree centrality for every node |
| [`eigenvector_centrality`](docs/api/eigenvector_centrality.md) | Eigenvector centrality for every node |
| [`louvain_communities`](docs/api/louvain_communities.md) | Louvain community detection on the edge graph |
| [`pagerank`](docs/api/pagerank.md) | PageRank centrality (influence scoring) |
| [`shortest_path`](docs/api/shortest_path.md) | Shortest path between two nodes in the edge graph |

### Statistical modeling

| Function / Class | Description |
|---|---|
| [`anova`](docs/api/anova.md) | One-way ANOVA via OLS, tidy output |
| [`chi_square`](docs/api/chi_square.md) | Chi-square test of independence between two categorical variables |
| [`correlation`](docs/api/correlation.md) | Pairwise correlations across a set of columns |
| [`logistic`](docs/api/logistic.md) | Fit a logistic regression. Returns (tidy, glance) |
| [`model_compare_stats`](docs/api/model_compare_stats.md) | Side-by-side comparison of multiple fitted models |
| [`ols`](docs/api/ols.md) | Fit an OLS linear regression. Returns (tidy, glance) |
| [`plot_coefficients`](docs/api/plot_coefficients.md) | Forest plot of regression coefficients with confidence intervals |
| [`plot_ols`](docs/api/plot_ols.md) | Per-predictor OLS scatterplots with fitted regression line |
| [`plot_residuals`](docs/api/plot_residuals.md) | Residual-diagnostic plots for each predictor |
| [`save_figure`](docs/api/save_figure.md) | Save a matplotlib figure and return the path |
| [`tidy_to_duckdb`](docs/api/tidy_to_duckdb.md) | Write a tidy model result into a DuckDB table |
| [`to_duckdb`](docs/api/to_duckdb.md) | Register a DataFrame or relation as a DuckDB table |
| [`vif`](docs/api/vif.md) | Variance Inflation Factors for the predictors in a formula |

### Graph export

| Function / Class | Description |
|---|---|
| [`to_igraph`](docs/api/to_igraph.md) | Convert an edge relation into an igraph.Graph via Arrow |
| [`to_networkx`](docs/api/to_networkx.md) | Convert an edge relation into a NetworkX graph via Arrow |

## Short aliases (optional)

For convenience, the four hierarchy functions are also available as
shorter names in `pyduck_ona.hierarchy`:

```python
from pyduck_ona.hierarchy import valid, long, wide, stats

issues = valid(rel, "employee_id", "supervisor_id")
chain = long(rel, "employee_id", "supervisor_id")
flat = wide(rel, "employee_id", "supervisor_id", max_depth=5)
metrics = stats(rel, "employee_id", "supervisor_id")
```

## API conventions & gotchas

A few things to know that are easy to hit the first time:

- **Column names in output relations vary by function.**
  - `hierarchy_valid` → `issue_type, employee_id, detail`
  - `hierarchy_long` → `employee_id, supervisor_id, depth, path`
  - `hierarchy_wide` → `employee_id, Level_1, Level_2, ...`
  - `hierarchy_stats` → **`manager_id`** (not `employee_id`), `direct_reports, indirect_reports, total_reports, team_size, levels_below`
  - `betweenness` / `pagerank` → **`node_id`** (not `node`)
  Join on these columns when enriching an employee table.

- **For ONA centrality, pass the direct edge relation, not the
  long-format transitive closure.** `hierarchy_long()` star-flattens
  the graph and makes `betweenness` / `pagerank` degenerate. Use
  `SELECT employee_id, supervisor_id FROM rel WHERE supervisor_id IS NOT NULL`.

- **`employee_id` must be unique.** Duplicate employee IDs raise
  `ValueError` immediately; deduplicate upstream before calling the
  hierarchy functions.

- **`supervisor_id` may be any DuckDB type.** Integer, UUID, or VARCHAR
  IDs all work. Empty strings are normalized to NULL (treated as roots).

- **Zero-row input relations return empty results** for all four
  hierarchy functions instead of raising.

- **Graph algorithms drop rows with NULL endpoints.** Passing a raw org
  relation that includes the root (NULL supervisor) no longer crashes
  `betweenness` / `pagerank` / `connected_components`.

- **`tidy_to_duckdb()` rewrites dotted column names on write.**
  `broom-sm` returns `p.value`, `conf.low` (R-style). DuckDB parses
  unquoted dots as struct field access. `tidy_to_duckdb` renames them
  to `p_value` / `conf_low` so you can query with unquoted identifiers:
  `SELECT term, p_value FROM model_tidy WHERE p_value < 0.05`.

- **`tidy_to_duckdb` and `to_duckdb` are different.**
  - `tidy_to_duckdb(tidy_df, con, table_name)` writes a broom-sm tidy
    DataFrame to a DuckDB table (with the dotted-name rewrite above).
  - `to_duckdb(data, table_name, con)` registers any DataFrame or
    relation as a DuckDB table (no rewrite).
  Both return `(table_name, con)` and validate `table_name` as a safe
  unquoted DuckDB identifier.

- **Empty-graph safety.** `betweenness` / `pagerank` /
  `connected_components` return an empty DataFrame on an empty edge
  relation — no crash.

- **`supervisor_id` is allowed to be NULL** (that's the root of the
  hierarchy). `employee_id` is required to be non-null; passing a
  relation with NULL employee IDs raises `ValueError` upfront.

## Graph export (ONA)

```python
# Zero-copy Arrow → NetworkX
G = pona.to_networkx(long_rel, "employee_id", "supervisor_id",
                     weight_col="interaction_count")

# Or to igraph for faster algorithms
g = pona.to_igraph(long_rel, "employee_id", "supervisor_id", directed=True)

# Graph algorithms (NetworkX backend, default)
# For betweenness/pagerank/connected_components, pass the *direct*
# edge relation (one row per manager → report), not the long-format
# transitive closure from hierarchy_long().
direct = duckdb.sql("""
    SELECT employee_id, supervisor_id
    FROM rel WHERE supervisor_id IS NOT NULL
""")

pona.graph.shortest_path(direct, "employee_id", "supervisor_id",
                         source="E1000", target="E001")
pona.graph.betweenness(direct, "employee_id", "supervisor_id")
pona.graph.pagerank(direct, "employee_id", "supervisor_id")
pona.graph.connected_components(direct, "employee_id", "supervisor_id")
```

### DuckPGQ backend (optional, currently unavailable)

Each `pyduck_ona.graph.*` function accepts `backend="duckpgq"` for a
DuckDB-native property-graph implementation. **DuckPGQ is not currently
installable** from the DuckDB community-extension registry (HTTP 404 on
current DuckDB releases; the extension is in flux after a major API
rewrite). The NetworkX backend is the default and always available. The
DuckPGQ slot is reserved so the API surface stays stable when it ships.

### ERGM (deferred)

Exponential Random Graph Models (ERGMs) model the network itself as the
dependent variable — for example, testing whether employees are more
likely to collaborate when they share a department, controlling for
reciprocity and transitivity. There is currently no production-grade
Python ERGM library. `pyduck-ona` stays Python-only, so ERGM is
deferred until a mature Python implementation exists. The recommended
gold-standard tooling remains R's `statnet`/`ergm`; clean DuckDB
relations from this package can be exported to R via Parquet if you need
ERGMs today.

## Statistical-model integration (broom-sm)

```python
import statsmodels.api as sm
import pyduck_ona as pona
import duckdb

# --- Correlation (pairwise, with p-values) ---
pona.correlation(hr_df, columns=["team_size", "tenure_yrs", "salary"])

# --- One-way ANOVA ---
pona.anova(hr_df, "salary ~ department")

# --- Chi-square test of independence ---
chi_table, chi_fig = pona.chi_square(hr_df, "department", "gender")
pona.save_figure(chi_fig, "dept_by_gender.png")

# --- OLS linear regression (tidy + glance) ---
tidy, glance = pona.ols(hr_df, "salary ~ team_size + tenure_yrs")
print(tidy[tidy["p.value"] < 0.05])

# --- Logistic regression ---
tidy, glance = pona.logistic(hr_df, "attrition ~ salary + tenure_yrs + team_size")
tidy["odds_ratio"] = pona.__import__("numpy").exp(tidy["estimate"])  # exp(beta) = OR

# --- Coefficient forest plot ---
fig, ax = pona.plot_coefficients(tidy)
pona.save_figure(fig, "salary_forest.png")

# --- OLS scatter with regression line + 95% CI ---
for label, fig in pona.plot_ols(hr_df, x=["team_size", "tenure_yrs"], y="salary"):
    pona.save_figure(fig, f"ols_{label}.png")

# --- DuckDB round-trip: tidy results as a queryable table ---
tidy, _ = pona.ols(hr_df, "salary ~ team_size + tenure_yrs")
table_name, con = pona.tidy_to_duckdb(tidy, table_name="salary_model")
duckdb_con = con  # use the same connection
duckdb_con.sql("SELECT term, estimate FROM salary_model WHERE p_value < 0.05")
```

## Architecture

```
pyduck_ona/
├── core.py            # hierarchy_valid / long / wide / stats
├── hierarchy.py       # short-form aliases (valid, long, wide, stats)
├── bridge.py          # to_networkx / to_igraph (Arrow-based export)
├── graph/             # shortest_path / betweenness / pagerank /
│                      # connected_components (NetworkX default,
│                      # DuckPGQ reserved slot)
└── stats/             # correlation / anova / ols / logistic /
                       # chi_square / plot_* / tidy_to_duckdb
                       # (broom-sm backed; optional [broom] extra)
```

## v0.3 API unification draft

See [`docs/v0.3_api_contract.md`](docs/v0.3_api_contract.md) for the proposed
uniform verb taxonomy, return-schema contract, and janitor bridge standards.

## SQL safety

All public functions validate column names against a strict regex
(`[A-Za-z_][A-Za-z0-9_]*`) and double-quote-escape anything outside that
pattern. SQL values are always bound via DuckDB's `?` parameter API, never
string-interpolated. This means untrusted column names are safe.

## License

MIT — see LICENSE.

## Author

John C. Vallier — `jcvallier.cpa@gmail.com`
Maintained by EzraAir555.

## Changelog

### 0.2.0 — DuckONATemporal (time-series ONA)

- Added `pyduck_ona.DuckONATemporal` class for time-series analysis of
  organizational networks across multiple HRIS snapshots.
  - 8 analytical methods: `compute_temporal_metrics`, `network_evolution`,
    `event_window`, `change_detection`, `mobility_leaderboard`,
    `career_trajectory`, `manager_chain`, `mobility_anomaly`,
    `manager_effectiveness`.
  - 20 query primitives under `dt.q.*` namespace across 5 categories:
    trajectory (`trajectory_at`, `trajectory_diff`, `trajectory_pivot`,
    `trajectory_rank`), hierarchy change (`edges_added`, `edges_removed`,
    `node_set_diff`, `hierarchy_drift`), subtree (`subtree_at`,
    `subtree_size_at`, `subtree_growth`, `subtree_overlap`), snapshot
    compare (`delta_table`, `new_centers`, `fallen_centers`,
    `cohort_compare`), and window aggregate (`window_mean`,
    `window_trend`, `window_rank_change`, `window_volatility`).
  - Mixed return style: DuckDB relations for traversals (compose with
    SQL), DataFrames for terminal aggregations.
- Added `DuckONA.predict_engagement` helper for engagement prediction
  via OLS or logistic regression; results registered as a DuckDB table.
- Added 88 integration tests across 5 files:
  - `test_temporal.py` (29): API contract for the 8 analytical methods.
  - `test_temporal_primitives.py` (29): API contract for the 20 primitives.
  - `test_temporal_simulation.py` (10): Principle #9 DGP tests —
    plant known signals and verify recovery.
  - `test_temporal_properties.py` (14): Hypothesis fuzzing + edge
    cases (NaN, unicode, integer IDs, single-period).
  - `test_temporal_performance.py` (6): Scaling benchmarks at 50/100/500
    employees; memory-bounded check.
  - `test_simulation.py` (added previously): OLS + logistic + MRQAP
    coefficient recovery from known DGP.
- Added 4 docs:
  - `docs/temporal_ona_tutorial.md`: usage walkthrough.
  - `docs/temporal_api_reference.md`: per-method API reference.
  - `docs/temporal_cookbook.md`: 10 real-world query recipes.
  - `docs/when_to_use_pyduck_ona.md`: one-page decision guide for
    People Analytics teams.
- Fixed `NA`-comparison bug in `mobility_leaderboard` exposed by
  property-based testing (NaN supervisor IDs caused
  `TypeError: boolean value of NA is ambiguous`).

### 0.1.5 — P2 polish: rename, docs, ERGM note

- Added `node_id_col` parameter to graph metric functions so callers can
  rename the output node-id column (e.g., to `employee_id`).
- Added `examples/hr_compensation_mobility_analysis.py` to the README
  Examples table.
- Added `docs/duckona_tutorial.md` covering the `DuckONA` class, HR table
  loaders, validation, centrality, models, temporal slicing, and MRQAP.
- Added an ERGM scope note explaining that ERGM is deferred pending a
  mature Python library.

### 0.1.4 — Matplotlib 3.11 compatibility

- Fixed `tests/integration/test_stats.py` to use `tick_labels=` on matplotlib ≥3.9 while falling back to `labels=` on older versions.

### 0.1.3 — DuckONA analysis layer

- Added `pyduck_ona.DuckONA` high-level class for end-to-end HR analytics:
  - Load HRIS, compensation, turnover, survey, retirement, promotion, skills,
    and attendance tables.
  - Validate keys, drop duplicates, and reject future/out-of-range dates.
  - Build org-chart edges and compute centrality metrics.
  - Join metrics back to HRIS and run OLS/logistic/ANOVA/chi-square models.
  - Slice HR tables temporally for before/after analysis.
- Added new graph metrics: `eigenvector_centrality`, `degree_centrality`,
  `louvain_communities`.
- Added pure-Python MRQAP helper for matrix regression under network
  dependence (no R dependency).
- Added `examples/hr_compensation_mobility_analysis.py` and
  `tests/integration/test_analysis.py`.

### 0.1.2 — Kimi review hardening

- Fixed empty-relation crash in all four hierarchy functions.
- Fixed non-string key-type crash (integer, UUID, etc.) by normalizing
  empty strings to NULL without forcing VARCHAR casts.
- Fixed graph algorithms to drop NULL-endpoint rows instead of raising
  `ValueError: None cannot be a node`.
- Replaced global temp-view counter in `_run_sql_on_default` with UUIDs.
- Added duplicate `employee_id` validation across hierarchy functions.
- Validated `table_name` in `tidy_to_duckdb` / `to_duckdb` to prevent
  SQL injection.
- Added 22 regression tests (empty relations, numeric keys, duplicate
  IDs, NULL-supervisor graph handling, table-name validation).

### 0.1.1 — qwen3.5 audit hardening

- Fixed DuckDB connection isolation (`_run_sql_on_default`).
- Added NULL `employee_id` validation and improved error guidance.
- Rewrote `p.value` / `conf.low` dotted names on DuckDB write.
- Added MIT LICENSE, CI workflow, and Changelog.

### 0.1.0 — Initial release

- Core hierarchy functions: `hierarchy_valid`, `hierarchy_long`,
  `hierarchy_wide`, `hierarchy_stats`.
- Graph algorithms: `betweenness`, `pagerank`, `connected_components`,
  `shortest_path` (NetworkX backend, DuckPGQ slot reserved).
- Stats integration via optional `broom-sm` extra.
