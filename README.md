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

(pona.from_relation(rel)                  # if/when janitor flavor ships
   .clean_names()
   .hierarchy_valid("employee_id", "supervisor_id")
   .filter("issue_type = 'broken_chain'"))
```

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
  Both return `(table_name, con)`.

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
duckdb_con.sql("SELECT term, estimate FROM salary_model WHERE \"p.value\" < 0.05")
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

## SQL safety

All public functions validate column names against a strict regex
(`[A-Za-z_][A-Za-z0-9_]*`) and double-quote-escape anything outside that
pattern. SQL values are always bound via DuckDB's `?` parameter API, never
string-interpolated. This means untrusted column names are safe.

## Changelog

### 0.1.1 (2026-06-20)

Audit + hardening release.

**Bug fixes**

- `hierarchy_valid` / `hierarchy_long` / `hierarchy_wide` / `hierarchy_stats` now correctly accept DuckDB relations created on a custom `con = duckdb.connect()` instance. (Previously raised `InvalidInputException: not suitable for replacement scan` — see `memory/2026-06-21_qwen_review.md` P0-1.)
- `_validate_columns(..., require_non_null=[employee_id])` fails loudly on NULL `employee_id` rows, instead of silently dropping them in recursive CTEs. `supervisor_id` is still allowed to be NULL (that's the root of the hierarchy).
- `tidy_to_duckdb(...)` now rewrites R-style dotted column names (`p.value` → `p_value`, `conf.low` → `conf_low`) on write, so the resulting DuckDB table is queryable without manual quoting.

**Docs**

- README gains a "Short aliases" section and a "API conventions & gotchas" section.
- `hierarchy_wide` and `shortest_path` docstrings document their non-obvious behaviors (silent depth truncation, `source == target` returns trivial self-path).
- `max_depth` validation now suggests the default (15) in its error message.

**Tests**

- 13 new regression tests added: `TestConnectionIsolation` (6), `TestNullValidation` (3), `TestErrorGuidance` (1), `TestCycleDetectionBoundary` (5), `TestTidyToDuckDBDottedNames` (4). 103 total pass.

### 0.1.0 (2026-06-20)

Initial public release.

## License

MIT — see [LICENSE](LICENSE).

## Author

John C. Vallier — `jcvallier.cpa@gmail.com`
Maintained by EzraAir555.