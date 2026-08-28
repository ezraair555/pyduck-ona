# DuckONATemporal API Reference

Complete reference for the `DuckONATemporal` class — temporal ONA
analytics for HRIS snapshots over time.

## Class signature

```python
from pyduck_ona import DuckONATemporal

dt = DuckONATemporal(db_path: str = ":memory:")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `db_path` | str | `:memory:` | DuckDB database path. Use a file path to persist across sessions. |

**Attributes after construction:**
- `dt.con` — the underlying `duckdb.DuckDBPyConnection`
- `dt.q` — the `_QueryPrimitives` namespace (20 tools)
- `dt.periods` — list of detected period labels (after `load_snapshots`)
- `dt.freq` — period frequency (M / Q / Y, after `load_snapshots`)

---

## Loading

### `load_snapshots(df, snapshot_date_col, employee_id_col, supervisor_id_col, freq, table_name)`

Load HRIS snapshot data.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pd.DataFrame` | required | One row per employee per snapshot. |
| `snapshot_date_col` | str | `"snapshot_date"` | Column holding the snapshot date. |
| `employee_id_col` | str | `"employee_id"` | Column holding the unique employee ID. |
| `supervisor_id_col` | str | `"supervisor_id"` | Column holding the manager ID (NULL for org root). |
| `freq` | str | `"Q"` | Period frequency: `"M"` / `"Q"` / `"Y"`. |
| `table_name` | str | `"snapshots"` | DuckDB table name. |

**Returns:** `list[str]` — sorted period labels detected in the data.

**Required columns:**
- `employee_id_col` — non-null unique employee identifier
- `supervisor_id_col` — manager identifier (NULL for root)
- `snapshot_date_col` — date of the HRIS extract

**Optional columns used by specific methods:**
- `job_level` — used by mobility_leaderboard, career_trajectory
- `department` — used by mobility_leaderboard, mobility_anomaly
- `engagement` — used by manager_effectiveness (or load via survey table)

**Example:**
```python
hris = pd.read_csv("hris_snapshots.csv")
dt = DuckONATemporal()
periods = dt.load_snapshots(hris, freq="Q")
# → ['2025-01-01', '2025-04-01', '2025-07-01', '2025-10-01']
```

### `load_survey(df, table_name="survey")`

Load an engagement / survey table. Must contain columns `employee_id`,
`snapshot_date`, and `engagement`.

### `load_promotions(df, table_name="promotions")`

Load a promotions event table.

---

## Analytical methods (8)

### `compute_temporal_metrics(metrics=None, employee_id_col=None, supervisor_id_col=None)`

Per-employee ONA metric time-series across all periods.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metrics` | `list[str]` | `["betweenness", "pagerank", "degree_centrality", "team_size"]` | Metric names. |
| `employee_id_col`, `supervisor_id_col` | str | use columns from `load_snapshots` | Override column names. |

**Returns:** `pd.DataFrame` with columns `period, employee_id, metric, value, prev_value, delta, pct_change`.

**Supported metrics:** `betweenness`, `pagerank`, `eigenvector_centrality`, `degree_centrality`, `connected_components`, `louvain_communities`, `team_size`.

**Gotcha:** Computing `connected_components` and `louvain_communities` per period is expensive; for large orgs use `network_evolution` instead.

### `network_evolution(employee_id_col=None, supervisor_id_col=None)`

Aggregate network-shape metrics per period.

**Returns:** `pd.DataFrame` with columns `period, n_employees, n_edges, density, centralization, n_components, avg_path_length`.

**Density** = `nx.density(G)` (fraction of possible edges present).
**Centralization** = Freeman's out-degree centralization (0-1, higher = more star-like).
**Avg path length** = average shortest path in the largest weakly-connected component.

### `event_window(event_date, pre_window=None, post_window=None, metrics=None)`

Before/after comparison around a specific event date.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `event_date` | str | required | ISO date. Periods before this are "pre"; after are "post". |
| `pre_window` | `(str, str)` | None | (start, end) inclusive bounds for pre-period. |
| `post_window` | `(str, str)` | None | (start, end) inclusive bounds for post-period. |
| `metrics` | `list[str]` | `["betweenness", "pagerank"]` | ONA metrics to compute. |

**Returns:** `pd.DataFrame` with columns `period, period_type, employee_id, metric, value`.

**Example:**
```python
ev = dt.event_window(event_date="2025-07-01", metrics=["betweenness"])
ev.groupby(["period_type", "metric"])["value"].mean()
```

### `change_detection(metric="betweenness", top_n=20, lookback="4Q")`

Top movers for a given metric over the lookback window.

**Returns:** `pd.DataFrame` with columns `employee_id, start_value, end_value, delta, pct_change, rolling_zscore, rank`.

### `mobility_leaderboard(lookback="4Q", top_n=20, w_promotion=1.0, w_lateral=0.5, w_dept_change=0.3, w_demotion=-1.0)`

Top movers by composite mobility score.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lookback` | str | `"4Q"` | Period window. |
| `top_n` | int | `20` | Number of top movers. |
| `w_promotion` | float | `1.0` | Weight for promotions. |
| `w_lateral` | float | `0.5` | Weight for lateral moves. |
| `w_dept_change` | float | `0.3` | Weight for department transfers. |
| `w_demotion` | float | `-1.0` | Weight for demotions (typically negative). |

**Returns:** `pd.DataFrame` with columns `employee_id, mobility_score, n_promotions, n_lateral_moves, n_dept_changes, n_demotions, n_manager_changes, first_level, last_level, first_dept, last_dept, rank`.

**Definition:**
- **Promotion** = level increase with supervisor change.
- **Lateral move** = supervisor change, level unchanged.
- **Department change** = department change, supervisor unchanged.
- **Demotion** = level decrease.

### `career_trajectory(employee_id, lookback="4Q")`

Per-employee career path across periods.

**Returns:** `pd.DataFrame` with columns `period, supervisor_id, job_level, department, promoted, transferred, manager_changed`.

### `manager_chain(employee_id, lookback="4Q")`

Managers along the way for one employee.

**Returns:** `pd.DataFrame` with columns `period, supervisor_id, supervisor_name, supervisor_level, supervisor_path_to_ceo`.

### `mobility_anomaly(lookback="4Q", peer_basis=None, stuckness_threshold=1.5)`

Peer-relative stuckness z-score per employee.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lookback` | str | `"4Q"` | Period window. |
| `peer_basis` | `list[str]` | `["job_level", "department"]` | Columns defining the peer group. |
| `stuckness_threshold` | float | `1.5` | Z-score above which an employee is flagged as "stuck". |

**Returns:** `pd.DataFrame` with columns `employee_id, starting_level, starting_dept, mobility_events, peer_median, peer_std, stuckness_zscore, is_stuck, is_mobility_leader`.

**Definition:**
- `mobility_events(E)` = count of (supervisor changes + level changes + dept changes) across the window.
- `stuckness_zscore(E)` = `(peer_median - mobility_events(E)) / peer_std`.
- `is_stuck` = zscore > `stuckness_threshold`.
- `is_mobility_leader` = zscore < -`stuckness_threshold`.

**Peer basis warning:** With small orgs (<500 employees), tight peer groups (level × dept × tenure) can have 1-2 people per cell — useless for a z-score. Default `["job_level", "department"]` is a reasonable balance.

### `manager_effectiveness(lookback="4Q", w_engagement=0.50, w_retention=0.25, w_promotion=0.15, w_span=0.10, survey_table=None, promotions_table=None)`

Composite manager effectiveness score.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lookback` | str | `"4Q"` | Period window. |
| `w_engagement` | float | `0.50` | Weight on engagement trend. |
| `w_retention` | float | `0.25` | Weight on retention rate. |
| `w_promotion` | float | `0.15` | Weight on promotion rate. |
| `w_span` | float | `0.10` | Weight on span efficiency. |
| `survey_table` | str | auto | Name of survey table. |
| `promotions_table` | str | auto | Name of promotions table. |

**Weights must sum to 1.0.**

**Returns:** `pd.DataFrame` with columns `manager_id, manager_level, n_periods_active, team_engagement_t1, team_engagement_tn, engagement_trend, retention_rate, promotion_rate, span_efficiency, peer_engagement_trend, peer_retention_rate, peer_promotion_rate, peer_span_efficiency, effectiveness_score, rank`.

**Definition (default):**
```
effectiveness(M) = 0.50 * z(engagement_trend(M))
                 + 0.25 * z(retention_rate(M))
                 + 0.15 * z(promotion_rate(M))
                 + 0.10 * z(span_efficiency(M))
```

Where:
- `engagement_trend(M)` = linear slope of team engagement over the window.
- `retention_rate(M)` = fraction of M's subtree that persists across periods.
- `promotion_rate(M)` = fraction of M's subtree promoted.
- `span_efficiency(M)` = 1 / average_subtree_size(M).

Z-scores are computed within each manager_level peer group.

---

## Query primitives (`dt.q.*`, 20 tools)

All primitives return either `pd.DataFrame` (terminal aggregations) or `DuckDBPyRelation` (composable traversals).

### Trajectory (4)

#### `dt.q.trajectory_at(employee_id, metric="betweenness", lookback="4Q")`

One employee, one metric, time-series. **Returns:** DataFrame with `period, value, delta, pct_change`.

#### `dt.q.trajectory_diff(employee_id, metric, period_t, period_t1)`

Single point diff between two periods. **Returns:** dict with `value_t, value_t1, delta, pct_change`.

#### `dt.q.trajectory_pivot(metric="betweenness", lookback="4Q")`

Wide format: rows = employees, columns = periods. **Returns:** DataFrame.

#### `dt.q.trajectory_rank(metric="pagerank", period=None, top_n=10)`

Top N at a single period. **Returns:** DataFrame with `rank, employee_id, value`.

### Hierarchy change (4)

#### `dt.q.edges_added(period_t, period_t1)`

New supervisor edges between two periods. **Returns:** Relation.

#### `dt.q.edges_removed(period_t, period_t1)`

Edges present at t but missing at t1. **Returns:** Relation with `employee_id, supervisor_id_at_t`.

#### `dt.q.node_set_diff(period_t, period_t1)`

Joined and left employees. **Returns:** dict `{"joined": DataFrame, "left": DataFrame}`.

#### `dt.q.hierarchy_drift(period_t, period_t1)`

Span-of-control changes per manager. **Returns:** DataFrame with `manager_id, direct_reports_t, direct_reports_t1, delta, total_reports_t, total_reports_t1, total_delta`.

### Subtree (4)

#### `dt.q.subtree_at(manager_id, period=None)`

All transitive descendants at a period. **Returns:** Relation with `manager_id, employee_id, depth, path`.

#### `dt.q.subtree_size_at(manager_id, period=None)`

Just the count. **Returns:** int.

#### `dt.q.subtree_growth(manager_id, lookback="4Q")`

Subtree size over time. **Returns:** DataFrame with `period, subtree_size, delta`.

#### `dt.q.subtree_overlap(mgr_a, mgr_b, period=None)`

Shared descendants between two managers. **Returns:** dict with `shared, only_a, only_b, jaccard` (float 0-1).

### Snapshot compare (4)

#### `dt.q.delta_table(period_t, period_t1, metric="betweenness")`

Per-employee delta for a metric between two periods. **Returns:** DataFrame with `employee_id, value_t, value_t1, delta, pct_change`.

#### `dt.q.new_centers(period_t, period_t1, metric="pagerank", top_n=10)`

Employees who joined the top-N list at t1 but not at t. **Returns:** DataFrame.

#### `dt.q.fallen_centers(period_t, period_t1, metric="pagerank", top_n=10)`

Inverse: dropped out of top-N. **Returns:** DataFrame.

#### `dt.q.cohort_compare(cohort_filter, metric, period_t, period_t1)`

Compare a SQL-filtered cohort across two periods.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cohort_filter` | str | required | DuckDB boolean expression, e.g. `"department = 'Engineering'"`. |
| `metric` | str | required | Metric name. |
| `period_t`, `period_t1` | str | required | Period labels. |

**Returns:** DataFrame with `employee_id, value_t, value_t1, delta, pct_change`.

### Window aggregate (4)

#### `dt.q.window_mean(metric, lookback="4Q")`

Org-wide mean per period. **Returns:** DataFrame with `period, mean_value, n_employees`.

#### `dt.q.window_trend(metric, lookback="4Q", aggregate="mean")`

Linear slope across the window.

**Returns:** dict with `slope, intercept, r_squared, direction ("up"|"down"|"flat"), periods`.

#### `dt.q.window_rank_change(metric, employee_id, lookback="4Q")`

Per-period rank for one employee. **Returns:** DataFrame with `period, value, rank, n_total`.

#### `dt.q.window_volatility(metric, lookback="4Q")`

Per-employee std-dev of a metric across the window. **Returns:** DataFrame with `employee_id, std_value, mean_value, n_periods`.

---

## Lookback string format

All lookback parameters accept strings like:

| Format | Example | Meaning |
|---|---|---|
| `<n>M` | `"6M"` | last 6 months |
| `<n>Q` | `"4Q"` | last 4 quarters |
| `<n>Y` | `"3Y"` | last 3 years |

If the lookback exceeds available periods, the method returns all available periods (no error).

---

## Return type conventions

| Return type | Used for | Examples |
|---|---|---|
| `DuckDBPyRelation` | Traversals that compose with SQL | `edges_added`, `edges_removed`, `subtree_at` |
| `pd.DataFrame` | Terminal aggregations | `trajectory_at`, `delta_table`, `mobility_leaderboard` |
| `int` | Single-number aggregates | `subtree_size_at` |
| `dict` | Mixed return shapes | `trajectory_diff`, `node_set_diff`, `window_trend`, `subtree_overlap` |

---

## Gotchas

1. **String vs integer employee IDs.** All public methods accept string IDs (`"E001"`) and integer IDs (`1`). The recursive CTE in `hierarchy_long` (used by `subtree_at`) has a known string-to-INT32 conversion issue with single-employee orgs. Workaround: load the data with integer IDs if you hit this.

2. **Empty period tables.** If a snapshot date has zero employees (e.g., between layoffs), that period is silently skipped. Methods that depend on the latest period fall back to the most recent non-empty period.

3. **Survey table is optional.** `manager_effectiveness` will use the survey table if loaded; otherwise it skips the engagement component and computes the rest with the configured weights renormalized. To get engagement-only scoring, set `w_engagement=1.0` and skip the survey load.

4. **Performance.** Betweenness and Louvain are O(n*m) for n nodes, m edges. For 10,000+ employees, expect seconds-to-minutes per period. Use `network_evolution` for cheaper aggregate metrics instead.

5. **Concurrent connections.** All methods use the single `dt.con` DuckDB connection. If you need to query the data from another Python process, register a new DuckDBPyConnection on the same file path.

---

*Added 2026-08-27. Companion to `pyduck-ona` v0.1.5+.*
