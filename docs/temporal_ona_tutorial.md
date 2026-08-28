# Temporal ONA: Analyzing Organizational Change Over Time

This tutorial walks through `DuckONATemporal`, pyduck-ona's temporal
analysis class. It answers the "deep questions" that require computing
network metrics across multiple HRIS snapshots and comparing them
period-over-period — things a normal SQL query cannot do.

## Quick start

```python
import pandas as pd
from pyduck_ona import DuckONATemporal

# Load your HRIS snapshots (one row per employee per snapshot date)
hris = pd.read_csv("hris_snapshots.csv")

dt = DuckONATemporal()
dt.load_snapshots(hris, snapshot_date_col="snapshot_date", freq="Q")

# Optional: load engagement survey and promotions for manager_effectiveness
dt.load_survey(survey_df)
dt.load_promotions(promotions_df)
```

## The eight methods

### 1. `compute_temporal_metrics` — per-employee ONA time-series

Computes betweenness, pagerank, degree centrality, or team_size for
every employee at every period, with delta and pct_change vs. the
previous period.

```python
ts = dt.compute_temporal_metrics(
    metrics=["betweenness", "pagerank", "team_size"]
)
# → period, employee_id, metric, value, prev_value, delta, pct_change

# "Who gained the most betweenness?"
ts[(ts.metric == "betweenness")].sort_values("delta", ascending=False).head(10)
```

### 2. `network_evolution` — structural drift

One row per period with aggregate network-shape metrics: density,
centralization, number of components, average path length.

```python
ev = dt.network_evolution()
# → period, n_employees, n_edges, density, centralization,
#   n_components, avg_path_length

# "Is the org centralizing or fragmenting?"
ev.plot(x="period", y=["density", "centralization"])
```

### 3. `event_window` — before/after a specific date

Computes ONA metrics for periods before and after an event (reorg,
acquisition, leadership change).

```python
ev = dt.event_window(
    event_date="2025-07-01",
    metrics=["betweenness", "pagerank"],
)
# → period, period_type ("pre"|"post"), employee_id, metric, value

# Compare pre vs. post means
ev.groupby(["period_type", "metric"])["value"].mean()
```

### 4. `change_detection` — top movers

Ranks employees by absolute change in a metric over the lookback window.

```python
movers = dt.change_detection(metric="betweenness", top_n=20, lookback="4Q")
# → employee_id, start_value, end_value, delta, pct_change,
#   rolling_zscore, rank
```

### 5. `mobility_leaderboard` — who moved up the most

Composite mobility score across periods:

```
score = 1.0 * n_promotions
      + 0.5 * n_lateral_moves
      + 0.3 * n_dept_changes
      - 1.0 * n_demotions
```

```python
lb = dt.mobility_leaderboard(lookback="4Q", top_n=20)
# → employee_id, mobility_score, n_promotions, n_lateral_moves,
#   n_dept_changes, n_demotions, n_manager_changes,
#   first_level, last_level, first_dept, last_dept, rank
```

Weights are tunable:

```python
# Engagement-focused org? Weight promotions heavier:
lb = dt.mobility_leaderboard(
    w_promotion=2.0, w_lateral=0.5, w_dept_change=0.3, w_demotion=-2.0
)
```

### 6. `career_trajectory` + `manager_chain` — the path

For a single employee, shows their full career path across periods:

```python
traj = dt.career_trajectory("E12345", lookback="4Q")
# → period, supervisor_id, job_level, department,
#   promoted, transferred, manager_changed

chain = dt.manager_chain("E12345", lookback="4Q")
# → period, supervisor_id, supervisor_name, supervisor_level,
#   supervisor_path_to_ceo
```

### 7. `mobility_anomaly` — who is stuck while peers move?

Peer-relative stuckness z-score. Peers are defined by starting level +
starting department (configurable).

```python
stuck = dt.mobility_anomaly(lookback="4Q")
# → employee_id, starting_level, starting_dept, mobility_events,
#   peer_median, peer_std, stuckness_zscore, is_stuck, is_mobility_leader

# Who is most stuck?
stuck[stuck.is_stuck].sort_values("stuckness_zscore", ascending=False)

# Per-department stuckness rate
stuck.groupby("starting_dept").agg(
    n=("employee_id", "count"),
    n_stuck=("is_stuck", "sum"),
    pct_stuck=("is_stuck", "mean"),
)
```

### 8. `manager_effectiveness` — the composite score

**This is the headline method.** For each manager, it:

1. Enumerates their transitive subtree (all indirect reports) at each
   period via `hierarchy_long`.
2. Aggregates team engagement from the survey table.
3. Computes the engagement trend (linear slope over time).
4. Computes retention rate (fraction of subtree that persists).
5. Computes promotion rate (fraction of subtree promoted).
6. Computes span efficiency (1 / average team size).
7. Peer-normalizes each metric against same-level managers.
8. Composites into a single score.

Default weights (engagement-dominant):

```
effectiveness = 0.50 * z(engagement_trend)
              + 0.25 * z(retention_rate)
              + 0.15 * z(promotion_rate)
              + 0.10 * z(span_efficiency)
```

```python
eff = dt.manager_effectiveness(lookback="4Q")
# → manager_id, manager_level, n_periods_active,
#   team_engagement_t1, team_engagement_tn, engagement_trend,
#   retention_rate, promotion_rate, span_efficiency,
#   peer_engagement_trend, peer_retention_rate,
#   peer_promotion_rate, effectiveness_score, rank

# "Which manager has the highest effectiveness score
#  AND is improving engagement with their indirects?"
eff.sort_values("effectiveness_score", ascending=False).head(10)
```

Custom weights — all exposed as kwargs:

```python
# Equal-weighted (simpler, more transparent)
eff = dt.manager_effectiveness(
    w_engagement=0.25, w_retention=0.25,
    w_promotion=0.25, w_span=0.25,
)

# Engagement-only (ignores retention, promotion, span)
eff = dt.manager_effectiveness(
    w_engagement=1.0, w_retention=0.0,
    w_promotion=0.0, w_span=0.0,
)
```

## Composing methods

The methods are designed to compose. Example: "Find managers whose
engagement is rising AND whose team has low mobility (stable + improving)":

```python
eff = dt.manager_effectiveness(lookback="4Q")
stuck = dt.mobility_anomaly(lookback="4Q")

# Managers with positive engagement trend
good_eng = eff[eff["engagement_trend"] > 0]

# Their team members with low mobility (stable)
# Join effectiveness → manager's subtree → mobility
# (requires a custom join, but the pieces are all DuckDB relations)
```

## Input format

`DuckONATemporal` accepts a single HRIS DataFrame with a
`snapshot_date` column — the realistic shape for Workday / SAP /
SuccessFactors delta extracts. One row per employee per snapshot.

Required columns:
- `employee_id` — unique employee identifier
- `supervisor_id` — manager identifier (NULL for root)
- `snapshot_date` — date of the HRIS extract

Optional columns (used by specific methods):
- `job_level` — used by mobility_leaderboard, career_trajectory
- `department` — used by mobility_leaderboard, mobility_anomaly
- `engagement` — used by manager_effectiveness (or load via survey table)
- `name` — used by manager_chain for supervisor display name

## Testing

```bash
python -m pytest tests/integration/test_temporal.py tests/integration/test_temporal_primitives.py -v
# 58 tests, all passing
```

The test suite includes a simulation test (Principle #9) that plants a
known engagement trend (one manager's team rising, another's declining)
and verifies that `manager_effectiveness` ranks them correctly.

---

## Query primitives (`dt.q.*`)

20 composable query primitives for hierarchy trends and over-time
changes. Organized under `dt.q` so they're discoverable separately
from the 8 analytical methods on `dt` itself.

### Trajectory primitives

```python
# One employee, one metric, time-series
ts = dt.q.trajectory_at("E12345", "betweenness", lookback="4Q")
# → period, value, delta, pct_change

# Single point diff between two periods
d = dt.q.trajectory_diff("E12345", "pagerank", "2025-01-01", "2025-10-01")
# → {employee_id, metric, value_t, value_t1, delta, pct_change}

# Wide format: one row per employee, columns = periods
piv = dt.q.trajectory_pivot("betweenness", lookback="4Q")

# Top-N at a single period
top = dt.q.trajectory_rank("pagerank", period="2025-10-01", top_n=10)
```

### Hierarchy-change primitives

```python
# New supervisor edges between two periods (relations)
new_edges = dt.q.edges_added("2025-01-01", "2025-10-01")
dropped = dt.q.edges_removed("2025-01-01", "2025-10-01")

# Employees who joined or left
diff = dt.q.node_set_diff("2025-01-01", "2025-10-01")
diff["joined"]  # DataFrame of new hires
diff["left"]    # DataFrame of departures

# Span-of-control changes per manager
drift = dt.q.hierarchy_drift("2025-01-01", "2025-10-01")
# → manager_id, direct_reports_t, direct_reports_t1, delta, total_delta
```

### Subtree primitives

```python
# All transitive descendants at a period (relation)
team = dt.q.subtree_at("VP0", "2025-10-01")

# Just the count
size = dt.q.subtree_size_at("VP0", "2025-10-01")  # → int

# Subtree size over time
growth = dt.q.subtree_growth("VP0", lookback="4Q")

# Shared descendants between two managers (Jaccard)
overlap = dt.q.subtree_overlap("VP0", "VP1", "2025-10-01")
overlap["jaccard"]  # → float in [0, 1]
```

### Snapshot-comparison primitives

```python
# Per-employee delta for any metric between two periods
deltas = dt.q.delta_table("2025-01-01", "2025-10-01", metric="pagerank")

# Employees who joined the top-N at t1 (but not at t)
new_top = dt.q.new_centers("2025-01-01", "2025-10-01", metric="pagerank", top_n=10)

# Inverse: dropped out of top-N
fallen = dt.q.fallen_centers("2025-01-01", "2025-10-01", metric="pagerank", top_n=10)

# Compare a SQL-filtered cohort across two periods
eng_pagerank = dt.q.cohort_compare(
    cohort_filter="department = 'Engineering'",
    metric="pagerank",
    period_t="2025-01-01",
    period_t1="2025-10-01",
)
```

### Time-window aggregate primitives

```python
# Org-wide mean per period
mean_b = dt.q.window_mean("betweenness", lookback="4Q")

# Linear slope across the window (is it trending up/down/flat?)
trend = dt.q.window_trend("pagerank", lookback="4Q")
trend["slope"]      # → float
trend["direction"]  # → "up" | "down" | "flat"

# Per-period rank for one employee
rank = dt.q.window_rank_change("pagerank", "E12345", lookback="4Q")
# → period, value, rank, n_total

# Per-employee volatility (std across periods)
vol = dt.q.window_volatility("betweenness", lookback="4Q")
```

### Composing primitives

The primitives compose with each other and with the analytical methods.
Example: "Find managers whose engagement is rising AND whose team
members are stuck (stable teams + improving engagement)":

```python
eff = dt.manager_effectiveness(lookback="4Q")        # analytical
stuck = dt.q.mobility_anomaly(lookback="4Q")         # analytical
team = dt.q.subtree_at("VP0", dt.periods[-1]).df()   # primitive

# Top 5 by effectiveness, all of whose subtree members are stuck
top_eff = eff.head(5)
for _, mgr in top_eff.iterrows():
    team_ids = team[team["manager_id"] == mgr["manager_id"]]["employee_id"]
    team_stuck = stuck[stuck["employee_id"].isin(team_ids)]
    print(mgr["manager_id"], "team_stuckness_rate:", team_stuck["is_stuck"].mean())
```

---

## See also

- **[API Reference](temporal_api_reference.md)** — full per-method signature, parameters, return types, gotchas.
- **[Cookbook](temporal_cookbook.md)** — 10 real-world query recipes (mobility leaderboard, stuckness, manager effectiveness, centralization trends, etc.).

## Testing

```bash
pytest tests/integration/test_temporal.py -v               # 29 analytical method tests
pytest tests/integration/test_temporal_primitives.py -v    # 29 query primitive tests
pytest tests/integration/test_temporal_simulation.py -v    # 10 Principle #9 DGP tests
pytest tests/integration/test_temporal_properties.py -v    # 14 hypothesis + edge-case tests
pytest tests/integration/test_temporal_performance.py -v   # 6 scaling benchmarks
```

---

*Added 2026-08-27. Companion to `pyduck-ona` v0.1.5+.*