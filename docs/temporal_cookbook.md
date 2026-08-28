# Temporal ONA Cookbook — 10 Real-World Recipes

Workforce questions you can answer with `DuckONATemporal`, with code.

All recipes assume:
```python
import pandas as pd
from pyduck_ona import DuckONATemporal

dt = DuckONATemporal()
dt.load_snapshots(hris_df, snapshot_date_col="snapshot_date", freq="Q")
dt.load_survey(survey_df)          # if needed
dt.load_promotions(promotions_df)  # if needed
```

---

## 1. Who has moved up the most in the last year?

The headline mobility question. Composite score weights promotions heaviest.

```python
top_movers = dt.mobility_leaderboard(lookback="4Q", top_n=20)
print(top_movers[["employee_id", "mobility_score", "n_promotions",
                   "n_lateral_moves", "first_level", "last_level", "rank"]])
```

**Tune for your org's culture:**
```python
# Innovation org — weight lateral moves more (cross-team is healthy)
lb_innovation = dt.mobility_leaderboard(
    w_promotion=1.0, w_lateral=1.0, w_dept_change=0.5, w_demotion=-2.0
)

# Conservative org — promotions only, lateral moves deemphasized
lb_conservative = dt.mobility_leaderboard(
    w_promotion=2.0, w_lateral=0.2, w_dept_change=0.1, w_demotion=-1.0
)
```

---

## 2. Who has the most interesting career trajectory in the last 2 years?

Pick someone with high mobility_score and inspect their full path.

```python
top = dt.mobility_leaderboard(lookback="8Q", top_n=5)
for emp_id in top["employee_id"]:
    print(f"\n=== {emp_id} ===")
    print(dt.career_trajectory(emp_id, lookback="8Q"))
    print(dt.manager_chain(emp_id, lookback="8Q"))
```

**Use the output for:**
- Promotion packets ("here's the case for promoting X")
- Career development reviews ("here's the trajectory you've been on")
- Succession planning ("who is being groomed for senior roles?")

---

## 3. Who is stuck while peers around them are moving?

Peer-relative stuckness. Flags employees in the bottom decile of peer mobility.

```python
stuck = dt.mobility_anomaly(lookback="4Q")
# Most stuck first
stuck_top = stuck[stuck.is_stuck].sort_values("stuckness_zscore", ascending=False)
print(stuck_top[["employee_id", "starting_level", "starting_dept",
                  "mobility_events", "peer_median", "stuckness_zscore"]])
```

**Department-level stuckness rate** (is it concentrated or distributed?):
```python
dept_stuck = stuck.groupby("starting_dept").agg(
    n=("employee_id", "count"),
    n_stuck=("is_stuck", "sum"),
    pct_stuck=("is_stuck", "mean"),
).sort_values("pct_stuck", ascending=False)
print(dept_stuck)
```

**Inverse:** who is moving faster than peers?
```python
leaders = stuck[stuck.is_mobility_leader].sort_values("stuckness_zscore")
print(leaders.head(10))
```

---

## 4. Which manager has the highest effectiveness with rising engagement in their indirects?

The headline manager-effectiveness question.

```python
eff = dt.manager_effectiveness(lookback="4Q")
# Top 10 with positive engagement trend
top_eff = eff[eff.engagement_trend > 0].sort_values("effectiveness_score", ascending=False).head(10)
print(top_eff[["manager_id", "manager_level", "engagement_trend",
                "retention_rate", "promotion_rate", "effectiveness_score", "rank"]])
```

**Compare against peers:**
```python
# Show how the top 10 rank relative to all managers in the same level
eff["rank_in_level"] = eff.groupby("manager_level")["effectiveness_score"].rank(ascending=False)
top_eng = eff[eff.engagement_trend > 0].sort_values("effectiveness_score", ascending=False).head(20)
print(top_eng[["manager_id", "manager_level", "effectiveness_score", "rank_in_level"]])
```

---

## 5. Is the org getting more centralized over time?

Aggregate metric trend.

```python
trend = dt.q.window_trend("centralization", lookback="4Q")
# Wait — centralization isn't a per-employee metric. Use network_evolution instead.
```

Correct approach (centralization is a network-shape metric, not per-employee):
```python
ev = dt.network_evolution()
print(ev[["period", "centralization", "density", "n_components"]])

# Visualize: is centralization rising?
# (User can plot: ev.plot(x="period", y="centralization"))
```

**Interpretation:**
- Centralization rising → org becoming more top-down / star-shaped.
- Density rising → org becoming more interconnected.
- n_components rising → org fragmenting (silos, disconnected sub-orgs).

---

## 6. Who joined the top-10 betweenness list this quarter?

Detect rising stars.

```python
new_centers = dt.q.new_centers(
    period_t=dt.periods[-2],
    period_t1=dt.periods[-1],
    metric="betweenness",
    top_n=10,
)
print(new_centers[["employee_id", "value_t", "value_t1", "delta"]])
```

**Inverse:** who dropped out?
```python
fallen = dt.q.fallen_centers(
    period_t=dt.periods[-2],
    period_t1=dt.periods[-1],
    metric="betweenness",
    top_n=10,
)
print(fallen[["employee_id", "value_t", "value_t1", "delta"]])
```

---

## 7. Show me one employee's betweenness trajectory

Single employee, single metric, time-series.

```python
ts = dt.q.trajectory_at("E12345", "betweenness", lookback="4Q")
print(ts)
# → period, value, delta, pct_change
```

**Wide format for comparison:**
```python
# Pivot: rows = employees, columns = periods
piv = dt.q.trajectory_pivot("betweenness", lookback="4Q")
# Plot a few interesting employees
for emp in ["E12345", "E67890", "E11111"]:
    if emp in piv.index:
        piv.loc[emp].plot(title=f"Betweenness trajectory: {emp}")
```

---

## 8. What changed in the org chart between Q2 and Q3?

Edge-level structural changes.

```python
q2 = dt.periods[-2]
q3 = dt.periods[-1]

added = dt.q.edges_added(q2, q3).df()
removed = dt.q.edges_removed(q2, q3).df()
diff = dt.q.node_set_diff(q2, q3)

print(f"New supervisor edges: {len(added)}")
print(f"Dropped supervisor edges: {len(removed)}")
print(f"New hires: {len(diff['joined'])}")
print(f"Departures: {len(diff['left'])}")

# Which managers gained the most reports?
drift = dt.q.hierarchy_drift(q2, q3)
print(drift.sort_values("delta", ascending=False).head(5))
```

---

## 9. Are we losing people from the top-N?

Top-N attrition detection.

```python
# Pagerank top-10 at Q1
top_q1 = set(dt.q.trajectory_rank("pagerank", period=dt.periods[0], top_n=10)["employee_id"])
# Pagerank top-10 at latest
top_now = set(dt.q.trajectory_rank("pagerank", period=dt.periods[-1], top_n=10)["employee_id"])

lost = top_q1 - top_now     # dropped out
gained = top_now - top_q1   # joined the top

print(f"Lost from top-10: {lost}")
print(f"Joined top-10: {gained}")
```

---

## 10. Which department's managers have the most rising brokers?

Cross-cuts: cohort filter + mobility_leaderboard.

```python
lb = dt.mobility_leaderboard(lookback="4Q", top_n=100)

# For each manager, look up their starting department from the first period
first_period_emp = hris_df[hris_df.snapshot_date == hris_df.snapshot_date.min()][
    ["employee_id", "department"]
]
lb_with_dept = lb.merge(first_period_emp, on="employee_id", how="left")

# Top managers per department
top_per_dept = (
    lb_with_dept
    .sort_values("mobility_score", ascending=False)
    .groupby("department")
    .head(3)
)
print(top_per_dept[["department", "employee_id", "mobility_score", "n_promotions"]])
```

**Cohort-specific comparison** (use `cohort_compare`):
```python
eng_pagerank = dt.q.cohort_compare(
    cohort_filter="department = 'Engineering'",
    metric="pagerank",
    period_t=dt.periods[0],
    period_t1=dt.periods[-1],
)
print(eng_pagerank.head(10))
```

---

## Bonus: composability with SQL

For arbitrary workforce questions, drop down to SQL:

```python
# What is the average betweenness of VPs in each quarter?
vp_avg = dt.con.sql("""
    SELECT
        date_trunc('quarter', t.snapshot_date) AS period,
        AVG(t.betweenness) AS avg_vp_betweenness
    FROM (
        SELECT * FROM dt_temp
        WHERE employee_id LIKE 'E_VP%'
    ) t
    GROUP BY period
    ORDER BY period
""").df()
```

(The above assumes you've registered a `dt_temp` table with betweenness values; see `compute_temporal_metrics` for the structure.)

---

*Added 2026-08-27. Companion to `pyduck-ona` v0.1.5+.*
