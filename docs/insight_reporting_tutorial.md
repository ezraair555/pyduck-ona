# Explainable ONA Insight Reports

`DuckONATemporal.insight_report()` turns a longitudinal HR hierarchy into an
aggregate-first brief. It is designed for questions such as:

- Which structural changes coincided with large network-position shifts?
- Did manager changes, exits, or department moves affect network metrics?
- Which demographic groups experienced different metric movement?
- Can the result be shared without exposing employee-level rows by default?

```python
from pyduck_ona import DuckONATemporal

ona = DuckONATemporal()
ona.load_snapshots(
    hris_snapshots,
    snapshot_date_col="snapshot_date",
    employee_id_col="employee_id",
    supervisor_id_col="supervisor_id",
    freq="Q",
)

report = ona.insight_report(
    lookback="8Q",
    metrics=["betweenness", "pagerank", "team_size"],
    demographic_columns=["department", "job_level", "gender"],
    min_group_size=10,
)

print(report.headline)
print(report.driver_summary)
print(report.driver_effects)
print(report.demographic_summary)
report.save("ona-brief.html")
```

## What the report computes

`driver_summary` counts endpoint changes: hires, exits, manager changes,
department changes, and level changes. `metric_changes` compares each
employee's first and last available metric values. `driver_effects` compares
metric movement for people affected and unaffected by each driver. These are
investigative associations, not causal estimates.

`demographic_summary` joins the final-period demographic attributes to metric
movement and reports group size, mean/median movement, and the share with
positive movement. Groups smaller than `min_group_size` are marked
`[suppressed]` and their aggregate values are withheld.

Markdown and HTML rendering are aggregate-first. Employee-level movers are
available to an authorized analyst with `include_individual=True`, but should
not be used for broad distribution without reviewing the fields and role
permissions.

## Why this is more than a metric table

A centrality score says where a person sits in a graph. An insight report puts
that movement beside the organizational events that could plausibly explain
it and shows whether the pattern is concentrated in a population. The report
still does not claim that a manager move caused a change; causal claims require
an explicit design, controls, and outcome timing.
