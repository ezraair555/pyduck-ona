# DuckONATemporal.insight_report

Build an [`ONAInsightReport`](../insight_reporting_tutorial.md) from a loaded
`DuckONATemporal` workspace. The report decomposes endpoint structural changes,
compares metric movement for affected and unaffected employees, and summarizes
movement by demographic fields with configurable small-cell suppression.

```python
report = dt.insight_report(
    lookback="8Q",
    metrics=["betweenness", "pagerank", "team_size"],
    demographic_columns=["department", "job_level"],
    min_group_size=10,
)
report.save("ona-brief.html")
```

Driver effects are descriptive associations and must not be interpreted as
causal estimates without an explicit identification design.
