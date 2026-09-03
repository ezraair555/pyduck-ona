# `summary_dashboard`

**Module:** `pyduck_ona.viz.dashboard`

## Signature

```python
summary_dashboard(hierarchy_stats'pd.DataFrame', betweenness'pd.DataFrame | None'=None, pagerank'pd.DataFrame | None'=None, diversity'pd.DataFrame | None'=None, attrition'pd.DataFrame | None'=None, id_col'str'='manager_id', direct_reports_col'str'='direct_reports', total_reports_col'str'='total_reports', levels_below_col'str'='levels_below', department_col'str | None'=None, title'str'='People Analytics Summary Dashboard', subtitle'str'='Organizational structure, span, centrality, diversity, attrition')
```

## Description

Build a single-page HTML summary dashboard

## Parameters

----------
hierarchy_stats
    The output of ``pyduck_ona.hierarchy_stats(...)`` after ``.df()``.
betweenness, pagerank
    Optional centrality frames to plot.
diversity
    Optional long-form diversity table with ``group_col`` and a count
    column. Auto-detects columns named ``group`` + ``count``.
attrition
    Optional attrition table, auto-detecting ``department`` / ``level``
    / ``rate`` / ``count`` columns.

## Returns

-------
str
    Full HTML document.

## Example

```python
import pyduck_ona as pona
# TODO: add a runnable example
```

---

[Back to API catalog](../README.md#api-catalog)
