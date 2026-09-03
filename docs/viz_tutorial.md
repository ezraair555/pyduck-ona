# pyduck_ona.viz — Visualization Tutorial

> Publication-quality visualizations for organizational chart analysis and
> people analytics. Integrated into `pyduck-ona` as the `pyduck_ona.viz`
> subpackage (formerly the standalone `pyduck-ona-viz` package).

`pyduck_ona.viz` takes the DuckDB-relation outputs of `pyduck-ona`
(hierarchy stats, centrality frames, communities, attrition tables…) and
turns them into polished, presentation-ready figures.

- **Static matplotlib figures** for embedding into reports and slide decks.
- **Interactive HTML** (D3 + Plotly + pyvis) for exploratory dashboards.

The design language is consistent across every function: a deep-blue /
warm-gray / coral palette, no chartjunk, 11 pt axis labels, 16 pt titles,
150 DPI for screen / 300 DPI for print.

---

## Installation

```bash
pip install pyduck-ona[viz]
```

This pulls in matplotlib, plotly, pyvis, seaborn, and numpy. The rest of
`pyduck-ona` works without any of these — visualization is strictly opt-in.

---

## Quick start

```python
import pyduck_ona as pona
import pyduck_ona.viz as viz

# pyduck-ona produces DuckDB relations; .df() gives us pandas DataFrames.
long_df  = pona.hierarchy_long(rel, "employee_id", "supervisor_id").df()
stats_df = pona.hierarchy_stats(rel, "employee_id", "supervisor_id").df()

# 1. Span-of-control bar chart
fig = viz.span_of_control(stats_df, top_n=20)

# 2. Interactive org chart (HTML string)
html = viz.org_chart_tree(long_df, metadata=employees_df)

# 3. Single-page executive dashboard
html = viz.summary_dashboard(stats_df, betweenness=b.df(), pagerank=pr.df())
```

Every public function is also reachable from the top level without touching
the subpackage explicitly (resolved lazily):

```python
from pyduck_ona import org_chart_tree  # same function
```

---

## Functions

| Function | Output | Use case |
|---|---|---|
| `org_chart_tree` | Interactive HTML (D3) | Executive org chart with collapsible nodes. |
| `reporting_chain_walk` | matplotlib Figure | Clean path from any employee up to the top. |
| `span_of_control` | Figure or Plotly HTML | Top managers by direct reports. |
| `span_vs_depth` | Figure | Quadrant bubble chart (efficient / top-heavy / flat / deep). |
| `hierarchy_depth_heatmap` | Figure | Matrix of employees × levels. |
| `centrality_dashboard` | Figure (2×2) | Compare betweenness / PageRank / eigenvector / degree. |
| `silo_map` | HTML or Figure | Community-coloured network map. |
| `attrition_heatmap` | Figure | Department × level attrition rates. |
| `compensation_equity` | Figure | Tenure / level vs salary, with regression + outliers. |
| `summary_dashboard` | HTML | One-page executive dashboard. |

Per-function reference pages live in [docs/api/](api/).

---

## Examples

### Interactive org chart

```python
import pyduck_ona as pona
import pyduck_ona.viz as viz

long_df = pona.hierarchy_long(rel, "employee_id", "supervisor_id").df()
metadata = employees_df  # must contain employee_id + name + title + department

html = viz.org_chart_tree(
    long_df,
    metadata=metadata,
    color_by="department",
    title="Acme Corp · Q4 2026",
)
Path("org.html").write_text(html)
```

### Span of control

```python
fig = viz.span_of_control(
    stats_df,
    metadata=employees_df,
    top_n=15,
    color_by_department=True,
)
fig.savefig("span.png", dpi=300, bbox_inches="tight")
```

### Centrality dashboard

```python
fig = viz.centrality_dashboard(
    betweenness=b.df(),
    pagerank=pr.df(),
    eigenvector=ev.df(),
    degree=dg.df(),
    metadata=employees_df,
    top_n=10,
)
```

### Silo map

```python
# Interactive HTML
html = viz.silo_map(edges_df, communities=comms.df(), return_html=True)

# Static fallback for a slide deck
fig = viz.silo_map(edges_df, communities=comms.df(), return_html=False)
```

### Compensation equity

```python
fig = viz.compensation_equity(
    comp_df,
    x_col="tenure_years",
    y_col="salary",
    group_col="gender",
)
```

### Summary dashboard

```python
html = viz.summary_dashboard(
    hierarchy_stats=stats_df,
    betweenness=b.df(),
    pagerank=pr.df(),
    diversity=diversity_df,
    attrition=attrition_df,
)
Path("dashboard.html").write_text(html)
```

A runnable end-to-end demo lives at `examples/viz_demo.py`: it builds a
synthetic 300-person org with realistic hierarchy stats, runs every
visualization, and writes output to `examples/output/`.

---

## Design language

All functions share a single visual identity defined in
`pyduck_ona.viz.theme`:

- **Palette**: deep blue (`#1F3A5F`), coral accent (`#E27D60`), warm gray
  text (`#4D4D4D`), sage success (`#5B9279`), brick danger (`#C44536`).
- **Typography**: DejaVu Sans throughout. Titles 16 pt semibold, axes 11 pt,
  ticks 10 pt, annotations 9 pt.
- **Layout**: `constrained_layout=True` everywhere; no top/right spines
  on bar charts; only horizontal grid on bar charts.
- **DPI**: 150 default; pass `dpi=300` to `savefig` for print.
- **Categorical colour cycling**: deterministic from
  `pyduck_ona.viz.CATEGORICAL`.

If you need a different brand palette, copy `theme.py` and override
`PALETTE` / `CATEGORICAL` — every function reads from there.

---

## API patterns

Every visualization function:

1. **Accepts a DataFrame** (the `.df()` of a pyduck-ona relation) plus
   optional `metadata=` DataFrame keyed by employee id.
2. **Returns either** a `matplotlib.figure.Figure` or a `str` of HTML.
   Nothing is ever rendered to screen — `plt.show()` is never called.
3. **Validates input** column names and raises `KeyError` / `TypeError`
   with clear messages.

Interactive variants are exposed via `return_html=True` on the functions
that support them (`span_of_control`, `silo_map`).

---

## Migration from the standalone package

If you previously installed `pyduck-ona-viz`:

```python
# Before
import pyduck_ona_viz as viz

# After
import pyduck_ona.viz as viz
```

Function names, signatures, and return types are unchanged. The standalone
`pyduck-ona-viz` repository is deprecated and will not receive further
updates; all development continues in `pyduck-ona`.