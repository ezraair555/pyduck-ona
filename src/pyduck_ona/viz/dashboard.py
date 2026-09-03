"""
One-page HTML summary dashboard combining key org metrics.

- :func:`summary_dashboard` - Returns a standalone HTML document with
  interactive Plotly panels for org structure, span metrics, centrality,
  diversity, and attrition.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .theme import PALETTE, category_colors


def summary_dashboard(
    hierarchy_stats: pd.DataFrame,
    *,
    betweenness: pd.DataFrame | None = None,
    pagerank: pd.DataFrame | None = None,
    diversity: pd.DataFrame | None = None,
    attrition: pd.DataFrame | None = None,
    id_col: str = "manager_id",
    direct_reports_col: str = "direct_reports",
    total_reports_col: str = "total_reports",
    levels_below_col: str = "levels_below",
    department_col: str | None = None,
    title: str = "People Analytics Summary Dashboard",
    subtitle: str = "Organizational structure, span, centrality, diversity, attrition",
) -> str:
    """Build a single-page HTML summary dashboard.

    Parameters
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

    Returns
    -------
    str
        Full HTML document.
    """

    panels_html: list[str] = []

    # ── 1. Headline KPIs ────────────────────────────────────────────────
    n_managers = int((hierarchy_stats[direct_reports_col].fillna(0) > 0).sum())
    total_headcount = int(hierarchy_stats[total_reports_col].fillna(0).sum()) + n_managers
    avg_span = float(hierarchy_stats[direct_reports_col].fillna(0).mean())
    max_depth = int(hierarchy_stats[levels_below_col].fillna(0).max())

    panels_html.append(_kpi_cards_html([
        ("Headcount",        f"{total_headcount:,}"),
        ("Managers",         f"{n_managers:,}"),
        ("Avg span",         f"{avg_span:.1f}"),
        ("Max depth",        f"{max_depth}"),
    ]))

    # ── 2. Span-of-control bar chart ────────────────────────────────────
    fig_span = _span_bar_fig(hierarchy_stats, direct_reports_col)
    panels_html.append(_panel_card(
        "Span of Control",
        "Top 20 managers by direct reports.",
        fig_span.to_html(include_plotlyjs=False, full_html=False, div_id="span_div"),
    ))

    # ── 3. Span distribution ─────────────────────────────────────────────
    fig_dist = _span_dist_fig(hierarchy_stats, direct_reports_col)
    panels_html.append(_panel_card(
        "Span Distribution",
        "Histogram of how many direct reports each manager carries.",
        fig_dist.to_html(include_plotlyjs=False, full_html=False, div_id="dist_div"),
    ))

    # ── 4. Top brokers / PageRank ────────────────────────────────────────
    if betweenness is not None and len(betweenness) > 0:
        fig_b = _centrality_fig(betweenness, "betweenness", "Top Brokers (Betweenness)",
                                 color=PALETTE["accent"])
        panels_html.append(_panel_card(
            "Top Brokers",
            "Highest betweenness centrality — information flow control.",
            fig_b.to_html(include_plotlyjs=False, full_html=False, div_id="betw_div"),
        ))

    if pagerank is not None and len(pagerank) > 0:
        fig_p = _centrality_fig(pagerank, "pagerank", "Most Influential (PageRank)",
                                color=PALETTE["primary"])
        panels_html.append(_panel_card(
            "Most Influential",
            "PageRank — random-walk influence.",
            fig_p.to_html(include_plotlyjs=False, full_html=False, div_id="pr_div"),
        ))

    # ── 5. Diversity breakdown ───────────────────────────────────────────
    if diversity is not None and len(diversity) > 0:
        fig_div = _diversity_fig(diversity)
        panels_html.append(_panel_card(
            "Diversity Mix",
            "Headcount by demographic group.",
            fig_div.to_html(include_plotlyjs=False, full_html=False, div_id="div_div"),
        ))

    # ── 6. Attrition heatmap ─────────────────────────────────────────────
    if attrition is not None and len(attrition) > 0:
        fig_attr = _attrition_fig(attrition)
        panels_html.append(_panel_card(
            "Attrition",
            "Department × job-level attrition rate.",
            fig_attr.to_html(include_plotlyjs=False, full_html=False, div_id="attr_div"),
            wide=True,
        ))

    return _wrap_html(title, subtitle, panels_html)


# ---------------------------------------------------------------------------
# Internal Plotly panel builders
# ---------------------------------------------------------------------------

def _span_bar_fig(stats: pd.DataFrame, col: str) -> Any:
    import plotly.graph_objects as go
    work = stats[[col]].dropna().sort_values(col, ascending=False).head(20)
    work = work.sort_values(col, ascending=True)
    fig = go.Figure(go.Bar(
        x=work[col], y=[str(i) for i in work.index],
        orientation="h",
        marker=dict(color=PALETTE["primary"]),
        hovertemplate="%{y}: %{x} direct reports<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=120, r=20, t=20, b=40),
        height=420, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#E5E5E5", zerolinecolor="#E5E5E5",
                   title="Direct reports"),
        yaxis=dict(title=""),
    )
    return fig


def _span_dist_fig(stats: pd.DataFrame, col: str) -> Any:
    import plotly.graph_objects as go
    vals = stats[col].fillna(0).astype(int)
    fig = go.Figure(go.Histogram(
        x=vals,
        nbinsx=min(20, max(5, vals.nunique())),
        marker=dict(color=PALETTE["secondary"]),
        hovertemplate="span %{x}: %{y} managers<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        height=320, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#E5E5E5", title="Direct reports"),
        yaxis=dict(gridcolor="#E5E5E5", title="Managers"),
        bargap=0.08,
    )
    return fig


def _centrality_fig(df: pd.DataFrame, col: str, name: str, *, color: str) -> Any:
    import plotly.graph_objects as go
    work = df.sort_values(col, ascending=False).head(15)
    work = work.sort_values(col, ascending=True)
    fig = go.Figure(go.Bar(
        x=work[col], y=work.iloc[:, 0].astype(str),
        orientation="h",
        marker=dict(color=color),
        hovertemplate="%{y}: %{x:.3g}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=120, r=20, t=20, b=40),
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#E5E5E5", title=col),
        yaxis=dict(title=""),
    )
    return fig


def _diversity_fig(df: pd.DataFrame) -> Any:
    import plotly.graph_objects as go

    # Heuristic: pick the first non-id string column as group, the first
    # numeric column as count.
    group_col = None
    count_col = None
    for c in df.columns:
        if c.lower() in {"group", "category", "department", "gender"} and group_col is None:
            group_col = c
    if group_col is None:
        for c in df.columns:
            if df[c].dtype == object:
                group_col = c
                break
    for c in df.columns:
        if c != group_col and pd.api.types.is_numeric_dtype(df[c]):
            count_col = c
            break
    if group_col is None or count_col is None:
        # Fallback: empty figure
        return go.Figure()

    work = df[[group_col, count_col]].dropna().sort_values(count_col, ascending=False)
    colors = category_colors(len(work))
    fig = go.Figure(go.Bar(
        x=work[group_col].astype(str),
        y=work[count_col],
        marker=dict(color=colors),
        hovertemplate="%{x}: %{y}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        height=320, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title=""),
        yaxis=dict(gridcolor="#E5E5E5", title=count_col),
    )
    return fig


def _attrition_fig(df: pd.DataFrame) -> Any:
    import plotly.graph_objects as go

    # Heuristic: department, level, rate
    dept_col = next((c for c in df.columns if "dept" in c.lower()), df.columns[0])
    level_col = next((c for c in df.columns if "level" in c.lower()), df.columns[1])
    rate_col = next((c for c in df.columns if c.lower() in {"rate", "attrition"}), None)
    if rate_col is None:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]) and c not in (dept_col, level_col):
                rate_col = c
                break
    if rate_col is None:
        return go.Figure()

    pivot = df.pivot(index=dept_col, columns=level_col, values=rate_col)
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[str(c) for c in pivot.index],
        colorscale=[[0.0, "#5B9279"], [0.5, "#F5E6D3"], [1.0, "#C44536"]],
        zmin=0, zmax=1,
        hovertemplate="%{y} · %{x}: %{z:.0%}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=120, r=20, t=20, b=40),
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title=level_col),
        yaxis=dict(title=dept_col, autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _panel_card(title: str, subtitle: str, plot_html: str, *, wide: bool = False) -> str:
    return f"""
    <section class="card{' wide' if wide else ''}">
      <h2>{title}</h2>
      <p class="sub">{subtitle}</p>
      <div class="plot">{plot_html}</div>
    </section>
    """


def _kpi_cards_html(kpis: list[tuple[str, str]]) -> str:
    cells = "\n".join(
        f'<div class="kpi"><div class="kpi-value">{v}</div><div class="kpi-label">{k}</div></div>'
        for k, v in kpis
    )
    return f'<section class="kpi-row">{cells}</section>'


def _wrap_html(title: str, subtitle: str, panels_html: list[str]) -> str:
    panels = "\n".join(panels_html)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{
    margin: 0;
    font-family: 'DejaVu Sans', 'Helvetica Neue', Arial, sans-serif;
    background: {PALETTE["background"]};
    color: {PALETTE["neutral"]};
  }}
  header {{
    padding: 24px 32px;
    background: white;
    border-bottom: 1px solid #E5E5E5;
  }}
  header h1 {{
    margin: 0;
    font-size: 22px;
    font-weight: 600;
    color: {PALETTE["primary"]};
  }}
  header p {{
    margin: 4px 0 0 0;
    font-size: 12px;
    color: {PALETTE["neutral_lt"]};
  }}
  main {{
    padding: 24px 32px;
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
  }}
  .card {{
    background: white;
    border: 1px solid #E5E5E5;
    border-radius: 8px;
    padding: 18px 20px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }}
  .card.wide {{ grid-column: span 2; }}
  .card h2 {{
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: {PALETTE["primary"]};
  }}
  .card .sub {{
    margin: 2px 0 12px 0;
    font-size: 11px;
    color: {PALETTE["neutral_lt"]};
  }}
  .kpi-row {{
    grid-column: span 2;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
  }}
  .kpi {{
    background: white;
    border: 1px solid #E5E5E5;
    border-radius: 8px;
    padding: 16px 20px;
  }}
  .kpi-value {{
    font-size: 26px;
    font-weight: 600;
    color: {PALETTE["primary"]};
  }}
  .kpi-label {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {PALETTE["neutral_lt"]};
    margin-top: 4px;
  }}
  footer {{
    padding: 16px 32px 32px;
    color: {PALETTE["neutral_lt"]};
    font-size: 11px;
  }}
</style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>{subtitle}</p>
  </header>
  <main>
    {panels}
  </main>
  <footer>Generated by pyduck-ona-viz.</footer>
</body>
</html>
"""


__all__ = ["summary_dashboard"]
