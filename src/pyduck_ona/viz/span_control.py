"""
Span-of-control visualizations.

- :func:`span_of_control`  - Horizontal bar chart of direct reports per
                              manager with optional Plotly interactive twin.
- :func:`span_vs_depth`    - Quadrant bubble chart of span × depth with
                              bubble size = team size.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .theme import (
    PALETTE,
    apply_default_style,
    category_colors,
    configure_axes,
    new_figure,
    style_axis_labels,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prepare_stats(
    df: pd.DataFrame,
    *,
    id_col: str = "manager_id",
    metric_col: str = "direct_reports",
    label_col: str | None = None,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame")
    if id_col not in df.columns:
        raise KeyError(f"Column '{id_col}' not found in DataFrame")
    if metric_col not in df.columns:
        raise KeyError(f"Column '{metric_col}' not found in DataFrame")
    return df


# ---------------------------------------------------------------------------
# Span of control
# ---------------------------------------------------------------------------

def span_of_control(
    df: pd.DataFrame,
    *,
    id_col: str = "manager_id",
    metric_col: str = "direct_reports",
    label_col: str | None = None,
    metadata: pd.DataFrame | None = None,
    name_col: str = "name",
    department_col: str | None = None,
    top_n: int = 20,
    color_by_department: bool = False,
    title: str | None = None,
    figsize: tuple[float, float] = (10.0, 7.0),
    return_html: bool = False,
) -> Any:
    """Plot span of control for the top ``top_n`` managers.

    Parameters
    ----------
    df
        DataFrame with one row per manager (typically the output of
        ``pyduck_ona.hierarchy_stats(...)`` after ``.df()``).
    id_col
        Manager identifier column.
    metric_col
        Numeric metric column (default ``"direct_reports"``). Could also be
        ``"total_reports"`` for total team size.
    label_col
        Optional explicit label column in ``df``.
    metadata
        Optional per-manager metadata. ``name_col`` and ``department_col``
        are looked up here.
    color_by_department
        If True, colour each bar by the manager's department.
    return_html
        If True, return a Plotly Figure's HTML string (interactive) instead
        of a matplotlib Figure.

    Returns
    -------
    matplotlib.figure.Figure or str (HTML)
    """
    df = _prepare_stats(df, id_col=id_col, metric_col=metric_col)

    # Resolve names from metadata if available
    name_lookup: dict[str, str] = {}
    dept_lookup: dict[str, str] = {}
    if metadata is not None and id_col in metadata.columns:
        for _, row in metadata.iterrows():
            key = str(row[id_col])
            if name_col in metadata.columns:
                name_lookup[key] = str(row[name_col])
            if department_col and department_col in metadata.columns:
                v = row[department_col]
                if not pd.isna(v):
                    dept_lookup[key] = str(v)

    # Build the working frame, sorted descending by the metric
    work = df[[id_col, metric_col]].copy()
    work[metric_col] = work[metric_col].fillna(0).astype(float)
    work = work.sort_values(metric_col, ascending=False).head(int(top_n))
    work = work.sort_values(metric_col, ascending=True)  # for plotting

    # Labels
    if label_col and label_col in work.columns:
        labels = work[label_col].astype(str).tolist()
    elif name_lookup:
        labels = [name_lookup.get(str(k), str(k)) for k in work[id_col]]
    else:
        labels = work[id_col].astype(str).tolist()

    values = work[metric_col].astype(float).tolist()

    # Colors
    bar_colors: str | list[str]
    if color_by_department and dept_lookup:
        depts = [dept_lookup.get(str(k), "Unknown") for k in work[id_col]]
        distinct = sorted(set(depts))
        palette = {d: category_colors(len(distinct))[i]
                   for i, d in enumerate(distinct)}
        bar_colors = [palette[d] for d in depts]
    else:
        bar_colors = PALETTE["primary"]

    title = title or f"Span of Control · Top {int(top_n)} managers"
    subtitle = f"Sorted by {metric_col.replace('_', ' ')}"

    if return_html:
        return _span_of_control_plotly(
            labels=labels, values=values, bar_colors=bar_colors,
            title=title, subtitle=subtitle, color_by_department=color_by_department,
        )

    apply_default_style()
    fig, ax = new_figure(figsize=figsize)
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=bar_colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # largest at the top
    style_axis_labels(ax, xlabel=metric_col.replace("_", " ").title(), title=title)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes,
            fontsize=10, color=PALETTE["neutral_lt"], ha="left", va="bottom")
    configure_axes(ax, grid=True, grid_axis="x")

    # Annotate values at the tip of each bar
    vmax = max(values) if values else 1.0
    for bar, v in zip(bars, values, strict=False):
        ax.text(
            bar.get_width() + vmax * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(v):,}",
            va="center", ha="left",
            fontsize=9, color=PALETTE["neutral"],
        )

    # Median reference line
    if values:
        med = float(np.median(values))
        ax.axvline(med, color=PALETTE["accent"], linestyle="--", linewidth=1.2,
                   alpha=0.8, zorder=2)
        ax.text(med, len(labels) - 0.4, f" median = {med:.1f}",
                color=PALETTE["accent"], fontsize=9, ha="left", va="bottom")

    return fig


def _span_of_control_plotly(
    *,
    labels: list[str],
    values: list[float],
    bar_colors: str | list[str],
    title: str,
    subtitle: str,
    color_by_department: bool,
) -> str:
    import plotly.graph_objects as go

    marker = dict(
        color=bar_colors if isinstance(bar_colors, list) else PALETTE["primary"],
        line=dict(color="white", width=0.5),
    )
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=marker, hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:11px;color:#A6A6A6'>{subtitle}</span>",
                   x=0.01, xanchor="left"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="DejaVu Sans", color=PALETTE["neutral"]),
        xaxis=dict(title="Direct Reports",
                   gridcolor="#E5E5E5", zerolinecolor="#E5E5E5"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=180, r=40, t=80, b=50),
        height=600,
    )
    return str(fig.to_html(include_plotlyjs="cdn", full_html=True))


# ---------------------------------------------------------------------------
# Span vs Depth bubble chart
# ---------------------------------------------------------------------------

def span_vs_depth(
    df: pd.DataFrame,
    *,
    id_col: str = "manager_id",
    span_col: str = "direct_reports",
    depth_col: str = "levels_below",
    team_col: str = "total_reports",
    label_col: str | None = None,
    metadata: pd.DataFrame | None = None,
    name_col: str = "name",
    title: str | None = None,
    figsize: tuple[float, float] = (10.0, 7.5),
) -> Any:
    """Quadrant bubble chart of span × depth with team size as bubble area.

    Each bubble represents one manager. The four quadrants are:

    - **Efficient managers**: high span, low depth (broad but shallow teams).
    - **Top-heavy**:     low span, high depth (many layers of small teams).
    - **Flat leaders**:  low span, low depth (small flat groups).
    - **Deep & broad**:  high span, high depth (large tall pyramids).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if id_col not in df.columns:
        raise KeyError(f"Column '{id_col}' not found in DataFrame")
    for c in (span_col, depth_col, team_col):
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in DataFrame")

    name_lookup: dict[str, str] = {}
    if metadata is not None and id_col in metadata.columns and name_col in metadata.columns:
        for _, row in metadata.iterrows():
            name_lookup[str(row[id_col])] = str(row[name_col])

    work = df[[id_col, span_col, depth_col, team_col]].copy()
    work = work.fillna(0)
    work[span_col] = work[span_col].astype(float)
    work[depth_col] = work[depth_col].astype(float)
    work[team_col] = work[team_col].astype(float)

    if label_col and label_col in work.columns:
        labels = work[label_col].astype(str).tolist()
    elif name_lookup:
        labels = [name_lookup.get(str(k), str(k)) for k in work[id_col]]
    else:
        labels = work[id_col].astype(str).tolist()

    x = work[span_col].to_numpy()
    y = work[depth_col].to_numpy()
    # Scale bubble area: sqrt scaling keeps area proportional to value.
    sizes = np.sqrt(np.maximum(work[team_col].to_numpy(), 1.0)) * 22.0

    apply_default_style()
    fig, ax = new_figure(figsize=figsize)
    ax.scatter(x, y, s=sizes, alpha=0.65,
               color=PALETTE["secondary"], edgecolor=PALETTE["primary"],
               linewidths=0.8, zorder=3)

    # Quadrant reference lines (median split)
    if x.size and y.size:
        x_med, y_med = float(np.median(x)), float(np.median(y))
        ax.axvline(x_med, color=PALETTE["neutral_lt"], linestyle="--", linewidth=0.9, zorder=1)
        ax.axhline(y_med, color=PALETTE["neutral_lt"], linestyle="--", linewidth=0.9, zorder=1)
        # Quadrant labels
        x_pad = (x.max() - x.min()) * 0.04 or 0.1
        y_pad = (y.max() - y.min()) * 0.04 or 0.1
        ax.text(x_med + x_pad, y.max() - y_pad, "DEEP & BROAD",
                fontsize=9, fontweight="bold", color=PALETTE["neutral_lt"], ha="left")
        ax.text(x.max() - x_pad, y_med + y_pad, "EFFICIENT",
                fontsize=9, fontweight="bold", color=PALETTE["success"], ha="right")
        ax.text(x_med - x_pad, y.max() - y_pad, "TOP-HEAVY",
                fontsize=9, fontweight="bold", color=PALETTE["accent"], ha="right")
        ax.text(x.min() + x_pad, y_med + y_pad, "FLAT",
                fontsize=9, fontweight="bold", color=PALETTE["neutral_lt"], ha="left")

    # Annotate the largest bubbles (top 5 by team size)
    top_idx = np.argsort(work[team_col].to_numpy())[-5:]
    for i in top_idx:
        ax.annotate(
            labels[i],
            (x[i], y[i]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=8.5, color=PALETTE["primary"],
        )

    title = title or "Span vs Depth"
    style_axis_labels(
        ax,
        xlabel=span_col.replace("_", " ").title(),
        ylabel=depth_col.replace("_", " ").title(),
        title=title,
        subtitle=f"Bubble size = {team_col.replace('_', ' ')}",
    )
    configure_axes(ax, grid=True, grid_axis="both")

    # Legend for bubble sizes
    if x.size:
        for size_label, size_val in [("10", 10), ("50", 50), ("200", 200)]:
            ax.scatter([], [], s=np.sqrt(size_val) * 22,
                       color=PALETTE["secondary"], alpha=0.65,
                       edgecolor=PALETTE["primary"], linewidths=0.8,
                       label=f"{team_col.replace('_', ' ')} = {size_label}")
        ax.legend(loc="lower right", title="Bubble size",
                  fontsize=8.5, title_fontsize=9, frameon=True,
                  facecolor="white", edgecolor="#E5E5E5")

    return fig


__all__ = ["span_of_control", "span_vs_depth"]
