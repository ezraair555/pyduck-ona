"""
People-analytics visualizations.

- :func:`attrition_heatmap`   - Department × job-level grid coloured by
                                attrition rate. Cells annotated with counts.
- :func:`compensation_equity` - Tenure / level vs salary scatter with
                                regression overlays and outlier flags.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .theme import (
    DIVERG_RYG,
    PALETTE,
    apply_default_style,
    category_colors,
    configure_axes,
    new_figure,
    style_axis_labels,
)

# ---------------------------------------------------------------------------
# Attrition heatmap
# ---------------------------------------------------------------------------

def attrition_heatmap(
    df: pd.DataFrame,
    *,
    department_col: str = "department",
    level_col: str = "job_level",
    attrition_col: str = "attrition",
    value_col: str | None = None,
    count_col: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10.0, 6.5),
    cmap: str | None = None,
) -> Any:
    """Render a department × job-level attrition heatmap.

    The DataFrame should already be aggregated to one row per (department,
    job_level) combination. If you pass the raw employee-level DataFrame,
    the function will aggregate it automatically using ``value_col`` for the
    rate and ``count_col`` for the cell counts (defaults: ``value_col`` is
    derived from ``attrition_col``; ``count_col`` defaults to row count).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame")
    if department_col not in df.columns or level_col not in df.columns:
        raise KeyError(f"Need both '{department_col}' and '{level_col}' in DataFrame")

    if value_col is None:
        if attrition_col in df.columns:
            # Aggregate raw rows
            agg = (
                df.groupby([department_col, level_col], dropna=False)
                  .agg(rate=(attrition_col, "mean"),
                       n=(attrition_col, "size"))
                  .reset_index()
            )
            value_col = "rate"
            count_col = "n"
        else:
            # Pre-aggregated: look for a sensible value column.
            candidates = [
                c for c in df.columns
                if c not in (department_col, level_col)
                and pd.api.types.is_numeric_dtype(df[c])
            ]
            if not candidates:
                raise KeyError(
                    f"Need '{attrition_col}' (raw) or pass value_col=... "
                    "(pre-aggregated)"
                )
            value_col = candidates[0]
            agg = df.copy()
    else:
        agg = df.copy()
    if count_col is None:
        candidates = [
            c for c in agg.columns
            if c not in (department_col, level_col, value_col)
            and pd.api.types.is_numeric_dtype(agg[c])
        ]
        if candidates:
            # Prefer a column literally named "count" or "n".
            for pref in ("count", "n", "headcount", "total"):
                hit = next((c for c in candidates if c.lower() == pref), None)
                if hit:
                    count_col = hit
                    break
        if count_col is None:
            agg["__count__"] = 1
            count_col = "__count__"

    if value_col not in agg.columns or count_col not in agg.columns:
        raise KeyError(f"Need columns '{value_col}' and '{count_col}' in aggregated frame")

    # Build a pivot
    pivot_val = agg.pivot(index=department_col, columns=level_col, values=value_col)
    pivot_cnt = agg.pivot(index=department_col, columns=level_col, values=count_col)

    # Sort departments by total headcount (largest first), levels ascending
    pivot_cnt = pivot_cnt.fillna(0)
    pivot_val = pivot_val.loc[pivot_cnt.sum(axis=1).sort_values(ascending=False).index]

    # Sort level columns numerically when possible
    try:
        sorted_cols = sorted(pivot_val.columns, key=lambda c: float(c))
    except (TypeError, ValueError):
        sorted_cols = sorted(pivot_val.columns, key=str)
    pivot_val = pivot_val[sorted_cols]
    pivot_cnt = pivot_cnt[sorted_cols]

    matrix = pivot_val.to_numpy(dtype=float)
    counts = pivot_cnt.to_numpy(dtype=float)

    apply_default_style()
    fig, ax = new_figure(figsize=figsize)
    cmap_obj = DIVERG_RYG if cmap is None else _resolve_cmap(cmap)
    # Use a fixed 0-1 range so the colour scale is comparable across charts.
    vmin, vmax = 0.0, 1.0
    im = ax.imshow(matrix, aspect="auto", cmap=cmap_obj,
                   interpolation="nearest", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(pivot_val.columns)))
    ax.set_xticklabels(pivot_val.columns)
    ax.set_yticks(np.arange(len(pivot_val.index)))
    ax.set_yticklabels(pivot_val.index)

    title = title or "Attrition Risk Heatmap"
    style_axis_labels(
        ax,
        xlabel=level_col.replace("_", " ").title(),
        ylabel=department_col.replace("_", " ").title(),
        title=title,
        subtitle="Cell colour = attrition rate; label = headcount & rate",
    )

    # Cell annotations: count + rate
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            n = counts[i, j]
            if pd.isna(v) or pd.isna(n) or n == 0:
                continue
            text_color = "white" if (v > 0.66 or v < 0.15) else PALETTE["neutral"]
            ax.text(j, i, f"{v:.0%}\nn={int(n)}",
                    ha="center", va="center",
                    fontsize=9, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("attrition rate", color=PALETTE["neutral"], fontsize=9)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8)

    configure_axes(ax, grid=False)
    ax.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="both", length=0)

    return fig


def _resolve_cmap(name: str) -> Any:
    import matplotlib.cm as cm
    try:
        return cm.get_cmap(name)
    except ValueError:
        return cm.get_cmap("viridis")


# ---------------------------------------------------------------------------
# Compensation equity scatter
# ---------------------------------------------------------------------------

def compensation_equity(
    df: pd.DataFrame,
    *,
    x_col: str = "tenure_years",
    y_col: str = "salary",
    group_col: str | None = "gender",
    label_col: str | None = None,
    metadata: pd.DataFrame | None = None,
    id_col: str = "employee_id",
    name_col: str = "name",
    title: str | None = None,
    figsize: tuple[float, float] = (11.0, 7.0),
    iqr_threshold: float = 1.5,
) -> Any:
    """Scatter of tenure (or level) vs salary, coloured by group, with regression.

    A linear regression line is fit per group (and an overall line), and any
    point whose residual from the overall fit exceeds ``iqr_threshold``
    interquartile ranges is flagged as an outlier.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame")
    for c in (x_col, y_col):
        if c not in df.columns:
            raise KeyError(f"Need '{c}' in DataFrame")

    work = df.copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=[x_col, y_col])

    name_lookup: dict[str, str] = {}
    if metadata is not None and id_col in metadata.columns and name_col in metadata.columns:
        for _, row in metadata.iterrows():
            name_lookup[str(row[id_col])] = str(row[name_col])

    # Resolve labels (use metadata name when label_col is not provided)
    if label_col and label_col in work.columns:
        labels = work[label_col].astype(str).tolist()
    elif name_lookup and id_col in work.columns:
        labels = [name_lookup.get(str(k), str(k)) for k in work[id_col]]
    else:
        labels = [""] * len(work)

    apply_default_style()
    fig, ax = new_figure(figsize=figsize)

    # Fit overall line and detect outliers
    x = work[x_col].to_numpy(dtype=float)
    y = work[y_col].to_numpy(dtype=float)
    if len(x) >= 2:
        coef = np.polyfit(x, y, 1)
        y_pred = np.polyval(coef, x)
        resid = y - y_pred
        q1, q3 = np.percentile(resid, [25, 75])
        iqr = q3 - q1
        outlier_mask = (resid > q3 + iqr_threshold * iqr) | (resid < q1 - iqr_threshold * iqr)
    else:
        coef = (0.0, float(y.mean()) if len(y) else 0.0)
        outlier_mask = np.zeros(len(x), dtype=bool)

    # Draw scatter by group
    if group_col and group_col in work.columns:
        groups = work[group_col].astype(str).fillna("Unknown").tolist()
        distinct = sorted(set(groups))
        palette = {g: category_colors(len(distinct))[i] for i, g in enumerate(distinct)}
        for g in distinct:
            mask = np.array([gg == g for gg in groups])
            ax.scatter(x[mask], y[mask],
                       color=palette[g], label=g,
                       s=42, alpha=0.7,
                       edgecolor="white", linewidth=0.6, zorder=3)
            # Per-group regression line
            if mask.sum() >= 2:
                gc = np.polyfit(x[mask], y[mask], 1)
                xs = np.linspace(x.min(), x.max(), 100)
                ax.plot(xs, np.polyval(gc, xs),
                        color=palette[g], linestyle="--", linewidth=1.4, alpha=0.8,
                        zorder=2)
        ax.legend(loc="upper left", fontsize=9, frameon=True,
                  facecolor="white", edgecolor="#E5E5E5", title=group_col)
    else:
        ax.scatter(x, y, color=PALETTE["primary"], s=42, alpha=0.7,
                   edgecolor="white", linewidth=0.6, zorder=3)

    # Overall regression line
    if len(x) >= 2:
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.polyval(coef, xs),
                color=PALETTE["accent"], linewidth=2.2,
                label=f"overall fit (slope={coef[0]:,.0f})", zorder=4)
        ax.legend(loc="upper left", fontsize=9, frameon=True,
                  facecolor="white", edgecolor="#E5E5E5")

    # Flag outliers
    if outlier_mask.any():
        ax.scatter(x[outlier_mask], y[outlier_mask],
                   facecolor="none", edgecolor=PALETTE["danger"],
                   s=130, linewidth=1.8, zorder=5,
                   label=f"outlier (IQR×{iqr_threshold})")
        ax.legend(loc="upper left", fontsize=9, frameon=True,
                  facecolor="white", edgecolor="#E5E5E5")
        for i in np.where(outlier_mask)[0]:
            ax.annotate(
                labels[i] or f"row {i}",
                (x[i], y[i]),
                xytext=(8, 8), textcoords="offset points",
                fontsize=8, color=PALETTE["danger"], fontweight="bold",
            )

    # Pay-gap annotation: median of one group vs another if exactly two groups
    if group_col and group_col in work.columns:
        groups_arr = work[group_col].astype(str).fillna("Unknown").to_numpy()
        if len(set(groups_arr)) == 2:
            uniq = sorted(set(groups_arr))
            a, b = uniq
            med_a = float(np.median(y[groups_arr == a]))
            med_b = float(np.median(y[groups_arr == b]))
            gap_pct = (med_a - med_b) / med_b * 100 if med_b else 0.0
            ax.text(
                0.98, 0.04,
                f"Median {y_col} gap ({a} vs {b}): {gap_pct:+.1f}%",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9.5, color=PALETTE["accent"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="white", edgecolor=PALETTE["accent"],
                          linewidth=0.8),
            )

    title = title or "Compensation Equity"
    style_axis_labels(
        ax,
        xlabel=x_col.replace("_", " ").title(),
        ylabel=y_col.replace("_", " ").title(),
        title=title,
        subtitle="Regression line per group + flagged outliers",
    )
    configure_axes(ax, grid=True, grid_axis="both")

    return fig


__all__ = ["attrition_heatmap", "compensation_equity"]
