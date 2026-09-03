"""
Shared visual design language for pyduck-ona-viz.

All visualization modules import palettes, fonts, and helpers from here so
that every figure in the package shares a consistent look suitable for
executive presentations and publication-quality output.

Design language:
    - Deep blue primary, warm gray neutrals, coral accent.
    - Consistent typography (12pt base, 16pt titles, 9pt annotations).
    - Minimal chartjunk: no default grid, pared-down spines.
    - DPI: 150 for screen, 300 for print.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

# Primary deep-blue / warm-gray / coral brand palette.
PALETTE: dict[str, str] = {
    "primary":    "#1F3A5F",  # deep blue
    "secondary":  "#4A6FA5",  # mid blue
    "tertiary":   "#7FA0C9",  # light blue
    "neutral":    "#4D4D4D",  # warm gray (text)
    "neutral_lt": "#A6A6A6",  # light warm gray
    "accent":     "#E27D60",  # coral
    "accent_lt":  "#F2B6A0",  # light coral
    "success":    "#5B9279",  # sage green
    "warning":    "#E8C547",  # amber
    "danger":     "#C44536",  # brick red
    "background": "#FAFAFA",  # near-white
}

# Ordered categorical palette for departments / communities. Cycles cleanly.
CATEGORICAL: list[str] = [
    "#1F3A5F",  # deep blue
    "#E27D60",  # coral
    "#5B9279",  # sage
    "#9C7CA5",  # muted purple
    "#E8C547",  # amber
    "#4A6FA5",  # mid blue
    "#C44536",  # brick red
    "#7FA0C9",  # light blue
    "#8C7853",  # taupe
    "#41B3A3",  # teal
    "#D4A5A5",  # dusty rose
    "#5E548E",  # deep purple
]

# Sequential blue ramp (light → deep) used for heatmaps and continuous bars.
BLUES_CMAP = LinearSegmentedColormap.from_list(
    "pona_blues",
    ["#E8EEF5", "#C3D2E5", "#9DB6D3", "#779AC0", "#547DAC", "#356197", "#1F3A5F"],
    N=256,
)

# Diverging palette: red (negative) → cream (neutral) → green (positive).
DIVERG_RYG = LinearSegmentedColormap.from_list(
    "pona_diverg",
    ["#C44536", "#E27D60", "#F5E6D3", "#A8C8A0", "#5B9279", "#2D5F4C"],
    N=256,
)


def category_colors(n: int) -> list[str]:
    """Return ``n`` colors from :data:`CATEGORICAL`, cycling if needed."""
    if n <= 0:
        return []
    return [CATEGORICAL[i % len(CATEGORICAL)] for i in range(n)]


def color_for(value: str, lookup: dict[str, str] | None = None) -> str:
    """Map a category string to a color (deterministic, lookup-aware)."""
    if lookup and value in lookup:
        return lookup[value]
    return CATEGORICAL[hash(value) % len(CATEGORICAL)]


# ---------------------------------------------------------------------------
# Typography and figure defaults
# ---------------------------------------------------------------------------

FONT_FAMILY = "DejaVu Sans"

FONT_SIZES: dict[str, int] = {
    "title":      16,
    "subtitle":   12,
    "axis":       11,
    "tick":       10,
    "legend":     10,
    "annotation": 9,
    "body":       9,
}


def apply_default_style() -> None:
    """Apply the package-wide matplotlib rcParams. Called by each function."""
    mpl.rcParams.update({
        "font.family":      FONT_FAMILY,
        "font.size":        FONT_SIZES["body"],
        "axes.titlesize":   FONT_SIZES["title"],
        "axes.titleweight": "semibold",
        "axes.labelsize":   FONT_SIZES["axis"],
        "axes.labelweight": "medium",
        "axes.labelcolor":  PALETTE["neutral"],
        "axes.edgecolor":   PALETTE["neutral_lt"],
        "axes.linewidth":   0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.facecolor":   PALETTE["background"],
        "figure.facecolor": "white",
        "xtick.labelsize":  FONT_SIZES["tick"],
        "ytick.labelsize":  FONT_SIZES["tick"],
        "xtick.color":      PALETTE["neutral"],
        "ytick.color":      PALETTE["neutral"],
        "legend.fontsize":  FONT_SIZES["legend"],
        "legend.frameon":   False,
        "grid.color":       "#E5E5E5",
        "grid.linewidth":   0.5,
        "savefig.dpi":      150,
        "savefig.bbox":     "tight",
        "savefig.facecolor": "white",
    })


def configure_axes(ax: Any, *, grid: bool = False, grid_axis: str = "y") -> None:
    """Apply common axis cosmetics (minimal spines, optional grid)."""
    if grid:
        ax.grid(axis=grid_axis, color="#E5E5E5", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(PALETTE["neutral_lt"])
    ax.spines["bottom"].set_color(PALETTE["neutral_lt"])


def annotate_bars(
    ax: Any,
    values: Iterable[float],
    *,
    fmt: str = "{:,.0f}",
    offset: float = 0.01,
    color: str | None = None,
) -> None:
    """Annotate the tip of each bar in a horizontal bar plot."""
    color = color or PALETTE["neutral"]
    ymax = max(values) if values else 1
    for bar, v in zip(ax.patches, values, strict=False):
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        x_text = width + (ymax * offset if ymax else 0.01)
        ax.text(
            x_text, y, fmt.format(v),
            va="center", ha="left",
            fontsize=FONT_SIZES["annotation"],
            color=color,
        )


def gradient_color(
    values: NDArray[np.float64] | list[float], cmap: str = "viridis",
) -> list[tuple[float, float, float, float]]:
    """Map an array of values to a list of RGBA tuples via a matplotlib colormap."""
    import matplotlib.cm as cm
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    lo, hi = float(arr.min()), float(arr.max())
    if hi == lo:
        hi = lo + 1.0
    norm = (arr - lo) / (hi - lo)
    return [to_rgba(cm.get_cmap(cmap)(v), alpha=1.0) for v in norm]


def gradient_color_palette(
    values: NDArray[np.float64] | list[float], cmap_name: str | None = None,
) -> list[tuple[float, float, float, float]]:
    """Gradient using the package's brand blue ramp by default."""
    cmap_name = cmap_name or "pona_blues"
    import matplotlib.cm as cm
    try:
        cm_obj = cm.get_cmap(cmap_name) if cmap_name != "pona_blues" else BLUES_CMAP
    except ValueError:
        cm_obj = BLUES_CMAP
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    lo, hi = float(arr.min()), float(arr.max())
    if hi == lo:
        hi = lo + 1.0
    norm = (arr - lo) / (hi - lo)
    return [to_rgba(cm_obj(v), alpha=1.0) for v in norm]


# ---------------------------------------------------------------------------
# Figure factory helpers
# ---------------------------------------------------------------------------

def new_figure(
    figsize: tuple[float, float] = (10.0, 6.0),
    *,
    dpi: int = 150,
    nrows: int = 1,
    ncols: int = 1,
    constrained_layout: bool = True,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Create a new figure with the package style applied."""
    apply_default_style()
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=figsize,
        dpi=dpi,
        constrained_layout=constrained_layout,
        **kwargs,
    )
    return fig, axes


def style_axis_labels(
    ax: Any,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
) -> None:
    """Apply axis labels and a (sub)title pair in the package style."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", pad=24 if subtitle else 12)
    if subtitle:
        ax.text(
            0.0, 1.02, subtitle,
            transform=ax.transAxes,
            fontsize=FONT_SIZES["subtitle"],
            color=PALETTE["neutral_lt"],
            ha="left", va="bottom",
        )


def figure_to_html(fig: Any, include_plotlyjs: bool = True) -> str:
    """Convert a Plotly figure to a standalone HTML string."""
    return str(fig.to_html(include_plotlyjs=include_plotlyjs, full_html=True))


__all__ = [
    "PALETTE",
    "CATEGORICAL",
    "BLUES_CMAP",
    "DIVERG_RYG",
    "FONT_FAMILY",
    "FONT_SIZES",
    "apply_default_style",
    "configure_axes",
    "annotate_bars",
    "gradient_color",
    "gradient_color_palette",
    "new_figure",
    "style_axis_labels",
    "figure_to_html",
    "category_colors",
    "color_for",
]
