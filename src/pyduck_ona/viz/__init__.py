"""
pyduck_ona.viz: Publication-quality visualizations for organizational analysis.

Visualization subpackage of ``pyduck_ona`` (integrated from the former
standalone ``pyduck-ona-viz`` package). Provides ten visualization entry points
that consume the DuckDB-relation outputs of pyduck-ona (after ``.df()``) and
return polished matplotlib Figures or standalone HTML strings.

Top-level functions:
    - :func:`org_chart_tree`         - Interactive D3-style HTML org chart.
    - :func:`reporting_chain_walk`   - Path from employee up to the top.
    - :func:`span_of_control`        - Horizontal bar chart of team sizes.
    - :func:`span_vs_depth`          - Span × depth bubble quadrant chart.
    - :func:`hierarchy_depth_heatmap` - Matrix view of who reports where.
    - :func:`centrality_dashboard`   - 2×2 grid of centrality rankings.
    - :func:`silo_map`               - Community-coloured network map.
    - :func:`attrition_heatmap`      - Department × level attrition grid.
    - :func:`compensation_equity`    - Tenure/level vs salary scatter.
    - :func:`summary_dashboard`      - One-page HTML dashboard of everything.

Every public function is also re-exported at the top level of the package so
that ``from pyduck_ona.viz import org_chart_tree`` is the canonical import.
"""
from __future__ import annotations

from importlib import metadata as _md

try:
    __version__ = _md.version("pyduck-ona")
except _md.PackageNotFoundError:  # pragma: no cover - editable install path
    __version__ = "0.3.0"

try:
    from .dashboard import summary_dashboard
    from .hierarchy_viz import hierarchy_depth_heatmap
    from .hr_viz import attrition_heatmap, compensation_equity
    from .network_viz import centrality_dashboard, silo_map
    from .org_chart import org_chart_tree, reporting_chain_walk
    from .span_control import span_of_control, span_vs_depth
    from .theme import (
        BLUES_CMAP,
        CATEGORICAL,
        DIVERG_RYG,
        PALETTE,
    )
except ModuleNotFoundError as _e:  # pragma: no cover - missing optional deps
    raise ImportError(
        "pyduck_ona.viz requires the visualization extras. "
        "Install with: pip install pyduck-ona[viz]"
    ) from _e

__all__ = [
    "org_chart_tree",
    "reporting_chain_walk",
    "span_of_control",
    "span_vs_depth",
    "hierarchy_depth_heatmap",
    "centrality_dashboard",
    "silo_map",
    "attrition_heatmap",
    "compensation_equity",
    "summary_dashboard",
    "PALETTE",
    "CATEGORICAL",
    "BLUES_CMAP",
    "DIVERG_RYG",
    "__version__",
]
