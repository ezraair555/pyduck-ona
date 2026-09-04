"""
pyduck_ona: DuckDB-native People Analytics and Organizational Network Analysis.

This library brings HR analytics to DuckDB's vectorized engine using recursive
CTEs, the relational API, and zero-copy Arrow transfers. It is designed to
compose with `pyduck-janitor` for chainable data-cleaning workflows.

Public API entry points:
    - hierarchy_valid / hierarchy_long / hierarchy_wide / hierarchy_stats
    - to_networkx / to_igraph (graph export)
    - broom_augment / broom_tidy (statistical-model integration)
    - pyduck_ona.viz (visualization subpackage; requires the [viz] extra)
"""
from __future__ import annotations

from importlib import metadata as _md
from typing import Any

try:
    __version__ = _md.version("pyduck-ona")
except _md.PackageNotFoundError:  # pragma: no cover - editable install path
    __version__ = "0.1.0"

from pyduck_ona import stats as _stats
from pyduck_ona.analysis import DuckONA
from pyduck_ona.bridge import to_igraph, to_networkx
from pyduck_ona.core import (
    hierarchy_long,
    hierarchy_stats,
    hierarchy_valid,
    hierarchy_wide,
)
from pyduck_ona.frame import DuckONAFrame
from pyduck_ona.graph import (
    betweenness,
    connected_components,
    degree_centrality,
    eigenvector_centrality,
    louvain_communities,
    pagerank,
    shortest_path,
)
from pyduck_ona.insights import ONAInsightReport, build_insight_report
from pyduck_ona.search import (
    build_fts_index,
    build_vector_index,
    drop_fts_index,
    drop_vector_index,
    fuzzy_join_vectors,
    text_search,
    vector_search,
)
from pyduck_ona.temporal import DuckONATemporal

# Re-export the public stats functions at the top level. The full set
# lives in pyduck_ona.stats (lazy import keeps the heavy broom_sm
# dependency optional).
correlation = _stats.correlation
anova = _stats.anova
ols = _stats.ols
logistic = _stats.logistic
chi_square = _stats.chi_square
plot_ols = _stats.plot_ols
plot_residuals = _stats.plot_residuals
plot_coefficients = _stats.plot_coefficients
vif = _stats.vif
model_compare_stats = _stats.model_compare
tidy_to_duckdb = _stats.tidy_to_duckdb
to_duckdb = _stats.to_duckdb
save_figure = _stats.save_figure

__all__ = [
    "hierarchy_valid",
    "hierarchy_long",
    "hierarchy_wide",
    "hierarchy_stats",
    "to_networkx",
    "to_igraph",
    "shortest_path",
    "betweenness",
    "pagerank",
    "eigenvector_centrality",
    "degree_centrality",
    "connected_components",
    "louvain_communities",
    "DuckONA",
    "DuckONAFrame",
    "DuckONATemporal",
    "correlation",
    "anova",
    "ols",
    "logistic",
    "chi_square",
    "plot_ols",
    "plot_residuals",
    "plot_coefficients",
    "vif",
    "model_compare_stats",
    "tidy_to_duckdb",
    "to_duckdb",
    "save_figure",
    "build_fts_index",
    "drop_fts_index",
    "text_search",
    "build_vector_index",
    "drop_vector_index",
    "vector_search",
    "fuzzy_join_vectors",
    "ONAInsightReport",
    "build_insight_report",
    "viz",
    "__version__",
]

# Visualization subpackage (integrated from pyduck-ona-viz v0.1.1).
# Exposed lazily via PEP 562 so `import pyduck_ona` stays light and the
# [viz] extras (matplotlib/pyvis/plotly) remain optional.
_VIZ_EXPORTS = frozenset(
    {
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
    }
)


def __getattr__(name: str) -> Any:
    # NOTE: must use importlib.import_module here. `from pyduck_ona import
    # viz` would re-enter __getattr__ (the attribute isn't set until the
    # submodule import completes) and recurse infinitely.
    if name == "viz" or name in _VIZ_EXPORTS:
        import importlib

        viz_mod = importlib.import_module("pyduck_ona.viz")
        if name == "viz":
            return viz_mod
        return getattr(viz_mod, name)
    raise AttributeError(f"module 'pyduck_ona' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _VIZ_EXPORTS | {"viz"})
