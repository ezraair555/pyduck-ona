"""
pyduck_ona: DuckDB-native People Analytics and Organizational Network Analysis.

This library brings HR analytics to DuckDB's vectorized engine using recursive
CTEs, the relational API, and zero-copy Arrow transfers. It is designed to
compose with `pyduck-janitor` for chainable data-cleaning workflows.

Public API entry points:
    - hierarchy_valid / hierarchy_long / hierarchy_wide / hierarchy_stats
    - to_networkx / to_igraph (graph export)
    - broom_augment / broom_tidy (statistical-model integration)
"""
from __future__ import annotations

from importlib import metadata as _md

try:
    __version__ = _md.version("pyduck-ona")
except _md.PackageNotFoundError:  # pragma: no cover - editable install path
    __version__ = "0.1.0"

from pyduck_ona.core import (
    hierarchy_long,
    hierarchy_stats,
    hierarchy_valid,
    hierarchy_wide,
)
from pyduck_ona.bridge import to_igraph, to_networkx
from pyduck_ona.graph import (
    betweenness,
    connected_components,
    degree_centrality,
    eigenvector_centrality,
    louvain_communities,
    pagerank,
    shortest_path,
)
from pyduck_ona.analysis import DuckONA
from pyduck_ona import stats as _stats

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
    "__version__",
]