# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `DuckONATemporal.insight_report()` builds an aggregate-first ONA brief with
  structural driver decomposition, affected-versus-unaffected metric movement,
  demographic summaries, small-cell suppression, and Markdown/HTML export.
- Added regression coverage and documentation for explainable temporal ONA.

### Changed
- Replaced deprecated table-form MIT license metadata and the deprecated MIT
  license classifier with the SPDX-style project license field.

## [0.3.0] - 2026-09-03

### Added (visualization subpackage)
- **`pyduck_ona.viz` subpackage**: the standalone `pyduck-ona-viz` v0.1.1
  package is now integrated into `pyduck-ona` as a first-party subpackage.
  Ten visualization entry points with unchanged names/signatures:
  `org_chart_tree`, `reporting_chain_walk`, `span_of_control`,
  `span_vs_depth`, `hierarchy_depth_heatmap`, `centrality_dashboard`,
  `silo_map`, `attrition_heatmap`, `compensation_equity`,
  `summary_dashboard`, plus the shared `theme` module (palette,
  `PALETTE` / `CATEGORICAL` / colormaps).
  - Lazy exposure via PEP 562: `import pyduck_ona` stays light;
    `pyduck_ona.viz` and its functions resolve on access and raise a
    clear `ImportError` ("pip install pyduck-ona[viz]") when the extras
    are missing.
  - `[viz]` extra now includes `numpy>=1.24` and `plotly>=5.18` (plotly
    backs the interactive paths in `span_of_control`, `silo_map`, and
    `summary_dashboard`).
  - Imports: `import pyduck_ona.viz as viz` (old `import
    pyduck_ona_viz` is deprecated — standalone repo frozen at v0.1.1).
  - New: `tests/unit/test_viz.py` (14 smoke tests),
    `examples/viz_demo.py`, `docs/viz_tutorial.md`, README install
    section updated, API catalog + generated per-function pages.
  - `scripts/generate_api_docs.py` now emits a Visualization section.
- Strict mypy across the new `src/pyduck_ona/viz` modules.

### Added
- `DuckONAFrame` v0.3 façade in `src/pyduck_ona/frame.py`: relation-first verbs with `as_pandas=True` materialization, `output=<name>` chaining, and canonical `entity_id`. Includes `pipeline()` combinator, `from_pandas`, `from_janitor`, and representatives from all five verb families.
- `DuckONA.from_janitor(...)` bridge for `pyduck-janitor` integration and a draft v0.3 API contract in `docs/v0.3_api_contract.md`.
- Career-path Markov modeling (`career_markov_matrix`, `career_markov_forecast`) on `DuckONATemporal`.
- Profile clustering (`profile_clusters`) on `DuckONA` with optional network features and k-means / GMM.
- Organizational-design analytics (`org_design_scorecard`, `org_design_change_alerts`) on `DuckONATemporal`.
- Regression tests for quarter-end and year-end snapshot frequencies in `q.*` temporal primitives.
- README now contains a full API catalog with links to 89 per-function reference pages under `docs/api/`.
- `scripts/generate_api_docs.py` auto-generates `docs/api/*.md` pages from live docstrings and signatures.
- Added `docs/ai_contributor_guide.md` and repo-level `AGENTS.md` as the AI/CI runbook and project-specific agent onboarding.
- Full-text search (`fts`) and vector similarity search (`vss`) helpers:
  - `text_search`, `build_fts_index`, `drop_fts_index`
  - `vector_search`, `build_vector_index`, `drop_vector_index`
  - `fuzzy_join_vectors`
  - `DuckONAFrame.search_text` and `DuckONAFrame.search_similar` v0.3 verbs.

### Fixed
- Temporal primitives now respect the parent `DuckONATemporal.freq` setting instead of hard-coding `date_trunc('month', ...)`, which caused empty results for quarterly and yearly snapshots.
- `DuckONATemporal.q._freq_word` is now read lazily from `parent.freq`, so `load_snapshots(freq=...)` changes propagate correctly.
- `DuckONA.build_org_edges(active_as_of=...)` no longer generates a duplicate `WHERE` clause.
- Weighted Louvain communities now correctly zips `(source, target, weight)` tuples.
- `ols()` and `logistic()` no longer leak `alpha=` into `broom_sm.stats_glance()`, removing a statsmodels `ValueWarning`.

### Changed
- Full ruff lint cleanup: 196 violations → 0.
- Public docstring coverage is now 100%.
- Strict mypy type-checking now passes across `src/pyduck_ona` and is enforced in CI.

### Added (DuckPGQ backend)
- **DuckPGQ backend for `pyduck_ona.graph`**: opt-in alternative to the
  default NetworkX backend, dispatched via `backend="duckpgq"`. Covers the
  four algorithms DuckPGQ v1.3.1 actually ships as table functions:
  `pagerank` (DuckPGQ PageRank), `connected_components` (DuckPGQ
  `weakly_connected_component`), `degree_centrality` (pure DuckDB SQL
  derived from the property-graph edge table — DuckPGQ does not expose a
  dedicated table function), and `local_clustering_coefficient` (bonus
  algorithm DuckPGQ exposes that NetworkX needs an undirected coercion to
  reproduce). The remaining four algorithms (`shortest_path`,
  `betweenness`, `eigenvector_centrality`, `louvain_communities`) keep the
  NetworkX backend — DuckPGQ v1.3.1 does not ship them as table functions.
- `pyduck_ona.graph.duckpgq_setup()` (and the lower-level
  `_duckpgq_backend.ensure_duckpgq`): fetches DuckPGQ from the official
  S3 mirror (`http://duckpgq.s3.eu-north-1.amazonaws.com`), installs +
  loads it on a DuckDB connection, and returns the (graph-name,
  vertex-label, edge-label) tuple that the v1.3.1 algorithm call shape
  requires.
- `pyduck_ona.graph.create_property_graph_for()`: convenience helper that
  turns a `(edges, vertices)` DuckDB relation pair into a registered
  DuckPGQ property graph with the right labels for the algorithm table
  functions.
- Optional `[graph]` extra in `pyproject.toml` pins `duckdb==1.3.1`
  because DuckPGQ v1.3.1's C++ ABI only loads on DuckDB 1.3.x (and the
  mirror publishes no v1.3.2 build). Calling `backend="duckpgq"` on
  DuckDB >=1.4 raises `ImportError` with install instructions rather
  than silently misbehaving.

### Fixed
- DuckPGQ integration tests no longer only assert that
  `backend="duckpgq"` raises. Tests for `pagerank`, `connected_components`,
  and `degree_centrality` now actually invoke the DuckPGQ path and
  cross-check against NetworkX within tolerance.
- `graph._require_duckpgq()` is no longer a hard-coded error path: it now
  attempts `LOAD` (and falls back to `INSTALL` from the DuckPGQ mirror)
  on a fresh in-memory connection. If the extension is unavailable
  (DuckDB >=1.4 ABI mismatch), the original `ImportError` is surfaced
  with current install instructions.
- `[graph]` extra tightened from `duckdb>=1.3,<1.4` to `duckdb==1.3.1`:
  the mirror has no v1.3.2 extension build, so fresh installs of the
  range pin previously failed with HTTP 404 on first `INSTALL duckpgq`.
- **PyPI metadata cleanup:** removed the `broom` extra (broom-sm is
  not on PyPI) and the `pyduck-janitor` git URL from the `dev` extra.
  Direct PEP 508 references in wheel metadata cause PyPI rejection.
  broom-sm and pyduck-janitor are now installed separately in CI from
  their GitHub repos. `package-data` now explicitly includes
  `viz/py.typed` alongside the top-level marker. Maintainer email
  normalized to `ezraair555@gmail.com`.
- Updated stats import error and README language to point users at the
  broom-sm GitHub repo instead of the removed `[broom]` extra.
- `is_duckpgq_supported_duckdb()` gate tightened from broad `1.3.x` to
  the exact supported semver (`1.3.1`), matching the mirror reality.
- `scipy>=1.11` added to base dependencies: `networkx.pagerank()`
  requires scipy, so `backend="networkx"` PageRank failed in clean
  environments with `ModuleNotFoundError`.

## [0.2.0]

## [0.1.5] - 2026-08-27

### Added
- Initial release of `pyduck-ona` with hierarchy validation, ONA graph metrics, span-of-control statistics, temporal mobility analysis, and statistical modeling helpers.
