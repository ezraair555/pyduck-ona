# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

## [0.1.5] - 2026-08-27

### Added
- Initial release of `pyduck-ona` with hierarchy validation, ONA graph metrics, span-of-control statistics, temporal mobility analysis, and statistical modeling helpers.
