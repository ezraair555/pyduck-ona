# AI Contributor Guide

This guide is written for agents and AI collaborators who need to modify,
extend, or maintain `pyduck-ona` without human hand-holding. Follow it and
the package stays green.

## Quick safety check

Before starting work, run the agent CI runner to confirm the repo is in a
known-good state:

```bash
python scripts/agent_ci.py --format json
```

Expected output: all checks pass (`ruff`, `pytest`, `mypy`). If anything
fails, fix it before adding new code.

Run it again before every commit.

## Project layout

| Path | What lives here |
|---|---|
| `src/pyduck_ona/` | Package source. One concept per file. |
| `src/pyduck_ona/core.py` | Hierarchy primitives (`hierarchy_*`). |
| `src/pyduck_ona/graph/__init__.py` | Graph metrics (`betweenness`, `pagerank`, ...). |
| `src/pyduck_ona/stats/__init__.py` | Statistical modeling (`ols`, `logistic`, `anova`, ...). |
| `src/pyduck_ona/temporal.py` | `DuckONATemporal` class and time-aware analytics. |
| `src/pyduck_ona/frame.py` | `DuckONAFrame` v0.3 façade. |
| `src/pyduck_ona/search.py` | FTS / VSS search helpers. |
| `src/pyduck_ona/sql_builder.py` | Central parameterized SQL / identifier helpers. |
| `tests/unit/` | Fast, pure-Python unit tests. |
| `tests/integration/` | DuckDB-backed tests (most tests live here). |
| `docs/api/` | Auto-generated per-function reference pages. |
| `docs/v0.3_api_contract.md` | Verb-family contract for the frame API. |
| `scripts/agent_ci.py` | Agent-friendly CI runner. |
| `scripts/generate_api_docs.py` | Regenerates `docs/api/*.md`. |

## Adding a new public function

1. **Pick the right module.** Match the domain (graph, stats, temporal,
   search, hierarchy, frame).

2. **Follow the v0.3 contract for frame verbs.** The six verb families are:
   - `prep_*` — data prep / validation
   - `graph_*` — network metrics
   - `temporal_*` — time-window analytics
   - `model_*` — statistical models
   - `report_*` — output packaging
   - `search_*` — text / vector search

   Frame methods must use this return contract:
   ```python
   def verb_name(
       self,
       ...,
       *,
       output: str | None = None,
       as_pandas: bool = False,
   ) -> DuckDBPyRelation | pd.DataFrame | DuckONAFrame:
       ...
   ```
   - Default return is `DuckDBPyRelation`.
   - `as_pandas=True` materializes to a pandas DataFrame.
   - `output="name"` registers the result as a view/table and returns `self`
     for chaining.

3. **Use canonical `entity_id`.** Employee-level outputs should rename the
   caller's id column to `entity_id`. Use
   `DuckONAFrame._canonical(rel, id_col, key="entity_id")`.

4. **Write a docstring.** Docstrings drive the generated API pages. Include:
   - One-line description.
   - `Parameters` section.
   - `Returns` section.
   - `Examples` section with runnable Python.

5. **Type-annotate everything.** Mypy runs in strict mode. If a DuckDB or
   pandas return is untyped in stubs, use `typing.cast` or a targeted
   `# type: ignore[error-code]` with a brief reason.

6. **Quote identifiers, parameterize values.** Never interpolate scalars into
   SQL strings. Use `pyduck_ona.sql_builder.quote_identifier` for table/column
   names and DuckDB `?` placeholders for values.

7. **Export from `src/pyduck_ona/__init__.py`.** Add the name to both the
   import block and `__all__`.

8. **Add tests.** Put unit tests in `tests/unit/` and integration tests in
   `tests/integration/`. Match the naming and fixture style of neighboring
   files.

9. **Regenerate docs and catalog.** After the API surface changes, run:
   ```bash
   python scripts/generate_api_docs.py
   ```
   Then paste the printed catalog tables into `README.md` under
   `## API catalog`.

10. **Run CI and commit.**
    ```bash
    python scripts/agent_ci.py --format json
    git add -A
    git commit -m "type(scope): concise description" -m "- ..." -m "- pytest -q = N passed; ruff clean; mypy clean."
    ```

## Adding a new DuckDB extension integration

Pattern established by `src/pyduck_ona/search.py`:

1. Create a helper that installs and loads the extension idempotently:
   ```python
   def _require_extension(ext: str, con: DuckDBPyConnection) -> None:
       try:
           con.execute(f"LOAD {ext};")
       except Exception:
           con.execute(f"INSTALL {ext};")
           con.execute(f"LOAD {ext};")
   ```

2. Test extension availability with a minimal DuckDB connect in a
   `pytest.fixture` so tests can be marked or skipped if the extension is
   unavailable in a given environment.

3. Keep extension-specific code isolated in its own module. Do not make the
   extension mandatory at import time.

4. Provide `build_*`, `drop_*`, and query functions. Accept an optional `con`
   keyword for connection reuse, defaulting to a fresh in-memory connection.

5. Handle type quirks explicitly (e.g., VSS wants `FLOAT[N]`, but pandas
   produces `DOUBLE[]`). Cast or transform before building indexes.

## SQL safety rules

- **Scalars:** use `con.sql("... ? ...", params=[value])`.
- **Identifiers:** use `quote_identifier(name)` from `pyduck_ona.sql_builder`.
  It validates against `^[A-Za-z_][A-Za-z0-9_]*$` and raises on invalid input.
- **No f-string SQL for values.** This is the fastest way to introduce an
  injection path.
- **No bare identifiers from user input.** If a function accepts a column
  name, run it through `quote_identifier`.

## Linting and type-checking

- **Ruff:** `ruff check .` must report zero errors. Do not disable rules with
  broad `# noqa` unless the rule conflict is unavoidable; prefer fixing the
  underlying issue.
- **Mypy strict:** `mypy src/pyduck_ona` must report zero errors. The
  `pyproject.toml` section enables `strict = true`, `disallow_untyped_defs = true`,
  and `ignore_missing_imports = true` (so missing external stubs do not drown
  out repo-code issues).
- **Tests:** `pytest -q` must pass. Add tests for new code; do not weaken
  existing tests to make them pass.

## Regenerating API docs

```bash
python scripts/generate_api_docs.py
```

This scans the live public API and writes one markdown page per
function/class/method under `docs/api/`. After running it:

1. Review the generated pages for any missing examples.
2. Paste the printed catalog tables into `README.md`.
3. Commit the generated pages alongside the source change.

## Commit message style

Use conventional commits:

```
type(scope): short description

- Bullet details.
- pytest -q = N passed; ruff clean; mypy clean.
```

Common types: `feat`, `fix`, `refactor`, `docs`, `test`, `type`, `ci`.

## Failure protocol

If CI fails on your change:

1. Re-run `python scripts/agent_ci.py --format json` locally to capture the
   full output.
2. Fix the first failure; many downstream errors are artifacts of the first.
3. Do not comment out tests or disable rules to green-light the build.
4. If you are genuinely blocked, report: the failing command, the first error,
   and what you already tried.
