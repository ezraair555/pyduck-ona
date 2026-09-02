# Agent Instructions for pyduck-ona

> This is the local agent-onboarding file for the `pyduck-ona` repository.
> It supplements the workspace-level `AGENTS.md` and `SOUL.md` with
> project-specific conventions.

## Start here

1. Read `docs/ai_contributor_guide.md` for the full maintenance and
   contribution playbook.
2. Read `docs/v0.3_api_contract.md` before adding or renaming frame verbs.
3. Run `python scripts/agent_ci.py --format json` to verify state before and
   after any change.

## Hard rules

- **No scalar SQL interpolation.** Use `?` placeholders or
  `pyduck_ona.sql_builder.quote_identifier`.
- **No public API renames without a deprecation plan.** Follow the v0.3
  contract's compatibility rules.
- **No new public function without a docstring and a test.**
- **No commit if `ruff`, `pytest`, or `mypy` fail.**

## Common tasks

| Task | Command / file |
|---|---|
| Run all CI checks | `python scripts/agent_ci.py --format json` |
| Regenerate API pages | `python scripts/generate_api_docs.py` |
| Add a frame verb | Edit `src/pyduck_ona/frame.py` and `docs/v0.3_api_contract.md` |
| Add a DuckDB extension integration | Follow the pattern in `src/pyduck_ona/search.py` |
| Update the public catalog | Paste generator output into `README.md` `## API catalog` |

## Escalation

If a change requires modifying `SOUL.md`, `OPERATING_MANIFEST.md`, or the
package dependency version pin in `pyproject.toml`, ask the owner before
proceeding.
