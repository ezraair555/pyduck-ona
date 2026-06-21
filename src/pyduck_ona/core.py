"""
Core hierarchy operations for pyduck_ona.

All public functions in this module accept a `duckdb.DuckDBPyRelation` and
return a new `DuckDBPyRelation`. SQL identifiers (column names) are
validated against a strict regex before being interpolated; values are
always bound via DuckDB's `?` parameter API.

The four exported functions mirror the `hR` R-package vocabulary, adapted
for DuckDB SQL and the relational API:

    hierarchy_valid()  ->  diagnostic report (loops, broken chains, roots)
    hierarchy_long()   ->  long format (employee, supervisor, depth)
    hierarchy_wide()   ->  wide format (one row per employee, supervisor levels as columns)
    hierarchy_stats()  ->  span-of-control metrics for every manager
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import duckdb

if TYPE_CHECKING:
    from duckdb import DuckDBPyRelation


# ─── Identifier validation ──────────────────────────────────────────────────
#
# We enforce a strict pattern for column names to prevent SQL injection.
# DuckDB identifiers may be quoted with double quotes to escape reserved
# words; this regex matches safe unquoted identifiers (letters, digits,
# underscores, starting with letter or underscore). Any column name that
# does not match must be quoted, in which case we double-quote-escape
# any embedded double quotes (the standard SQL identifier-escape rule).

_IDENT_SAFE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_ident(name: str) -> str:
    """Return a safely-quoted DuckDB identifier.

    >>> _quote_ident("supervisor_id")
    '"supervisor_id"'
    >>> _quote_ident('weird"name')
    '"weird""name"'
    >>> _quote_ident("a b")  # spaces force quoting
    '"a b"'
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"column name must be a non-empty string, got {name!r}")
    if _IDENT_SAFE_RE.match(name):
        return f'"{name}"'
    # Quote and escape embedded double quotes by doubling them
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _validate_relation(rel: duckdb.DuckDBPyRelation) -> None:
    """Ensure `rel` is a non-empty DuckDB relation."""
    if not isinstance(rel, duckdb.DuckDBPyRelation):
        raise TypeError(
            f"expected duckdb.DuckDBPyRelation, got {type(rel).__name__}"
        )


def _validate_columns(
    rel: duckdb.DuckDBPyRelation,
    *columns: str,
    require_non_null: "bool | list[str] | None" = None,
) -> None:
    """Verify each column name exists in the relation's schema.

    Parameters
    ----------
    rel
    *columns
        Column names that must exist on ``rel``.
    require_non_null
        - ``None`` (default): no null check.
        - ``True``: every column in ``*columns`` is required to be non-null.
        - ``list[str]``: only the columns in this list are required to be
          non-null. Use this when some columns in ``*columns`` are
          allowed to be NULL (e.g. ``supervisor_id`` for the root of the
          hierarchy).

        NULL key columns silently corrupt recursive-CTE joins (NULL !=
        anything, including itself), so we fail loudly with a count of
        offending rows. Fix the HRIS export upstream rather than
        papering over it here.
    """
    schema_names = set(rel.columns)
    missing = [c for c in columns if c not in schema_names]
    if missing:
        available = sorted(schema_names)
        raise ValueError(
            f"column(s) not found in relation: {missing}. "
            f"Available columns: {available}"
        )
    if not require_non_null:
        return
    if require_non_null is True:
        null_check_cols = list(columns)
    else:
        null_check_cols = list(require_non_null)
    if not null_check_cols:
        return
    # Build a single COUNT(*) per column to avoid N separate queries.
    # We use the helper so this also works when the caller is on a
    # non-default connection.
    null_selects = ", ".join(
        f'SUM(CASE WHEN {_quote_ident(c)} IS NULL THEN 1 ELSE 0 END) AS "{c}_nulls"'
        for c in null_check_cols
    )
    counts = _run_sql_on_default(
        rel,
        f"SELECT {null_selects} FROM {{INPUT}}",
    ).df()
    for c in null_check_cols:
        n_null = int(counts.iloc[0][f"{c}_nulls"])
        if n_null > 0:
            raise ValueError(
                f"column {c!r} contains {n_null} NULL value(s); "
                f"key columns must be non-null. "
                f"Filter or fix the HRIS export upstream."
            )


# ─── Cross-connection execution ───────────────────────────────────────────
#
# Every public function in this module builds SQL that references the
# input relation as ``FROM df`` and runs it via ``duckdb.sql(...)``. That
# works when the caller used the default in-memory connection (the one
# ``duckdb.sql`` itself uses) — but fails with
# ``InvalidInputException: not suitable for replacement scan`` when the
# caller created the relation on a custom ``duckdb.connect()`` instance
# (the relation carries a reference to *its* connection, not the
# default one).
#
# Fix: materialize the relation to a pyarrow-backed DataFrame, register
# it on the default connection under a unique alias, run the SQL there,
# and drop the alias. This is the only portable bridge between two
# in-memory DuckDB connections; the round-trip cost is one Arrow
# conversion, which is the same cost the caller already pays the moment
# they call ``.df()`` on the result.

_TEMP_VIEW_COUNTER = 0


def _run_sql_on_default(
    rel: duckdb.DuckDBPyRelation,
    sql: str,
    params: list | None = None,
) -> duckdb.DuckDBPyRelation:
    """Run SQL against ``rel`` on the default DuckDB connection.

    The package's public functions build SQL that references the input
    relation as ``FROM df`` (or ``{INPUT}``) and run it via the
    default connection's ``duckdb.sql`` machinery. That works when the
    caller used the default in-memory connection — but fails with
    ``InvalidInputException: not suitable for replacement scan`` when
    the caller created the relation on a custom ``duckdb.connect()``
    instance (DuckDB relations carry a reference to *their* connection,
    not the default one).

    Fix: materialize ``rel`` to a pyarrow-backed DataFrame, register
    it on the default connection under a unique temp view, run the SQL
    against that view, **materialize the result via Arrow (which forces
    execution against the still-live view)**, then drop the view and
    return a fresh relation from the materialized arrow data. This
    keeps the returned relation independent of the dropped view.

    The SQL string may reference the input as ``{INPUT}`` (case-sensitive
    placeholder) or use the conventional ``FROM df``; both are bound to
    the temp view name before execution.

    See P0-1 in ``memory/2026-06-21_qwen_review.md`` for the original bug.
    """
    global _TEMP_VIEW_COUNTER
    _TEMP_VIEW_COUNTER += 1
    view_name = f"_pona_df_{_TEMP_VIEW_COUNTER}_{id(rel) & 0xFFFFFFFF:x}"

    arrow_table = rel.arrow()
    # DuckDB 1.3+ returns a streaming RecordBatchReader; materialize.
    if hasattr(arrow_table, "read_all"):
        arrow_table = arrow_table.read_all()
    df = arrow_table.to_pandas()

    bound_sql = (
        sql.replace("{INPUT}", view_name) if "{INPUT}" in sql
        else sql.replace("FROM df", f"FROM {view_name}")
    )

    try:
        duckdb.register(view_name, df)
        result = duckdb.sql(bound_sql, params=params) if params else duckdb.sql(bound_sql)
        # Force execution while the view is still alive. Arrow round-trip
        # materializes all rows; the result is no longer dependent on
        # the temp view existing.
        result_arrow = result.arrow()
        if hasattr(result_arrow, "read_all"):
            result_arrow = result_arrow.read_all()
    finally:
        try:
            duckdb.sql(f"DROP VIEW IF EXISTS {view_name}")
        except Exception:
            pass  # Best-effort cleanup; view is per-connection.

    # Re-wrap the materialized arrow data as a fresh relation on the
    # default connection. The caller can call .df() / .arrow() / etc.
    # on this without it re-executing against the dropped view.
    return duckdb.from_arrow(result_arrow)


# ─── hierarchy_valid ────────────────────────────────────────────────────────

def hierarchy_valid(
    df: duckdb.DuckDBPyRelation,
    employee_id: str,
    supervisor_id: str,
) -> duckdb.DuckDBPyRelation:
    """Diagnose the integrity of an organizational reporting structure.

    Detects four classes of issue:

      1. **Loops** — cycles where A reports to B who reports to A (directly
         or transitively). Most-common in mis-keyed HRIS extracts.
      2. **Broken chains** — supervisor IDs that don't correspond to any
         employee in the dataset.
      3. **Multiple roots** — more than one employee with no supervisor
         (only one is expected in a normal hierarchy).
      4. **Self-references** — an employee who reports to themselves
         (a degenerate form of loop).

    Parameters
    ----------
    df : duckdb.DuckDBPyRelation
        Input relation with at minimum two columns.
    employee_id : str
        Name of the column holding unique employee identifiers.
    supervisor_id : str
        Name of the column holding the supervisor's employee identifier
        (NULL/empty for the top of the hierarchy).

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with columns `(issue_type, employee_id, detail)`. One row
        per detected issue. Empty relation if the hierarchy is clean.

    Examples
    --------
    >>> import duckdb
    >>> rel = duckdb.from_df(some_pandas_df)
    >>> issues = hierarchy_valid(rel, "employee_id", "supervisor_id")
    >>> issues.fetchall()
    [('multiple_roots', None, 'Found 2 employees with no supervisor'),
     ('loop', 'emp_42', 'Cycle detected at depth 4')]
    """
    _validate_relation(df)
    _validate_columns(df, employee_id, supervisor_id, require_non_null=[employee_id])

    emp = _quote_ident(employee_id)
    sup = _quote_ident(supervisor_id)

    sql = f"""
    WITH RECURSIVE
    base AS (SELECT {emp} AS emp, {sup} AS sup FROM df),
    -- 1. self-references
    self_refs AS (
        SELECT 'self_reference' AS issue_type, emp AS employee_id,
               'Employee reports to themselves' AS detail
        FROM base WHERE emp = sup
    ),
    -- 2. broken chains: supervisor_id not in employee_id set (excluding NULL)
    all_emp AS (SELECT emp FROM base),
    broken_chains AS (
        SELECT 'broken_chain' AS issue_type, b.emp AS employee_id,
               'Supervisor ID ' || COALESCE(CAST(b.sup AS VARCHAR), 'NULL') ||
               ' does not appear as any employee' AS detail
        FROM base b
        WHERE b.sup IS NOT NULL AND b.sup NOT IN (SELECT emp FROM all_emp)
    ),
    -- 3. multiple roots: count of employees with no supervisor
    root_count AS (
        SELECT COUNT(*) AS n FROM base WHERE sup IS NULL OR sup = ''
    ),
    multiple_roots AS (
        SELECT 'multiple_roots' AS issue_type, CAST(NULL AS VARCHAR) AS employee_id,
               'Found ' || CAST(n AS VARCHAR) || ' employees with no supervisor' AS detail
        FROM root_count WHERE n > 1
    ),
    -- 4. loops: walk the tree; if any walk exceeds N rows, it's a cycle.
    --    We use a recursive CTE and bound depth by the row count + 1.
    recursive_walk AS (
        SELECT emp, sup, 1 AS depth
        FROM base WHERE sup IS NOT NULL AND sup <> ''
        UNION ALL
        SELECT rw.emp, b.sup, rw.depth + 1
        FROM recursive_walk rw JOIN base b ON rw.sup = b.emp
        WHERE rw.depth < (SELECT COUNT(*) FROM base) + 1
    ),
    loop_emps AS (
        SELECT DISTINCT emp AS employee_id FROM recursive_walk
        WHERE depth > (SELECT COUNT(*) FROM base)
    ),
    loops AS (
        SELECT 'loop' AS issue_type, employee_id,
               'Cycle detected — walk exceeded ' ||
               CAST((SELECT COUNT(*) FROM base) AS VARCHAR) || ' levels' AS detail
        FROM loop_emps
    )
    SELECT * FROM self_refs
    UNION ALL SELECT * FROM broken_chains
    UNION ALL SELECT * FROM multiple_roots
    UNION ALL SELECT * FROM loops
    """

    return _run_sql_on_default(df, sql)


# ─── hierarchy_long ─────────────────────────────────────────────────────────

def hierarchy_long(
    df: duckdb.DuckDBPyRelation,
    employee_id: str,
    supervisor_id: str,
    max_depth: int = 50,
) -> duckdb.DuckDBPyRelation:
    """Unroll the org tree into long format via a recursive CTE.

    For every employee, emits one row per ancestor in their reporting chain
    (not including themselves). The top-of-hierarchy (no supervisor) gets
    zero rows.

    Parameters
    ----------
    df : duckdb.DuckDBPyRelation
    employee_id, supervisor_id : str
    max_depth : int, default 50
        Safety bound on recursion. If the org has more than `max_depth`
        levels, deeper ancestors will be silently truncated. 50 covers
        every realistic organization (Amazon has ~10).

    Returns
    -------
    duckdb.DuckDBPyRelation
        Columns: `(employee_id, supervisor_id, depth, path)`.
        - `depth`: 1 = direct manager, 2 = manager's manager, ...
        - `path`: Arrow-style "->" delimited ancestor chain ending at this
          supervisor (useful for debugging cycles visually).

    Examples
    --------
    >>> long = hierarchy_long(rel, "emp_id", "mgr_id").df()
    >>> long.head()
       employee_id supervisor_id  depth              path
    0        E001          E010      1          E001->E010
    1        E001          E005      2     E001->E010->E005
    """
    _validate_relation(df)
    _validate_columns(df, employee_id, supervisor_id, require_non_null=[employee_id])

    emp = _quote_ident(employee_id)
    sup = _quote_ident(supervisor_id)

    sql = f"""
    WITH RECURSIVE
    base AS (SELECT {emp} AS emp, {sup} AS sup FROM df),
    chain AS (
        -- Anchor: every (employee, supervisor) edge except NULL supervisors
        SELECT emp AS employee_id, sup AS supervisor_id, 1 AS depth,
               emp || '->' || sup AS path
        FROM base
        WHERE sup IS NOT NULL AND sup <> ''

        UNION ALL

        -- Recursion: extend the chain by walking up one more level
        SELECT c.employee_id, b.sup AS supervisor_id, c.depth + 1,
               c.path || '->' || b.sup AS path
        FROM chain c JOIN base b ON c.supervisor_id = b.emp
        WHERE c.depth < ? AND b.sup IS NOT NULL AND b.sup <> ''
    )
    SELECT employee_id, supervisor_id, depth, path FROM chain
    ORDER BY employee_id, depth
    """
    return _run_sql_on_default(df, sql, params=[max_depth])


# ─── hierarchy_wide ─────────────────────────────────────────────────────────

def hierarchy_wide(
    df: duckdb.DuckDBPyRelation,
    employee_id: str,
    supervisor_id: str,
    max_depth: int = 15,
    level_prefix: str = "Level_",
) -> duckdb.DuckDBPyRelation:
    """Flatten the reporting chain into a single row per employee.

    Produces columns `Level_1`, `Level_2`, ..., `Level_N` where each is
    the ID of the supervisor at that distance above the employee. Level_1
    is the direct manager; the highest level is the top of the hierarchy.

    Parameters
    ----------
    df : duckdb.DuckDBPyRelation
    employee_id, supervisor_id : str
    max_depth : int, default 15
        Number of level-columns to produce. Raises if exceeded.
    level_prefix : str, default "Level_"
        Prefix for generated column names. Output columns will be
        `{prefix}1`, `{prefix}2`, ..., `{prefix}{max_depth}`. The prefix
        is validated as a safe identifier prefix.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Columns: `{employee_id}, {level_prefix}1, ..., {level_prefix}{max_depth}`.

    Notes
    -----
    Implemented as PIVOT over the long-format chain. DuckDB's PIVOT
    requires explicit value columns, so we generate the pivot IN-list
    programmatically from `max_depth` and the validated `level_prefix`.

    Examples
    --------
    >>> wide = hierarchy_wide(rel, "emp_id", "mgr_id", max_depth=5).df()
    >>> wide.head()
       emp_id    Level_1    Level_2    Level_3   Level_4   Level_5
    0   E001        E010        E005       E002      E001      None
    """
    _validate_relation(df)
    _validate_columns(df, employee_id, supervisor_id, require_non_null=[employee_id])
    if not (1 <= max_depth <= 100):
        raise ValueError(
            f"max_depth must be 1..100, got {max_depth}. "
            f"Org charts rarely exceed 15 levels; the default (15) covers "
            f"every realistic organization. Raise it only if you have a "
            f"specific reason to expect deeper chains."
        )
    if not _IDENT_SAFE_RE.match(level_prefix):
        raise ValueError(
            f"level_prefix must match {_IDENT_SAFE_RE.pattern}, got {level_prefix!r}"
        )

    emp = _quote_ident(employee_id)
    sup = _quote_ident(supervisor_id)

    # Build the PIVOT-IN list and level-column names.
    pivot_levels = list(range(1, max_depth + 1))
    pivot_level_quoted = ", ".join(str(d) for d in pivot_levels)
    level_cols = [
        f"{level_prefix}{d}" for d in pivot_levels
    ]
    level_cols_quoted = ", ".join(_quote_ident(c) for c in level_cols)

    # Inline the long-format recursive CTE directly into the PIVOT query
    # rather than calling ``hierarchy_long()`` as a sub-step. This keeps
    # the entire query in one SQL string so it can be passed through
    # ``_run_sql_on_default`` without depending on a Python-side Relation
    # in the helper's scope. See P0-1 in
    # ``memory/2026-06-21_qwen_review.md`` for context.
    sql = f"""
    WITH RECURSIVE
    base AS (SELECT {emp} AS emp, {sup} AS sup FROM {{INPUT}}),
    chain AS (
        SELECT emp AS employee_id, sup AS supervisor_id, 1 AS depth
        FROM base WHERE sup IS NOT NULL AND sup <> ''
        UNION ALL
        SELECT c.employee_id, b.sup AS supervisor_id, c.depth + 1
        FROM chain c JOIN base b ON c.supervisor_id = b.emp
        WHERE c.depth < ? AND b.sup IS NOT NULL AND b.sup <> ''
    ),
    deduped AS (
        SELECT DISTINCT employee_id, depth, supervisor_id
        FROM chain
    ),
    pivot_step AS (
        PIVOT deduped
        ON depth IN ({pivot_level_quoted})
        USING FIRST(supervisor_id) AS lvl
    ),
    aliased AS (
        SELECT
            {emp} AS {_quote_ident(employee_id)},
            {', '.join(f'"{d}_lvl" AS {_quote_ident(c)}' for d, c in zip(pivot_levels, level_cols))}
        FROM pivot_step
    )
    SELECT * FROM aliased
    ORDER BY {_quote_ident(employee_id)}
    """
    return _run_sql_on_default(df, sql, params=[max_depth])


# ─── hierarchy_stats ────────────────────────────────────────────────────────

def hierarchy_stats(
    df: duckdb.DuckDBPyRelation,
    employee_id: str,
    supervisor_id: str,
    max_depth: int = 50,
) -> duckdb.DuckDBPyRelation:
    """Calculate span-of-control metrics for every manager.

    For each manager (defined as any employee who appears as someone else's
    supervisor), compute:

      - `direct_reports`: count of employees whose supervisor_id = this person
      - `indirect_reports`: count of employees below this manager (any depth)
      - `total_reports`: direct + indirect
      - `team_size`: same as total_reports (alias for HR convention)
      - `levels_below`: deepest level in the chain (proxy for org height
        under this manager; useful for detecting flat vs deep teams)

    Parameters
    ----------
    df : duckdb.DuckDBPyRelation
    employee_id, supervisor_id : str
    max_depth : int, default 50

    Returns
    -------
    duckdb.DuckDBPyRelation
        Columns: `(manager_id, direct_reports, indirect_reports,
        total_reports, team_size, levels_below)`.

    Examples
    --------
    >>> stats = hierarchy_stats(rel, "emp_id", "mgr_id").df()
    >>> stats.sort_values("direct_reports", ascending=False).head()
       manager_id  direct_reports  indirect_reports  total_reports  team_size  levels_below
    0        E010              12                47             59         59             5
    """
    _validate_relation(df)
    _validate_columns(df, employee_id, supervisor_id, require_non_null=[employee_id])

    emp = _quote_ident(employee_id)
    sup = _quote_ident(supervisor_id)

    sql = f"""
    WITH RECURSIVE
    base AS (SELECT {emp} AS emp, {sup} AS sup FROM df),
    -- The long format gives us every (manager, report) edge with depth
    chain AS (
        SELECT sup AS manager_id, emp AS report_id, 1 AS depth
        FROM base WHERE sup IS NOT NULL AND sup <> ''
        UNION ALL
        SELECT c.manager_id, b.emp, c.depth + 1
        FROM chain c JOIN base b ON c.report_id = b.sup
        WHERE c.depth < ? AND b.sup IS NOT NULL AND b.sup <> ''
    ),
    -- Direct: depth = 1
    direct AS (
        SELECT manager_id, COUNT(DISTINCT report_id) AS direct_reports
        FROM chain WHERE depth = 1
        GROUP BY manager_id
    ),
    -- Total / indirect: all depths
    totals AS (
        SELECT manager_id,
               COUNT(DISTINCT report_id) AS total_reports,
               MAX(depth) AS levels_below
        FROM chain
        GROUP BY manager_id
    )
    SELECT
        t.manager_id,
        COALESCE(d.direct_reports, 0) AS direct_reports,
        t.total_reports - COALESCE(d.direct_reports, 0) AS indirect_reports,
        t.total_reports,
        t.total_reports AS team_size,
        t.levels_below
    FROM totals t
    LEFT JOIN direct d USING (manager_id)
    ORDER BY t.total_reports DESC
    """
    return _run_sql_on_default(df, sql, params=[max_depth])