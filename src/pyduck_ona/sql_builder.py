"""Central, safe SQL construction helpers for pyduck-ona.

Goals:

1. Identifier validation — table/column names are never interpolated blindly.
2. Parameter binding — scalar values are passed via ``?`` placeholders, not
   string formatting.
3. Reusable fragments — period filtering, edge selection, etc.

This is the first step toward closing the README's promise that every
predicate value is parameter-bound. Migration is incremental; new code
uses these helpers, and existing hard-coded SQL is refactored as it is
 touched.
"""

from __future__ import annotations

import re

# Only ASCII identifiers that look like typical SQL names.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_identifier(name: str) -> str:
    """Return a safely quoted SQL identifier.

    Raises
    ------
    ValueError
        If ``name`` contains characters outside the safe identifier set.
    """
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"invalid SQL identifier: {name!r}")
    return f'"{name}"'


def period_filter(date_col: str, freq_word: str, period: str) -> tuple[str, list[str]]:
    """Return a parameterized period filter and its parameter list.

    Parameters
    ----------
    date_col : str
        Name of the date column.
    freq_word : str
        DuckDB ``date_trunc`` frequency word (``month``, ``quarter``, ``year``).
    period : str
        Period label to match (e.g., ``'2026-04-01'``).

    Returns
    -------
    tuple[str, list[str]]
        ``(sql_fragment, [period])``.
    """
    expr = f"date_trunc('{freq_word}', CAST({quote_identifier(date_col)} AS DATE))"
    return f"{expr} = ?", [period]


def build_period_edges_query(
    table: str,
    emp_col: str,
    sup_col: str,
    date_col: str,
    period: str,
    freq_word: str,
) -> tuple[str, list[str]]:
    """Build a parameterized SELECT for one snapshot period.

    Returns
    -------
    tuple[str, list[str]]
        ``(sql, [period])``.
    """
    filter_sql, params = period_filter(date_col, freq_word, period)
    sql = (
        f"SELECT {quote_identifier(emp_col)}, {quote_identifier(sup_col)}\n"
        f"FROM {quote_identifier(table)}\n"
        f"WHERE {filter_sql}"
    )
    return sql, params
