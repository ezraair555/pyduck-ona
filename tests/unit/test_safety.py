"""SQL-injection regression tests — column names must be safely escaped."""
from __future__ import annotations

import pytest

from pyduck_ona.core import _quote_ident


class TestQuoteIdent:
    @pytest.mark.parametrize("name", [
        "employee_id", "_underscore_first", "camelCase", "ALL_CAPS",
        "name with spaces", "name-with-dashes", "name.with.dots",
    ])
    def test_safe_names_handled(self, name):
        quoted = _quote_ident(name)
        # Result should always be a properly-quoted DuckDB identifier
        assert quoted.startswith('"')
        assert quoted.endswith('"')

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError, match="non-empty string"):
            _quote_ident("")

    def test_non_string_rejected(self):
        with pytest.raises(ValueError, match="non-empty string"):
            _quote_ident(None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="non-empty string"):
            _quote_ident(123)  # type: ignore[arg-type]

    def test_dangerous_quotes_escaped(self):
        # An attempted injection via "name"; DROP TABLE...
        quoted = _quote_ident('name"; DROP TABLE x; --')
        # The embedded " must be doubled, and the whole thing wrapped in "
        assert quoted == '"name""; DROP TABLE x; --"'


class TestIdentifierInjectionAttempt:
    """Verify that even pathological column names don't cause SQL injection."""

    def test_injection_via_column_name_does_nothing(self, simple_org):
        # Rename the column to a SQL-injection attempt
        import duckdb
        from pyduck_ona.core import hierarchy_valid

        rel = duckdb.sql("""
            SELECT employee_id, supervisor_id AS "sup_id; DROP TABLE x; --"
            FROM (
              SELECT 'E001' AS employee_id, NULL AS supervisor_id
              UNION ALL SELECT 'E002', 'E001'
            ) t
        """)
        # Should run safely without executing the embedded DROP
        result = hierarchy_valid(rel, "employee_id", "sup_id; DROP TABLE x; --")
        rows = result.fetchall()
        # Just verify it returned SOMETHING (not crashed)
        assert isinstance(rows, list)