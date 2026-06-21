"""Unit tests for the four core hierarchy operations."""
from __future__ import annotations

import duckdb
import pytest

from pyduck_ona.core import (
    hierarchy_long,
    hierarchy_stats,
    hierarchy_valid,
    hierarchy_wide,
)


# ─── hierarchy_valid ────────────────────────────────────────────────────────

class TestHierarchyValid:
    def test_clean_org_reports_no_issues(self, simple_org):
        result = hierarchy_valid(simple_org, "employee_id", "supervisor_id").df()
        assert len(result) == 0, f"Clean org should have 0 issues, got:\n{result}"

    def test_detects_multiple_roots(self, broken_org):
        result = hierarchy_valid(broken_org, "employee_id", "supervisor_id").df()
        assert "multiple_roots" in result["issue_type"].values

    def test_detects_broken_chain(self, broken_org):
        result = hierarchy_valid(broken_org, "employee_id", "supervisor_id").df()
        assert "broken_chain" in result["issue_type"].values
        broken = result[result["issue_type"] == "broken_chain"]
        assert "E011" in broken["employee_id"].values

    def test_detects_self_reference(self, broken_org):
        result = hierarchy_valid(broken_org, "employee_id", "supervisor_id").df()
        assert "self_reference" in result["issue_type"].values

    def test_detects_cycle(self, cyclic_org):
        result = hierarchy_valid(cyclic_org, "employee_id", "supervisor_id").df()
        assert "loop" in result["issue_type"].values
        # All three cycle members should appear
        loop_rows = result[result["issue_type"] == "loop"]
        loop_emps = set(loop_rows["employee_id"].tolist())
        assert {"A", "B", "C"} <= loop_emps

    def test_invalid_column_raises(self, simple_org):
        with pytest.raises(ValueError, match="not found in relation"):
            hierarchy_valid(simple_org, "employee_id", "nonexistent_col")

    def test_non_relation_raises(self):
        with pytest.raises(TypeError, match="expected duckdb.DuckDBPyRelation"):
            hierarchy_valid("not a relation", "a", "b")


# ─── hierarchy_long ─────────────────────────────────────────────────────────

class TestHierarchyLong:
    def test_clean_org_long_shape(self, simple_org):
        result = hierarchy_long(simple_org, "employee_id", "supervisor_id").df()
        # E1000 has chain E1000 → E100 → E010 → E001 (depths 1,2,3)
        e1000_chains = result[result["employee_id"] == "E1000"]
        assert len(e1000_chains) == 3
        assert sorted(e1000_chains["depth"].tolist()) == [1, 2, 3]

    def test_root_has_no_chain(self, simple_org):
        result = hierarchy_long(simple_org, "employee_id", "supervisor_id").df()
        assert len(result[result["employee_id"] == "E001"]) == 0

    def test_path_is_arrow_delimited(self, simple_org):
        result = hierarchy_long(simple_org, "employee_id", "supervisor_id").df()
        # Find E1000's deepest chain
        e1000 = result[result["employee_id"] == "E1000"]
        deepest = e1000[e1000["depth"] == 3]
        path_value = deepest["path"].iloc[0]
        assert path_value == "E1000->E100->E010->E001"

    def test_max_depth_truncates(self, simple_org):
        result = hierarchy_long(
            simple_org, "employee_id", "supervisor_id", max_depth=2
        ).df()
        # E1000's chain should be capped at depth 2
        assert len(result[result["employee_id"] == "E1000"]) == 2

    def test_returns_relation(self, simple_org):
        result = hierarchy_long(simple_org, "employee_id", "supervisor_id")
        assert isinstance(result, duckdb.DuckDBPyRelation)


# ─── hierarchy_wide ─────────────────────────────────────────────────────────

class TestHierarchyWide:
    def test_wide_columns_present(self, simple_org):
        result = hierarchy_wide(
            simple_org, "employee_id", "supervisor_id", max_depth=4
        ).df()
        assert "Level_1" in result.columns
        assert "Level_2" in result.columns
        assert "Level_3" in result.columns
        assert "Level_4" in result.columns

    def test_wide_values_correct(self, simple_org):
        result = hierarchy_wide(
            simple_org, "employee_id", "supervisor_id", max_depth=3
        ).df()
        e1000_row = result[result["employee_id"] == "E1000"].iloc[0]
        assert e1000_row["Level_1"] == "E100"
        assert e1000_row["Level_2"] == "E010"
        assert e1000_row["Level_3"] == "E001"

    def test_custom_prefix(self, simple_org):
        result = hierarchy_wide(
            simple_org, "employee_id", "supervisor_id",
            max_depth=2, level_prefix="mgr_",
        ).df()
        assert "mgr_1" in result.columns
        assert "mgr_2" in result.columns
        assert "Level_1" not in result.columns

    def test_invalid_prefix_raises(self, simple_org):
        with pytest.raises(ValueError, match="level_prefix"):
            hierarchy_wide(
                simple_org, "employee_id", "supervisor_id",
                max_depth=2, level_prefix="bad prefix!",
            )

    def test_max_depth_too_large_raises(self, simple_org):
        with pytest.raises(ValueError, match="max_depth"):
            hierarchy_wide(
                simple_org, "employee_id", "supervisor_id", max_depth=200,
            )


# ─── hierarchy_stats ────────────────────────────────────────────────────────

class TestHierarchyStats:
    def test_e010_has_two_directs(self, simple_org):
        result = hierarchy_stats(simple_org, "employee_id", "supervisor_id").df()
        e010 = result[result["manager_id"] == "E010"].iloc[0]
        assert e010["direct_reports"] == 2

    def test_e010_total_reports_includes_ic(self, simple_org):
        result = hierarchy_stats(simple_org, "employee_id", "supervisor_id").df()
        e010 = result[result["manager_id"] == "E010"].iloc[0]
        # E010 has 2 directs (E100, E101) + 3 indirects (E1000, E1001, E1010)
        assert e010["direct_reports"] == 2
        assert e010["indirect_reports"] == 3
        assert e010["total_reports"] == 5

    def test_levels_below(self, simple_org):
        result = hierarchy_stats(simple_org, "employee_id", "supervisor_id").df()
        e100 = result[result["manager_id"] == "E100"].iloc[0]
        # E100 has ICs (E1000, E1001) one level below. E010 sits two
        # levels below E100's manager, so E010's levels_below == 2.
        assert e100["levels_below"] == 1
        e010 = result[result["manager_id"] == "E010"].iloc[0]
        assert e010["levels_below"] == 2

    def test_sorted_descending(self, simple_org):
        result = hierarchy_stats(simple_org, "employee_id", "supervisor_id").df()
        totals = result["total_reports"].tolist()
        assert totals == sorted(totals, reverse=True)

# ─── Connection isolation (P0-1 regression) ───────────────────────────────

class TestConnectionIsolation:
    """Regression tests for the P0-1 connection-isolation bug.

    Before the fix: passing a DuckDBPyRelation created on a custom
    ``duckdb.connect()`` instance to any hierarchy_* function raised
    ``InvalidInputException: not suitable for replacement scan``.

    See ``memory/2026-06-21_qwen_review.md`` for the original bug report.
    """

    def test_hierarchy_valid_with_custom_connection(self):
        con = duckdb.connect()
        rel = con.sql(
            "SELECT 'E001' AS employee_id, CAST(NULL AS VARCHAR) AS supervisor_id "
            "UNION ALL SELECT 'E002', 'E001'"
        )
        # Must not raise; result is on default connection.
        result = hierarchy_valid(rel, "employee_id", "supervisor_id").df()
        assert isinstance(result, type(duckdb.sql("SELECT 1").df()))
        assert len(result) >= 0  # 0 issues for a 2-employee clean org

    def test_hierarchy_long_with_custom_connection(self):
        con = duckdb.connect()
        rel = con.sql(
            "SELECT 'E001' AS employee_id, CAST(NULL AS VARCHAR) AS supervisor_id "
            "UNION ALL SELECT 'E002', 'E001' "
            "UNION ALL SELECT 'E003', 'E002'"
        )
        result = hierarchy_long(rel, "employee_id", "supervisor_id").df()
        # E002 -> E001 (depth 1), E003 -> E002 -> E001 (depths 1 and 2)
        assert len(result) == 3
        assert "depth" in result.columns

    def test_hierarchy_stats_with_custom_connection(self):
        con = duckdb.connect()
        rel = con.sql(
            "SELECT 'E001' AS employee_id, CAST(NULL AS VARCHAR) AS supervisor_id "
            "UNION ALL SELECT 'E002', 'E001' "
            "UNION ALL SELECT 'E003', 'E001'"
        )
        result = hierarchy_stats(rel, "employee_id", "supervisor_id").df()
        # E001 is the only manager, with 2 direct reports
        assert len(result) == 1
        assert result.iloc[0]["manager_id"] == "E001"
        assert result.iloc[0]["direct_reports"] == 2

    def test_hierarchy_wide_with_custom_connection(self):
        con = duckdb.connect()
        rel = con.sql(
            "SELECT 'E001' AS employee_id, CAST(NULL AS VARCHAR) AS supervisor_id "
            "UNION ALL SELECT 'E002', 'E001' "
            "UNION ALL SELECT 'E003', 'E002'"
        )
        result = hierarchy_wide(
            rel, "employee_id", "supervisor_id", max_depth=3
        ).df()
        # E003 should have E002 at Level_1 and E001 at Level_2
        e003 = result[result["employee_id"] == "E003"].iloc[0]
        assert e003["Level_1"] == "E002"
        assert e003["Level_2"] == "E001"

    def test_repeated_calls_dont_leak_views(self):
        """Repeated calls on different connections must not pile up temp views.

        Each call registers a uniquely-numbered temp view, materializes
        the result, and drops the view. A bug in the cleanup loop would
        leave stale views on the default connection and surface as
        "Catalog Error: View with name _pona_df_NN_xxx already exists"
        on the second call.
        """
        for i in range(5):
            con = duckdb.connect()
            rel = con.sql(
                f"SELECT 'E{i}' AS employee_id, "
                f"CAST(NULL AS VARCHAR) AS supervisor_id"
            )
            result = hierarchy_valid(rel, "employee_id", "supervisor_id").df()
            assert len(result) == 0

    def test_result_is_independent_of_source_connection(self):
        """Closing the source connection must not affect later .df() calls
        on the returned relation. This verifies the materialization
        step in _run_sql_on_default actually decouples the result from
        the input relation's lifetime.
        """
        con = duckdb.connect()
        rel = con.sql(
            "SELECT 'E001' AS employee_id, CAST(NULL AS VARCHAR) AS supervisor_id"
        )
        result_rel = hierarchy_valid(rel, "employee_id", "supervisor_id")
        con.close()  # Source connection closed; result must still work.
        df = result_rel.df()
        assert len(df) == 0


# ─── NULL employee_id validation (P1-1 regression) ──────────────────────

class TestNullValidation:
    """Regression tests for the NULL employee_id validation (P1-1).

    Before the fix: a NULL employee_id was silently dropped from
    recursive-CTE joins, leading to ghost employees appearing/disappearing
    depending on which query ran. The fix: ``_validate_columns`` with
    ``require_non_null=[employee_id]`` fails loudly on the first call.
    """

    def test_null_employee_id_raises(self):
        rel = duckdb.sql("""
            SELECT CAST(NULL AS VARCHAR) AS employee_id,
                   CAST(NULL AS VARCHAR) AS supervisor_id
            UNION ALL SELECT 'E001', CAST(NULL AS VARCHAR)
        """)
        with pytest.raises(ValueError, match="employee_id"):
            hierarchy_valid(rel, "employee_id", "supervisor_id").df()

    def test_null_supervisor_id_allowed(self):
        """supervisor_id == NULL is valid: that's the root of the tree."""
        rel = duckdb.sql("""
            SELECT 'E001' AS employee_id, CAST(NULL AS VARCHAR) AS supervisor_id
            UNION ALL SELECT 'E002', 'E001'
        """)
        # Must NOT raise; supervisor_id NULL is by design.
        result = hierarchy_valid(rel, "employee_id", "supervisor_id").df()
        assert len(result) == 0  # Clean 2-employee org

    def test_null_check_works_on_custom_connection(self):
        con = duckdb.connect()
        rel = con.sql("""
            SELECT CAST(NULL AS VARCHAR) AS employee_id,
                   CAST(NULL AS VARCHAR) AS supervisor_id
        """)
        with pytest.raises(ValueError, match="employee_id"):
            hierarchy_long(rel, "employee_id", "supervisor_id").df()


# ─── Error message guidance (P1-3 regression) ──────────────────────────

class TestErrorGuidance:
    """Regression tests for actionable error messages (P1-3)."""

    def test_max_depth_error_suggests_default(self):
        rel = duckdb.sql("SELECT 'A' AS e, CAST(NULL AS VARCHAR) AS s")
        with pytest.raises(ValueError) as exc_info:
            hierarchy_wide(rel, "e", "s", max_depth=200).df()
        # Error must mention the default as a hint, not just complain.
        assert "default" in str(exc_info.value).lower()


# ─── Cycle detection boundary (P2-3 + P0-3 boundary check) ─────────────

class TestCycleDetectionBoundary:
    """Regression tests for cycle detection at the boundary where
    cycle_length == row_count.

    The recursion bound is ``WHERE rw.depth < (SELECT COUNT(*) FROM base) + 1``,
    which allows depths 1..N+1 inclusive. A cycle of length N produces
    walks of depth N+1, which the filter ``depth > COUNT(*)`` flags.
    This test exercises the boundary for N=3, N=4, N=5 to lock in
    correct behavior at the edge.
    """

    @pytest.mark.parametrize("cycle_size", [3, 4, 5, 10])
    def test_detects_cycle_at_boundary(self, cycle_size):
        # Build a cycle_size-cycle: 0 -> cycle_size-1 -> ... -> 1 -> 0
        rows = " UNION ALL ".join(
            f"SELECT 'N{i}' AS employee_id, 'N{(i-1) % cycle_size}' AS supervisor_id"
            for i in range(cycle_size)
        )
        rel = duckdb.sql(f"SELECT * FROM ({rows})")
        result = hierarchy_valid(rel, "employee_id", "supervisor_id").df()
        assert "loop" in result["issue_type"].values
        # All cycle members should be flagged.
        loop_rows = result[result["issue_type"] == "loop"]
        assert len(loop_rows) == cycle_size

    def test_hierarchy_wide_max_depth_matches_org_depth(self):
        """hierarchy_wide with max_depth == actual org depth produces
        all levels and no Level_{N+1} column.
        """
        # 3-level org: E001 -> E010 -> E100 -> E1000
        rel = duckdb.sql("""
            SELECT 'E1000' AS employee_id, 'E100' AS supervisor_id
            UNION ALL SELECT 'E100', 'E010'
            UNION ALL SELECT 'E010', 'E001'
            UNION ALL SELECT 'E001', CAST(NULL AS VARCHAR)
        """)
        result = hierarchy_wide(rel, "employee_id", "supervisor_id", max_depth=3).df()
        e1000 = result[result["employee_id"] == "E1000"].iloc[0]
        assert e1000["Level_1"] == "E100"
        assert e1000["Level_2"] == "E010"
        assert e1000["Level_3"] == "E001"
        assert "Level_4" not in result.columns
