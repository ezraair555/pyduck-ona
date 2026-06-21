"""Shared pytest fixtures for pyduck_ona tests."""
from __future__ import annotations

import duckdb
import pytest


@pytest.fixture
def simple_org() -> duckdb.DuckDBPyRelation:
    """A simple 3-level org: CEO → VP → Director → IC.

    Hierarchy:
        E001 (CEO)
         └─ E010 (VP)
              ├─ E100 (Director)
              │    ├─ E1000 (IC)
              │    └─ E1001 (IC)
              └─ E101 (Director)
                   └─ E1010 (IC)
    """
    rows = [
        ("E001", None),     # CEO — root
        ("E010", "E001"),   # VP
        ("E100", "E010"),   # Director
        ("E101", "E010"),   # Director
        ("E1000", "E100"),  # IC
        ("E1001", "E100"),  # IC
        ("E1010", "E101"),  # IC
    ]
    return duckdb.sql(
        "SELECT * FROM (VALUES (CAST(? AS VARCHAR), CAST(? AS VARCHAR)), "
        "(CAST(? AS VARCHAR), CAST(? AS VARCHAR)), "
        "(CAST(? AS VARCHAR), CAST(? AS VARCHAR)), "
        "(CAST(? AS VARCHAR), CAST(? AS VARCHAR)), "
        "(CAST(? AS VARCHAR), CAST(? AS VARCHAR)), "
        "(CAST(? AS VARCHAR), CAST(? AS VARCHAR)), "
        "(CAST(? AS VARCHAR), CAST(? AS VARCHAR))) "
        "t(employee_id, supervisor_id)",
        params=[
            "E001", None,
            "E010", "E001",
            "E100", "E010",
            "E101", "E010",
            "E1000", "E100",
            "E1001", "E100",
            "E1010", "E101",
        ],
    )


@pytest.fixture
def broken_org() -> duckdb.DuckDBPyRelation:
    """An org with all four classes of issue for testing hierarchy_valid."""
    rows = [
        ("E001", None),       # root (legitimate)
        ("E002", None),       # second root → multiple_roots
        ("E010", "E001"),     # OK
        ("E011", "E999"),     # broken_chain (E999 doesn't exist)
        ("E012", "E012"),     # self_reference
    ]
    return duckdb.sql(
        "SELECT * FROM (VALUES (CAST(? AS VARCHAR), CAST(? AS VARCHAR)), "
        "(CAST(? AS VARCHAR), CAST(? AS VARCHAR)), "
        "(CAST(? AS VARCHAR), CAST(? AS VARCHAR)), "
        "(CAST(? AS VARCHAR), CAST(? AS VARCHAR)), "
        "(CAST(? AS VARCHAR), CAST(? AS VARCHAR))) "
        "t(employee_id, supervisor_id)",
        params=[
            "E001", None,
            "E002", None,
            "E010", "E001",
            "E011", "E999",
            "E012", "E012",
        ],
    )


@pytest.fixture
def cyclic_org() -> duckdb.DuckDBPyRelation:
    """A 3-cycle: A → B → C → A."""
    return duckdb.sql(
        "SELECT * FROM (VALUES (CAST(? AS VARCHAR), CAST(? AS VARCHAR)), "
        "(CAST(? AS VARCHAR), CAST(? AS VARCHAR)), "
        "(CAST(? AS VARCHAR), CAST(? AS VARCHAR))) "
        "t(employee_id, supervisor_id)",
        params=["A", "C", "B", "A", "C", "B"],
    )