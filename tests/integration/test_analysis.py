"""Integration tests for the high-level DuckONA analysis class.

Covers the HR-analytics workflow described in the re-scoped plan:
loading HR tables, key/date validation, org-chart edge construction,
graph metrics, HRIS joins, model helpers, temporal slicing, and the
pure-Python MRQAP helper.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest

from pyduck_ona import DuckONA
from pyduck_ona.graph import (
    degree_centrality,
    eigenvector_centrality,
    louvain_communities,
)


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def ona() -> DuckONA:
    """A fresh in-memory DuckONA workspace."""
    return DuckONA(":memory:")


@pytest.fixture
def hris_df() -> pd.DataFrame:
    """Small org: CEO (E001) → two VPs (E010/E011) → four ICs."""
    return pd.DataFrame(
        {
            "employee_id": ["E001", "E010", "E011", "E100", "E101", "E110", "E111"],
            "supervisor_id": [None, "E001", "E001", "E010", "E010", "E011", "E011"],
            "department": ["People", "Eng", "Sales", "Eng", "Eng", "Sales", "Sales"],
            "job_level": [4, 3, 3, 1, 1, 1, 1],
            "hire_date": pd.to_datetime(
                ["2018-01-15", "2019-03-01", "2019-04-01", "2020-06-15", "2021-02-01",
                 "2020-09-01", "2021-05-01"]
            ),
        }
    )


@pytest.fixture
def compensation_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "employee_id": ["E001", "E010", "E011", "E100", "E101", "E110", "E111"],
            "salary": [220_000, 165_000, 160_000, 92_000, 95_000, 88_000, 90_000],
            "bonus": [40_000, 25_000, 22_000, 8_000, 9_000, 7_500, 8_000],
            "currency": ["USD"] * 7,
            "effective_date": pd.to_datetime(["2026-01-01"] * 7),
        }
    )


@pytest.fixture
def survey_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "employee_id": ["E001", "E010", "E011", "E100", "E101", "E110", "E111"],
            "engagement": [8.5, 7.2, 7.5, 6.8, 7.0, 6.5, 7.8],
            "survey_date": pd.to_datetime(["2026-03-15"] * 7),
        }
    )


@pytest.fixture
def turnover_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "employee_id": ["E100"],
            "termination_date": pd.to_datetime(["2026-05-01"]),
            "reason": ["voluntary"],
        }
    )


@pytest.fixture
def promotions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "employee_id": ["E101", "E110"],
            "promotion_date": pd.to_datetime(["2026-02-01", "2026-04-01"]),
            "from_level": [1, 1],
            "to_level": [2, 2],
        }
    )


@pytest.fixture
def skills_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "employee_id": ["E010", "E100", "E101", "E110"],
            "skill": ["python", "python", "sql", "python"],
            "proficiency": [5, 4, 3, 4],
        }
    )


@pytest.fixture
def attendance_df() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=14, freq="D")
    rows = []
    for eid in ["E001", "E010", "E100", "E101"]:
        for d in dates:
            rows.append({"employee_id": eid, "date": d, "present": int(d.dayofweek < 5)})
    return pd.DataFrame(rows)


@pytest.fixture
def retirement_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "employee_id": ["E001", "E010"],
            "eligible_date": pd.to_datetime(["2040-01-01", "2035-06-01"]),
            "plan": ["401k", "401k"],
        }
    )


# ─── Table loading ──────────────────────────────────────────────────────────

class TestLoading:
    def test_load_hris_registers_table(self, ona: DuckONA, hris_df: pd.DataFrame):
        rel = ona.load_hris(hris_df)
        assert isinstance(rel, duckdb.DuckDBPyRelation)
        assert rel.count("*").fetchone()[0] == 7

    def test_load_compensation(self, ona: DuckONA, compensation_df: pd.DataFrame):
        rel = ona.load_compensation(compensation_df)
        assert rel.columns == ["employee_id", "salary", "bonus", "currency", "effective_date"]

    def test_load_turnover(self, ona: DuckONA, turnover_df: pd.DataFrame):
        rel = ona.load_turnover(turnover_df)
        assert rel.count("*").fetchone()[0] == 1

    def test_load_survey(self, ona: DuckONA, survey_df: pd.DataFrame):
        rel = ona.load_survey(survey_df)
        assert "engagement" in rel.columns

    def test_load_retirement(self, ona: DuckONA, retirement_df: pd.DataFrame):
        rel = ona.load_retirement(retirement_df)
        assert rel.count("*").fetchone()[0] == 2

    def test_load_promotions(self, ona: DuckONA, promotions_df: pd.DataFrame):
        rel = ona.load_promotions(promotions_df)
        assert rel.count("*").fetchone()[0] == 2

    def test_load_skills(self, ona: DuckONA, skills_df: pd.DataFrame):
        rel = ona.load_skills(skills_df)
        assert set(rel.df()["skill"].tolist()) == {"python", "sql"}

    def test_load_attendance(self, ona: DuckONA, attendance_df: pd.DataFrame):
        rel = ona.load_attendance(attendance_df)
        assert rel.count("*").fetchone()[0] == 56


# ─── Validation ─────────────────────────────────────────────────────────────

class TestValidation:
    def test_validation_passes_clean_hris(self, ona: DuckONA, hris_df: pd.DataFrame):
        ona.load_hris(hris_df)
        ona.validate_keys("hris")

    def test_validation_catches_null_employee_id(self, ona: DuckONA, hris_df: pd.DataFrame):
        bad = hris_df.copy()
        bad.loc[0, "employee_id"] = None
        ona.load_hris(bad)
        with pytest.raises(ValueError, match="NULL"):
            ona.validate_keys("hris")

    def test_validation_catches_duplicate_employee_id(self, ona: DuckONA, hris_df: pd.DataFrame):
        bad = pd.concat([hris_df, hris_df.iloc[[0]]], ignore_index=True)
        ona.load_hris(bad)
        with pytest.raises(ValueError, match="duplicate"):
            ona.validate_keys("hris")

    def test_validation_catches_future_date(self, ona: DuckONA, compensation_df: pd.DataFrame):
        bad = compensation_df.copy()
        bad.loc[0, "effective_date"] = pd.Timestamp("2099-01-01")
        ona.load_compensation(bad)
        with pytest.raises(ValueError, match="future"):
            ona.validate_keys("compensation", date_col="effective_date")

    def test_validation_catches_duplicate_per_date(
        self, ona: DuckONA, compensation_df: pd.DataFrame
    ):
        bad = pd.concat([compensation_df, compensation_df.iloc[[0]]], ignore_index=True)
        ona.load_compensation(bad)
        with pytest.raises(ValueError, match="duplicate"):
            ona.validate_keys("compensation", date_col="effective_date")


# ─── Noise filtering / deduplication helpers ─────────────────────────────────

class TestCleaningHelpers:
    def test_filter_noise_drops_test_ids(self):
        df = pd.DataFrame(
            {"employee_id": ["E001", "TEST01", "E002"], "value": [1, 2, 3]}
        )
        out = DuckONA.filter_noise(df, test_ids=["TEST01"])
        assert out["employee_id"].tolist() == ["E001", "E002"]

    def test_filter_noise_drops_null_keys(self):
        df = pd.DataFrame({"employee_id": ["E001", None, "E002"], "value": [1, 2, 3]})
        out = DuckONA.filter_noise(df)
        assert out["employee_id"].tolist() == ["E001", "E002"]

    def test_deduplicate_keeps_last(self):
        df = pd.DataFrame(
            {
                "employee_id": ["E001", "E001", "E002"],
                "salary": [100, 110, 200],
            }
        )
        out = DuckONA.deduplicate(df)
        assert out["salary"].tolist() == [110, 200]


# ─── Org edges ──────────────────────────────────────────────────────────────

class TestOrgEdges:
    def test_build_org_edges_shape(self, ona: DuckONA, hris_df: pd.DataFrame):
        ona.load_hris(hris_df)
        edges = ona.build_org_edges()
        df = edges.df()
        # 7 employees, 1 root, expect 6 directed edges
        assert len(df) == 6
        assert set(df.columns) == {"employee_id", "supervisor_id"}
        assert "E001" not in df["employee_id"].tolist()

    def test_build_org_edges_respects_custom_columns(self, ona: DuckONA):
        df = pd.DataFrame({"emp": ["A", "B"], "mgr": [None, "A"]})
        ona.load_hris(df)
        edges = ona.build_org_edges("emp", "mgr")
        assert set(edges.df().columns) == {"emp", "mgr"}
        assert edges.df()["emp"].tolist() == ["B"]


# ─── Graph metrics ──────────────────────────────────────────────────────────

class TestGraphMetrics:
    @pytest.fixture
    def edges(self, ona: DuckONA, hris_df: pd.DataFrame) -> duckdb.DuckDBPyRelation:
        ona.load_hris(hris_df)
        return ona.build_org_edges()

    def test_betweenness_returns_relation(self, ona: DuckONA, edges: duckdb.DuckDBPyRelation):
        result = ona.betweenness(edges, "employee_id", "supervisor_id")
        assert isinstance(result, duckdb.DuckDBPyRelation)
        assert set(result.df().columns) == {"node_id", "betweenness"}

    def test_pagerank_returns_all_employees(self, ona: DuckONA, edges: duckdb.DuckDBPyRelation):
        df = ona.pagerank(edges, "employee_id", "supervisor_id").df()
        assert len(df) == 7
        assert abs(df["pagerank"].sum() - 1.0) < 1e-6

    def test_eigenvector_centrality_returns_scores_for_all_nodes(
        self, ona: DuckONA, edges: duckdb.DuckDBPyRelation
    ):
        df = ona.eigenvector_centrality(edges, "employee_id", "supervisor_id").df()
        assert len(df) == 7
        # The root has no incoming edges in a tree digraph, so it may
        # not be top; we simply assert scores are non-negative.
        assert (df["eigenvector"] >= 0).all()

    def test_degree_centrality_modes(self, ona: DuckONA, edges: duckdb.DuckDBPyRelation):
        out = ona.degree_centrality(edges, "employee_id", "supervisor_id", mode="out").df()
        assert out[out["node_id"] == "E001"]["degree_centrality"].iloc[0] == 0.0
        inn = ona.degree_centrality(edges, "employee_id", "supervisor_id", mode="in").df()
        assert inn[inn["node_id"] == "E001"]["degree_centrality"].iloc[0] > 0

    def test_connected_components_single_component(
        self, ona: DuckONA, edges: duckdb.DuckDBPyRelation
    ):
        df = ona.connected_components(edges, "employee_id", "supervisor_id").df()
        assert len(df) == 1
        assert df.iloc[0]["size"] == 7

    def test_louvain_communities_returns_communities(
        self, ona: DuckONA, edges: duckdb.DuckDBPyRelation
    ):
        df = ona.louvain_communities(edges, "employee_id", "supervisor_id").df()
        assert set(df.columns) == {"node_id", "community_id"}
        assert len(df) == 7
        assert df["community_id"].nunique() >= 1


# ─── HRIS join ──────────────────────────────────────────────────────────────

class TestJoinHris:
    def test_join_hris_shape(self, ona: DuckONA, hris_df: pd.DataFrame):
        ona.load_hris(hris_df)
        edges = ona.build_org_edges()
        metrics = ona.pagerank(edges, "employee_id", "supervisor_id")
        joined = ona.join_hris(metrics)
        df = joined.df()
        assert "pagerank" in df.columns
        assert "department" in df.columns
        assert len(df) == 7


# ─── Model helpers ──────────────────────────────────────────────────────────

class TestModelHelpers:
    @pytest.fixture
    def analysis_df(self, ona: DuckONA, hris_df: pd.DataFrame, compensation_df: pd.DataFrame):
        ona.load_hris(hris_df)
        ona.load_compensation(compensation_df)
        # Build a simple analysis DataFrame in pandas.
        df = hris_df.merge(compensation_df, on="employee_id").copy()
        df["tenure_yrs"] = (pd.Timestamp("2026-06-01") - df["hire_date"]).dt.days / 365.25
        return df

    def test_ols_output_shapes(self, ona: DuckONA, analysis_df: pd.DataFrame):
        tidy, glance = ona.ols(analysis_df, "salary ~ job_level + tenure_yrs")
        assert "estimate" in tidy.columns
        assert "p.value" in tidy.columns
        assert "rsquared" in glance.columns
        assert "nobs" in glance.columns

    def test_logistic_output_shapes(self, ona: DuckONA, analysis_df: pd.DataFrame):
        analysis_df["high_paid"] = (analysis_df["salary"] > 100_000).astype(int)
        tidy, glance = ona.logistic(analysis_df, "high_paid ~ job_level + tenure_yrs")
        assert "estimate" in tidy.columns
        assert "p.value" in tidy.columns
        assert "deviance" in glance.columns or "logLik" in glance.columns


# ─── Temporal slicing ────────────────────────────────────────────────────────

class TestTemporalSlices:
    def test_build_temporal_slices_monthly(self, ona: DuckONA, attendance_df: pd.DataFrame):
        ona.load_attendance(attendance_df)
        slices = ona.build_temporal_slices("attendance", "date", freq="M")
        assert len(slices) >= 1
        for label, start, end, rel in slices:
            assert isinstance(start, (dt.date, pd.Timestamp))
            assert isinstance(end, (dt.date, pd.Timestamp))
            assert isinstance(rel, duckdb.DuckDBPyRelation)
            assert start <= end
            # Label should contain year-month
            assert isinstance(label, str) and len(label) >= 4

    def test_build_temporal_slices_daily(self, ona: DuckONA, attendance_df: pd.DataFrame):
        ona.load_attendance(attendance_df)
        slices = ona.build_temporal_slices("attendance", "date", freq="D")
        assert len(slices) == 14


# ─── MRQAP ────────────────────────────────────────────────────────────────

class TestMRQAP:
    def test_mrqap_returns_p_values_in_zero_one(self):
        n = 12
        rng = np.random.default_rng(seed=1)
        Y = rng.random((n, n))
        X = [rng.random((n, n)) for _ in range(2)]
        result = DuckONA.mrqap(Y, X, n_permutations=200)
        assert "coefficients" in result
        assert "p_values" in result
        assert "r2" in result
        for p in result["p_values"]:
            assert 0.0 <= p <= 1.0

    def test_mrqap_invalid_shape(self):
        with pytest.raises(ValueError):
            DuckONA.mrqap(np.zeros((3, 3)), [np.zeros((2, 2))], n_permutations=50)


# ─── End-to-end smoke ──────────────────────────────────────────────────────

class TestEndToEnd:
    def test_full_hr_analytics_smoke(
        self,
        ona: DuckONA,
        hris_df: pd.DataFrame,
        compensation_df: pd.DataFrame,
        survey_df: pd.DataFrame,
        turnover_df: pd.DataFrame,
        promotions_df: pd.DataFrame,
        skills_df: pd.DataFrame,
        attendance_df: pd.DataFrame,
        retirement_df: pd.DataFrame,
    ):
        ona.load_hris(hris_df)
        ona.load_compensation(compensation_df)
        ona.load_survey(survey_df)
        ona.load_turnover(turnover_df)
        ona.load_promotions(promotions_df)
        ona.load_skills(skills_df)
        ona.load_attendance(attendance_df)
        ona.load_retirement(retirement_df)

        ona.validate_keys("hris")
        ona.validate_keys("compensation", date_col="effective_date")

        edges = ona.build_org_edges()
        metrics = ona.pagerank(edges, "employee_id", "supervisor_id")
        joined = ona.join_hris(metrics)
        assert len(joined.df()) == 7

        slices = ona.build_temporal_slices("attendance", "date", freq="W")
        assert len(slices) >= 2
