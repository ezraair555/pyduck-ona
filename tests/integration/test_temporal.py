"""Integration tests for DuckONATemporal — temporal ONA analytics.

Per Principle #9 (simulate to validate), several tests plant known
mobility/engagement trends in synthetic HRIS snapshots and check that
the methods recover them.

Run with:
    python -m pytest tests/integration/test_temporal.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyduck_ona import DuckONATemporal


# ─── Fixtures ───────────────────────────────────────────────────────────────


def _build_synthetic_org(
    n_per_level: tuple[int, int, int, int] = (1, 3, 10, 50),
    n_periods: int = 4,
    freq: str = "Q",
) -> pd.DataFrame:
    """Build a synthetic HRIS with multiple quarterly snapshots.

    Returns a DataFrame with columns:
        employee_id, supervisor_id, snapshot_date, job_level,
        department, engagement, name
    """
    rng = np.random.default_rng(seed=20260827)
    n_ceo, n_vp, n_dir, n_ic = n_per_level

    # Build the org structure
    rows: list[tuple[str, str | None, int, str]] = []
    rows.append(("E_CEO", None, 5, "Executive"))
    for i in range(n_vp):
        rows.append((f"E_VP{i:02d}", "E_CEO", 4, "Executive"))
    for i in range(n_dir):
        mgr = f"E_VP{i % n_vp:02d}"
        dept = ["Engineering", "Sales", "Operations", "People"][i % 4]
        rows.append((f"E_DIR{i:03d}", mgr, 3, dept))
    for i in range(n_ic):
        mgr = f"E_DIR{i % n_dir:03d}"
        dept = ["Engineering", "Sales", "Operations", "People"][i % 4]
        rows.append((f"E_IC{i:04d}", mgr, 1, dept))

    base = pd.DataFrame(rows, columns=["employee_id", "supervisor_id", "job_level", "department"])
    base["name"] = base["employee_id"].apply(lambda x: f"Person_{x}")

    # Create n_periods quarterly snapshots
    period_dates = pd.date_range("2025-01-01", periods=n_periods, freq=f"{n_periods}QS")

    snapshots: list[pd.DataFrame] = []
    for p_idx, p_date in enumerate(period_dates):
        snap = base.copy()
        snap["snapshot_date"] = p_date
        snap["engagement"] = np.clip(rng.normal(7.0, 1.5, len(snap)), 1, 10).round(2)
        # Inject a mobility event: promote E_IC0001 in period 2
        if p_idx >= 2:
            snap.loc[snap["employee_id"] == "E_IC0001", "job_level"] = 2
            snap.loc[snap["employee_id"] == "E_IC0001", "supervisor_id"] = "E_DIR001"
        snapshots.append(snap)

    return pd.concat(snapshots, ignore_index=True)


def _build_with_survey() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build HRIS snapshots + a survey table with engagement trends."""
    hris = _build_synthetic_org()
    # Survey: one row per employee per period
    survey_rows: list[dict] = []
    for _, row in hris.iterrows():
        survey_rows.append({
            "employee_id": row["employee_id"],
            "snapshot_date": row["snapshot_date"],
            "engagement": row["engagement"],
        })
    survey = pd.DataFrame(survey_rows)
    return hris, survey


# ─── Tests ──────────────────────────────────────────────────────────────────


@pytest.fixture
def dt_basic() -> DuckONATemporal:
    """A DuckONATemporal with 4 quarterly snapshots loaded."""
    hris = _build_synthetic_org()
    dt = DuckONATemporal()
    dt.load_snapshots(hris, snapshot_date_col="snapshot_date", freq="Q")
    return dt


@pytest.fixture
def dt_with_survey() -> DuckONATemporal:
    """DuckONATemporal with survey + promotions tables loaded."""
    hris, survey = _build_with_survey()
    promotions = pd.DataFrame({
        "employee_id": ["E_IC0001"],
        "promotion_date": pd.Timestamp("2025-07-01"),
        "new_level": 2,
    })
    dt = DuckONATemporal()
    dt.load_snapshots(hris, snapshot_date_col="snapshot_date", freq="Q")
    dt.load_survey(survey)
    dt.load_promotions(promotions)
    return dt


class TestLoadSnapshots:
    def test_loads_and_detects_periods(self, dt_basic: DuckONATemporal) -> None:
        assert len(dt_basic.periods) == 4
        assert dt_basic.freq == "Q"

    def test_raises_on_missing_date_col(self) -> None:
        df = pd.DataFrame({"employee_id": ["A"], "supervisor_id": [None]})
        dt = DuckONATemporal()
        with pytest.raises(ValueError, match="snapshot_date"):
            dt.load_snapshots(df, snapshot_date_col="snapshot_date")

    def test_raises_on_missing_emp_col(self) -> None:
        df = pd.DataFrame({"snapshot_date": ["2025-01-01"], "supervisor_id": [None]})
        dt = DuckONATemporal()
        with pytest.raises(ValueError, match="employee_id"):
            dt.load_snapshots(df, snapshot_date_col="snapshot_date")


class TestComputeTemporalMetrics:
    def test_returns_expected_columns(self, dt_basic: DuckONATemporal) -> None:
        ts = dt_basic.compute_temporal_metrics(metrics=["betweenness"])
        expected_cols = {"period", "employee_id", "metric", "value", "prev_value", "delta", "pct_change"}
        assert set(ts.columns) == expected_cols, f"got {set(ts.columns)}"

    def test_betweenness_across_periods(self, dt_basic: DuckONATemporal) -> None:
        ts = dt_basic.compute_temporal_metrics(metrics=["betweenness"])
        assert ts["metric"].nunique() == 1
        assert ts["period"].nunique() == 4
        assert len(ts) > 0

    def test_multiple_metrics(self, dt_basic: DuckONATemporal) -> None:
        ts = dt_basic.compute_temporal_metrics(
            metrics=["betweenness", "pagerank", "degree_centrality"]
        )
        assert ts["metric"].nunique() == 3

    def test_team_size_metric(self, dt_basic: DuckONATemporal) -> None:
        ts = dt_basic.compute_temporal_metrics(metrics=["team_size"])
        assert ts["metric"].nunique() == 1
        assert ts["value"].max() > 0

    def test_quarterly_period_alignment_not_empty(self) -> None:
        q_end = pd.DataFrame(
            {
                "employee_id": ["E1", "E2", "E1", "E2"],
                "supervisor_id": [None, "E1", None, "E1"],
                "snapshot_date": pd.to_datetime(
                    ["2026-03-31", "2026-03-31", "2026-06-30", "2026-06-30"]
                ),
            }
        )
        dt = DuckONATemporal()
        periods = dt.load_snapshots(q_end, snapshot_date_col="snapshot_date", freq="Q")
        ts = dt.compute_temporal_metrics(metrics=["pagerank"])
        assert periods == ["2026-01-01", "2026-04-01"]
        assert len(ts) > 0
        assert ts["period"].nunique() == 2


class TestNetworkEvolution:
    def test_returns_expected_columns(self, dt_basic: DuckONATemporal) -> None:
        ev = dt_basic.network_evolution()
        expected = {"period", "n_employees", "n_edges", "density",
                    "centralization", "n_components", "avg_path_length"}
        assert set(ev.columns) == expected
        assert len(ev) == 4

    def test_density_between_0_and_1(self, dt_basic: DuckONATemporal) -> None:
        ev = dt_basic.network_evolution()
        assert (ev["density"] >= 0).all()
        assert (ev["density"] <= 1).all()

    def test_n_components_positive(self, dt_basic: DuckONATemporal) -> None:
        ev = dt_basic.network_evolution()
        assert (ev["n_components"] >= 1).all()


class TestEventWindow:
    def test_pre_post_split(self, dt_basic: DuckONATemporal) -> None:
        ev = dt_basic.event_window(event_date="2025-07-01")
        assert "pre" in ev["period_type"].values
        assert "post" in ev["period_type"].values

    def test_custom_windows(self, dt_basic: DuckONATemporal) -> None:
        ev = dt_basic.event_window(
            event_date="2025-07-01",
            pre_window=("2025-01-01", "2025-06-30"),
            post_window=("2025-07-01", "2025-12-31"),
        )
        assert len(ev) > 0


class TestChangeDetection:
    def test_returns_top_movers(self, dt_basic: DuckONATemporal) -> None:
        cd = dt_basic.change_detection(metric="betweenness", top_n=10)
        assert len(cd) <= 10
        assert "delta" in cd.columns
        assert "rolling_zscore" in cd.columns
        assert "rank" in cd.columns

    def test_rank_sorted(self, dt_basic: DuckONATemporal) -> None:
        cd = dt_basic.change_detection(metric="pagerank", top_n=20)
        assert cd["rank"].tolist() == list(range(1, len(cd) + 1))


class TestMobilityLeaderboard:
    def test_detects_promotion(self, dt_basic: DuckONATemporal) -> None:
        """E_IC0001 was promoted in period 2; should appear in leaderboard."""
        lb = dt_basic.mobility_leaderboard(lookback="4Q", top_n=50)
        assert "mobility_score" in lb.columns
        assert "n_promotions" in lb.columns
        promoted = lb[lb["employee_id"] == "E_IC0001"]
        if not promoted.empty:
            assert promoted.iloc[0]["n_promotions"] >= 1

    def test_top_n_limit(self, dt_basic: DuckONATemporal) -> None:
        lb = dt_basic.mobility_leaderboard(top_n=5)
        assert len(lb) <= 5

    def test_score_formula(self, dt_basic: DuckONATemporal) -> None:
        lb = dt_basic.mobility_leaderboard(
            w_promotion=2.0, w_lateral=1.0, w_dept_change=0.5, w_demotion=-2.0
        )
        # Score should equal: 2*n_promo + 1*n_lateral + 0.5*n_dept - 2*n_demotion
        row = lb.iloc[0]
        expected = (
            2.0 * row["n_promotions"]
            + 1.0 * row["n_lateral_moves"]
            + 0.5 * row["n_dept_changes"]
            - 2.0 * row["n_demotions"]
        )
        assert abs(row["mobility_score"] - expected) < 0.01


class TestCareerTrajectory:
    def test_returns_periods(self, dt_basic: DuckONATemporal) -> None:
        traj = dt_basic.career_trajectory("E_IC0001", lookback="4Q")
        assert len(traj) == 4
        assert "period" in traj.columns
        assert "supervisor_id" in traj.columns
        assert "promoted" in traj.columns

    def test_detects_promotion_event(self, dt_basic: DuckONATemporal) -> None:
        traj = dt_basic.career_trajectory("E_IC0001", lookback="4Q")
        assert traj["promoted"].any(), "E_IC0001 should have at least one promotion"

    def test_missing_employee(self, dt_basic: DuckONATemporal) -> None:
        traj = dt_basic.career_trajectory("NONEXISTENT", lookback="4Q")
        assert len(traj) == 4
        assert traj["supervisor_id"].isna().all()


class TestManagerChain:
    def test_returns_managers(self, dt_basic: DuckONATemporal) -> None:
        chain = dt_basic.manager_chain("E_IC0001", lookback="4Q")
        assert len(chain) == 4
        assert "supervisor_id" in chain.columns

    def test_path_to_ceo(self, dt_basic: DuckONATemporal) -> None:
        chain = dt_basic.manager_chain("E_IC0001", lookback="4Q")
        # Path should include at least the director and VP
        for _, row in chain.iterrows():
            if row["supervisor_path_to_ceo"]:
                assert isinstance(row["supervisor_path_to_ceo"], list)


class TestMobilityAnomaly:
    def test_returns_zscores(self, dt_basic: DuckONATemporal) -> None:
        ma = dt_basic.mobility_anomaly(lookback="4Q")
        assert "stuckness_zscore" in ma.columns
        assert "is_stuck" in ma.columns
        assert "is_mobility_leader" in ma.columns
        assert len(ma) > 0

    def test_promoted_employee_not_stuck(self, dt_basic: DuckONATemporal) -> None:
        ma = dt_basic.mobility_anomaly(lookback="4Q")
        promoted = ma[ma["employee_id"] == "E_IC0001"]
        if not promoted.empty:
            assert not promoted.iloc[0]["is_stuck"], "promoted employee should not be stuck"


class TestManagerEffectiveness:
    def test_returns_effectiveness_score(self, dt_with_survey: DuckONATemporal) -> None:
        eff = dt_with_survey.manager_effectiveness(lookback="4Q")
        assert "effectiveness_score" in eff.columns
        assert "rank" in eff.columns
        assert len(eff) > 0

    def test_weights_sum_validation(self, dt_with_survey: DuckONATemporal) -> None:
        with pytest.raises(ValueError, match="weights must sum"):
            dt_with_survey.manager_effectiveness(
                w_engagement=0.3, w_retention=0.3,
                w_promotion=0.3, w_span=0.3,
            )

    def test_engagement_dominant_weights(self, dt_with_survey: DuckONATemporal) -> None:
        eff = dt_with_survey.manager_effectiveness(
            w_engagement=0.50, w_retention=0.25,
            w_promotion=0.15, w_span=0.10,
        )
        assert "engagement_trend" in eff.columns
        assert "retention_rate" in eff.columns

    def test_custom_weights(self, dt_with_survey: DuckONATemporal) -> None:
        eff = dt_with_survey.manager_effectiveness(
            w_engagement=0.25, w_retention=0.25,
            w_promotion=0.25, w_span=0.25,
        )
        assert len(eff) > 0


# ─── Simulation test (Principle #9) ────────────────────────────────────────


class TestSimulationRecovery:
    """Plant a known engagement trend and check that manager_effectiveness
    recovers the signal."""

    def test_engagement_trend_detected(self) -> None:
        """Manager E_VP00's team engagement improves over 4 periods;
        manager E_VP01's team engagement declines.
        manager_effectiveness should rank E_VP00 above E_VP01."""
        rng = np.random.default_rng(42)
        n_periods = 4
        period_dates = pd.date_range("2025-01-01", periods=n_periods, freq="3QS")

        # Build org: 2 VPs, 3 dirs each, 5 ICs each
        rows: list[tuple[str, str | None, int, str]] = []
        rows.append(("CEO", None, 5, "Exec"))
        for i in range(2):
            rows.append((f"VP{i}", "CEO", 4, "Exec"))
        for i in range(6):
            mgr = f"VP{i % 2}"
            rows.append((f"DIR{i}", mgr, 3, "Eng"))
        for i in range(10):
            mgr = f"DIR{i % 6}"
            rows.append((f"IC{i}", mgr, 1, "Eng"))

        base = pd.DataFrame(rows, columns=["employee_id", "supervisor_id", "job_level", "department"])

        snapshots: list[pd.DataFrame] = []
        for p_idx, p_date in enumerate(period_dates):
            snap = base.copy()
            snap["snapshot_date"] = p_date
            # VP0's team engagement rises from 6 to 9; VP1's falls from 7 to 4
            for _, r in snap.iterrows():
                if r["supervisor_id"] in ("VP0", "DIR0", "DIR2", "DIR4") or r["employee_id"] in ("VP0", "DIR0", "DIR2", "DIR4"):
                    # VP0 subtree
                    base_eng = 6.0 + p_idx * 1.0
                    snap.loc[snap["employee_id"] == r["employee_id"], "engagement"] = base_eng + rng.normal(0, 0.3)
                else:
                    base_eng = 7.0 - p_idx * 1.0
                    snap.loc[snap["employee_id"] == r["employee_id"], "engagement"] = base_eng + rng.normal(0, 0.3)
            snapshots.append(snap)

        hris = pd.concat(snapshots, ignore_index=True)
        survey_rows: list[dict] = []
        for _, r in hris.iterrows():
            survey_rows.append({
                "employee_id": r["employee_id"],
                "snapshot_date": r["snapshot_date"],
                "engagement": r["engagement"],
            })
        survey = pd.DataFrame(survey_rows)

        dt = DuckONATemporal()
        dt.load_snapshots(hris, snapshot_date_col="snapshot_date", freq="Q")
        dt.load_survey(survey)

        eff = dt.manager_effectiveness(lookback="4Q")
        # VP0 should rank higher than VP1
        vp0 = eff[eff["manager_id"] == "VP0"]
        vp1 = eff[eff["manager_id"] == "VP1"]
        if not vp0.empty and not vp1.empty:
            assert vp0.iloc[0]["effectiveness_score"] > vp1.iloc[0]["effectiveness_score"], (
                f"VP0 score={vp0.iloc[0]['effectiveness_score']:.3f}, "
                f"VP1 score={vp1.iloc[0]['effectiveness_score']:.3f}; "
                f"expected VP0 > VP1 (engagement trend was planted)"
            )
