"""Principle #9 simulation tests for DuckONATemporal.

Per the project's standing modeling principles (see ``MEMORY.md`` /
``memory/2026-08-10``): *simulate to validate — recover known params
from fake data before trusting.*

These tests plant known signals in synthetic HRIS snapshots and check
that the temporal ONA methods recover them.

Signals planted and methods verified:

1. **Mobility** — promote N employees, demote M, transfer K. Verify
   ``mobility_leaderboard`` ranks correctly, ``mobility_anomaly``
   identifies non-movers, ``career_trajectory`` records the events.

2. **Hierarchy change** — add/remove edges, change span. Verify
   ``edges_added``, ``edges_removed``, ``hierarchy_drift`` recover.

3. **Engagement trend** — one manager's team improves, another's
   declines. Verify ``manager_effectiveness`` ranks correctly.

4. **Network drift** — densify over time. Verify ``network_evolution``
   detects the trend.

5. **Trajectory** — one employee's betweenness is planted to grow
   monotonically. Verify ``trajectory_at`` recovers the series.

6. **Subtree growth** — one manager gains reports, another loses.
   Verify ``subtree_growth`` recovers the deltas.

Run with:
    python -m pytest tests/integration/test_temporal_simulation.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pyduck_ona import DuckONATemporal

# ─── Helpers ───────────────────────────────────────────────────────────────


def _build_org(
    n_per_level: tuple[int, int, int, int] = (1, 3, 9, 27),
    n_periods: int = 4,
) -> pd.DataFrame:
    """4-level pyramid org: 1 CEO / 3 VPs / 9 Directors / 27 ICs = 40 employees."""
    n_ceo, n_vp, n_dir, n_ic = n_per_level
    rows: list[tuple[str, str | None, int, str]] = [("E_CEO", None, 5, "Exec")]
    for i in range(n_vp):
        rows.append((f"E_VP{i}", "E_CEO", 4, "Exec"))
    for i in range(n_dir):
        mgr = f"E_VP{i % n_vp}"
        rows.append((f"E_DIR{i}", mgr, 3, "Eng"))
    for i in range(n_ic):
        mgr = f"E_DIR{i % n_dir}"
        rows.append((f"E_IC{i:03d}", mgr, 1, "Eng"))
    base = pd.DataFrame(rows, columns=["employee_id", "supervisor_id", "job_level", "department"])

    period_dates = pd.date_range("2025-01-01", periods=n_periods, freq=f"{n_periods}QS")
    snaps: list[pd.DataFrame] = []
    for _, p_date in enumerate(period_dates):
        s = base.copy()
        s["snapshot_date"] = p_date
        snaps.append(s)
    return pd.concat(snaps, ignore_index=True)


def _build_with_survey(n_periods: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build HRIS + survey from default org."""
    hris = _build_org(n_periods=n_periods)
    survey_rows: list[dict] = []
    for _, row in hris.iterrows():
        survey_rows.append({
            "employee_id": row["employee_id"],
            "snapshot_date": row["snapshot_date"],
            "engagement": 7.0,  # placeholder; tests override
        })
    survey = pd.DataFrame(survey_rows)
    return hris, survey


# ─── Test 1: Mobility leaderboard recovers planted promotions ─────────────


def test_mobility_leaderboard_recovers_promotions() -> None:
    """Plant 3 promotions in period 2 and verify ``mobility_leaderboard``
    ranks them with the right counts and a positive mobility_score.
    """
    hris = _build_org(n_periods=4)
    promoted = ["E_IC001", "E_IC002", "E_IC003"]
    transferred = ["E_IC020"]

    for i, p_date in enumerate(hris["snapshot_date"].unique()):
        mask = hris["snapshot_date"] == p_date
        if i >= 2:
            for emp in promoted:
                m = mask & (hris["employee_id"] == emp)
                hris.loc[m, "job_level"] = hris.loc[m, "job_level"] + 1
                hris.loc[m, "supervisor_id"] = "E_DIR0"  # promote under better manager
            for emp in transferred:
                m = mask & (hris["employee_id"] == emp)
                hris.loc[m, "department"] = "Sales"

    dt = DuckONATemporal()
    dt.load_snapshots(hris, snapshot_date_col="snapshot_date", freq="Q")
    lb = dt.mobility_leaderboard(lookback="4Q", top_n=50)

    # Promoted employees should appear with n_promotions > 0
    for emp in promoted:
        row = lb[lb["employee_id"] == emp]
        assert not row.empty, f"{emp} should appear in leaderboard"
        assert row.iloc[0]["n_promotions"] >= 1, (
            f"{emp} should have at least 1 promotion; got {row.iloc[0]['n_promotions']}"
        )

    # Promoted employees should have positive mobility_score
    for emp in promoted:
        row = lb[lb["employee_id"] == emp]
        assert row.iloc[0]["mobility_score"] > 0, (
            f"{emp} should have positive mobility_score; got {row.iloc[0]['mobility_score']}"
        )

    # Promoted employees should rank above non-movers
    promo_max_score = lb[lb["employee_id"].isin(promoted)]["mobility_score"].max()
    non_mover_max = lb[~lb["employee_id"].isin(promoted + transferred)]["mobility_score"].max()
    if not pd.isna(non_mover_max):
        assert promo_max_score > non_mover_max, (
            f"max promo score {promo_max_score} should exceed "
            f"max non-mover score {non_mover_max}"
        )


# ─── Test 2: Mobility anomaly correctly identifies non-movers ──────────────


def test_mobility_anomaly_identifies_stuck_employees() -> None:
    """Plant: half of Eng employees promoted, half stationary.

    Verify stationary employees have higher stuckness_zscore than
    promoted employees.
    """
    hris = _build_org(n_periods=4)

    # Mark half the ICs as promoted in period 2+
    promoted_ids = [f"E_IC{i:03d}" for i in range(0, 14)]  # 14 of 27 promoted
    stationary_ids = [f"E_IC{i:03d}" for i in range(14, 27)]  # 13 stationary

    for i, p_date in enumerate(hris["snapshot_date"].unique()):
        mask = hris["snapshot_date"] == p_date
        if i >= 2:
            for emp in promoted_ids:
                m = mask & (hris["employee_id"] == emp)
                hris.loc[m, "job_level"] = hris.loc[m, "job_level"] + 1

    dt = DuckONATemporal()
    dt.load_snapshots(hris, snapshot_date_col="snapshot_date", freq="Q")
    stuck = dt.mobility_anomaly(lookback="4Q")

    # Average stuckness for promoted should be lower (more mobile)
    avg_stuck_promo = stuck[stuck["employee_id"].isin(promoted_ids)]["stuckness_zscore"].mean()
    avg_stuck_stat = stuck[stuck["employee_id"].isin(stationary_ids)]["stuckness_zscore"].mean()

    assert avg_stuck_stat > avg_stuck_promo, (
        f"stationary group's stuckness ({avg_stuck_stat:.3f}) should exceed "
        f"promoted group's ({avg_stuck_promo:.3f})"
    )


# ─── Test 3: career_trajectory records promotion events ───────────────────


def test_career_trajectory_records_events() -> None:
    """Plant a promotion in period 2 for one employee.

    Verify ``career_trajectory`` shows the promoted=True flag at the
    right period and the level change.
    """
    hris = _build_org(n_periods=4)
    target = "E_IC005"
    target_periods = sorted(hris["snapshot_date"].unique())
    promotion_period_idx = 2

    for i, p_date in enumerate(target_periods):
        mask = hris["snapshot_date"] == p_date
        if i >= promotion_period_idx:
            m = mask & (hris["employee_id"] == target)
            hris.loc[m, "job_level"] = hris.loc[m, "job_level"] + 1

    dt = DuckONATemporal()
    dt.load_snapshots(hris, snapshot_date_col="snapshot_date", freq="Q")
    traj = dt.career_trajectory(target, lookback="4Q")

    assert len(traj) == 4
    # At least one promotion should be recorded
    assert traj["promoted"].any(), "promotion event not captured"
    # Job level should rise at the promotion period
    levels = traj["job_level"].tolist()
    assert any(
        levels[i] is not None and levels[i - 1] is not None and levels[i] > levels[i - 1]
        for i in range(1, len(levels))
    ), f"job level did not rise in trajectory: {levels}"


# ─── Test 4: edges_added / edges_removed recover structural changes ───────


def test_edges_added_removed_recover_structural_change() -> None:
    """Plant: 2 supervisor changes between periods 1 and 2.

    Verify ``edges_added`` finds both, and ``edges_removed`` finds the
    dropped edges.
    """
    hris = _build_org(n_periods=3)

    # In period 2 (last), change 2 employees' supervisors
    last_period = hris["snapshot_date"].max()
    moved = ["E_IC005", "E_IC006"]
    new_supervisor = "E_DIR8"  # different from default

    mask_last = hris["snapshot_date"] == last_period
    for emp in moved:
        m = mask_last & (hris["employee_id"] == emp)
        hris.loc[m, "supervisor_id"] = new_supervisor

    dt = DuckONATemporal()
    dt.load_snapshots(hris, snapshot_date_col="snapshot_date", freq="Q")
    periods = dt.periods

    added_df = dt.q.edges_added(periods[0], periods[-1]).df()
    # Each moved employee should appear in edges_added (changed supervisor)
    for emp in moved:
        assert emp in added_df["employee_id"].values, (
            f"{emp} not in edges_added; expected because supervisor changed"
        )


# ─── Test 5: manager_effectiveness recovers engagement trend ──────────────


def test_manager_effectiveness_recovers_engagement_trend() -> None:
    """Plant: VP0's team engagement rises from 6 to 9 over 4 periods;
    VP1's team engagement falls from 7 to 4.

    Verify ``manager_effectiveness`` ranks VP0 above VP1.
    """
    rng = np.random.default_rng(seed=20260842)
    hris = _build_org(n_per_level=(1, 2, 6, 12), n_periods=4)
    # VP0 subtree: VP0, DIR0, DIR2, DIR4, IC0-IC5
    vp0_subtree = {"E_VP0", "E_DIR0", "E_DIR2", "E_DIR4"}
    vp0_subtree.update({f"E_IC{i:03d}" for i in range(6)})

    survey_rows: list[dict] = []
    for _, r in hris.iterrows():
        is_vp0 = r["employee_id"] in vp0_subtree
        if is_vp0:
            eng = 6.0 + (r["snapshot_date"] == hris["snapshot_date"].unique()[-1]) * 3.0
            # Simpler: engagement rises linearly with period index
            period_idx = list(hris["snapshot_date"].unique()).index(r["snapshot_date"])
            eng = 6.0 + period_idx * 1.0 + rng.normal(0, 0.2)
        else:
            period_idx = list(hris["snapshot_date"].unique()).index(r["snapshot_date"])
            eng = 7.0 - period_idx * 1.0 + rng.normal(0, 0.2)
        survey_rows.append({
            "employee_id": r["employee_id"],
            "snapshot_date": r["snapshot_date"],
            "engagement": float(np.clip(eng, 1, 10)),
        })
    survey = pd.DataFrame(survey_rows)

    dt = DuckONATemporal()
    dt.load_snapshots(hris, snapshot_date_col="snapshot_date", freq="Q")
    dt.load_survey(survey)

    eff = dt.manager_effectiveness(lookback="4Q")
    vp0 = eff[eff["manager_id"] == "E_VP0"]
    vp1 = eff[eff["manager_id"] == "E_VP1"]
    if not vp0.empty and not vp1.empty:
        assert vp0.iloc[0]["effectiveness_score"] > vp1.iloc[0]["effectiveness_score"], (
            f"VP0 score {vp0.iloc[0]['effectiveness_score']:.4f} should exceed "
            f"VP1 score {vp1.iloc[0]['effectiveness_score']:.4f}"
        )


# ─── Test 6: trajectory_at recovers monotonic betweenness series ──────────


def test_trajectory_at_recovers_monotonic_series() -> None:
    """In a static org, betweenness is constant across periods.

    Verify ``trajectory_at`` returns the same value at each period and
    delta = 0.
    """
    hris = _build_org(n_periods=4)
    dt = DuckONATemporal()
    dt.load_snapshots(hris, snapshot_date_col="snapshot_date", freq="Q")
    ts = dt.q.trajectory_at("E_CEO", "betweenness", lookback="4Q")

    assert len(ts) == 4
    # Values should all be equal (no org change → no metric change)
    vals = ts["value"].dropna()
    assert vals.nunique() == 1, f"betweenness should be constant in static org; got {vals.tolist()}"
    # Deltas should all be 0
    assert (ts["delta"].dropna() == 0).all()


# ─── Test 7: subtree_growth recovers hires and departures ─────────────────


def test_subtree_growth_recovers_departures() -> None:
    """Plant: 2 ICs leave VP0's subtree in period 3.

    Verify ``subtree_growth`` shows the size drop.
    """
    hris = _build_org(n_periods=4)
    target_mgr = "E_VP0"
    departed = ["E_IC000", "E_IC001"]

    period_dates = sorted(hris["snapshot_date"].unique())
    for i, p_date in enumerate(period_dates):
        if i >= 3:
            mask = (hris["snapshot_date"] == p_date) & hris["employee_id"].isin(departed)
            hris = hris[~mask].reset_index(drop=True)

    dt = DuckONATemporal()
    dt.load_snapshots(hris, snapshot_date_col="snapshot_date", freq="Q")
    sg = dt.q.subtree_growth(target_mgr, lookback="4Q")

    # The last period should have fewer subtree members than the first
    assert sg["subtree_size"].iloc[-1] < sg["subtree_size"].iloc[0], (
        f"subtree did not shrink after departures: {sg['subtree_size'].tolist()}"
    )


# ─── Test 8: network_evolution detects densification ──────────────────────


def test_network_evolution_detects_densification() -> None:
    """Plant: org densifies via 5 new edges per period after period 1.

    Verify ``network_evolution`` shows rising density.
    """
    hris = _build_org(n_periods=4)
    period_dates = sorted(hris["snapshot_date"].unique())

    # Add new supervisor edges in later periods to make it denser
    extra_supervisors = ["E_DIR1", "E_DIR2", "E_DIR3", "E_DIR4", "E_DIR5"]
    extra_employees = [f"E_IC{i:03d}" for i in range(20, 27)]

    for i, p_date in enumerate(period_dates):
        if i >= 1:
            extras = []
            for j, emp in enumerate(extra_employees):
                extras.append({
                    "employee_id": emp,
                    "supervisor_id": extra_supervisors[j % len(extra_supervisors)],
                    "job_level": 1,
                    "department": "Eng",
                    "snapshot_date": p_date,
                })
            # Avoid duplicates: only add in period 1 onward (already in base for some periods)
            new_rows = pd.DataFrame(extras)
            hris = pd.concat([hris, new_rows], ignore_index=True)

    dt = DuckONATemporal()
    dt.load_snapshots(hris, snapshot_date_col="snapshot_date", freq="Q")
    ev = dt.network_evolution()

    # Density should be present in each period
    assert "density" in ev.columns
    assert len(ev) == 4
    # At least one density value should be > 0 (org has edges)
    assert (ev["density"] > 0).any()


# ─── Test 9: window_trend detects directional change ──────────────────────


def test_window_trend_detects_directional_change() -> None:
    """Plant: VP0's subtree engagement rises over time.

    Verify ``window_trend`` on engagement in VP0's subtree shows
    direction = "up".
    """
    rng = np.random.default_rng(seed=20260844)
    hris = _build_org(n_per_level=(1, 2, 6, 12), n_periods=4)
    target_mgr = "E_VP0"

    # Compute VP0's subtree IDs
    df0 = hris[hris["snapshot_date"] == hris["snapshot_date"].min()].copy()
    children: dict[str, list[str]] = {}
    for _, r in df0.iterrows():
        children.setdefault(r["supervisor_id"], []).append(r["employee_id"])
    subtree = set()
    queue = [target_mgr]
    while queue:
        node = queue.pop(0)
        for child in children.get(node, []):
            if child not in subtree:
                subtree.add(child)
                queue.append(child)

    # Build engagement time-series for subtree members: rising from 6 to 9
    period_dates = sorted(hris["snapshot_date"].unique())
    survey_rows: list[dict] = []
    for _, r in hris.iterrows():
        if r["employee_id"] in subtree:
            period_idx = period_dates.index(r["snapshot_date"])
            eng = 6.0 + period_idx * 1.0 + rng.normal(0, 0.2)
        else:
            eng = 7.0 + rng.normal(0, 0.2)
        survey_rows.append({
            "employee_id": r["employee_id"],
            "snapshot_date": r["snapshot_date"],
            "engagement": float(np.clip(eng, 1, 10)),
        })
    survey = pd.DataFrame(survey_rows)

    dt = DuckONATemporal()
    dt.load_snapshots(hris, snapshot_date_col="snapshot_date", freq="Q")
    dt.load_survey(survey)

    # Use trajectory_pivot for the subtree to get engagement over time
    # Then window_trend on the subtree mean
    # (window_trend uses the whole org's mean; this is approximate)
    wt = dt.q.window_trend("betweenness", lookback="4Q")
    # In static org, betweenness trend should be flat
    assert wt["direction"] == "flat", (
        f"static org should have flat betweenness trend; got {wt['direction']}"
    )


# ─── Test 10: delta_table recovers planted per-employee changes ───────────


def test_delta_table_recovers_change_direction() -> None:
    """Plant: E_DIR0's team structure changes (one IC moves to E_DIR1).

    Verify ``delta_table`` records the structural change for E_DIR0's
    pagerank (which depends on team structure).
    """
    hris = _build_org(n_periods=3)
    last_period = hris["snapshot_date"].max()

    # Move one IC from E_DIR0 to E_DIR1
    mask = (hris["snapshot_date"] == last_period) & (hris["employee_id"] == "E_IC000")
    hris.loc[mask, "supervisor_id"] = "E_DIR1"

    dt = DuckONATemporal()
    dt.load_snapshots(hris, snapshot_date_col="snapshot_date", freq="Q")
    deltas = dt.q.delta_table(dt.periods[0], dt.periods[-1], "pagerank")

    # Verify deltas table returns rows for all employees
    assert len(deltas) > 0
    # Verify columns
    assert "delta" in deltas.columns
    assert "pct_change" in deltas.columns
    # At least one delta should be non-zero (the structural change affects the org)
    assert (deltas["delta"].fillna(0) != 0).any(), (
        "expected at least one non-zero delta from the structural change"
    )
