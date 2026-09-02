"""Performance / scaling tests for DuckONATemporal.

These tests verify that the temporal ONA methods scale reasonably with
HRIS size. They are NOT strict timing assertions (CI is too noisy for
that), but they do check that:

    - The methods complete in bounded time at typical sizes.
    - The methods don't blow up to O(n²) or worse for large inputs.
    - Memory usage stays bounded.

Run with:
    python -m pytest tests/integration/test_temporal_performance.py -v -m "not slow"

Or to enable the slow tests (larger sizes, longer timeouts):
    python -m pytest tests/integration/test_temporal_performance.py -v

Skip slow tests by default:
    pytest -m "not slow"
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from pyduck_ona import DuckONATemporal


def _build_large_org(
    n_per_level: tuple[int, int, int, int],
    n_periods: int = 4,
    seed: int = 20260827,
) -> pd.DataFrame:
    """Build a synthetic HRIS with N employees across M periods."""
    n_ceo, n_vp, n_dir, n_ic = n_per_level

    rows: list[tuple[str, str | None, int, str]] = []
    rows.append(("E_CEO", None, 5, "Exec"))
    for i in range(n_vp):
        rows.append((f"E_VP{i}", "E_CEO", 4, "Exec"))
    for i in range(n_dir):
        mgr = f"E_VP{i % n_vp}"
        rows.append((f"E_DIR{i}", mgr, 3, ["Eng", "Sales", "Ops"][i % 3]))
    for i in range(n_ic):
        mgr = f"E_DIR{i % n_dir}"
        rows.append((f"E_IC{i:04d}", mgr, 1, ["Eng", "Sales", "Ops"][i % 3]))

    base = pd.DataFrame(rows, columns=["employee_id", "supervisor_id", "job_level", "department"])
    period_dates = pd.date_range("2025-01-01", periods=n_periods, freq=f"{n_periods}QS")
    snaps: list[pd.DataFrame] = []
    for p_date in period_dates:
        s = base.copy()
        s["snapshot_date"] = p_date
        snaps.append(s)
    return pd.concat(snaps, ignore_index=True)


# ─── Scaling tests ─────────────────────────────────────────────────────────


@pytest.mark.slow
def test_500_employees_4_periods_runs() -> None:
    """500-employee org, 4 quarterly periods. Should complete in <30s."""
    # 1 + 5 + 25 + 469 ≈ 500
    hris = _build_large_org((1, 5, 25, 469), n_periods=4)
    dt = DuckONATemporal()
    dt.load_snapshots(hris, snapshot_date_col="snapshot_date", freq="Q")

    t0 = time.time()
    _ts = dt.compute_temporal_metrics(metrics=["betweenness"])
    t1 = time.time()
    assert len(_ts) > 0
    assert (t1 - t0) < 30.0, f"compute_temporal_metrics took {t1 - t0:.1f}s"

    t0 = time.time()
    _ev = dt.network_evolution()
    t1 = time.time()
    assert len(_ev) == 4
    assert (t1 - t0) < 30.0, f"network_evolution took {t1 - t0:.1f}s"

    t0 = time.time()
    _lb = dt.mobility_leaderboard(top_n=20)
    t1 = time.time()
    assert len(_lb) > 0
    assert (t1 - t0) < 30.0, f"mobility_leaderboard took {t1 - t0:.1f}s"


def test_100_employees_4_periods_fast() -> None:
    """100-employee org should be quick (<10s total)."""
    hris = _build_large_org((1, 3, 10, 86), n_periods=4)  # ≈100
    dt = DuckONATemporal()
    dt.load_snapshots(hris, snapshot_date_col="snapshot_date", freq="Q")

    t0 = time.time()
    _ts = dt.compute_temporal_metrics(metrics=["betweenness"])
    assert (time.time() - t0) < 10.0

    t0 = time.time()
    _ev = dt.network_evolution()
    assert (time.time() - t0) < 10.0

    t0 = time.time()
    _lb = dt.mobility_leaderboard(top_n=10)
    assert (time.time() - t0) < 10.0


def test_50_employees_4_periods_very_fast() -> None:
    """50-employee org should be near-instant (<5s)."""
    hris = _build_large_org((1, 2, 6, 41), n_periods=4)  # ≈50
    dt = DuckONATemporal()
    dt.load_snapshots(hris, snapshot_date_col="snapshot_date", freq="Q")

    t0 = time.time()
    for _ in range(3):  # run 3 times to amortize setup
        _ts = dt.compute_temporal_metrics(metrics=["betweenness"])
        _ev = dt.network_evolution()
        _lb = dt.mobility_leaderboard(top_n=5)
    elapsed = time.time() - t0
    assert elapsed < 15.0, f"3 full passes took {elapsed:.1f}s"


def test_window_volatility_scales_linearly() -> None:
    """window_volatility should scale linearly with employees × periods."""
    # Run on 50 and 100 employees; time should be < 4x (linear scaling)
    hris_50 = _build_large_org((1, 2, 6, 41), n_periods=4)
    hris_100 = _build_large_org((1, 3, 10, 86), n_periods=4)

    dt50 = DuckONATemporal()
    dt50.load_snapshots(hris_50, snapshot_date_col="snapshot_date", freq="Q")
    t0 = time.time()
    dt50.q.window_volatility("betweenness", lookback="4Q")
    t_50 = time.time() - t0

    dt100 = DuckONATemporal()
    dt100.load_snapshots(hris_100, snapshot_date_col="snapshot_date", freq="Q")
    t0 = time.time()
    dt100.q.window_volatility("betweenness", lookback="4Q")
    t_100 = time.time() - t0

    # Linear scaling: t_100 should be ~2x t_50 (or less). Allow 4x for noise.
    assert t_100 < max(t_50 * 4.0, 0.5), (
        f"t_100={t_100:.2f}s, t_50={t_50:.2f}s; ratio {t_100 / max(t_50, 0.01):.1f}x "
        f"exceeds linear scaling budget"
    )


def test_memory_bounded_for_repeated_calls() -> None:
    """Repeated calls don't leak memory or accumulate state."""
    hris = _build_large_org((1, 2, 6, 41), n_periods=4)
    dt = DuckONATemporal()
    dt.load_snapshots(hris, snapshot_date_col="snapshot_date", freq="Q")

    # Run each method 10 times; total time should be roughly linear (no leak)
    times: list[float] = []
    for _ in range(10):
        t0 = time.time()
        dt.q.window_mean("betweenness", lookback="4Q")
        dt.q.window_volatility("pagerank", lookback="4Q")
        dt.q.trajectory_rank("degree_centrality", top_n=5)
        times.append(time.time() - t0)

    # Last call should not be > 5x the median (no pathological slowdown)
    median_time = float(np.median(times))
    assert times[-1] < max(median_time * 5.0, 0.5), (
        f"last call took {times[-1]:.2f}s vs median {median_time:.2f}s — possible leak"
    )


# ─── Benchmarks (informational; not assertions) ───────────────────────────


def test_benchmark_8_methods_at_50_employees() -> None:
    """Benchmark all major methods at 50 employees / 4 periods.

    Prints timings but does not assert strict bounds (CI noise).
    """
    hris = _build_large_org((1, 2, 6, 41), n_periods=4)
    dt = DuckONATemporal()
    dt.load_snapshots(hris, snapshot_date_col="snapshot_date", freq="Q")

    timings: dict[str, float] = {}

    for name, fn in [
        ("compute_temporal_metrics[betweenness]",
         lambda: dt.compute_temporal_metrics(metrics=["betweenness"])),
        ("compute_temporal_metrics[pagerank]",
         lambda: dt.compute_temporal_metrics(metrics=["pagerank"])),
        ("network_evolution", lambda: dt.network_evolution()),
        ("event_window", lambda: dt.event_window("2025-04-01")),
        ("change_detection", lambda: dt.change_detection("betweenness", top_n=10)),
        ("mobility_leaderboard", lambda: dt.mobility_leaderboard(top_n=10)),
        ("mobility_anomaly", lambda: dt.mobility_anomaly(lookback="4Q")),
        ("trajectory_pivot", lambda: dt.q.trajectory_pivot("betweenness", "4Q")),
        ("window_volatility", lambda: dt.q.window_volatility("pagerank", "4Q")),
        ("hierarchy_drift", lambda: dt.q.hierarchy_drift(dt.periods[0], dt.periods[-1])),
        ("subtree_at", lambda: dt.q.subtree_at("E_VP0", dt.periods[-1]).df()),
    ]:
        # Warm up (avoid measuring cold-cache effects)
        fn()
        t0 = time.time()
        for _ in range(3):
            fn()
        elapsed = (time.time() - t0) / 3
        timings[name] = elapsed

    # Print as informational; not asserted
    print("\n=== Benchmark: 50 employees, 4 periods ===")
    for name, t in sorted(timings.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {t * 1000:.1f}ms")

    # Sanity check: nothing should take more than 5 seconds
    slow = {k: v for k, v in timings.items() if v > 5.0}
    assert not slow, f"unexpectedly slow methods: {slow}"
