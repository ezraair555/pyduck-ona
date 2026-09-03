"""
Smoke tests for pyduck-ona.

Each test feeds a synthetic DataFrame into one of the visualization
functions and verifies that the return type and minimum-quality
attributes are correct. These tests intentionally avoid checking
visual output pixel-by-pixel — they exist to catch import errors,
API regressions, and broken column-name contracts.
"""
from __future__ import annotations

import matplotlib

# Use a non-interactive backend so figures never try to render.
matplotlib.use("Agg")

import matplotlib.figure as mfigure
import numpy as np
import pandas as pd
import pytest

import pyduck_ona.viz as viz
from pyduck_ona.viz import (
    attrition_heatmap,
    centrality_dashboard,
    compensation_equity,
    hierarchy_depth_heatmap,
    org_chart_tree,
    reporting_chain_walk,
    silo_map,
    span_of_control,
    span_vs_depth,
    summary_dashboard,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _make_hierarchy(n_per_level: list[int]) -> pd.DataFrame:
    """Build a long-form hierarchy with n_per_level employees per level."""
    rows: list[tuple[str, str | None]] = []
    prev_level: list[str] = []
    counter = 0
    for level_idx, count in enumerate(n_per_level):
        current_level: list[str] = []
        for _ in range(count):
            counter += 1
            eid = f"E{counter:04d}"
            if level_idx == 0:
                rows.append((eid, None))
            else:
                # Round-robin attach to previous level
                parent = prev_level[counter % len(prev_level)]
                rows.append((eid, parent))
            current_level.append(eid)
        prev_level = current_level
    return pd.DataFrame(rows, columns=["employee_id", "supervisor_id"])


def _make_metadata(hierarchy: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    depts = ["Engineering", "Sales", "Marketing", "Finance", "HR", "Operations"]
    levels = ["L1", "L2", "L3", "L4", "L5"]
    rows = []
    for eid in hierarchy["employee_id"]:
        rows.append({
            "employee_id": eid,
            "name":        f"Person-{eid}",
            "title":       rng.choice(["Manager", "Director", "IC", "VP", "SVP"]),
            "department":  rng.choice(depts),
            "level":       rng.choice(levels),
            "gender":      rng.choice(["F", "M"]),
            "tenure_years": float(rng.integers(0, 20)),
            "salary":      float(rng.normal(120_000, 35_000)),
        })
    return pd.DataFrame(rows)


def _make_hierarchy_stats(hierarchy: pd.DataFrame) -> pd.DataFrame:
    """Build a hierarchy_stats-shaped frame from a long-form hierarchy."""
    parents = hierarchy["supervisor_id"].dropna().tolist()
    out: list[dict] = []
    for parent in sorted(set(parents)):
        direct = int((hierarchy["supervisor_id"] == parent).sum())
        out.append({
            "manager_id":      parent,
            "direct_reports":  direct,
            "indirect_reports": max(direct * 2, 0),
            "total_reports":   direct * 3,
            "team_size":       direct + direct * 2,
            "levels_below":    int(np.random.default_rng(0).integers(1, 5)),
        })
    return pd.DataFrame(out)


def _make_centrality_frames(hierarchy: pd.DataFrame):
    rng = np.random.default_rng(7)
    ids = hierarchy["employee_id"].tolist()
    n = len(ids)
    return {
        "betweenness": pd.DataFrame({
            "node_id":    ids,
            "betweenness": rng.random(n) / n,
        }),
        "pagerank": pd.DataFrame({
            "node_id":  ids,
            "pagerank": rng.random(n) / n,
        }),
        "eigenvector": pd.DataFrame({
            "node_id":    ids,
            "eigenvector": rng.random(n) / n,
        }),
        "degree": pd.DataFrame({
            "node_id":   ids,
            "degree":     rng.integers(0, 10, n).astype(float),
            "in_degree":  rng.integers(0, 5, n).astype(float),
            "out_degree": rng.integers(0, 5, n).astype(float),
        }),
    }


@pytest.fixture(scope="module")
def synthetic():
    hierarchy = _make_hierarchy([1, 4, 12, 30, 60])
    metadata  = _make_metadata(hierarchy)
    stats     = _make_hierarchy_stats(hierarchy)
    centrality = _make_centrality_frames(hierarchy)
    return {
        "hierarchy":   hierarchy,
        "metadata":    metadata,
        "stats":       stats,
        **centrality,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_org_chart_tree_returns_html(synthetic):
    html = org_chart_tree(
        synthetic["hierarchy"],
        metadata=synthetic["metadata"],
        color_by="department",
    )
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
    assert "d3" in html.lower()


def test_reporting_chain_walk_returns_figure(synthetic):
    employee_id = synthetic["hierarchy"]["employee_id"].iloc[-1]
    fig = reporting_chain_walk(
        synthetic["hierarchy"], employee_id,
        metadata=synthetic["metadata"],
    )
    assert isinstance(fig, mfigure.Figure)
    assert len(fig.axes) == 1


def test_span_of_control_returns_figure(synthetic):
    fig = span_of_control(synthetic["stats"], top_n=10)
    assert isinstance(fig, mfigure.Figure)


def test_span_of_control_returns_html_when_requested(synthetic):
    html = span_of_control(synthetic["stats"], top_n=10, return_html=True)
    assert isinstance(html, str)
    assert "Plotly" in html or "plotly" in html.lower()


def test_span_vs_depth_returns_figure(synthetic):
    fig = span_vs_depth(synthetic["stats"])
    assert isinstance(fig, mfigure.Figure)


def test_hierarchy_depth_heatmap_returns_figure(synthetic):
    # Build a wide-form hierarchy from the long one.
    edges = synthetic["hierarchy"].dropna(subset=["supervisor_id"]).copy()
    wide = (
        edges.assign(dummy=1)
        .pivot_table(index="employee_id", columns="supervisor_id",
                     values="dummy", fill_value=0)
    )
    # The above is meaningless for depth — we directly synthesize a wide frame:
    n = 20
    wide = pd.DataFrame({
        "employee_id": [f"X{i}" for i in range(n)],
        "Level_1": [None] + [f"X{i % 2}" for i in range(n - 1)],
        "Level_2": [None, None] + [f"X{i % 4}" for i in range(n - 2)],
        "Level_3": [None] * 3 + [f"X{i % 8}" for i in range(n - 3)],
    })
    fig = hierarchy_depth_heatmap(wide)
    assert isinstance(fig, mfigure.Figure)


def test_centrality_dashboard_returns_figure(synthetic):
    fig = centrality_dashboard(
        betweenness=synthetic["betweenness"],
        pagerank=synthetic["pagerank"],
        eigenvector=synthetic["eigenvector"],
        degree=synthetic["degree"],
        metadata=synthetic["metadata"],
        top_n=5,
    )
    assert isinstance(fig, mfigure.Figure)
    assert len(fig.axes) == 4


def test_silo_map_returns_html(synthetic):
    html = silo_map(
        synthetic["hierarchy"],
        communities=synthetic["pagerank"].assign(community=lambda d: (d["pagerank"] * 5).astype(int)),
        metadata=synthetic["metadata"],
        return_html=True,
    )
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html


def test_silo_map_returns_figure(synthetic):
    fig = silo_map(
        synthetic["hierarchy"],
        communities=synthetic["pagerank"].assign(community=lambda d: (d["pagerank"] * 5).astype(int)),
        metadata=synthetic["metadata"],
        return_html=False,
    )
    assert isinstance(fig, mfigure.Figure)


def test_attrition_heatmap_returns_figure(synthetic):
    rng = np.random.default_rng(0)
    rows = []
    depts = ["Eng", "Sales", "Marketing"]
    levels = [1, 2, 3, 4]
    for d in depts:
        for lv in levels:
            n = int(rng.integers(5, 30))
            rate = float(rng.uniform(0.05, 0.45))
            rows.append({"department": d, "job_level": lv, "rate": rate, "count": n})
    fig = attrition_heatmap(pd.DataFrame(rows))
    assert isinstance(fig, mfigure.Figure)


def test_compensation_equity_returns_figure(synthetic):
    # Build a synthetic compensation frame.
    rng = np.random.default_rng(2)
    rows = {
        "tenure_years": rng.uniform(0, 20, 100),
        "salary":       rng.normal(120_000, 25_000, 100),
        "gender":       rng.choice(["F", "M"], 100),
        "employee_id":  [f"E{i:04d}" for i in range(100)],
    }
    df = pd.DataFrame(rows)
    fig = compensation_equity(df, group_col="gender")
    assert isinstance(fig, mfigure.Figure)


def test_summary_dashboard_returns_html(synthetic):
    # Diversity mini-frame
    diversity = pd.DataFrame({
        "group": ["F", "M", "NB"],
        "count": [120, 175, 5],
    })
    # Attrition mini-frame
    attrition = pd.DataFrame({
        "department": ["Eng"] * 4 + ["Sales"] * 4,
        "job_level":  [1, 2, 3, 4] * 2,
        "rate":       [0.10, 0.15, 0.22, 0.30, 0.18, 0.20, 0.28, 0.35],
        "count":      [10, 8, 6, 4, 12, 9, 5, 3],
    })
    html = summary_dashboard(
        hierarchy_stats=synthetic["stats"],
        betweenness=synthetic["betweenness"],
        pagerank=synthetic["pagerank"],
        diversity=diversity,
        attrition=attrition,
    )
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
    assert "Plotly" in html or "plotly" in html.lower()


def test_theme_palette_is_well_formed():
    # The package exposes the palette dict and a categorical cycle list.
    assert isinstance(viz.PALETTE, dict)
    assert "primary" in viz.PALETTE
    assert all(c.startswith("#") for c in viz.PALETTE.values())
    assert len(viz.CATEGORICAL) >= 8


def test_invalid_input_raises():
    # Bad column name should produce KeyError, not silent garbage.
    with pytest.raises(KeyError):
        span_of_control(pd.DataFrame({"foo": [1, 2, 3]}))
    with pytest.raises((KeyError, ValueError)):
        hierarchy_depth_heatmap(pd.DataFrame({"employee_id": ["a", "b"]}))
