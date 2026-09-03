"""
End-to-end demo of the pyduck_ona.viz subpackage.

Builds a synthetic 300-person organization with realistic hierarchy
metrics, then renders every visualization in the package and saves the
output to ``examples/output/``.

Run with:
    python examples/full_viz_demo.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import pyduck_ona.viz as viz

OUT = Path(__file__).resolve().parent / "output"


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def build_org(n_per_level: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a long-form hierarchy plus a per-employee metadata frame."""
    rng = np.random.default_rng(42)
    rows: list[tuple[str, str | None]] = []
    counter = 0
    prev_level: list[str] = []
    depts = ["Engineering", "Sales", "Marketing", "Finance", "HR", "Operations"]
    levels = ["L1", "L2", "L3", "L4", "L5"]

    for level_idx, count in enumerate(n_per_level):
        current_level: list[str] = []
        for _ in range(count):
            counter += 1
            eid = f"E{counter:04d}"
            if level_idx == 0:
                rows.append((eid, None))
            else:
                parent = prev_level[counter % len(prev_level)]
                rows.append((eid, parent))
            current_level.append(eid)
        prev_level = current_level

    hierarchy = pd.DataFrame(rows, columns=["employee_id", "supervisor_id"])
    meta_rows = []
    for eid in hierarchy["employee_id"]:
        meta_rows.append({
            "employee_id":  eid,
            "name":         f"Person-{eid}",
            "title":        rng.choice(["Manager", "Director", "IC", "VP", "SVP"]),
            "department":   rng.choice(depts),
            "level":        rng.choice(levels),
            "gender":       rng.choice(["F", "M"]),
            "tenure_years": float(rng.integers(0, 20)),
            "salary":       float(rng.normal(130_000, 30_000)),
        })
    metadata = pd.DataFrame(meta_rows)
    return hierarchy, metadata


def compute_stats(hierarchy: pd.DataFrame) -> pd.DataFrame:
    """Compute manager-level stats from the long-form hierarchy."""
    parent_col = "supervisor_id"
    parents = hierarchy[parent_col].dropna().unique().tolist()
    rows = []
    for parent in sorted(parents):
        direct = int((hierarchy[parent_col] == parent).sum())
        rows.append({
            "manager_id":       parent,
            "direct_reports":   direct,
            "indirect_reports": max(direct * 2, 0),
            "total_reports":    direct * 3,
            "team_size":        direct + direct * 2,
            "levels_below":     int(np.random.default_rng(0).integers(1, 5)),
        })
    return pd.DataFrame(rows).sort_values("direct_reports", ascending=False)


def compute_centrality(hierarchy: pd.DataFrame):
    """Compute centrality via NetworkX (matches pyduck-ona output schema)."""
    graph = nx.DiGraph()
    for _, row in hierarchy.dropna(subset=["supervisor_id"]).iterrows():
        graph.add_edge(row["employee_id"], row["supervisor_id"])
    undirected = graph.to_undirected()
    nodes = sorted(graph.nodes())

    bet = nx.betweenness_centrality(undirected)
    pr = nx.pagerank(graph)
    try:
        eig = nx.eigenvector_centrality_numpy(undirected)
    except Exception:
        eig = nx.eigenvector_centrality(undirected, max_iter=500)
    deg = dict(undirected.degree())

    betweenness = pd.DataFrame({"node_id": nodes, "betweenness": [bet[n] for n in nodes]})
    pagerank_df = pd.DataFrame({"node_id": nodes, "pagerank": [pr[n] for n in nodes]})
    eigen       = pd.DataFrame({"node_id": nodes, "eigenvector": [eig[n] for n in nodes]})
    degree_df   = pd.DataFrame({
        "node_id": nodes,
        "degree":    [deg[n] for n in nodes],
        "in_degree": [graph.in_degree(n) for n in nodes],
        "out_degree":[graph.out_degree(n) for n in nodes],
    })
    comms = nx.community.louvain_communities(undirected, seed=42)
    cid: dict[str, int] = {}
    for i, c in enumerate(comms):
        for n in c:
            cid[n] = i
    communities = pd.DataFrame({"node_id": nodes, "community": [cid[n] for n in nodes]})
    return betweenness, pagerank_df, eigen, degree_df, communities


def compute_attrition(metadata: pd.DataFrame) -> pd.DataFrame:
    """Aggregate attrition-style data from the metadata table."""
    rng = np.random.default_rng(7)
    rows = []
    for d in metadata["department"].unique():
        for lv in metadata["level"].unique():
            sub = metadata[(metadata["department"] == d) & (metadata["level"] == lv)]
            n = len(sub)
            if n == 0:
                continue
            rate = float(np.clip(rng.normal(0.18, 0.10), 0.02, 0.5))
            rows.append({"department": d, "job_level": lv, "rate": rate, "count": n})
    return pd.DataFrame(rows)


def compute_diversity(metadata: pd.DataFrame) -> pd.DataFrame:
    return (
        metadata.groupby("gender").size().reset_index(name="count").rename(columns={"gender": "group"})
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Building synthetic organization ...")
    hierarchy, metadata = build_org([1, 4, 14, 40, 90, 150])
    stats = compute_stats(hierarchy)
    betweenness, pagerank_df, eigen, degree_df, communities = compute_centrality(hierarchy)
    attrition = compute_attrition(metadata)
    diversity = compute_diversity(metadata)

    print(f"  {len(hierarchy)} employees, {len(stats)} managers, "
          f"{communities['community'].nunique()} communities")

    # 1. Span of control -----------------------------------------------------
    print("[1/10] span_of_control ...")
    fig = viz.span_of_control(stats, metadata=metadata, top_n=15,
                              color_by_department=True)
    fig.savefig(OUT / "01_span_of_control.png", dpi=200)
    plt.close(fig)

    # Interactive variant ----------------------------------------------------
    html = viz.span_of_control(stats, top_n=15, return_html=True)
    (OUT / "01_span_of_control.html").write_text(html)

    # 2. Span vs depth -------------------------------------------------------
    print("[2/10] span_vs_depth ...")
    fig = viz.span_vs_depth(stats, metadata=metadata)
    fig.savefig(OUT / "02_span_vs_depth.png", dpi=200)
    plt.close(fig)

    # 3. Hierarchy depth heatmap --------------------------------------------
    print("[3/10] hierarchy_depth_heatmap ...")
    n = 30
    wide = pd.DataFrame({
        "employee_id": [f"X{i}" for i in range(n)],
        "Level_1":     [None] + [f"X{i % 2}" for i in range(n - 1)],
        "Level_2":     [None, None] + [f"X{i % 4}" for i in range(n - 2)],
        "Level_3":     [None] * 3 + [f"X{i % 8}" for i in range(n - 3)],
    })
    fig = viz.hierarchy_depth_heatmap(wide)
    fig.savefig(OUT / "03_depth_heatmap.png", dpi=200)
    plt.close(fig)

    # 4. Centrality dashboard ------------------------------------------------
    print("[4/10] centrality_dashboard ...")
    fig = viz.centrality_dashboard(
        betweenness=betweenness, pagerank=pagerank_df,
        eigenvector=eigen, degree=degree_df,
        metadata=metadata, top_n=10,
    )
    fig.savefig(OUT / "04_centrality_dashboard.png", dpi=200)
    plt.close(fig)

    # 5. Silo map ------------------------------------------------------------
    print("[5/10] silo_map ...")
    fig = viz.silo_map(hierarchy, communities=communities,
                       metadata=metadata, return_html=False)
    fig.savefig(OUT / "05_silo_map.png", dpi=200)
    plt.close(fig)
    html = viz.silo_map(hierarchy, communities=communities,
                        metadata=metadata, return_html=True)
    (OUT / "05_silo_map.html").write_text(html)

    # 6. Attrition heatmap ---------------------------------------------------
    print("[6/10] attrition_heatmap ...")
    fig = viz.attrition_heatmap(attrition)
    fig.savefig(OUT / "06_attrition_heatmap.png", dpi=200)
    plt.close(fig)

    # 7. Compensation equity -------------------------------------------------
    print("[7/10] compensation_equity ...")
    comp = metadata[["tenure_years", "salary", "gender", "employee_id"]].copy()
    fig = viz.compensation_equity(comp, group_col="gender")
    fig.savefig(OUT / "07_compensation_equity.png", dpi=200)
    plt.close(fig)

    # 8. Reporting chain walk ------------------------------------------------
    print("[8/10] reporting_chain_walk ...")
    target = hierarchy["employee_id"].iloc[-1]
    fig = viz.reporting_chain_walk(hierarchy, target, metadata=metadata)
    fig.savefig(OUT / "08_reporting_chain.png", dpi=200)
    plt.close(fig)

    # 9. Org chart tree ------------------------------------------------------
    print("[9/10] org_chart_tree ...")
    html = viz.org_chart_tree(
        hierarchy, metadata=metadata,
        color_by="department",
        title="Acme Corp · Q4 2026",
    )
    (OUT / "09_org_chart.html").write_text(html)

    # 10. Summary dashboard --------------------------------------------------
    print("[10/10] summary_dashboard ...")
    html = viz.summary_dashboard(
        hierarchy_stats=stats,
        betweenness=betweenness,
        pagerank=pagerank_df,
        diversity=diversity,
        attrition=attrition,
    )
    (OUT / "10_summary_dashboard.html").write_text(html)

    print(f"\nDone. Outputs written to {OUT}/")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
