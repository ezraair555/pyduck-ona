"""
Example: Full ONA workflow with pyduck-ona.

This script demonstrates a complete People Analytics pipeline:
  1. Load a sample org chart
  2. Diagnose hierarchy integrity
  3. Compute span-of-control stats
  4. Build an ONA graph and run broker detection
  5. Walk the reporting chain to a target employee

Run with:
    python examples/full_workflow.py
"""
from __future__ import annotations

import duckdb

import pyduck_ona as pona


def main() -> None:
    # ── 1. Load sample org ───────────────────────────────────────────────
    rel = duckdb.sql("""
        SELECT * FROM (VALUES
            ('E001',  CAST(NULL AS VARCHAR)),
            ('E010',  'E001'),
            ('E020',  'E001'),
            ('E100',  'E010'),
            ('E101',  'E010'),
            ('E200',  'E020'),
            ('E201',  'E020'),
            ('E1000', 'E100'),
            ('E1001', 'E100'),
            ('E1010', 'E101'),
            ('E2000', 'E200'),
            ('E2001', 'E200'),
            ('E2010', 'E201')
        ) AS t(employee_id, supervisor_id)
    """)

    print(f"Loaded org with {rel.count('*').fetchone()[0]} employees")

    # ── 2. Diagnose ──────────────────────────────────────────────────────
    issues = pona.hierarchy_valid(rel, "employee_id", "supervisor_id").df()
    print(f"\nHierarchy issues found: {len(issues)}")
    if len(issues):
        print(issues.to_string(index=False))

    # ── 3. Span-of-control stats ─────────────────────────────────────────
    print("\nTop managers by team size:")
    stats = pona.hierarchy_stats(rel, "employee_id", "supervisor_id").df()
    print(stats.head(5).to_string(index=False))

    # ── 4. Build ONA graph ───────────────────────────────────────────────
    long_rel = pona.hierarchy_long(rel, "employee_id", "supervisor_id")
    G = pona.to_networkx(long_rel, "employee_id", "supervisor_id")
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── 5. Graph algorithms ──────────────────────────────────────────────
    # For meaningful betweenness/pagerank, use the *direct* edge
    # relation (one row per manager → report). The transitive-closure
    # edges from hierarchy_long() flatten the graph into a star.
    direct = duckdb.sql(
        "SELECT employee_id, supervisor_id "
        "FROM rel WHERE supervisor_id IS NOT NULL"
    )

    # Shortest path through the chain
    path = pona.shortest_path(
        direct, "employee_id", "supervisor_id",
        source="E1000", target="E001",
    ).df()
    print(f"\nReporting chain E1000 → E001:")
    print(path.to_string(index=False))

    # Broker detection
    brokers = pona.betweenness(direct, "employee_id", "supervisor_id").df()
    print(f"\nTop 5 brokers (betweenness centrality):")
    print(brokers.head(5).to_string(index=False))

    # Influence
    influence = pona.pagerank(direct, "employee_id", "supervisor_id").df()
    print(f"\nTop 5 most-influential (PageRank):")
    print(influence.head(5).to_string(index=False))

    # Organizational silos
    components = pona.connected_components(
        direct, "employee_id", "supervisor_id"
    ).df()
    print(f"\nConnected components: {len(components)}")
    print(components.to_string(index=False))


if __name__ == "__main__":
    main()