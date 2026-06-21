"""Integration tests for graph algorithms.

These tests use NetworkX as the default backend. They run against the
real DuckDB relation pipeline — materializing the edge relation via
Arrow, building the in-memory graph, and running NX algorithms. No
mocks.

The DuckPGQ backend is exercised via a single smoke test confirming the
clear-ImportError path; the extension itself is not installable from
the community registry on current DuckDB releases.
"""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pyduck_ona.core import hierarchy_long
from pyduck_ona.graph import (
    betweenness,
    connected_components,
    pagerank,
    shortest_path,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _direct_edges(simple_org):
    """One row per (employee, supervisor) — the direct reporting graph.

    Used by betweenness/pagerank/connected_components tests because
    those algorithms are meaningful on the *chain* structure, not on the
    transitive closure produced by ``hierarchy_long()``.
    """
    return duckdb.sql(
        "SELECT employee_id, supervisor_id "
        "FROM simple_org WHERE supervisor_id IS NOT NULL"
    )


# ─── shortest_path ──────────────────────────────────────────────────────────

class TestShortestPath:
    def test_path_on_long_format_uses_direct_closure_edge(self, simple_org):
        """On the transitive-closure edges from ``hierarchy_long()``,
        every (employee, ancestor) is a single edge, so the shortest
        path length between any descendant and any ancestor is 1."""
        long = hierarchy_long(simple_org, "employee_id", "supervisor_id")
        result = shortest_path(long, "employee_id", "supervisor_id",
                                source="E1000", target="E001").df()
        assert len(result) == 1
        row = result.iloc[0]
        assert row["path_length"] == 1
        assert row["path"] == "E1000->E001"

    def test_path_walks_up_chain_on_direct_edges(self, simple_org):
        """On the *direct* edge relation, paths must walk up the chain."""
        direct = _direct_edges(simple_org)
        result = shortest_path(direct, "employee_id", "supervisor_id",
                                source="E1000", target="E001").df()
        row = result.iloc[0]
        # E1000 → E100 → E010 → E001 (3 edges in the chain graph)
        assert row["path_length"] == 3
        assert row["path"] == "E1000->E100->E010->E001"

    def test_path_to_self(self, simple_org):
        long = hierarchy_long(simple_org, "employee_id", "supervisor_id")
        result = shortest_path(long, "employee_id", "supervisor_id",
                                source="E001", target="E001").df()
        assert len(result) == 1
        assert result.iloc[0]["path_length"] == 0
        assert result.iloc[0]["path"] == "E001"

    def test_path_to_missing_node_returns_null_length(self, simple_org):
        long = hierarchy_long(simple_org, "employee_id", "supervisor_id")
        result = shortest_path(long, "employee_id", "supervisor_id",
                                source="E001", target="GHOST").df()
        assert len(result) == 1
        # GHOST isn't in the graph → no path possible; length is NA.
        assert pd.isna(result.iloc[0]["path_length"])
        assert result.iloc[0]["path"] == ""

    def test_returns_relation_not_dataFrame(self, simple_org):
        long = hierarchy_long(simple_org, "employee_id", "supervisor_id")
        result = shortest_path(long, "employee_id", "supervisor_id",
                                source="E001", target="E1000")
        assert isinstance(result, duckdb.DuckDBPyRelation)


# ─── betweenness ────────────────────────────────────────────────────────────

class TestBetweenness:
    def test_returns_all_nodes_on_chain_graph(self, simple_org):
        direct = _direct_edges(simple_org)
        result = betweenness(direct, "employee_id", "supervisor_id").df()
        # 7 nodes total: 6 edge endpoints + E001 (root, reachable as a
        # sink because edges point up to it). NetworkX adds every node
        # that appears as either source or target.
        assert len(result) == 7

    def test_top_node_is_subtree_bridge_on_chain_graph(self, simple_org):
        """On the direct-edge digraph, the CEO is a *sink* (no outgoing
        edges) so no shortest path between other nodes passes through
        it. The actual broker is E010, which sits between the two
        Director subtrees — every inter-subtree shortest path goes
        through E010."""
        direct = _direct_edges(simple_org)
        result = betweenness(direct, "employee_id", "supervisor_id").df()
        top = result.iloc[0]
        assert top["node_id"] == "E010"
        assert top["betweenness"] > 0

    def test_root_sink_has_zero_betweenness(self, simple_org):
        """The CEO has no outgoing edges, so no inter-node path passes
        through it; betweenness = 0. (This is true even though the CEO
        is on every path *to itself from elsewhere* — betweenness
        counts paths between *distinct* source/target pairs.)"""
        direct = _direct_edges(simple_org)
        result = betweenness(direct, "employee_id", "supervisor_id").df()
        root_row = result[result["node_id"] == "E001"].iloc[0]
        assert root_row["betweenness"] == 0

    def test_leaves_have_zero_betweenness(self, simple_org):
        direct = _direct_edges(simple_org)
        result = betweenness(direct, "employee_id", "supervisor_id").df()
        leaves = {"E1000", "E1001", "E1010"}
        leaf_scores = result[result["node_id"].isin(leaves)]
        assert (leaf_scores["betweenness"] == 0).all()

    def test_long_format_collapse_yields_zero_betweenness(self, simple_org):
        """Passing the transitive-closure edges from ``hierarchy_long()``
        flattens the graph into a star — every node reaches every
        ancestor in one step — so betweenness is 0 everywhere. Document
        this so callers know to pass direct edges for broker detection.
        """
        long = hierarchy_long(simple_org, "employee_id", "supervisor_id")
        result = betweenness(long, "employee_id", "supervisor_id").df()
        assert (result["betweenness"] == 0).all()


# ─── pagerank ───────────────────────────────────────────────────────────────

class TestPagerank:
    def test_returns_all_nodes_with_positive_scores(self, simple_org):
        direct = _direct_edges(simple_org)
        result = pagerank(direct, "employee_id", "supervisor_id").df()
        # 7 nodes (all employees including the CEO sink)
        assert len(result) == 7
        # PageRank is strictly positive for all nodes
        assert (result["pagerank"] > 0).all()
        # Scores sum to ~1.0
        assert abs(result["pagerank"].sum() - 1.0) < 1e-6

    def test_sink_has_highest_pagerank_in_chain_graph(self, simple_org):
        """In a tree digraph, every edge flows toward the root, so the
        root (sink) absorbs all PageRank mass and scores highest. (For
        influence scoring in an org chart this is usually the desired
        semantic — but for *collaboration* networks you'd reverse the
        edge direction.)"""
        direct = _direct_edges(simple_org)
        result = pagerank(direct, "employee_id", "supervisor_id").df()
        top = result.iloc[0]
        assert top["node_id"] == "E001"

    def test_pagerank_respects_damping(self, simple_org):
        direct = _direct_edges(simple_org)
        r85 = pagerank(direct, "employee_id", "supervisor_id", damping=0.85).df()
        r50 = pagerank(direct, "employee_id", "supervisor_id", damping=0.50).df()
        # Different damping should produce different distributions.
        a = r85.sort_values("node_id")["pagerank"].to_numpy()
        b = r50.sort_values("node_id")["pagerank"].to_numpy()
        assert not (a == b).all()


# ─── connected_components ───────────────────────────────────────────────────

class TestConnectedComponents:
    def test_healthy_org_has_one_component(self, simple_org):
        direct = _direct_edges(simple_org)
        result = connected_components(direct, "employee_id", "supervisor_id").df()
        assert len(result) == 1
        # 7 nodes total in simple_org, all in one weakly-connected component
        assert result.iloc[0]["size"] == 7

    def test_disconnected_org_has_multiple_components(self):
        """Two disconnected sub-trees should yield two components."""
        rel = duckdb.sql("""
            SELECT * FROM (VALUES
                ('A1', CAST(NULL AS VARCHAR)),
                ('A2', 'A1'),
                ('B1', CAST(NULL AS VARCHAR)),
                ('B2', 'B1')
            ) t(employee_id, supervisor_id)
        """)
        direct = duckdb.sql(
            "SELECT employee_id, supervisor_id "
            "FROM rel WHERE supervisor_id IS NOT NULL"
        )
        result = connected_components(direct, "employee_id", "supervisor_id").df()
        assert len(result) == 2
        assert set(result["size"].tolist()) == {2}


# ─── NULL-supervisor handling (P0-3 regression) ───────────────────────────

class TestGraphNullSupervisors:
    """Regression tests for passing raw org relations with NULL supervisors.

    Before the fix: ``betweenness`` raised ``ValueError: None cannot be a
    node`` because the edge materializer included root rows where
    ``supervisor_id`` was NULL.
    """

    @pytest.fixture
    def raw_org_with_null_sup(self):
        return duckdb.sql(
            "SELECT * FROM (VALUES "
            "('E001', CAST(NULL AS VARCHAR)), "
            "('E002', 'E001'), "
            "('E003', 'E001')) t(employee_id, supervisor_id)"
        )

    def test_betweenness_ignores_null_supervisors(self, raw_org_with_null_sup):
        result = betweenness(
            raw_org_with_null_sup, "employee_id", "supervisor_id"
        ).df()
        assert len(result) == 3
        assert set(result["node_id"].tolist()) == {"E001", "E002", "E003"}

    def test_pagerank_ignores_null_supervisors(self, raw_org_with_null_sup):
        result = pagerank(
            raw_org_with_null_sup, "employee_id", "supervisor_id"
        ).df()
        assert len(result) == 3
        assert (result["pagerank"] > 0).all()

    def test_connected_components_ignores_null_supervisors(self, raw_org_with_null_sup):
        result = connected_components(
            raw_org_with_null_sup, "employee_id", "supervisor_id"
        ).df()
        assert len(result) == 1
        assert result.iloc[0]["size"] == 3

    def test_shortest_path_source_target_not_in_graph(self):
        rel = duckdb.sql(
            "SELECT * FROM (VALUES "
            "('E001', CAST(NULL AS VARCHAR)), "
            "('E002', 'E001')) t(employee_id, supervisor_id)"
        )
        result = shortest_path(
            rel, "employee_id", "supervisor_id", "E001", "GHOST"
        ).df()
        assert len(result) == 1
        assert pd.isna(result.iloc[0]["path_length"])
        assert result.iloc[0]["path"] == ""


# ─── DuckPGQ backend (smoke tests for the not-installable path) ─────────────

class TestDuckPGQBackend:
    def test_requesting_duckpgq_raises_clear_error(self, simple_org):
        long = hierarchy_long(simple_org, "employee_id", "supervisor_id")
        # Even with a valid relation, selecting backend='duckpgq' must
        # raise ImportError (extension not available), not silently fall
        # back or return wrong data.
        with pytest.raises(ImportError, match="DuckPGQ"):
            shortest_path(long, "employee_id", "supervisor_id",
                          source="E001", target="E999", backend="duckpgq")

    def test_pgq_betweenness_raises(self, simple_org):
        direct = _direct_edges(simple_org)
        with pytest.raises(ImportError, match="DuckPGQ"):
            betweenness(direct, "employee_id", "supervisor_id", backend="duckpgq")

    def test_pgq_pagerank_raises(self, simple_org):
        direct = _direct_edges(simple_org)
        with pytest.raises(ImportError, match="DuckPGQ"):
            pagerank(direct, "employee_id", "supervisor_id", backend="duckpgq")

    def test_pgq_components_raises(self, simple_org):
        direct = _direct_edges(simple_org)
        with pytest.raises(ImportError, match="DuckPGQ"):
            connected_components(direct, "employee_id", "supervisor_id",
                                 backend="duckpgq")


# ─── node_id_col rename ─────────────────────────────────────────────────────

class TestNodeIdRename:
    """Graph functions accept a custom output node-id column name."""

    def test_betweenness_rename(self, simple_org):
        direct = _direct_edges(simple_org)
        result = betweenness(direct, "employee_id", "supervisor_id",
                              node_id_col="employee_id").df()
        assert "employee_id" in result.columns
        assert "betweenness" in result.columns
        assert "node_id" not in result.columns

    def test_pagerank_rename(self, simple_org):
        direct = _direct_edges(simple_org)
        result = pagerank(direct, "employee_id", "supervisor_id",
                          node_id_col="employee_id").df()
        assert "employee_id" in result.columns
        assert "pagerank" in result.columns
        assert "node_id" not in result.columns

    def test_eigenvector_rename(self, simple_org):
        from pyduck_ona.graph import eigenvector_centrality
        direct = _direct_edges(simple_org)
        result = eigenvector_centrality(direct, "employee_id", "supervisor_id",
                                         node_id_col="employee_id").df()
        assert "employee_id" in result.columns
        assert "eigenvector" in result.columns

    def test_degree_rename(self, simple_org):
        from pyduck_ona.graph import degree_centrality
        direct = _direct_edges(simple_org)
        result = degree_centrality(direct, "employee_id", "supervisor_id",
                                   node_id_col="employee_id").df()
        assert "employee_id" in result.columns
        assert "degree_centrality" in result.columns

    def test_louvain_rename(self, simple_org):
        from pyduck_ona.graph import louvain_communities
        direct = _direct_edges(simple_org)
        result = louvain_communities(direct, "employee_id", "supervisor_id",
                                    node_id_col="employee_id").df()
        assert "employee_id" in result.columns
        assert "community_id" in result.columns
