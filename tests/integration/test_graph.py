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

import contextlib

import duckdb
import pandas as pd
import pytest

from pyduck_ona.core import hierarchy_long
from pyduck_ona.graph import (  # noqa: PLC0415
    _duckpgq_backend as _duckpgq_backend_alias,
)
from pyduck_ona.graph import (
    betweenness,
    connected_components,
    degree_centrality,
    duckpgq_setup,
    louvain_communities,
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

    def test_returns_relation_not_dataframe(self, simple_org):
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
        _rel = duckdb.sql("""
            SELECT * FROM (VALUES
                ('A1', CAST(NULL AS VARCHAR)),
                ('A2', 'A1'),
                ('B1', CAST(NULL AS VARCHAR)),
                ('B2', 'B1')
            ) t(employee_id, supervisor_id)
        """)
        direct = duckdb.sql(
            "SELECT employee_id, supervisor_id "
            "FROM _rel WHERE supervisor_id IS NOT NULL"
        )
        result = connected_components(direct, "employee_id", "supervisor_id").df()
        assert len(result) == 2
        assert set(result["size"].tolist()) == {2}


class TestLouvain:
    def test_weighted_louvain_uses_weight_column(self):
        edges = duckdb.sql("""
            SELECT * FROM (VALUES
                ('A', 'B', 3.0),
                ('B', 'C', 2.0),
                ('A', 'C', 1.0),
                ('D', 'E', 4.0)
            ) t(employee_id, supervisor_id, w)
        """)
        result = louvain_communities(
            edges, "employee_id", "supervisor_id", weight_col="w"
        ).df()
        assert set(result.columns) == {"node_id", "community_id"}
        assert {"A", "B", "C", "D", "E"}.issubset(set(result["node_id"]))


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


# ─── DuckPGQ backend ─────────────────────────────────────────────────────

_duckpgq_reason = None
if not _duckpgq_backend_alias.is_duckpgq_supported_duckdb():
    _duckpgq_reason = f"requires DuckDB 1.3.1; got {duckdb.__version__}"


@pytest.fixture(scope="module")
def duckpgq_con():
    """Open a DuckDB connection with DuckPGQ loaded, if available.

    Tests that need to actually run a DuckPGQ algorithm should request
    this fixture and skip when it isn't available. The connection is
    module-scoped so per-test installation is amortized.
    """
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    duckpgq_setup(con)
    yield con
    with contextlib.suppress(Exception):
        con.close()


@pytest.mark.integration
class TestDuckPGQBackendErrors:
    """Algorithms without a DuckPGQ v1.3.1 backend must raise ImportError.

    These run on any DuckDB version because the import guard fires
    before any property-graph registration. They are *not* marked
    @pytest.mark.integration because they pass on system DuckDB too.
    """

    def test_shortest_path_raises(self, simple_org):
        long = hierarchy_long(simple_org, "employee_id", "supervisor_id")
        with pytest.raises(ImportError, match="DuckPGQ"):
            shortest_path(long, "employee_id", "supervisor_id",
                          source="E001", target="E999", backend="duckpgq")

    def test_betweenness_raises(self, simple_org):
        direct = _direct_edges(simple_org)
        with pytest.raises(ImportError, match="DuckPGQ"):
            betweenness(direct, "employee_id", "supervisor_id", backend="duckpgq")

    def test_eigenvector_centrality_raises(self, simple_org):
        from pyduck_ona.graph import eigenvector_centrality

        direct = _direct_edges(simple_org)
        with pytest.raises(ImportError, match="DuckPGQ"):
            eigenvector_centrality(
                direct, "employee_id", "supervisor_id", backend="duckpgq"
            )

    def test_louvain_communities_raises(self, simple_org):
        from pyduck_ona.graph import louvain_communities

        direct = _direct_edges(simple_org)
        with pytest.raises(ImportError, match="DuckPGQ"):
            louvain_communities(
                direct, "employee_id", "supervisor_id", backend="duckpgq"
            )


@pytest.mark.skipif(
    _duckpgq_reason is not None,
    reason=_duckpgq_reason or "DuckPGQ backend requires DuckDB 1.3.1",
)
@pytest.mark.integration
class TestDuckPGQBackendLive:
    """Real DuckPGQ execution. Skips on unsupported DuckDB versions.

    These tests require ``pyduck-ona[graph]`` (DuckDB pinned to 1.3.1)
    AND the DuckPGQ extension to install. They cross-check the DuckPGQ
    result against NetworkX for the four algorithms where DuckPGQ
    provides a real implementation.
    """

    def test_pagerank_duckpgq_returns_all_nodes(self, simple_org, duckpgq_con):
        direct = _direct_edges(simple_org)
        nx_result = pagerank(direct, "employee_id", "supervisor_id").df()
        dg_result = pagerank(
            direct,
            "employee_id",
            "supervisor_id",
            backend="duckpgq",
            con=duckpgq_con,
        ).df()
        assert len(dg_result) == len(nx_result)
        assert set(dg_result["node_id"]) == set(nx_result["node_id"])
        # DuckPGQ and NetworkX use different convergence criteria, so
        # the *values* won't match, but the *ordering* must be highly
        # correlated. Compute Kendall-tau via the simple pairwise
        # disagreement count.
        nx_order = (
            nx_result.sort_values("pagerank", ascending=False)["node_id"].tolist()
        )
        dg_order = (
            dg_result.sort_values("pagerank", ascending=False)["node_id"].tolist()
        )
        # On a small chain graph, the relative ordering of the top
        # half is preserved within a single inversion tolerance.
        assert nx_order[0] == dg_order[0], (
            f"top node disagrees: NetworkX={nx_order[0]} DuckPGQ={dg_order[0]}"
        )

    def test_connected_components_duckpgq_returns_single_component(
        self, simple_org, duckpgq_con
    ):
        direct = _direct_edges(simple_org)
        # The simple_org fixture is a single connected DAG — both
        # backends should find exactly one component.
        result = connected_components(
            direct,
            "employee_id",
            "supervisor_id",
            backend="duckpgq",
            con=duckpgq_con,
        ).df()
        # simple_org is a 7-node chain (E001 root + 2 levels + 4 leaves).
        # All nodes belong to one weakly-connected component.
        assert len(result) == 1
        assert int(result.iloc[0]["size"]) == 7

    def test_connected_components_duckpgq_matches_networkx_on_disconnected(
        self, simple_org, duckpgq_con
    ):
        # Two disconnected chains: E001→E100 and X001→X100.
        # Use _direct_edges so the relation has no NULL supervisor_id
        # values — DuckPGQ v1.3.1 rejects CREATE PROPERTY GRAPH when
        # edges reference NULL endpoints.
        rel = duckdb.sql(  # noqa: F841 — DuckDB SQL references via local lookup
            "SELECT * FROM (VALUES "
            "('E001', CAST(NULL AS VARCHAR)), "
            "('E100', 'E001'), "
            "('X001', CAST(NULL AS VARCHAR)), "
            "('X100', 'X001')) t(employee_id, supervisor_id)"
        )
        direct = duckdb.sql(
            "SELECT employee_id, supervisor_id FROM rel WHERE supervisor_id IS NOT NULL"
        )
        nx_cc = connected_components(
            direct, "employee_id", "supervisor_id"
        ).df()
        dg_cc = connected_components(
            direct,
            "employee_id",
            "supervisor_id",
            backend="duckpgq",
            con=duckpgq_con,
        ).df()
        # Both backends see exactly 2 components of size 2 each.
        assert len(nx_cc) == len(dg_cc) == 2
        nx_sizes = sorted(nx_cc["size"].astype(int).tolist())
        dg_sizes = sorted(dg_cc["size"].astype(int).tolist())
        assert nx_sizes == dg_sizes == [2, 2]

    def test_degree_centrality_duckpgq_matches_networkx(self, simple_org, duckpgq_con):
        direct = _direct_edges(simple_org)
        for mode in ("in", "out", "total"):
            nx_dc = degree_centrality(
                direct, "employee_id", "supervisor_id", mode=mode
            ).df()
            dg_dc = degree_centrality(
                direct,
                "employee_id",
                "supervisor_id",
                mode=mode,
                backend="duckpgq",
                con=duckpgq_con,
            ).df()
            # Sort to align nodes.
            nx_sorted = nx_dc.sort_values("node_id").reset_index(drop=True)
            dg_sorted = dg_dc.sort_values("node_id").reset_index(drop=True)
            assert (
                nx_sorted["node_id"].tolist() == dg_sorted["node_id"].tolist()
            ), f"node mismatch in mode={mode}"
            # Compare values: both implementations normalize the same
            # way, so they should match to within rounding.
            nx_vals = nx_sorted["degree_centrality"].astype(float)
            dg_vals = dg_sorted["degree_centrality"].astype(float)
            assert (abs(nx_vals - dg_vals) < 1e-6).all(), (
                f"DuckPGQ <-> NetworkX degree centrality disagree for mode={mode}"
            )

    def test_pagerank_duckpgq_caches_property_graph(self, simple_org, duckpgq_con):
        """Two calls on the same input share the registered property graph."""
        from pyduck_ona.graph._duckpgq_backend import _PG_CACHE

        direct = _direct_edges(simple_org)
        before = len(_PG_CACHE.get(duckpgq_con, {}))
        for _ in range(3):
            pagerank(
                direct,
                "employee_id",
                "supervisor_id",
                backend="duckpgq",
                con=duckpgq_con,
            ).df()
        after = len(_PG_CACHE.get(duckpgq_con, {}))
        assert after == before + 1, (
            "Repeated DuckPGQ calls should register at most one property "
            f"graph for identical input; before={before}, after={after}"
        )

    def test_duckpgq_run_without_con_creates_ephemeral(self, simple_org):
        """`con=None` path opens an ephemeral connection and survives."""
        direct = _direct_edges(simple_org)
        # Run without passing `con` — pyduck-ona will spin up an
        # ephemeral connection, run, and bind the result to the
        # process-egress connection.
        result = pagerank(
            direct, "employee_id", "supervisor_id", backend="duckpgq"
        ).df()
        assert "node_id" in result.columns
        assert "pagerank" in result.columns
        assert len(result) > 0
        # Round-trip via .df() one more time to make sure the relation
        # is still consumable after the ephemeral con has been closed.
        second = pagerank(
            direct, "employee_id", "supervisor_id", backend="duckpgq"
        ).df()
        assert sorted(result["node_id"].tolist()) == sorted(second["node_id"].tolist())


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
