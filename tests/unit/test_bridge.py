"""Unit tests for the graph-export bridge layer (NetworkX / igraph)."""
from __future__ import annotations

import pytest

from pyduck_ona.bridge import to_igraph, to_networkx


# ─── to_networkx ────────────────────────────────────────────────────────────

class TestToNetworkX:
    def test_basic_digraph(self, simple_org):
        from pyduck_ona.core import hierarchy_long
        long_rel = hierarchy_long(simple_org, "employee_id", "supervisor_id")
        G = to_networkx(long_rel, "employee_id", "supervisor_id", graph_type="DiGraph")
        # long format: 14 ancestor-pair edges across the 6 non-root employees
        # (E010×1, E100×2, E101×2, E1000×3, E1001×3, E1010×3)
        assert G.number_of_edges() == 14
        # 7 distinct employees
        assert G.number_of_nodes() == 7
        assert G.is_directed()

    def test_undirected(self, simple_org):
        from pyduck_ona.core import hierarchy_long
        long_rel = hierarchy_long(simple_org, "employee_id", "supervisor_id")
        G = to_networkx(long_rel, "employee_id", "supervisor_id", graph_type="Graph")
        assert not G.is_directed()

    def test_invalid_graph_type(self, simple_org):
        from pyduck_ona.core import hierarchy_long
        long_rel = hierarchy_long(simple_org, "employee_id", "supervisor_id")
        with pytest.raises(ValueError, match="graph_type"):
            to_networkx(long_rel, "employee_id", "supervisor_id", graph_type="BadType")


# ─── to_igraph ──────────────────────────────────────────────────────────────

class TestToIgraph:
    def test_basic_digraph(self, simple_org):
        try:
            import igraph  # noqa: F401
        except ImportError:
            pytest.skip("igraph not installed")
        from pyduck_ona.core import hierarchy_long
        long_rel = hierarchy_long(simple_org, "employee_id", "supervisor_id")
        g = to_igraph(long_rel, "employee_id", "supervisor_id", directed=True)
        assert g.vcount() == 7
        # long format: 14 ancestor-pair edges (matches NetworkX test)
        assert g.ecount() == 14
        assert g.is_directed()

    def test_node_names_preserved(self, simple_org):
        try:
            import igraph  # noqa: F401
        except ImportError:
            pytest.skip("igraph not installed")
        from pyduck_ona.core import hierarchy_long
        long_rel = hierarchy_long(simple_org, "employee_id", "supervisor_id")
        g = to_igraph(long_rel, "employee_id", "supervisor_id")
        assert "E001" in g.vs["name"]
        assert "E1010" in g.vs["name"]