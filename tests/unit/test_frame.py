"""Unit tests for the v0.3 DuckONAFrame façade."""

from __future__ import annotations

import pandas as pd
import pytest

from pyduck_ona.frame import DuckONAFrame


@pytest.fixture
def flat_org() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "employee_id": ["CEO", "VP1", "VP2", "M1", "M2", "IC1", "IC2"],
            "supervisor_id": [None, "CEO", "CEO", "VP1", "VP1", "M1", "M2"],
            "department": ["Exec", "Sales", "Ops", "Sales", "Sales", "Sales", "Ops"],
            "job_level": [7, 6, 6, 5, 5, 3, 3],
        }
    )


class TestFrameConstructors:
    def test_from_pandas(self, flat_org: pd.DataFrame) -> None:
        frame = DuckONAFrame.from_pandas(flat_org, "hris")
        rel = frame.relation()
        assert rel.count("*").fetchone()[0] == len(flat_org)

    def test_from_janitor(self, flat_org: pd.DataFrame) -> None:
        pytest.importorskip("pyduck_janitor")
        from pyduck_janitor import DuckJanitor

        dj = DuckJanitor.from_pandas(flat_org).clean_names()
        frame = DuckONAFrame.from_janitor(dj, "hris")
        assert frame.source == "hris"
        assert frame.relation().count("*").fetchone()[0] == len(flat_org)


class TestFramePrep:
    def test_prep_validate_returns_entity_id(self, flat_org: pd.DataFrame) -> None:
        frame = DuckONAFrame.from_pandas(flat_org, "hris")
        rel = frame.prep_validate()
        df = rel.df()
        assert "entity_id" in df.columns
        assert "employee_id" not in df.columns

    def test_prep_validate_output_returns_frame(self, flat_org: pd.DataFrame) -> None:
        frame = DuckONAFrame.from_pandas(flat_org, "hris")
        result = frame.prep_validate(output="validated")
        assert isinstance(result, DuckONAFrame)
        assert result.source == "validated"

    def test_prep_long(self, flat_org: pd.DataFrame) -> None:
        frame = DuckONAFrame.from_pandas(flat_org, "hris")
        rel = frame.prep_long()
        df = rel.df()
        assert "employee_id" in df.columns
        assert "supervisor_id" in df.columns
        assert "depth" in df.columns

    def test_prep_wide(self, flat_org: pd.DataFrame) -> None:
        frame = DuckONAFrame.from_pandas(flat_org, "hris")
        rel = frame.prep_wide(max_depth=4)
        df = rel.df()
        assert "Level_1" in df.columns
        assert "employee_id" in df.columns


class TestFrameGraph:
    def test_graph_pagerank_returns_entity_id(self, flat_org: pd.DataFrame) -> None:
        frame = DuckONAFrame.from_pandas(flat_org, "hris")
        rel = frame.graph_pagerank()
        df = rel.df()
        assert "entity_id" in df.columns
        assert "pagerank" in df.columns
        assert "node_id" not in df.columns

    def test_graph_betweenness_output_returns_frame(self, flat_org: pd.DataFrame) -> None:
        frame = DuckONAFrame.from_pandas(flat_org, "hris")
        result = frame.graph_betweenness(output="bc")
        assert isinstance(result, DuckONAFrame)
        assert result.source == "bc"


class TestFramePipeline:
    def test_pipeline_composes_steps(self, flat_org: pd.DataFrame) -> None:
        frame = DuckONAFrame.from_pandas(flat_org, "hris")
        result = frame.pipeline(
            [
                lambda f: f.graph_pagerank(output="pr"),
                lambda f: f.report_export("pagerank_out"),
            ]
        )
        assert result.source == "pagerank_out"
        assert result.relation().count("*").fetchone()[0] == len(flat_org)

    def test_pipeline_rejects_non_frame_returns(self, flat_org: pd.DataFrame) -> None:
        frame = DuckONAFrame.from_pandas(flat_org, "hris")
        with pytest.raises(TypeError):
            frame.pipeline([lambda f: f.graph_pagerank()])
