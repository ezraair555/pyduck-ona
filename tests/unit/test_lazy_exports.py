"""Tests for the lazy (PEP 562) viz exports on the pyduck_ona top level.

Regression tests for the RecursionError found during the v0.3.0 viz
integration: `from pyduck_ona import viz` inside `__getattr__` re-entered
`__getattr__` before the submodule import completed, blowing the stack.
The fix uses `importlib.import_module("pyduck_ona.viz")`, which never
triggers the parent package's `__getattr__`.
"""
from __future__ import annotations

import pytest

import pyduck_ona


def test_from_import_viz_subpackage() -> None:
    """`from pyduck_ona import viz` must resolve without recursion."""
    from pyduck_ona import viz

    assert hasattr(viz, "span_of_control")
    assert hasattr(viz, "org_chart_tree")


def test_from_import_viz_function() -> None:
    """`from pyduck_ona import <viz function>` resolves via lazy export."""
    from pyduck_ona import span_of_control

    assert callable(span_of_control)


def test_attribute_access_matches_subpackage() -> None:
    """Lazy attribute resolves to the same object as the submodule import."""
    import pyduck_ona.viz as direct
    from pyduck_ona import viz as lazy

    assert lazy is direct
    from pyduck_ona import org_chart_tree as lazy_fn

    assert lazy_fn is direct.org_chart_tree


def test_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(pyduck_ona, "not_a_real_export")  # noqa: B009


def test_dir_includes_viz_names() -> None:
    listing = dir(pyduck_ona)
    assert "viz" in listing
    assert "span_of_control" in listing
