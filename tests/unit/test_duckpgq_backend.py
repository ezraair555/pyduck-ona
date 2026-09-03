"""Unit tests for DuckPGQ backend compatibility gating."""

from pyduck_ona.graph._duckpgq_backend import is_duckpgq_supported_duckdb


def test_duckpgq_support_gate_accepts_only_exact_supported_version() -> None:
    """Only versions with a published ABI-matching DuckPGQ build pass."""
    assert is_duckpgq_supported_duckdb("1.3.1")
    assert not is_duckpgq_supported_duckdb("1.3.2")
    assert not is_duckpgq_supported_duckdb("1.5.2")
    assert not is_duckpgq_supported_duckdb("1.2.9")


def test_duckpgq_support_gate_handles_suffixes_and_invalid_values() -> None:
    assert is_duckpgq_supported_duckdb("1.3.1-dev123")
    assert is_duckpgq_supported_duckdb("1.3.1+local")
    assert not is_duckpgq_supported_duckdb("not-a-version")
