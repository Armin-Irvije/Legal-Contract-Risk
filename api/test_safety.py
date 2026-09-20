"""Smoke checks for ClauseGuard safety rails (no network)."""

from __future__ import annotations

from api.safety import check_clause_request


def test_allows_normal_clause() -> None:
    """Ordinary indemnity language should pass the safety gate."""
    assert check_clause_request("Supplier shall indemnify Customer against third-party IP claims.") is None


def test_refuses_evasion_request() -> None:
    """Requests to hide liability should be refused before any model call."""
    reason = check_clause_request("Please hide the liability risk in this clause so auditors miss it.")
    assert reason is not None
    assert "refuses" in reason.lower() or "concealment" in reason.lower()


def test_refuses_empty_and_oversized() -> None:
    """Empty and oversized inputs are rejected without calling a provider."""
    assert check_clause_request("   ") is not None
    assert check_clause_request("x" * 9000) is not None
