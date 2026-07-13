"""Tests for standalone call instrumentation."""

import logging

from word_forge.observability import eidosian


def test_decorator_is_passthrough_by_default(monkeypatch) -> None:
    monkeypatch.delenv("WORD_FORGE_TRACE_CALLS", raising=False)
    monkeypatch.delenv("WORD_FORGE_SLOW_CALL_MS", raising=False)

    @eidosian()
    def add(left: int, right: int) -> int:
        return left + right

    assert add(2, 3) == 5
    assert add.__name__ == "add"
    assert add.__eidosian_config__["standalone"] is True  # type: ignore[attr-defined]


def test_decorator_warns_about_slow_calls(monkeypatch, caplog) -> None:
    monkeypatch.setenv("WORD_FORGE_SLOW_CALL_MS", "0.000001")

    @eidosian()
    def answer() -> int:
        return 42

    with caplog.at_level(logging.WARNING):
        assert answer() == 42
    assert "slow-call threshold" in caplog.text
