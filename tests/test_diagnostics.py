"""Integration tests for installation diagnostics."""

from word_forge.diagnostics import render_diagnostics, run_diagnostics
from word_forge.utils.nltk_utils import ensure_nltk_data


def test_diagnostics_report_core_environment_ready() -> None:
    ensure_nltk_data()

    report = run_diagnostics()
    serialized = report.to_dict()

    assert report.ok is True
    assert serialized["ok"] is True
    assert any(check["name"] == "sqlite" for check in serialized["checks"])
    assert render_diagnostics(report).endswith("Result: ready")
