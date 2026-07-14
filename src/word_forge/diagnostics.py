"""Fast, non-invasive environment diagnostics for Word Forge."""

from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

from word_forge.config import DATA_ROOT
from word_forge.utils.nltk_utils import (
    ensure_nltk_data,
    get_missing_nltk_resources,
)

MINIMUM_PYTHON = (3, 10)


class DiagnosticStatus(str, Enum):
    """Outcome of an individual diagnostic check."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class DiagnosticCheckDict(TypedDict):
    """Serialized diagnostic check."""

    name: str
    status: str
    message: str
    required: bool


class DiagnosticReportDict(TypedDict):
    """Serialized environment diagnostic report."""

    ok: bool
    checks: List[DiagnosticCheckDict]


@dataclass(frozen=True)
class DiagnosticCheck:
    """One environment capability check.

    Attributes:
        name: Stable human-readable check name.
        status: Check outcome.
        message: Concise evidence or remediation guidance.
        required: Whether failure makes the core installation unusable.
    """

    name: str
    status: DiagnosticStatus
    message: str
    required: bool = True

    def to_dict(self) -> DiagnosticCheckDict:
        """Serialize this check for machine-readable output."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "required": self.required,
        }


@dataclass(frozen=True)
class DiagnosticReport:
    """Complete Word Forge environment diagnostic result."""

    checks: Tuple[DiagnosticCheck, ...]

    @property
    def ok(self) -> bool:
        """Return whether every required check passed."""
        return all(
            check.status is not DiagnosticStatus.FAIL
            for check in self.checks
            if check.required
        )

    def to_dict(self) -> DiagnosticReportDict:
        """Serialize the report for JSON output."""
        return {
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
        }


OPTIONAL_FEATURES: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "audio": (("ffmpeg", "whisper"), "audio"),
    "emotion": (("textblob",), "emotion"),
    "lexical RDF": (("rdflib",), "lexical"),
    "local LLM": (("torch", "transformers"), "llm"),
    "spaCy NLP": (("spacy",), "nlp"),
    "vector search": (("sentence_transformers", "chromadb", "faiss"), "vector"),
    "visualization": (("plotly", "pyvis"), "visualization"),
}


def _package_check(distribution: str) -> DiagnosticCheck:
    """Check an installed core Python distribution."""
    try:
        installed_version = version(distribution)
    except PackageNotFoundError:
        return DiagnosticCheck(
            name=f"package:{distribution}",
            status=DiagnosticStatus.FAIL,
            message="not installed",
        )
    return DiagnosticCheck(
        name=f"package:{distribution}",
        status=DiagnosticStatus.PASS,
        message=installed_version,
    )


def _optional_feature_check(
    name: str, modules: Tuple[str, ...], extra: str
) -> DiagnosticCheck:
    """Check an optional feature without importing its dependencies."""
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    if not missing:
        return DiagnosticCheck(
            name=f"optional:{name}",
            status=DiagnosticStatus.PASS,
            message="available",
            required=False,
        )
    return DiagnosticCheck(
        name=f"optional:{name}",
        status=DiagnosticStatus.WARNING,
        message=(
            f"missing {', '.join(missing)}; install with "
            f'python -m pip install "word_forge[{extra}]"'
        ),
        required=False,
    )


def _sqlite_check() -> DiagnosticCheck:
    """Verify SQLite can execute a transaction."""
    try:
        with sqlite3.connect(":memory:") as connection:
            value = connection.execute("SELECT 1").fetchone()
        if value != (1,):
            raise sqlite3.DatabaseError(f"unexpected result: {value!r}")
    except sqlite3.Error as exc:
        return DiagnosticCheck("sqlite", DiagnosticStatus.FAIL, str(exc))
    return DiagnosticCheck("sqlite", DiagnosticStatus.PASS, sqlite3.sqlite_version)


def _storage_check() -> DiagnosticCheck:
    """Verify the configured runtime data directory is writable."""
    data_root = Path(DATA_ROOT)
    try:
        data_root.mkdir(parents=True, exist_ok=True)
        descriptor, probe_path = tempfile.mkstemp(prefix=".doctor-", dir=data_root)
        os.close(descriptor)
        Path(probe_path).unlink()
    except OSError as exc:
        return DiagnosticCheck("storage", DiagnosticStatus.FAIL, str(exc))
    return DiagnosticCheck("storage", DiagnosticStatus.PASS, str(data_root.resolve()))


def run_diagnostics(*, fix: bool = False) -> DiagnosticReport:
    """Inspect core and optional Word Forge capabilities.

    Args:
        fix: Download missing NLTK resources before reporting their status.

    Returns:
        Immutable report containing required and optional capability checks.
    """
    if fix:
        ensure_nltk_data()

    python_ok = sys.version_info >= MINIMUM_PYTHON
    checks: List[DiagnosticCheck] = [
        DiagnosticCheck(
            "python",
            DiagnosticStatus.PASS if python_ok else DiagnosticStatus.FAIL,
            sys.version.split()[0],
        ),
        _package_check("nltk"),
        _package_check("networkx"),
        _package_check("numpy"),
        _sqlite_check(),
        _storage_check(),
    ]

    missing_nltk = get_missing_nltk_resources()
    checks.append(
        DiagnosticCheck(
            "nltk-data",
            DiagnosticStatus.FAIL if missing_nltk else DiagnosticStatus.PASS,
            (
                f"missing {', '.join(missing_nltk)}; run word_forge doctor --fix"
                if missing_nltk
                else "all parser resources available"
            ),
        )
    )

    for name, (modules, extra) in OPTIONAL_FEATURES.items():
        checks.append(_optional_feature_check(name, modules, extra))

    return DiagnosticReport(tuple(checks))


def render_diagnostics(report: DiagnosticReport) -> str:
    """Render a compact terminal report.

    Args:
        report: Diagnostic report to render.

    Returns:
        Multi-line human-readable report.
    """
    labels = {
        DiagnosticStatus.PASS: "PASS",
        DiagnosticStatus.WARNING: "INFO",
        DiagnosticStatus.FAIL: "FAIL",
    }
    lines = ["Word Forge doctor"]
    lines.extend(
        f"[{labels[check.status]:4}] {check.name}: {check.message}"
        for check in report.checks
    )
    lines.append("Result: ready" if report.ok else "Result: action required")
    return "\n".join(lines)


__all__ = [
    "DiagnosticCheck",
    "DiagnosticReport",
    "DiagnosticStatus",
    "render_diagnostics",
    "run_diagnostics",
]
