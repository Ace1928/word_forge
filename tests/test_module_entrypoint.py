"""Smoke tests for module-based command invocation."""

import os
import subprocess
import sys
from pathlib import Path

from word_forge import __main__


def test_module_entrypoint_exports_main() -> None:
    assert callable(__main__.main)


def test_package_metadata_import_does_not_initialize_heavy_stacks() -> None:
    script = """
import sys
import word_forge
for name in ('networkx', 'nltk', 'numpy', 'torch', 'transformers'):
    assert name not in sys.modules, name
assert word_forge.__version__
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
