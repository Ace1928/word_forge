"""Cross-platform tests for Word Forge's runtime filesystem policy."""

from __future__ import annotations

from pathlib import Path

from word_forge.paths import default_home_directory


def test_linux_home_honors_xdg_data_home(tmp_path: Path) -> None:
    assert (
        default_home_directory(
            environ={"XDG_DATA_HOME": str(tmp_path)},
            platform="linux",
            home=Path("/unused"),
        )
        == tmp_path / "word-forge"
    )


def test_windows_home_honors_local_app_data(tmp_path: Path) -> None:
    assert (
        default_home_directory(
            environ={"LOCALAPPDATA": str(tmp_path)},
            platform="win32",
            home=Path("C:/unused"),
        )
        == tmp_path / "WordForge"
    )


def test_explicit_home_has_precedence(tmp_path: Path) -> None:
    configured = tmp_path / "portable"
    assert (
        default_home_directory(
            environ={"WORD_FORGE_HOME": str(configured), "XDG_DATA_HOME": "/ignored"},
            platform="linux",
            home=Path("/unused"),
        )
        == configured
    )
