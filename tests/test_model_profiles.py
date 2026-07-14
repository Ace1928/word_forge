"""Tests for resource-aware local language-model profiles."""

from __future__ import annotations

import pytest

from word_forge.parser.model_profiles import (
    MODEL_PROFILES,
    ModelProfileError,
    RuntimeResources,
    get_model_profile,
    recommend_model_profile,
    resolve_model_profile,
    version_at_least,
)


def _runtime(
    *,
    available_ram_gib: float = 8.0,
    accelerator: str = "cpu",
    accelerator_memory_gib: float | None = None,
    torch_version: str | None = "2.8.0",
    transformers_version: str | None = "5.5.0",
) -> RuntimeResources:
    return RuntimeResources(
        total_ram_gib=max(available_ram_gib, 8.0),
        available_ram_gib=available_ram_gib,
        cpu_threads=8,
        accelerator=accelerator,  # type: ignore[arg-type]
        accelerator_memory_gib=accelerator_memory_gib,
        torch_version=torch_version,
        transformers_version=transformers_version,
    )


def test_catalog_has_offline_portable_and_gemma_profiles() -> None:
    assert tuple(MODEL_PROFILES) == (
        "off",
        "portable",
        "gemma3-tiny",
        "gemma4-edge",
    )
    assert MODEL_PROFILES["portable"].model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert MODEL_PROFILES["gemma4-edge"].model_id == "google/gemma-4-E2B-it"


def test_cpu_runtime_recommends_ungated_portable_model() -> None:
    assert recommend_model_profile(_runtime()).name == "portable"


def test_accelerated_runtime_recommends_gemma4() -> None:
    resources = _runtime(
        available_ram_gib=24.0,
        accelerator="cuda",
        accelerator_memory_gib=16.0,
    )
    assert recommend_model_profile(resources).name == "gemma4-edge"


def test_missing_dependencies_keep_automatic_selection_offline() -> None:
    resources = _runtime(torch_version=None, transformers_version=None)
    assert recommend_model_profile(resources).name == "off"


def test_profile_readiness_reports_memory_and_version_requirements() -> None:
    resources = _runtime(available_ram_gib=4.0, transformers_version="5.4.0")
    ready, issues = MODEL_PROFILES["gemma4-edge"].readiness(resources)

    assert ready is False
    assert any("Transformers >=5.5.0" in issue for issue in issues)
    assert any("14 GiB" in issue for issue in issues)


def test_resolve_auto_and_profile_name_normalization() -> None:
    assert resolve_model_profile("AUTO", _runtime()).name == "portable"
    assert get_model_profile("gemma3_tiny").name == "gemma3-tiny"


def test_required_profile_readiness_rejects_unsafe_runtime() -> None:
    resources = _runtime(
        available_ram_gib=1.0,
        torch_version=None,
        transformers_version=None,
    )

    with pytest.raises(ModelProfileError, match="not ready") as error:
        resolve_model_profile("portable", resources, require_ready=True)

    message = str(error.value)
    assert "word_forge[llm]" in message
    assert "2.5 GiB" in message
    assert "word_forge models list" in message


def test_off_profile_is_always_ready() -> None:
    resources = _runtime(
        available_ram_gib=0.0,
        torch_version=None,
        transformers_version=None,
    )

    assert resolve_model_profile("off", resources, require_ready=True).name == "off"


def test_unknown_profile_has_actionable_error() -> None:
    with pytest.raises(ModelProfileError, match="Available profiles"):
        get_model_profile("enormous")


def test_parser_refiner_is_offline_by_default(tmp_path) -> None:
    """Constructing the core parser must not allocate or download a model."""
    from word_forge.database.database_manager import DBManager
    from word_forge.parser.parser_refiner import ParserRefiner
    from word_forge.queue.queue_manager import QueueManager

    database = DBManager(db_path=tmp_path / "offline.db")
    queue: QueueManager[str] = QueueManager()
    parser = ParserRefiner(db_manager=database, queue_manager=queue)
    try:
        assert parser.llm_state is None
    finally:
        parser.shutdown()
        database.close()


def test_parser_refiner_resolves_explicit_profile(tmp_path) -> None:
    """An explicit profile selects a model without eagerly initializing it."""
    from word_forge.database.database_manager import DBManager
    from word_forge.parser.parser_refiner import ParserRefiner
    from word_forge.queue.queue_manager import QueueManager

    database = DBManager(db_path=tmp_path / "profile.db")
    queue: QueueManager[str] = QueueManager()
    parser = ParserRefiner(
        db_manager=database, queue_manager=queue, model_profile="portable"
    )
    try:
        assert parser.llm_state is not None
        assert parser.llm_state.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert parser.llm_state.is_initialized() is False
    finally:
        parser.shutdown()
        database.close()


@pytest.mark.parametrize(
    ("installed", "required", "expected"),
    [
        ("5.5.0", "5.5.0", True),
        ("5.13.1", "5.5.0", True),
        ("5.4.9", "5.5.0", False),
        ("5.5.0.dev0", "5.5.0", True),
        (None, "4.37.0", False),
    ],
)
def test_version_at_least(installed: str | None, required: str, expected: bool) -> None:
    assert version_at_least(installed, required) is expected
