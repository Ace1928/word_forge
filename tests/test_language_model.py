"""Integration tests for word_forge.parser.language_model using real dependencies."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

# Check if LLM dependencies are available
_LLM_AVAILABLE = (
    importlib.util.find_spec("transformers") is not None
    and importlib.util.find_spec("torch") is not None
)

from word_forge.parser.language_model import MAX_AUTOMATIC_NEW_TOKENS, ModelState

TEST_MODEL = "sshleifer/tiny-gpt2"


def test_import_does_not_load_optional_inference_stack() -> None:
    """Importing model support must remain cheap for lexical-only processes."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import word_forge.parser.language_model; "
                "from word_forge.parser.model_profiles import resolve_model_profile; "
                "resolve_model_profile('off', require_ready=True); "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    not _LLM_AVAILABLE, reason="LLM dependencies (transformers, torch) not installed"
)
def test_initialize_and_generate_text() -> None:
    state = ModelState(model_name=TEST_MODEL)
    assert state.initialize() is True

    output = state.generate_text("Hello world", max_new_tokens=8)
    assert output is not None
    assert isinstance(output, str)


@pytest.mark.skipif(
    not _LLM_AVAILABLE, reason="LLM dependencies (transformers, torch) not installed"
)
def test_query_uses_model() -> None:
    state = ModelState(model_name=TEST_MODEL)
    assert state.initialize() is True

    output = state.query("Tell me a short phrase.", max_new_tokens=8)
    assert output is not None
    assert isinstance(output, str)


def test_set_model_resets_initialization() -> None:
    state = ModelState(model_name=TEST_MODEL)
    state._initialized = True
    state.set_model(TEST_MODEL)
    assert state._initialized is False


@pytest.mark.parametrize(
    ("prompt", "max_new_tokens", "temperature", "num_beams"),
    [
        ("", 8, 0.7, 1),
        ("prompt", 0, 0.7, 1),
        ("prompt", 8, -0.1, 1),
        ("prompt", 8, 0.7, 0),
    ],
)
def test_generate_text_rejects_invalid_parameters(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    num_beams: int,
) -> None:
    state = ModelState(model_name=TEST_MODEL)

    with pytest.raises(ValueError):
        state.generate_text(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_beams=num_beams,
        )


def test_automatic_generation_limit_is_conservative() -> None:
    assert MAX_AUTOMATIC_NEW_TOKENS == 256


def test_invalid_model_triggers_failure_tracking() -> None:
    state = ModelState(model_name="invalid/does-not-exist")
    result = state.initialize()
    assert result is False
    assert state._inference_failures >= 1
