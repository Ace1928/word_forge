"""Tests for optional-service endpoint resolution."""

import pytest

from word_forge.services import get_service_url


def test_defaults_are_portable(monkeypatch) -> None:
    monkeypatch.delenv("WORD_FORGE_OLLAMA_HTTP_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert (
        get_service_url(
            "ollama_http",
            default_host="localhost",
            default_port=11434,
            default_path="/api/generate",
        )
        == "http://localhost:11434/api/generate"
    )


def test_override_keeps_the_call_specific_path(monkeypatch) -> None:
    monkeypatch.setenv("WORD_FORGE_OLLAMA_HTTP_URL", "https://models.example:443")
    assert (
        get_service_url("ollama_http", default_port=11434, default_path="/api/embeddings")
        == "https://models.example:443/api/embeddings"
    )


def test_invalid_port_is_rejected() -> None:
    with pytest.raises(ValueError, match="default_port"):
        get_service_url("ollama_http", default_port=0)
