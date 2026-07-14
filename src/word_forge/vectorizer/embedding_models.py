"""Deterministic contracts for supported sentence-embedding models.

The vector pipeline accepts arbitrary Sentence Transformers models, but known
retrieval models require exact query/document prefixes. Keeping those rules in
one dependency-free module prevents the indexer and search path from drifting.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Literal, Optional

EmbeddingPromptStyle = Literal["none", "e5", "e5-instruct"]

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_RETRIEVAL_TASK = (
    "Given a lexical search query, retrieve relevant terms, definitions, and examples"
)
MAX_COLLECTION_NAME_LENGTH = 63


@dataclass(frozen=True, slots=True)
class EmbeddingModelSpec:
    """Stable operational metadata for a supported embedding model."""

    model_id: str
    dimension: int
    prompt_style: EmbeddingPromptStyle
    languages: str
    license_name: str
    description: str


KNOWN_EMBEDDING_MODELS: Dict[str, EmbeddingModelSpec] = {
    DEFAULT_EMBEDDING_MODEL.casefold(): EmbeddingModelSpec(
        model_id=DEFAULT_EMBEDDING_MODEL,
        dimension=384,
        prompt_style="e5",
        languages="100 (low-resource quality varies)",
        license_name="MIT",
        description="Portable multilingual retrieval model with 12 transformer layers.",
    ),
    "intfloat/multilingual-e5-base": EmbeddingModelSpec(
        model_id="intfloat/multilingual-e5-base",
        dimension=768,
        prompt_style="e5",
        languages="100 (low-resource quality varies)",
        license_name="MIT",
        description="Balanced multilingual E5 retrieval model.",
    ),
    "intfloat/multilingual-e5-large": EmbeddingModelSpec(
        model_id="intfloat/multilingual-e5-large",
        dimension=1024,
        prompt_style="e5",
        languages="100 (low-resource quality varies)",
        license_name="MIT",
        description="High-quality multilingual E5 retrieval model.",
    ),
    "intfloat/multilingual-e5-large-instruct": EmbeddingModelSpec(
        model_id="intfloat/multilingual-e5-large-instruct",
        dimension=1024,
        prompt_style="e5-instruct",
        languages="100 (low-resource quality varies)",
        license_name="MIT",
        description="Instruction-tuned multilingual E5 retrieval model.",
    ),
}


def get_embedding_model_spec(model_name: str) -> Optional[EmbeddingModelSpec]:
    """Return known metadata for ``model_name`` without loading model weights."""

    normalized = _validate_model_name(model_name).casefold()
    return KNOWN_EMBEDDING_MODELS.get(normalized)


def get_prompt_style(model_name: str) -> EmbeddingPromptStyle:
    """Infer the prompt family for a model, including compatible E5 variants."""

    normalized = _validate_model_name(model_name).casefold()
    known = KNOWN_EMBEDDING_MODELS.get(normalized)
    if known is not None:
        return known.prompt_style
    if "multilingual-e5" in normalized:
        return "e5-instruct" if normalized.endswith("-instruct") else "e5"
    return "none"


def format_embedding_text(
    model_name: str,
    text: str,
    *,
    is_query: bool,
    task: Optional[str] = None,
) -> str:
    """Format text according to the selected model's retrieval contract.

    Args:
        model_name: Hugging Face model identifier.
        text: Query or document content.
        is_query: Whether the content is a retrieval query.
        task: Optional one-sentence task for E5-instruct query formatting.

    Returns:
        Model-ready text with the exact required prefix, or unchanged stripped
        text for models without a registered prompt contract.

    Raises:
        ValueError: If the model name or text is empty.
    """

    _validate_model_name(model_name)
    clean_text = text.strip() if isinstance(text, str) else ""
    if not clean_text:
        raise ValueError("text must be a non-empty string")

    prompt_style = get_prompt_style(model_name)
    if prompt_style == "e5":
        prefix = "query: " if is_query else "passage: "
        return (
            clean_text
            if clean_text.casefold().startswith(prefix)
            else prefix + clean_text
        )
    if prompt_style == "e5-instruct" and is_query:
        clean_task = (task or DEFAULT_RETRIEVAL_TASK).strip()
        if not clean_task:
            raise ValueError("E5-instruct queries require a non-empty task")
        return f"Instruct: {clean_task}\nQuery: {clean_text}"
    return clean_text


def collection_name_for_model(model_name: str) -> str:
    """Return a collision-resistant Chroma-compatible collection name."""

    clean_name = _validate_model_name(model_name)
    normalized = clean_name.casefold()
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_") or "model"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    suffix = f"_{digest}"
    prefix = "wf_"
    available = MAX_COLLECTION_NAME_LENGTH - len(prefix) - len(suffix)
    return f"{prefix}{slug[:available].rstrip('_')}{suffix}"


def _validate_model_name(model_name: str) -> str:
    """Return a stripped model identifier or raise a precise validation error."""

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    return model_name.strip()


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_RETRIEVAL_TASK",
    "EmbeddingModelSpec",
    "KNOWN_EMBEDDING_MODELS",
    "MAX_COLLECTION_NAME_LENGTH",
    "collection_name_for_model",
    "format_embedding_text",
    "get_embedding_model_spec",
    "get_prompt_style",
]
