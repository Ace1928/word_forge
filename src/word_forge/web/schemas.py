"""Validated request contracts for the local Word Forge HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    """Strict base model that rejects misspelled or unknown fields."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, str_strip_whitespace=True)


class EntryCreate(ApiModel):
    """Create or update one language-aware lexical entry."""

    term: str = Field(min_length=1, max_length=512)
    definition: str = Field(default="", max_length=100_000)
    part_of_speech: str = Field(default="", max_length=100)
    language: str = Field(default="en", min_length=2, max_length=35)
    examples: list[str] = Field(default_factory=list, max_length=100)
    source: str = Field(default="manual", max_length=500)
    base_term: str = Field(default="", max_length=512)
    expected_revision: int | None = Field(default=None, ge=0)


class RelationshipCreate(ApiModel):
    """Create one typed base-language relationship."""

    term: str = Field(min_length=1, max_length=512)
    target: str = Field(min_length=1, max_length=512)
    kind: str = Field(default="related", min_length=1, max_length=100)


class TranslationCreate(ApiModel):
    """Create a translation between two language-aware entries."""

    term: str = Field(min_length=1, max_length=512)
    source_language: str = Field(min_length=2, max_length=35)
    target_term: str = Field(min_length=1, max_length=512)
    target_language: str = Field(min_length=2, max_length=35)
    source: str = Field(default="manual", max_length=500)
    bidirectional: bool = True


class SenseWrite(ApiModel):
    """Create or revise one meaning attached to a lexical entry."""

    term: str = Field(min_length=1, max_length=512)
    language: str = Field(default="en", min_length=2, max_length=35)
    definition: str = Field(min_length=1, max_length=20_000)
    part_of_speech: str = Field(default="", max_length=100)
    domain: str = Field(default="", max_length=200)
    register_name: str = Field(default="", alias="register", max_length=100)
    source: str = Field(default="manual", max_length=500)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    review_state: str = Field(default="unreviewed", max_length=20)
    sense_id: str | None = Field(default=None, max_length=64)
    expected_revision: int | None = Field(default=None, ge=0)


class BackupCreate(ApiModel):
    """Optional human label for a managed local snapshot."""

    label: str = Field(default="", max_length=80, pattern=r"^[A-Za-z0-9 _.-]*$")


class RestoreRequest(ApiModel):
    """Explicit confirmation for restoring one managed snapshot."""

    backup_name: str = Field(min_length=1, max_length=180)
    confirmation: str = Field(min_length=1, max_length=20)


__all__ = [
    "BackupCreate",
    "EntryCreate",
    "RelationshipCreate",
    "RestoreRequest",
    "SenseWrite",
    "TranslationCreate",
]
