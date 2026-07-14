"""Validated, source-preserving records for normalized lexical imports."""

from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional, Sequence, Tuple

from word_forge.parser.linguistics import (
    canonicalize_language_tag,
    infer_script,
    normalize_term,
)

GlossKind = Literal["definition", "gloss", "raw"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: object) -> str:
    """Serialize JSON deterministically for equality keys and reproducibility."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Value is not portable JSON: {exc}") from exc


def _text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return unicodedata.normalize("NFC", value.strip())


def _optional_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("optional text values must be strings")
    return unicodedata.normalize("NFC", value.strip())


def _string_tuple(values: Sequence[str], field_name: str) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of strings")
    result = []
    for value in values:
        normalized = _text(value, field_name)
        if normalized not in result:
            result.append(normalized)
    return tuple(result)


def _confidence(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("confidence must be finite and between 0.0 and 1.0")
    return result


def _position(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("position must be a non-negative integer")
    return value


def _https_url(value: str, field_name: str, *, optional: bool = False) -> str:
    result = _optional_text(value)
    if optional and not result:
        return ""
    if not result.startswith("https://"):
        raise ValueError(f"{field_name} must use HTTPS")
    return result


def _metadata_json(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("metadata_json must be a string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"metadata_json must contain valid JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("metadata_json must contain a JSON object")
    return canonical_json(parsed)


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Reproducible identity and legal metadata for one source snapshot."""

    source_id: str
    source_version: str
    source_url: str
    license_name: str
    license_url: str
    attribution: str
    importer_version: str
    retrieved_at: float = field(default_factory=time.time)
    artifact_sha256: Optional[str] = None
    artifact_bytes: Optional[int] = None
    metadata_json: str = "{}"

    def __post_init__(self) -> None:
        """Canonicalize and validate the provenance envelope."""

        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        object.__setattr__(
            self, "source_version", _text(self.source_version, "source_version")
        )
        object.__setattr__(
            self, "source_url", _https_url(self.source_url, "source_url")
        )
        object.__setattr__(
            self, "license_name", _text(self.license_name, "license_name")
        )
        object.__setattr__(
            self, "license_url", _https_url(self.license_url, "license_url")
        )
        object.__setattr__(self, "attribution", _text(self.attribution, "attribution"))
        object.__setattr__(
            self,
            "importer_version",
            _text(self.importer_version, "importer_version"),
        )
        retrieved_at = float(self.retrieved_at)
        if not math.isfinite(retrieved_at) or retrieved_at < 0.0:
            raise ValueError("retrieved_at must be a finite non-negative timestamp")
        object.__setattr__(self, "retrieved_at", retrieved_at)

        digest = self.artifact_sha256
        if digest is not None:
            normalized_digest = digest.strip().casefold()
            if _SHA256.fullmatch(normalized_digest) is None:
                raise ValueError("artifact_sha256 must contain 64 hexadecimal digits")
            object.__setattr__(self, "artifact_sha256", normalized_digest)
        if self.artifact_bytes is not None:
            if (
                isinstance(self.artifact_bytes, bool)
                or not isinstance(self.artifact_bytes, int)
                or self.artifact_bytes < 0
            ):
                raise ValueError("artifact_bytes must be a non-negative integer")
            if digest is None:
                raise ValueError("artifact_bytes requires artifact_sha256")
        object.__setattr__(self, "metadata_json", _metadata_json(self.metadata_json))


@dataclass(frozen=True, slots=True)
class LexicalGlossRecord:
    """One ordered human-readable gloss for a lexical sense."""

    text: str
    language: str
    kind: GlossKind = "definition"
    generated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _text(self.text, "gloss text"))
        object.__setattr__(self, "language", canonicalize_language_tag(self.language))
        if self.kind not in {"definition", "gloss", "raw"}:
            raise ValueError(f"Unsupported gloss kind: {self.kind!r}")


@dataclass(frozen=True, slots=True)
class LexicalExampleRecord:
    """One source-authored or explicitly generated sense example."""

    text: str
    language: str
    source_example_id: str = ""
    translation: str = ""
    translation_language: str = ""
    reference: str = ""
    generated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _text(self.text, "example text"))
        object.__setattr__(self, "language", canonicalize_language_tag(self.language))
        for field_name in ("source_example_id", "translation", "reference"):
            object.__setattr__(
                self, field_name, _optional_text(getattr(self, field_name))
            )
        translation_language = _optional_text(self.translation_language)
        if translation_language:
            translation_language = canonicalize_language_tag(translation_language)
        if self.translation and not translation_language:
            raise ValueError("translated examples require translation_language")
        object.__setattr__(self, "translation_language", translation_language)


@dataclass(frozen=True, slots=True)
class LexicalSenseRecord:
    """A source-distinct sense with ordered glosses and examples."""

    source_sense_id: str
    position: int
    glosses: Tuple[LexicalGlossRecord, ...] = ()
    examples: Tuple[LexicalExampleRecord, ...] = ()
    concept_id: str = ""
    tags: Tuple[str, ...] = ()
    metadata_json: str = "{}"
    confidence: float = 1.0
    generated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_sense_id", _text(self.source_sense_id, "source_sense_id")
        )
        object.__setattr__(self, "position", _position(self.position))
        object.__setattr__(self, "concept_id", _optional_text(self.concept_id))
        object.__setattr__(self, "tags", _string_tuple(self.tags, "sense tags"))
        object.__setattr__(self, "metadata_json", _metadata_json(self.metadata_json))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        if len({gloss.text for gloss in self.glosses}) != len(self.glosses):
            raise ValueError("sense gloss text must be unique")
        if len({(example.text, example.language) for example in self.examples}) != len(
            self.examples
        ):
            raise ValueError("sense examples must be unique by text and language")


@dataclass(frozen=True, slots=True)
class LexicalFormRecord:
    """An inflected, alternate, or script-specific form of an entry."""

    form: str
    language: str
    position: int
    source_form_id: str = ""
    script: Optional[str] = None
    features: Tuple[str, ...] = ()
    tags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized_form = _text(self.form, "form")
        object.__setattr__(self, "form", normalized_form)
        object.__setattr__(self, "language", canonicalize_language_tag(self.language))
        object.__setattr__(self, "position", _position(self.position))
        object.__setattr__(self, "source_form_id", _optional_text(self.source_form_id))
        object.__setattr__(self, "script", self.script or infer_script(normalized_form))
        object.__setattr__(
            self, "features", _string_tuple(self.features, "form features")
        )
        object.__setattr__(self, "tags", _string_tuple(self.tags, "form tags"))


@dataclass(frozen=True, slots=True)
class LexicalPronunciationRecord:
    """A raw source pronunciation optionally attached to a specific form."""

    transcription: str
    notation: str
    language: str
    position: int
    source_record_id: str = ""
    form_source_id: str = ""
    dialect: str = ""
    tags: Tuple[str, ...] = ()
    audio_url: str = ""
    confidence: float = 1.0
    generated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "transcription", _text(self.transcription, "transcription")
        )
        object.__setattr__(self, "notation", _text(self.notation, "notation"))
        object.__setattr__(self, "language", canonicalize_language_tag(self.language))
        object.__setattr__(self, "position", _position(self.position))
        for field_name in ("source_record_id", "form_source_id", "dialect"):
            object.__setattr__(
                self, field_name, _optional_text(getattr(self, field_name))
            )
        object.__setattr__(self, "tags", _string_tuple(self.tags, "pronunciation tags"))
        object.__setattr__(
            self,
            "audio_url",
            _https_url(self.audio_url, "audio_url", optional=True),
        )
        object.__setattr__(self, "confidence", _confidence(self.confidence))


@dataclass(frozen=True, slots=True)
class LexicalRelationRecord:
    """A typed entry- or sense-scoped relation to a textual target."""

    relationship_type: str
    target_term: str
    target_language: str
    position: int
    source_sense_id: Optional[str] = None
    target_source_entry_id: str = ""
    source_record_id: str = ""
    confidence: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relationship_type",
            _text(self.relationship_type, "relationship_type"),
        )
        object.__setattr__(self, "target_term", _text(self.target_term, "target_term"))
        object.__setattr__(
            self,
            "target_language",
            canonicalize_language_tag(self.target_language),
        )
        object.__setattr__(self, "position", _position(self.position))
        if self.source_sense_id is not None:
            object.__setattr__(
                self,
                "source_sense_id",
                _text(self.source_sense_id, "source_sense_id"),
            )
        object.__setattr__(
            self,
            "target_source_entry_id",
            _optional_text(self.target_source_entry_id),
        )
        object.__setattr__(
            self, "source_record_id", _optional_text(self.source_record_id)
        )
        object.__setattr__(self, "confidence", _confidence(self.confidence))


@dataclass(frozen=True, slots=True)
class LexicalEntryRecord:
    """A complete source entry suitable for atomic, idempotent persistence."""

    source_entry_id: str
    lemma: str
    language: str
    part_of_speech: str = ""
    lexical_category: str = ""
    script: Optional[str] = None
    etymology: str = ""
    tags: Tuple[str, ...] = ()
    metadata_json: str = "{}"
    confidence: float = 1.0
    generated: bool = False
    forms: Tuple[LexicalFormRecord, ...] = ()
    senses: Tuple[LexicalSenseRecord, ...] = ()
    pronunciations: Tuple[LexicalPronunciationRecord, ...] = ()
    relations: Tuple[LexicalRelationRecord, ...] = ()

    def __post_init__(self) -> None:
        lemma = _text(self.lemma, "lemma")
        object.__setattr__(
            self, "source_entry_id", _text(self.source_entry_id, "source_entry_id")
        )
        object.__setattr__(self, "lemma", lemma)
        object.__setattr__(self, "language", canonicalize_language_tag(self.language))
        object.__setattr__(self, "part_of_speech", _optional_text(self.part_of_speech))
        object.__setattr__(
            self, "lexical_category", _optional_text(self.lexical_category)
        )
        object.__setattr__(self, "script", self.script or infer_script(lemma))
        object.__setattr__(self, "etymology", _optional_text(self.etymology))
        object.__setattr__(self, "tags", _string_tuple(self.tags, "entry tags"))
        object.__setattr__(self, "metadata_json", _metadata_json(self.metadata_json))
        object.__setattr__(self, "confidence", _confidence(self.confidence))

        sense_ids = [sense.source_sense_id for sense in self.senses]
        if len(sense_ids) != len(set(sense_ids)):
            raise ValueError("source_sense_id values must be unique within an entry")
        _require_unique_positions(self.senses, "sense")
        form_keys = [
            (normalize_term(form.form), form.language, form.features)
            for form in self.forms
        ]
        if len(form_keys) != len(set(form_keys)):
            raise ValueError("forms must be unique by form, language, and features")
        _require_unique_positions(self.forms, "form")
        form_source_ids = [
            form.source_form_id for form in self.forms if form.source_form_id
        ]
        if len(form_source_ids) != len(set(form_source_ids)):
            raise ValueError("non-empty source_form_id values must be unique")
        valid_sense_ids = set(sense_ids)
        for relation in self.relations:
            if (
                relation.source_sense_id is not None
                and relation.source_sense_id not in valid_sense_ids
            ):
                raise ValueError(
                    "relation source_sense_id must identify a sense in the entry"
                )
        form_ids = {form.source_form_id for form in self.forms if form.source_form_id}
        _require_unique_positions(self.pronunciations, "pronunciation")
        pronunciation_keys = [
            (
                item.notation,
                item.transcription,
                item.dialect,
                item.source_record_id,
            )
            for item in self.pronunciations
        ]
        if len(pronunciation_keys) != len(set(pronunciation_keys)):
            raise ValueError("pronunciations must have unique source identities")
        for pronunciation in self.pronunciations:
            if (
                pronunciation.form_source_id
                and pronunciation.form_source_id not in form_ids
            ):
                raise ValueError(
                    "pronunciation form_source_id must identify a form in the entry"
                )
        _require_unique_positions(self.relations, "relation")
        relation_keys = [
            (
                item.source_sense_id or "entry",
                item.relationship_type,
                normalize_term(item.target_term),
                item.target_language,
                item.target_source_entry_id,
                item.source_record_id,
            )
            for item in self.relations
        ]
        if len(relation_keys) != len(set(relation_keys)):
            raise ValueError("relations must have unique source identities")


def _require_unique_positions(values: Sequence[object], label: str) -> None:
    """Require child records exposing ``position`` to have stable ordering."""

    positions = [getattr(value, "position") for value in values]
    if len(positions) != len(set(positions)):
        raise ValueError(f"{label} positions must be unique")


__all__ = [
    "GlossKind",
    "LexicalEntryRecord",
    "LexicalExampleRecord",
    "LexicalFormRecord",
    "LexicalGlossRecord",
    "LexicalPronunciationRecord",
    "LexicalRelationRecord",
    "LexicalSenseRecord",
    "SourceSnapshot",
    "canonical_json",
]
