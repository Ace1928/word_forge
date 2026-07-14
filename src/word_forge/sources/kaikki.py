"""Streaming, resumable importer for Kaikki/Wiktextract JSON Lines data."""

from __future__ import annotations

import bz2
import gzip
import hashlib
import importlib.metadata
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    IO,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from word_forge.database.lexical_repository import (
    LexicalRepository,
    LexicalRepositoryError,
    LexicalWriteReport,
)
from word_forge.lexicon import (
    LexicalEntryRecord,
    LexicalExampleRecord,
    LexicalFormRecord,
    LexicalGlossRecord,
    LexicalPronunciationRecord,
    LexicalRelationRecord,
    LexicalSenseRecord,
    SourceSnapshot,
    canonical_json,
)
from word_forge.parser.linguistics import canonicalize_language_tag, normalize_term
from word_forge.sources.registry import get_source

_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_MAX_BATCH_SIZE = 10_000
_CHECKPOINT_SCHEMA_VERSION = 2
_IMPORT_FORMAT_VERSION = "kaikki-v1"

_RELATION_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("synonyms", "synonym"),
    ("antonyms", "antonym"),
    ("hypernyms", "hypernym"),
    ("hyponyms", "hyponym"),
    ("holonyms", "holonym"),
    ("meronyms", "meronym"),
    ("troponyms", "troponym"),
    ("coordinate_terms", "coordinate-term"),
    ("derived", "derived"),
    ("related", "related"),
)


class KaikkiImportError(RuntimeError):
    """Raised when a source artifact cannot be imported safely."""

    def __init__(
        self,
        message: str,
        *,
        line_number: Optional[int] = None,
        committed_through: int = 0,
    ) -> None:
        super().__init__(message)
        self.line_number = line_number
        self.committed_through = committed_through


class KaikkiLicenseError(KaikkiImportError):
    """Raised until share-alike source terms are explicitly acknowledged."""


def _normalize_sha256(value: str, *, field_name: str) -> str:
    """Validate and normalize one SHA-256 hexadecimal digest."""

    normalized = value.strip().casefold()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field_name} must contain 64 hexadecimal digits")
    return normalized


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """Immutable local artifact identity used before any database writes."""

    path: Path
    sha256: str
    byte_size: int
    modified_ns: int

    def __post_init__(self) -> None:
        normalized_digest = _normalize_sha256(self.sha256, field_name="sha256")
        if self.byte_size < 0 or self.modified_ns < 0:
            raise ValueError("artifact size and modification time must be non-negative")
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())
        object.__setattr__(self, "sha256", normalized_digest)

    def assert_unchanged(self) -> None:
        """Reject a file replaced between inspection and import."""

        try:
            stat = self.path.stat()
        except OSError as exc:
            raise KaikkiImportError(f"Cannot stat source artifact: {exc}") from exc
        if stat.st_size != self.byte_size or stat.st_mtime_ns != self.modified_ns:
            raise KaikkiImportError(
                "Source artifact changed after hashing; inspect it again before import"
            )


@dataclass(frozen=True, slots=True)
class KaikkiCheckpoint:
    """Durable position after the last successfully committed batch."""

    schema_version: int
    artifact_sha256: str
    configuration_sha256: str
    snapshot_id: int
    next_line: int
    imported_entries: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_sha256",
            _normalize_sha256(self.artifact_sha256, field_name="artifact_sha256"),
        )
        object.__setattr__(
            self,
            "configuration_sha256",
            _normalize_sha256(
                self.configuration_sha256, field_name="configuration_sha256"
            ),
        )
        if self.snapshot_id <= 0:
            raise ValueError("snapshot_id must be positive")
        if self.next_line < 1 or self.imported_entries < 0:
            raise ValueError("checkpoint counters must be non-negative")

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_sha256": self.artifact_sha256,
            "configuration_sha256": self.configuration_sha256,
            "snapshot_id": self.snapshot_id,
            "next_line": self.next_line,
            "imported_entries": self.imported_entries,
        }


@dataclass(frozen=True, slots=True)
class KaikkiImportReport:
    """Operational and persistence counts for one import invocation."""

    snapshot_id: int
    artifact_sha256: str
    artifact_bytes: int
    first_line: int
    last_line: int
    lines_read: int
    parsed_entries: int
    skipped_entries: int
    batches: int
    write_report: LexicalWriteReport
    elapsed_seconds: float

    def to_dict(self) -> Dict[str, object]:
        """Return a stable, JSON-serializable operational report."""

        writes = self.write_report
        return {
            "snapshot_id": self.snapshot_id,
            "artifact_sha256": self.artifact_sha256,
            "artifact_bytes": self.artifact_bytes,
            "first_line": self.first_line,
            "last_line": self.last_line,
            "lines_read": self.lines_read,
            "parsed_entries": self.parsed_entries,
            "skipped_entries": self.skipped_entries,
            "batches": self.batches,
            "write_report": {
                "attempted": writes.attempted,
                "inserted": writes.inserted,
                "updated": writes.updated,
                "forms": writes.forms,
                "senses": writes.senses,
                "glosses": writes.glosses,
                "examples": writes.examples,
                "pronunciations": writes.pronunciations,
                "relations": writes.relations,
            },
            "elapsed_seconds": self.elapsed_seconds,
        }


def inspect_artifact(
    path: Path,
    *,
    chunk_bytes: int = _HASH_CHUNK_BYTES,
    expected_sha256: Optional[str] = None,
) -> ArtifactIdentity:
    """Hash a local artifact in bounded memory and return its stable identity."""

    source_path = Path(path).expanduser().resolve()
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    try:
        stat = source_path.stat()
    except OSError as exc:
        raise KaikkiImportError(f"Cannot inspect source artifact: {exc}") from exc
    if not source_path.is_file():
        raise KaikkiImportError(f"Source artifact is not a regular file: {source_path}")

    digest = hashlib.sha256()
    try:
        with source_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_bytes), b""):
                digest.update(chunk)
    except OSError as exc:
        raise KaikkiImportError(f"Cannot hash source artifact: {exc}") from exc
    try:
        final_stat = source_path.stat()
    except OSError as exc:
        raise KaikkiImportError(f"Cannot re-inspect source artifact: {exc}") from exc
    if final_stat.st_size != stat.st_size or final_stat.st_mtime_ns != stat.st_mtime_ns:
        raise KaikkiImportError("Source artifact changed while it was being hashed")
    identity = ArtifactIdentity(
        path=source_path,
        sha256=digest.hexdigest(),
        byte_size=stat.st_size,
        modified_ns=stat.st_mtime_ns,
    )
    if expected_sha256 is not None:
        expected = _normalize_sha256(expected_sha256, field_name="expected_sha256")
        if identity.sha256 != expected:
            raise KaikkiImportError(
                "Source artifact SHA-256 mismatch: "
                f"expected {expected}, calculated {identity.sha256}"
            )
    return identity


class KaikkiImporter:
    """Convert Kaikki records and persist them in bounded atomic batches."""

    def __init__(
        self,
        repository: LexicalRepository,
        *,
        source_version: str,
        source_url: str,
        accept_source_license: bool,
        batch_size: int = 500,
        languages: Sequence[str] = (),
    ) -> None:
        if not accept_source_license:
            raise KaikkiLicenseError(
                "Kaikki contains Wiktionary-derived data. Re-run only after "
                "accepting the applicable attribution/share-alike source terms."
            )
        if not 1 <= batch_size <= _MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must be between 1 and {_MAX_BATCH_SIZE:,}")
        normalized_source_version = source_version.strip()
        normalized_source_url = source_url.strip()
        if not normalized_source_version:
            raise ValueError("source_version must be non-empty")
        if not normalized_source_url.startswith("https://"):
            raise ValueError("source_url must use HTTPS")
        self.repository = repository
        self.source_version = normalized_source_version
        self.source_url = normalized_source_url
        self.batch_size = batch_size
        self.languages = frozenset(
            canonicalize_language_tag(language) for language in languages
        )
        configuration_json = canonical_json(
            {
                "database_path": str(
                    self.repository.database.db_path.expanduser().resolve()
                ),
                "format_version": _IMPORT_FORMAT_VERSION,
                "languages": sorted(self.languages),
                "source_url": self.source_url,
                "source_version": self.source_version,
            }
        )
        self.configuration_sha256 = hashlib.sha256(
            configuration_json.encode("utf-8")
        ).hexdigest()

    def import_artifact(
        self,
        artifact: ArtifactIdentity,
        *,
        checkpoint_path: Optional[Path] = None,
        max_entries: Optional[int] = None,
    ) -> KaikkiImportReport:
        """Import a previously inspected artifact and checkpoint committed batches."""

        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be positive when provided")
        artifact.assert_unchanged()
        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint is not None:
            if checkpoint.artifact_sha256 != artifact.sha256:
                raise KaikkiImportError(
                    "Checkpoint artifact digest does not match the selected file"
                )
            if checkpoint.configuration_sha256 != self.configuration_sha256:
                raise KaikkiImportError(
                    "Checkpoint import configuration does not match the selected "
                    "database, source metadata, or language filters"
                )
        source = get_source("kaikki-wiktionary")
        snapshot = SourceSnapshot(
            source_id=source.source_id,
            source_version=self.source_version,
            source_url=self.source_url,
            retrieved_at=time.time(),
            artifact_sha256=artifact.sha256,
            artifact_bytes=artifact.byte_size,
            license_name=source.license_name,
            license_url=source.license_url,
            attribution=source.attribution,
            importer_version=_importer_version(),
            metadata_json=canonical_json(
                {
                    "format": "kaikki-wiktextract-jsonl",
                    "compression": _compression_name(artifact.path),
                    "license_acknowledged": True,
                }
            ),
        )
        snapshot_id = self.repository.register_snapshot(snapshot)
        start_line = 1
        imported_before = 0
        if checkpoint is not None:
            if checkpoint.artifact_sha256 != artifact.sha256:
                raise KaikkiImportError(
                    "Checkpoint artifact digest does not match the selected file"
                )
            if checkpoint.snapshot_id != snapshot_id:
                raise KaikkiImportError(
                    "Checkpoint snapshot does not match the registered source"
                )
            start_line = checkpoint.next_line
            imported_before = checkpoint.imported_entries

        started = time.perf_counter()
        lines_read = 0
        parsed_entries = 0
        skipped_entries = 0
        batches = 0
        last_line = start_line - 1
        batch: List[LexicalEntryRecord] = []
        total = _zero_write_report()
        committed_through = start_line - 1

        try:
            with _open_jsonl(artifact.path) as handle:
                for line_number, line in enumerate(handle, start=1):
                    if line_number < start_line:
                        continue
                    if max_entries is not None and parsed_entries >= max_entries:
                        break
                    last_line = line_number
                    lines_read += 1
                    if not line.strip():
                        skipped_entries += 1
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise KaikkiImportError(
                            f"Invalid JSON on line {line_number}: {exc}",
                            line_number=line_number,
                            committed_through=committed_through,
                        ) from exc
                    if not isinstance(raw, Mapping):
                        raise KaikkiImportError(
                            f"Line {line_number} is not a JSON object",
                            line_number=line_number,
                            committed_through=committed_through,
                        )
                    try:
                        record = parse_kaikki_record(raw, line_number=line_number)
                    except (TypeError, ValueError) as exc:
                        raise KaikkiImportError(
                            f"Invalid lexical record on line {line_number}: {exc}",
                            line_number=line_number,
                            committed_through=committed_through,
                        ) from exc
                    if self.languages and record.language not in self.languages:
                        skipped_entries += 1
                        continue
                    batch.append(record)
                    parsed_entries += 1
                    if len(batch) >= self.batch_size:
                        written = self.repository.upsert_entries(snapshot_id, batch)
                        total = _add_reports(total, written)
                        batches += 1
                        batch.clear()
                        committed_through = line_number
                        _save_checkpoint(
                            checkpoint_path,
                            KaikkiCheckpoint(
                                schema_version=_CHECKPOINT_SCHEMA_VERSION,
                                artifact_sha256=artifact.sha256,
                                configuration_sha256=self.configuration_sha256,
                                snapshot_id=snapshot_id,
                                next_line=line_number + 1,
                                imported_entries=imported_before + total.attempted,
                            ),
                        )
                if batch:
                    written = self.repository.upsert_entries(snapshot_id, batch)
                    total = _add_reports(total, written)
                    batches += 1
                    batch.clear()
                committed_through = last_line
                _save_checkpoint(
                    checkpoint_path,
                    KaikkiCheckpoint(
                        schema_version=_CHECKPOINT_SCHEMA_VERSION,
                        artifact_sha256=artifact.sha256,
                        configuration_sha256=self.configuration_sha256,
                        snapshot_id=snapshot_id,
                        next_line=last_line + 1,
                        imported_entries=imported_before + total.attempted,
                    ),
                )
        except KaikkiImportError:
            raise
        except (LexicalRepositoryError, ValueError) as exc:
            raise KaikkiImportError(
                f"Database import failed at line {last_line}: {exc}",
                line_number=last_line or None,
                committed_through=committed_through,
            ) from exc
        except (OSError, UnicodeError) as exc:
            raise KaikkiImportError(
                f"Failed reading source artifact: {exc}",
                line_number=last_line or None,
                committed_through=committed_through,
            ) from exc

        artifact.assert_unchanged()
        return KaikkiImportReport(
            snapshot_id=snapshot_id,
            artifact_sha256=artifact.sha256,
            artifact_bytes=artifact.byte_size,
            first_line=start_line,
            last_line=last_line,
            lines_read=lines_read,
            parsed_entries=parsed_entries,
            skipped_entries=skipped_entries,
            batches=batches,
            write_report=total,
            elapsed_seconds=time.perf_counter() - started,
        )


def parse_kaikki_record(
    raw: Mapping[str, object], *, line_number: int
) -> LexicalEntryRecord:
    """Convert one Wiktextract record without retaining the full source object."""

    lemma = _required_string(raw, "word")
    language = canonicalize_language_tag(_string(raw.get("lang_code")) or "und")
    part_of_speech = _string(raw.get("pos"))
    source_entry_id = _source_entry_id(raw, language, part_of_speech, line_number)
    senses = _parse_senses(raw.get("senses"), language)
    forms = _parse_forms(raw.get("forms"), language)
    pronunciations = _parse_pronunciations(raw.get("sounds"), language)
    relations = list(_parse_relations(raw, language, source_sense_id=None))
    relations.extend(_parse_translations(raw.get("translations"), None))
    for sense, source_sense in zip(_mapping_list(raw.get("senses")), senses):
        relations.extend(
            _parse_relations(sense, language, source_sense.source_sense_id)
        )
        relations.extend(
            _parse_translations(sense.get("translations"), source_sense.source_sense_id)
        )
    relations = [
        LexicalRelationRecord(
            relationship_type=relation.relationship_type,
            target_term=relation.target_term,
            target_language=relation.target_language,
            position=position,
            source_sense_id=relation.source_sense_id,
            target_source_entry_id=relation.target_source_entry_id,
            source_record_id=relation.source_record_id,
            confidence=relation.confidence,
        )
        for position, relation in enumerate(_deduplicate_relations(relations))
    ]

    metadata = {
        key: raw[key]
        for key in (
            "lang",
            "pos_title",
            "etymology_number",
            "categories",
            "topics",
            "wikidata",
            "wikipedia",
            "redirect",
        )
        if key in raw
    }
    metadata["source_line"] = line_number
    return LexicalEntryRecord(
        source_entry_id=source_entry_id,
        lemma=lemma,
        language=language,
        part_of_speech=part_of_speech,
        lexical_category=_string(raw.get("pos_title")),
        etymology=_string(raw.get("etymology_text")),
        tags=_string_sequence(raw.get("tags")),
        metadata_json=canonical_json(metadata),
        forms=forms,
        senses=senses,
        pronunciations=pronunciations,
        relations=tuple(relations),
    )


def _parse_forms(value: object, language: str) -> Tuple[LexicalFormRecord, ...]:
    result: List[LexicalFormRecord] = []
    seen = set()
    for item in _mapping_list(value):
        form = _string(item.get("form"))
        if not form or form == "-":
            continue
        features = _string_sequence(item.get("tags"))
        key = (normalize_term(form), features)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            LexicalFormRecord(
                form=form,
                language=language,
                position=len(result),
                source_form_id=_string(item.get("id")) or f"form:{len(result)}",
                features=features,
                tags=_string_sequence(item.get("raw_tags")),
            )
        )
    return tuple(result)


def _parse_senses(value: object, language: str) -> Tuple[LexicalSenseRecord, ...]:
    result: List[LexicalSenseRecord] = []
    for item in _mapping_list(value):
        source_sense_id = _string(item.get("id")) or f"sense:{len(result)}"
        glosses: List[LexicalGlossRecord] = []
        for text in _string_sequence(item.get("glosses")):
            if text not in {gloss.text for gloss in glosses}:
                glosses.append(LexicalGlossRecord(text, language, "definition"))
        for text in _string_sequence(item.get("raw_glosses")):
            if text not in {gloss.text for gloss in glosses}:
                glosses.append(LexicalGlossRecord(text, language, "raw"))
        examples = _parse_examples(item.get("examples"), language, len(result))
        metadata = {
            key: item[key]
            for key in ("categories", "topics", "wikidata", "wikipedia", "senseid")
            if key in item
        }
        result.append(
            LexicalSenseRecord(
                source_sense_id=source_sense_id,
                position=len(result),
                glosses=tuple(glosses),
                examples=examples,
                concept_id=_string(item.get("wikidata")),
                tags=_string_sequence(item.get("tags")),
                metadata_json=canonical_json(metadata),
            )
        )
    return tuple(result)


def _parse_examples(
    value: object, language: str, sense_position: int
) -> Tuple[LexicalExampleRecord, ...]:
    result: List[LexicalExampleRecord] = []
    seen = set()
    for item in _mapping_list(value):
        text = _string(item.get("text"))
        if not text or text in seen:
            continue
        seen.add(text)
        translation = _string(item.get("english"))
        translation_language = "en" if translation else ""
        if not translation:
            candidate = _string(item.get("translation"))
            candidate_language = _string(item.get("translation_language"))
            if candidate and candidate_language:
                translation = candidate
                translation_language = candidate_language
        result.append(
            LexicalExampleRecord(
                text=text,
                language=language,
                source_example_id=_string(item.get("id"))
                or f"example:{sense_position}:{len(result)}",
                translation=translation,
                translation_language=translation_language,
                reference=_string(item.get("ref")),
            )
        )
    return tuple(result)


def _parse_pronunciations(
    value: object, language: str
) -> Tuple[LexicalPronunciationRecord, ...]:
    result: List[LexicalPronunciationRecord] = []
    seen = set()
    for item in _mapping_list(value):
        notation = ""
        transcription = ""
        if _string(item.get("ipa")):
            notation = "ipa"
            transcription = _string(item.get("ipa"))
        elif _string(item.get("enpr")):
            notation = "enpr"
            transcription = _string(item.get("enpr"))
        if not transcription:
            continue
        key = (notation, transcription)
        if key in seen:
            continue
        seen.add(key)
        audio_url = next(
            (
                candidate
                for candidate in (
                    _string(item.get("ogg_url")),
                    _string(item.get("mp3_url")),
                )
                if candidate.startswith("https://")
            ),
            "",
        )
        result.append(
            LexicalPronunciationRecord(
                transcription=transcription,
                notation=notation,
                language=language,
                position=len(result),
                source_record_id=_string(item.get("id")) or f"sound:{len(result)}",
                tags=_string_sequence(item.get("tags")),
                audio_url=audio_url,
            )
        )
    return tuple(result)


def _parse_relations(
    raw: Mapping[str, object], language: str, source_sense_id: Optional[str]
) -> Iterator[LexicalRelationRecord]:
    for field, relationship_type in _RELATION_FIELDS:
        for item in _target_items(raw.get(field)):
            target = _target_word(item)
            if not target:
                continue
            yield LexicalRelationRecord(
                relationship_type=relationship_type,
                target_term=target,
                target_language=canonicalize_language_tag(
                    _string(item.get("code"))
                    or _string(item.get("lang_code"))
                    or language
                ),
                position=0,
                source_sense_id=source_sense_id,
                target_source_entry_id=_string(item.get("id")),
            )


def _parse_translations(
    value: object, source_sense_id: Optional[str]
) -> Iterator[LexicalRelationRecord]:
    for item in _mapping_list(value):
        target = _target_word(item)
        if not target:
            continue
        target_language = _string(item.get("code")) or _string(item.get("lang_code"))
        if not target_language:
            target_language = "und"
        yield LexicalRelationRecord(
            relationship_type="translation",
            target_term=target,
            target_language=canonicalize_language_tag(target_language),
            position=0,
            source_sense_id=source_sense_id,
            target_source_entry_id=_string(item.get("id")),
        )


def _deduplicate_relations(
    relations: Iterable[LexicalRelationRecord],
) -> Tuple[LexicalRelationRecord, ...]:
    result = []
    seen = set()
    for relation in relations:
        key = (
            relation.source_sense_id,
            relation.relationship_type,
            normalize_term(relation.target_term),
            relation.target_language,
            relation.target_source_entry_id,
        )
        if key not in seen:
            seen.add(key)
            result.append(relation)
    return tuple(result)


def _source_entry_id(
    raw: Mapping[str, object],
    language: str,
    part_of_speech: str,
    line_number: int,
) -> str:
    explicit = next(
        (
            value
            for value in (
                _string(raw.get("id")),
                _string(raw.get("entry_id")),
                _string(raw.get("source_id")),
            )
            if value
        ),
        "",
    )
    if explicit:
        return explicit
    lemma = _required_string(raw, "word")
    etymology_number = _string(raw.get("etymology_number")) or "0"
    return f"{language}:{lemma}:{part_of_speech or 'unknown'}:{etymology_number}:{line_number}"


def _target_items(value: object) -> Iterator[Mapping[str, object]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            if isinstance(item, Mapping):
                yield item
            elif isinstance(item, str):
                yield {"word": item}


def _target_word(item: Mapping[str, object]) -> str:
    return _string(item.get("word")) or _string(item.get("name"))


def _mapping_list(value: object) -> Tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_sequence(value: object) -> Tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    result = []
    for item in value:
        text = _string(item)
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _required_string(raw: Mapping[str, object], key: str) -> str:
    value = _string(raw.get(key))
    if not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


@contextmanager
def _open_jsonl(path: Path) -> Iterator[IO[str]]:
    suffix = path.suffix.casefold()
    if suffix == ".bz2":
        with bz2.open(path, "rt", encoding="utf-8", errors="strict") as handle:
            yield handle
    elif suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="strict") as handle:
            yield handle
    else:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            yield handle


def _compression_name(path: Path) -> str:
    if path.suffix.casefold() == ".bz2":
        return "bzip2"
    if path.suffix.casefold() == ".gz":
        return "gzip"
    return "none"


def _load_checkpoint(path: Optional[Path]) -> Optional[KaikkiCheckpoint]:
    if path is None:
        return None
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None
    try:
        raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("checkpoint root must be an object")
        checkpoint = KaikkiCheckpoint(
            schema_version=int(raw["schema_version"]),
            artifact_sha256=str(raw["artifact_sha256"]),
            configuration_sha256=str(raw["configuration_sha256"]),
            snapshot_id=int(raw["snapshot_id"]),
            next_line=int(raw["next_line"]),
            imported_entries=int(raw["imported_entries"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise KaikkiImportError(f"Invalid import checkpoint: {exc}") from exc
    if checkpoint.schema_version != _CHECKPOINT_SCHEMA_VERSION:
        raise KaikkiImportError(
            f"Unsupported checkpoint schema {checkpoint.schema_version}; "
            "restart the import with a new checkpoint"
        )
    return checkpoint


def _save_checkpoint(path: Optional[Path], checkpoint: KaikkiCheckpoint) -> None:
    if path is None:
        return
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(checkpoint.to_dict(), indent=2, sort_keys=True) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, checkpoint_path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise KaikkiImportError(f"Cannot write import checkpoint: {exc}") from exc


def _zero_write_report() -> LexicalWriteReport:
    return LexicalWriteReport(0, 0, 0, 0, 0, 0, 0, 0, 0)


def _add_reports(
    left: LexicalWriteReport, right: LexicalWriteReport
) -> LexicalWriteReport:
    return LexicalWriteReport(
        attempted=left.attempted + right.attempted,
        inserted=left.inserted + right.inserted,
        updated=left.updated + right.updated,
        forms=left.forms + right.forms,
        senses=left.senses + right.senses,
        glosses=left.glosses + right.glosses,
        examples=left.examples + right.examples,
        pronunciations=left.pronunciations + right.pronunciations,
        relations=left.relations + right.relations,
    )


def _importer_version() -> str:
    try:
        version = importlib.metadata.version("word_forge")
    except importlib.metadata.PackageNotFoundError:
        version = "source"
    return f"word_forge/{version}:kaikki-v1"


__all__ = [
    "ArtifactIdentity",
    "KaikkiCheckpoint",
    "KaikkiImportError",
    "KaikkiImportReport",
    "KaikkiImporter",
    "KaikkiLicenseError",
    "inspect_artifact",
    "parse_kaikki_record",
]
