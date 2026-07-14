"""Typed serialization for provenance-preserving graph assertions."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Optional, cast

from word_forge.graph.graph_config import (
    GraphAssertion,
    RelationshipDimension,
    RelType,
    WordId,
)
from word_forge.parser.linguistics import canonicalize_language_tag

GraphAssertionIdentity = tuple[WordId, WordId, RelType, RelationshipDimension, str]

_RELATIONSHIP_DIMENSIONS = frozenset(
    {"lexical", "emotional", "affective", "connotative", "contextual"}
)


def create_graph_assertion(
    source_id: WordId,
    target_id: WordId,
    relationship: RelType,
    *,
    dimension: RelationshipDimension,
    source: str,
    confidence: float,
    related_language: str,
    valence: Optional[float] = None,
    arousal: Optional[float] = None,
) -> GraphAssertion:
    """Create one validated, JSON-safe directed relationship assertion.

    Args:
        source_id: Graph node that makes the assertion.
        target_id: Graph node targeted by the assertion.
        relationship: Lexical or semantic relationship type.
        dimension: Relationship dimension used for filtering and styling.
        source: Stable lexical source identifier.
        confidence: Source confidence in the inclusive range ``[0, 1]``.
        related_language: Canonical BCP 47 language tag for the target.
        valence: Optional finite emotional valence.
        arousal: Optional finite emotional arousal.

    Returns:
        A fully validated assertion suitable for canonical JSON encoding.

    Raises:
        ValueError: If text fields are empty or numeric values are invalid.
        TypeError: If node identifiers are not integers.
    """

    if isinstance(source_id, bool) or not isinstance(source_id, int):
        raise TypeError("source_id must be an integer")
    if isinstance(target_id, bool) or not isinstance(target_id, int):
        raise TypeError("target_id must be an integer")
    if not isinstance(relationship, str):
        raise TypeError("relationship must be a string")
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    if not isinstance(related_language, str):
        raise TypeError("related_language must be a string")

    normalized_relationship = relationship.strip()
    if not normalized_relationship:
        raise ValueError("relationship must be non-empty")
    normalized_source = source.strip()
    if not normalized_source:
        raise ValueError("source must be non-empty")
    normalized_dimension = str(dimension).strip()
    if normalized_dimension not in _RELATIONSHIP_DIMENSIONS:
        raise ValueError(f"unsupported relationship dimension: {dimension!r}")

    validated_confidence = _finite_float(confidence, "confidence")
    if not 0.0 <= validated_confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    return {
        "source_id": source_id,
        "target_id": target_id,
        "relationship": normalized_relationship,
        "dimension": cast(RelationshipDimension, normalized_dimension),
        "source": normalized_source,
        "confidence": validated_confidence,
        "related_language": canonicalize_language_tag(related_language),
        "valence": _optional_finite_float(valence, "valence"),
        "arousal": _optional_finite_float(arousal, "arousal"),
    }


def decode_edge_assertions(
    edge_data: Mapping[str, object],
    *,
    default_source_id: WordId,
    default_target_id: WordId,
) -> list[GraphAssertion]:
    """Decode assertion JSON, falling back to legacy scalar edge metadata.

    Invalid individual records are ignored. If no valid serialized record is
    available, a legacy edge is represented as one directed assertion so old
    GEXF files and callers remain readable.

    Args:
        edge_data: NetworkX edge attributes.
        default_source_id: Source used by legacy records without direction.
        default_target_id: Target used by legacy records without direction.

    Returns:
        Valid directed assertions in stored order.
    """

    serialized = edge_data.get("assertions_json")
    if isinstance(serialized, str) and serialized:
        try:
            decoded: object = json.loads(serialized)
        except (json.JSONDecodeError, TypeError):
            decoded = None
        if isinstance(decoded, list):
            assertions = [
                assertion
                for item in decoded
                if isinstance(item, Mapping)
                and (
                    assertion := _decode_assertion(
                        item,
                        default_source_id=default_source_id,
                        default_target_id=default_target_id,
                    )
                )
                is not None
            ]
            if assertions:
                return assertions

    legacy_assertion = _decode_assertion(
        edge_data,
        default_source_id=default_source_id,
        default_target_id=default_target_id,
    )
    return [legacy_assertion] if legacy_assertion is not None else []


def encode_graph_assertions(assertions: Sequence[GraphAssertion]) -> str:
    """Encode assertions as deterministic, standards-compliant JSON.

    Args:
        assertions: Validated directed assertions.

    Returns:
        Canonical compact JSON with no non-standard NaN values.
    """

    return json.dumps(
        sort_graph_assertions(assertions),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sort_graph_assertions(
    assertions: Sequence[GraphAssertion],
) -> list[GraphAssertion]:
    """Return assertions in a deterministic provenance-first order."""

    return sorted(
        assertions,
        key=lambda item: (
            item["source_id"],
            item["target_id"],
            item["relationship"],
            item["dimension"],
            item["source"],
            item["confidence"],
            item["related_language"],
            item["valence"] is None,
            item["valence"] or 0.0,
            item["arousal"] is None,
            item["arousal"] or 0.0,
        ),
    )


def graph_assertion_identity(assertion: GraphAssertion) -> GraphAssertionIdentity:
    """Return the stable identity used to replace a refreshed assertion."""

    return (
        assertion["source_id"],
        assertion["target_id"],
        assertion["relationship"],
        assertion["dimension"],
        assertion["source"],
    )


def _decode_assertion(
    values: Mapping[str, object],
    *,
    default_source_id: WordId,
    default_target_id: WordId,
) -> Optional[GraphAssertion]:
    """Coerce one serialized or legacy assertion without raising."""

    relationship_value = values.get("relationship")
    if not isinstance(relationship_value, str) or not relationship_value.strip():
        return None

    source_id = values.get("source_id", default_source_id)
    target_id = values.get("target_id", default_target_id)
    if (
        isinstance(source_id, bool)
        or not isinstance(source_id, int)
        or isinstance(target_id, bool)
        or not isinstance(target_id, int)
    ):
        return None
    if (source_id, target_id) not in {
        (default_source_id, default_target_id),
        (default_target_id, default_source_id),
    }:
        return None

    dimension_value = values.get("dimension", "lexical")
    source_value = values.get("source", "unknown")
    language_value = values.get("related_language", "und")
    if (
        not isinstance(dimension_value, str)
        or dimension_value not in _RELATIONSHIP_DIMENSIONS
        or not isinstance(source_value, str)
    ):
        return None
    if not isinstance(language_value, str):
        return None

    try:
        confidence = _finite_float(values.get("confidence", 1.0), "confidence")
        if not 0.0 <= confidence <= 1.0:
            return None
        valence = _optional_finite_float(values.get("valence"), "valence")
        arousal = _optional_finite_float(values.get("arousal"), "arousal")
        language = canonicalize_language_tag(language_value)
    except (TypeError, ValueError):
        return None

    return {
        "source_id": source_id,
        "target_id": target_id,
        "relationship": relationship_value.strip(),
        "dimension": cast(RelationshipDimension, dimension_value),
        "source": source_value.strip() or "unknown",
        "confidence": confidence,
        "related_language": language,
        "valence": valence,
        "arousal": arousal,
    }


def _finite_float(value: object, field_name: str) -> float:
    """Return a finite float or raise a precise validation error."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _optional_finite_float(value: object, field_name: str) -> Optional[float]:
    """Validate an optional finite float."""

    return None if value is None else _finite_float(value, field_name)


__all__ = [
    "create_graph_assertion",
    "decode_edge_assertions",
    "encode_graph_assertions",
    "graph_assertion_identity",
    "sort_graph_assertions",
]
