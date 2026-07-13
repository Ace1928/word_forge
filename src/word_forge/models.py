"""Stable public data models returned by the high-level Word Forge API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Relationship:
    """A named connection from one lexical entry to another."""

    target: str
    kind: str
    language: str | None = None
    id: str = ""
    revision: int = 1
    source: str = ""
    confidence: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class Sense:
    """One independently reviewable meaning attached to a lexeme."""

    id: str
    definition: str = ""
    part_of_speech: str = ""
    domain: str = ""
    register: str = ""
    source: str = ""
    confidence: float = 1.0
    review_state: str = "unreviewed"
    revision: int = 1

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class LexicalEntry:
    """A language-aware lexical entry and its known relationships."""

    id: str
    term: str
    language: str
    normalized_term: str = ""
    script: str = "Zyyy"
    definition: str = ""
    part_of_speech: str = ""
    examples: tuple[str, ...] = ()
    relationships: tuple[Relationship, ...] = ()
    senses: tuple[Sense, ...] = ()
    source: str = ""
    base_term: str = ""
    revision: int = 1
    confidence: float = 1.0
    review_state: str = "unreviewed"
    created_at: float = 0.0
    updated_at: float = 0.0
    last_refreshed: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for CLI and API users."""

        payload = asdict(self)
        payload["examples"] = list(self.examples)
        payload["relationships"] = [item.as_dict() for item in self.relationships]
        payload["senses"] = [item.as_dict() for item in self.senses]
        return payload


@dataclass(frozen=True, slots=True)
class LexiconStatistics:
    """A compact snapshot of the records stored in a lexicon."""

    words: int = 0
    lexemes: int = 0
    languages: int = 0
    relationships: int = 0
    translations: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class NetworkNode:
    """One lexical node prepared for interactive graph rendering."""

    id: str
    term: str
    language: str
    definition: str = ""
    part_of_speech: str = ""
    weight: int = 1

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class NetworkEdge:
    """One typed, directed connection in an interactive graph snapshot."""

    id: str
    source: str
    target: str
    kind: str

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class NetworkSnapshot:
    """A bounded graph projection suitable for browser visualization."""

    nodes: tuple[NetworkNode, ...]
    edges: tuple[NetworkEdge, ...]
    truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "nodes": [node.as_dict() for node in self.nodes],
            "edges": [edge.as_dict() for edge in self.edges],
            "truncated": self.truncated,
        }


__all__ = [
    "LexicalEntry",
    "LexiconStatistics",
    "NetworkEdge",
    "NetworkNode",
    "NetworkSnapshot",
    "Relationship",
    "Sense",
]
