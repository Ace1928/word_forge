"""
Handles building and updating the knowledge graph from database sources.

Encapsulates logic for fetching data, adding nodes/edges, ensuring sample data,
and verifying database integrity related to graph construction. Adheres to
Eidosian principles of modularity, precision, and structural integrity.

Architecture:
    ┌──────────────────┐      ┌──────────────────┐
    │  GraphManager    │◄────►│   GraphBuilder   │
    │ (Orchestrator)   │      │ (Data Fetching & │
    └────────┬─────────┘      │  Graph Assembly) │
             │                └──────────────────┘
             ▼
    ┌──────────────────┐
    │   DBManager      │
    │ (Database Conn)  │
    └──────────────────┘
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

# Import necessary components (adjust paths as needed)
from word_forge.exceptions import GraphDataError, GraphError
from word_forge.graph.graph_assertions import (
    create_graph_assertion,
    decode_edge_assertions,
    encode_graph_assertions,
    graph_assertion_identity,
    sort_graph_assertions,
)
from word_forge.graph.graph_config import (
    GraphRelationship,
    GraphWord,
    RelationshipDimension,
    RelType,
    WordId,
)
from word_forge.parser.linguistics import normalize_term
from word_forge.relationships import RelationshipProperties

# Type hint for the main GraphManager to avoid circular imports
if TYPE_CHECKING:
    from .graph_manager import GraphManager


@dataclass(frozen=True)
class GraphUpdateMetrics:
    """Lightweight snapshot of the most recent graph update."""

    new_nodes: int = 0
    updated_nodes: int = 0
    new_edges: int = 0
    new_relationships: int = 0
    updated_relationships: int = 0
    processed_words: int = 0
    max_last_refreshed: float = 0.0
    full_rebuild: bool = False


class AssertionMutation(str, Enum):
    """Result of merging one source assertion into a graph edge."""

    UNCHANGED = "unchanged"
    ADDED = "added"
    UPDATED = "updated"

    @property
    def changed(self) -> bool:
        """Return whether edge metadata changed."""

        return self is not AssertionMutation.UNCHANGED


class GraphBuilder:
    """
    Manages graph construction and updates from the database.

    Responsible for fetching word and relationship data, constructing the
    NetworkX graph structure, ensuring the presence of sample data if needed,
    and verifying the underlying database schema. Delegates layout computations
    to the GraphLayout module via the GraphManager.

    Attributes:
        manager: Reference to the main GraphManager for state access.
        logger: Logger instance for this module.
    """

    def __init__(self, manager: GraphManager) -> None:
        """
        Initialize the GraphBuilder with a reference to the GraphManager.

        Args:
            manager: The orchestrating GraphManager instance.
        """
        self.manager: GraphManager = manager
        self.logger: logging.Logger = logging.getLogger(__name__)
        # Use config from manager for consistency
        self._config = self.manager.config
        self._metadata_key = "graph_last_refresh"
        self._last_update_metrics: GraphUpdateMetrics = GraphUpdateMetrics()
        self._last_refresh_watermark: float = self._load_last_refresh_watermark()

    @property
    def last_update_metrics(self) -> GraphUpdateMetrics:
        """Return metrics captured during the most recent update cycle."""

        return self._last_update_metrics

    def _set_last_update_metrics(
        self,
        *,
        new_nodes: int,
        updated_nodes: int,
        new_edges: int,
        new_relationships: int,
        updated_relationships: int,
        processed_words: int,
        max_last_refreshed: float,
        full_rebuild: bool,
    ) -> None:
        self._last_update_metrics = GraphUpdateMetrics(
            new_nodes=new_nodes,
            updated_nodes=updated_nodes,
            new_edges=new_edges,
            new_relationships=new_relationships,
            updated_relationships=updated_relationships,
            processed_words=processed_words,
            max_last_refreshed=max_last_refreshed,
            full_rebuild=full_rebuild,
        )

    def _ensure_metadata_table(self, conn: sqlite3.Connection) -> None:
        """Ensure the metadata table needed for graph watermarks exists."""

        conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """)

    def _load_last_refresh_watermark(self) -> float:
        """Load the persisted watermark indicating the last processed timestamp."""

        try:
            with self.manager._db_connection() as conn:
                self._ensure_metadata_table(conn)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT value FROM graph_metadata WHERE key = ?",
                    (self._metadata_key,),
                )
                row = cursor.fetchone()
                if row and row[0] is not None:
                    return float(row[0])
        except (sqlite3.Error, TypeError, ValueError) as exc:
            self.logger.debug(
                "Unable to load graph watermark from metadata table: %s", exc
            )
        return 0.0

    def _persist_watermark(self, value: float) -> None:
        """Persist the latest processed timestamp for incremental updates."""

        if value <= self._last_refresh_watermark:
            return

        try:
            with self.manager._db_connection() as conn:
                self._ensure_metadata_table(conn)
                conn.execute(
                    """
                    INSERT INTO graph_metadata (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value=excluded.value,
                        updated_at=excluded.updated_at
                    """,
                    (self._metadata_key, str(value), time.time()),
                )
                conn.commit()
                self._last_refresh_watermark = value
        except sqlite3.Error as exc:
            self.logger.warning(
                "Failed to persist graph watermark: %s",
                exc,
                exc_info=self.logger.isEnabledFor(logging.DEBUG),
            )

    def build_graph(self, *, compute_layout: bool = True) -> None:
        """
        Construct the graph from the database, replacing any existing graph.

        Fetches all words and relationships, adds them as nodes and edges
        to the manager's graph object, builds the term-to-ID mapping,
        calculates relationship counts, and triggers a full layout computation.

        Args:
            compute_layout: Whether to position the complete graph after loading.
                Disable this when a bounded visualization will compute its own
                layout, avoiding unnecessary whole-graph work.

        Raises:
            GraphDataError: If fetching data from the database fails.
            GraphError: For other graph construction issues.
        """
        self.logger.info("Initiating full graph build process.")
        # Clear existing graph state held by the manager
        self.manager.g.clear()
        self.manager._clear_node_indexes()
        self.manager._positions.clear()
        self.manager._relationship_counts.clear()  # Use clear() for consistency

        try:
            words, relationships, latest_refresh = self._fetch_data()
            self.logger.info(
                f"Fetched {len(words)} words and {len(relationships)} relationships."
            )
        except sqlite3.Error as db_err:
            raise GraphDataError(
                "Database error during data fetch.", db_err
            ) from db_err
        except Exception as e:
            raise GraphDataError(f"Unexpected error fetching data: {e}", e) from e

        # --- Node Addition ---
        total_words = len(words)
        for idx, word in enumerate(words, start=1):
            # Ensure term is not None or empty before adding
            if word.term:
                # Add node with term and ID attributes for consistency
                self.manager.g.add_node(
                    word.word_id,
                    term=word.term,
                    normalized_term=word.normalized_term,
                    language=word.language,
                    script=word.script,
                    source=word.source,
                    is_stub=word.is_stub,
                    last_refreshed=word.last_refreshed,
                    id=word.word_id,
                )
                self.manager._index_node(
                    word.word_id,
                    word.term,
                    word.language,
                    word.normalized_term,
                )
            else:
                self.logger.warning(
                    f"Skipping node with ID {word.word_id} due to missing term."
                )
            if idx % max(total_words // 10, 1) == 0:
                self.logger.info("Node build progress: %d/%d", idx, total_words)

        # --- Edge Addition ---
        total_edges = len(relationships)
        relationships_added = 0
        for idx, relationship in enumerate(relationships, start=1):
            # Validate source node exists
            if relationship.word_id not in self.manager.g.nodes():
                self.logger.debug(
                    "Skipping edge from non-existent node ID %s.",
                    relationship.word_id,
                )
                continue

            related_id = self.manager._identity_to_id.get(
                (
                    relationship.related_normalized_term,
                    relationship.related_language,
                )
            )

            # Validate target node exists
            if related_id is None or related_id not in self.manager.g.nodes():
                self.logger.debug(
                    "Skipping edge to non-existent term %r (%s).",
                    relationship.related_term,
                    relationship.related_language,
                )
                continue

            # Prevent self-loops unless explicitly allowed by config (future)
            if relationship.word_id == related_id:
                self.logger.debug(
                    "Skipping self-loop for node ID %s.", relationship.word_id
                )
                continue

            # Add edge with calculated properties
            mutation = self._add_relationship_edge(
                relationship.word_id,
                related_id,
                relationship.relationship_type,
                dimension=relationship.dimension,
                valence=relationship.valence,
                arousal=relationship.arousal,
                source=relationship.source,
                confidence=relationship.confidence,
                related_language=relationship.related_language,
            )
            if mutation is AssertionMutation.ADDED:
                relationships_added += 1
            if idx % max(total_edges // 10, 1) == 0:
                self.logger.info("Edge build progress: %d/%d", idx, total_edges)

        self.logger.info(
            f"Graph built: {self.manager.g.number_of_nodes()} nodes, {self.manager.g.number_of_edges()} edges."
        )

        # Delegate layout computation via the manager
        if self.manager.g.number_of_nodes() > 0 and compute_layout:
            self.logger.info("Triggering full graph layout computation.")
            self.manager.layout.compute_layout()
        elif self.manager.g.number_of_nodes() > 0:
            self.logger.info("Skipping full layout for deferred bounded rendering.")
        else:
            self.logger.info("Graph is empty, skipping layout computation.")

        self._set_last_update_metrics(
            new_nodes=self.manager.g.number_of_nodes(),
            updated_nodes=0,
            new_edges=self.manager.g.number_of_edges(),
            new_relationships=relationships_added,
            updated_relationships=0,
            processed_words=len(words),
            max_last_refreshed=latest_refresh,
            full_rebuild=True,
        )
        self._persist_watermark(latest_refresh)

    def update_graph(self) -> int:
        """
        Incrementally update the existing graph with new data from the database.

        Fetches all data and compares against the current graph state. Adds new
        nodes and edges. Triggers an incremental layout update if changes occurred.
        If the graph is initially empty, performs a full build instead.

        Returns:
            int: The number of new nodes added during the update.

        Raises:
            GraphDataError: If fetching data from the database fails.
            GraphError: For other graph update issues.
        """
        if self.manager.g.number_of_nodes() == 0:
            self.logger.info(
                "Graph is empty, performing initial build instead of update."
            )
            self.build_graph()
            return int(self.manager.g.number_of_nodes())

        self.logger.info("Initiating incremental graph update.")
        since = (
            self._last_refresh_watermark if self._last_refresh_watermark > 0 else None
        )
        if since is None:
            self.logger.debug(
                "No persisted watermark detected; falling back to full dataset fetch."
            )
        try:
            all_words, all_relationships, latest_refresh = self._fetch_data(since=since)
        except sqlite3.Error as db_err:
            raise GraphDataError(
                "Database error during data fetch for update.", db_err
            ) from db_err
        except Exception as e:
            raise GraphDataError(
                f"Unexpected error fetching data for update: {e}", e
            ) from e

        current_node_ids: Set[WordId] = set(self.manager.g.nodes())
        new_nodes_added: List[WordId] = []
        updated_node_count = 0
        new_edges_added_count = 0

        # --- Add New Nodes ---
        for word in all_words:
            if word.term and word.word_id not in current_node_ids:
                self.manager.g.add_node(
                    word.word_id,
                    term=word.term,
                    normalized_term=word.normalized_term,
                    language=word.language,
                    script=word.script,
                    source=word.source,
                    is_stub=word.is_stub,
                    last_refreshed=word.last_refreshed,
                    id=word.word_id,
                )
                self.manager._index_node(
                    word.word_id,
                    word.term,
                    word.language,
                    word.normalized_term,
                )
                new_nodes_added.append(word.word_id)
            elif word.term and word.word_id in current_node_ids:
                current = self.manager.g.nodes[word.word_id]
                current_identity = (
                    str(
                        current.get(
                            "normalized_term", normalize_term(str(current["term"]))
                        )
                    ),
                    str(current.get("language", word.language)),
                )
                incoming_identity = (word.normalized_term, word.language)
                if current_identity != incoming_identity:
                    raise GraphDataError(
                        f"Node ID {word.word_id} changed lexical identity from "
                        f"{current_identity!r} to {incoming_identity!r}"
                    )
                refreshed_attributes = {
                    "term": word.term,
                    "normalized_term": word.normalized_term,
                    "language": word.language,
                    "script": word.script,
                    "source": word.source,
                    "is_stub": word.is_stub,
                    "last_refreshed": word.last_refreshed,
                    "id": word.word_id,
                }
                if any(
                    current.get(name) != value
                    for name, value in refreshed_attributes.items()
                    if name != "last_refreshed"
                ):
                    updated_node_count += 1
                current.update(refreshed_attributes)
                self.manager._index_node(
                    word.word_id,
                    word.term,
                    word.language,
                    word.normalized_term,
                )

        new_node_count = len(new_nodes_added)

        # --- Add New Edges ---
        new_relationships_added_count = 0
        updated_relationships_count = 0
        for relationship in all_relationships:
            related_id = self.manager._identity_to_id.get(
                (
                    relationship.related_normalized_term,
                    relationship.related_language,
                )
            )
            # Ensure both nodes exist in the potentially updated graph
            if (
                relationship.word_id in self.manager.g
                and related_id is not None
                and related_id in self.manager.g
            ):
                if relationship.word_id == related_id:
                    continue
                edge_existed = self.manager.g.has_edge(relationship.word_id, related_id)
                mutation = self._add_relationship_edge(
                    relationship.word_id,
                    related_id,
                    relationship.relationship_type,
                    dimension=relationship.dimension,
                    valence=relationship.valence,
                    arousal=relationship.arousal,
                    source=relationship.source,
                    confidence=relationship.confidence,
                    related_language=relationship.related_language,
                )
                if mutation is AssertionMutation.ADDED:
                    new_relationships_added_count += 1
                    if not edge_existed:
                        new_edges_added_count += 1
                elif mutation is AssertionMutation.UPDATED:
                    updated_relationships_count += 1

        # --- Post-Update Actions ---
        if (
            new_node_count > 0
            or updated_node_count > 0
            or new_edges_added_count > 0
            or new_relationships_added_count > 0
            or updated_relationships_count > 0
        ):
            self.logger.info(
                "Graph updated: +%d nodes, %d nodes refreshed, +%d edges, "
                "+%d assertions, %d assertions refreshed.",
                new_node_count,
                updated_node_count,
                new_edges_added_count,
                new_relationships_added_count,
                updated_relationships_count,
            )
            # Delegate incremental layout update only if nodes were added
            if new_node_count > 0:
                self.logger.info("Triggering incremental layout update.")
                self.manager.layout.update_layout_incrementally(new_nodes_added)
            else:
                self.logger.debug("No new nodes; retaining the current layout.")
                # Optionally trigger full re-layout if edge changes significantly impact structure
                # self.manager.layout.compute_layout()
        else:
            self.logger.info(
                "Graph update: no topology or provenance changes detected."
            )

        self._set_last_update_metrics(
            new_nodes=new_node_count,
            updated_nodes=updated_node_count,
            new_edges=new_edges_added_count,
            new_relationships=new_relationships_added_count,
            updated_relationships=updated_relationships_count,
            processed_words=len(all_words),
            max_last_refreshed=latest_refresh,
            full_rebuild=False,
        )
        self._persist_watermark(latest_refresh)

        return new_node_count

    def _add_relationship_edge(
        self,
        source_id: WordId,
        target_id: WordId,
        rel_type: RelType,
        *,
        dimension: Optional[RelationshipDimension] = None,
        valence: Optional[float] = None,
        arousal: Optional[float] = None,
        source: str = "unknown",
        confidence: float = 1.0,
        related_language: str = "en",
    ) -> AssertionMutation:
        """
        Adds a single relationship edge to the graph with calculated attributes.

        Internal helper method to encapsulate edge creation logic, including
        determining dimension, properties, and updating counts.

        Args:
            source_id: The ID of the source node.
            target_id: The ID of the target node.
            rel_type: The type of the relationship.
        """
        resolved_dimension = dimension or self.manager._determine_dimension(rel_type)

        assertion = create_graph_assertion(
            source_id,
            target_id,
            rel_type,
            dimension=resolved_dimension,
            source=source,
            confidence=confidence,
            related_language=related_language,
            valence=valence,
            arousal=arousal,
        )
        existing_data = self.manager.g.get_edge_data(source_id, target_id) or {}
        assertions = decode_edge_assertions(
            existing_data,
            default_source_id=source_id,
            default_target_id=target_id,
        )
        assertion_identity = graph_assertion_identity(assertion)
        mutation = AssertionMutation.ADDED
        for index, existing_assertion in enumerate(assertions):
            if graph_assertion_identity(existing_assertion) != assertion_identity:
                continue
            if existing_assertion == assertion:
                return AssertionMutation.UNCHANGED
            assertions[index] = assertion
            mutation = AssertionMutation.UPDATED
            break
        else:
            assertions.append(assertion)
        assertions = sort_graph_assertions(assertions)

        relationship_types = list(
            dict.fromkeys(str(item["relationship"]) for item in assertions)
        )
        dimensions = list(dict.fromkeys(str(item["dimension"]) for item in assertions))
        sources = list(dict.fromkeys(str(item["source"]) for item in assertions))
        assertion_properties: List[RelationshipProperties] = [
            self.manager._get_relationship_properties(item["relationship"])
            for item in assertions
        ]
        primary_assertion = assertions[0]
        primary_properties = assertion_properties[0]
        directions = {(item["source_id"], item["target_id"]) for item in assertions}
        has_opposite_directions = any(
            (target, source_node) in directions for source_node, target in directions
        )
        is_bidirectional = has_opposite_directions or any(
            bool(properties.get("bidirectional", False))
            for properties in assertion_properties
        )
        source_term_text = self.manager.g.nodes[primary_assertion["source_id"]].get(
            "term", f"ID:{primary_assertion['source_id']}"
        )
        target_term_text = self.manager.g.nodes[primary_assertion["target_id"]].get(
            "term", f"ID:{primary_assertion['target_id']}"
        )

        # Construct GEXF-compatible scalar edge attributes. Complete assertions
        # remain available as canonical JSON instead of an unserializable list.
        edge_attrs = {
            "relationship": relationship_types[0],
            "relationship_types": "|".join(relationship_types),
            "weight": max(
                float(properties.get("weight", 1.0)) * item["confidence"]
                for item, properties in zip(assertions, assertion_properties)
            ),
            "color": primary_properties.get(
                "color", self._config.relationship_colors.get("default", "#aaaaaa")
            ),
            "bidirectional": is_bidirectional,
            "dimension": dimensions[0],
            "dimensions": "|".join(dimensions),
            "source": sources[0],
            "sources": "|".join(sources),
            "confidence": max(float(item["confidence"]) for item in assertions),
            "assertion_count": len(assertions),
            "assertions_json": encode_graph_assertions(assertions),
            "title": (
                f"{', '.join(relationship_types)}: {source_term_text or '?'} "
                f"{'↔' if is_bidirectional else '→'} "
                f"{target_term_text or '?'}"
            ),
        }

        valences = [
            float(item["valence"]) for item in assertions if item["valence"] is not None
        ]
        arousals = [
            float(item["arousal"]) for item in assertions if item["arousal"] is not None
        ]
        if valences:
            edge_attrs["valence"] = max(valences, key=abs)
        else:
            existing_data.pop("valence", None)
        if arousals:
            edge_attrs["arousal"] = max(arousals)
        else:
            existing_data.pop("arousal", None)

        # Add the edge to the manager's graph
        self.manager.g.add_edge(source_id, target_id, **edge_attrs)

        # Update relationship counts held by the manager
        # Use Counter's update method for clarity
        if mutation is AssertionMutation.ADDED:
            self.manager._relationship_counts.update([rel_type])
        return mutation

    def _fetch_data(
        self, since: Optional[float] = None
    ) -> Tuple[List[GraphWord], List[GraphRelationship], float]:
        """
        Fetch words and relationships from the database.

        Uses the manager's database connection context manager for safe access.
        When ``since`` is provided, only rows newer than the watermark are
        returned.

        Returns:
            Tuple containing a list of word tuples (id, term) and a list of
            relationship tuples (word_id, related_term, relationship_type),
            along with the highest ``last_refreshed`` timestamp observed.

        Raises:
            GraphDataError: If database tables are missing or query fails.
            sqlite3.Error: For underlying database connection or query errors.
        """
        self.logger.debug("Fetching graph data from database.")
        words: List[GraphWord] = []
        relationships: List[GraphRelationship] = []
        latest_refresh = self._last_refresh_watermark

        try:
            # Use manager's context manager for connection safety
            with self.manager._db_connection() as conn:
                cursor = conn.cursor()

                # Verify 'words' table exists
                cursor.execute(self._config.sql_templates["check_words_table"])
                if not cursor.fetchone():
                    raise GraphDataError(
                        "Database table 'words' not found. Cannot build graph."
                    )

                # Fetch words: Ensure id and term are not NULL
                word_query_key = (
                    "fetch_words_since" if since is not None else "fetch_all_words"
                )
                word_query = self._config.sql_templates.get(word_query_key)
                if word_query is None:
                    raise GraphDataError(f"SQL template '{word_query_key}' is missing.")
                params: Tuple[float, ...] = (since,) if since is not None else tuple()
                cursor.execute(word_query, params)
                words_raw = cursor.fetchall()
                words = [
                    GraphWord(
                        word_id=int(row["id"]),
                        term=str(row["term"]),
                        normalized_term=str(row["normalized_term"]),
                        language=str(row["language"]),
                        script=str(row["script"]),
                        source=str(row["source"]),
                        is_stub=bool(row["is_stub"]),
                        last_refreshed=float(row["last_refreshed"]),
                    )
                    for row in words_raw
                    if row["id"] is not None and row["term"] is not None
                ]
                for row in words_raw:
                    try:
                        refreshed = float(row["last_refreshed"] or 0.0)
                        latest_refresh = max(latest_refresh, refreshed)
                    except (TypeError, ValueError):
                        continue
                self.logger.debug(f"Fetched {len(words)} valid word entries.")

                # Verify 'relationships' table exists
                cursor.execute(self._config.sql_templates["check_relationships_table"])
                if not cursor.fetchone():
                    self.logger.warning(
                        "Database table 'relationships' not found. Graph will have no edges."
                    )
                    # Return words only if relationships table is missing
                    return words, [], latest_refresh

                # Fetch relationships: Ensure all parts are not NULL
                rel_query_key = (
                    "fetch_relationships_since"
                    if since is not None
                    else "fetch_all_relationships"
                )
                rel_query = self._config.sql_templates.get(rel_query_key)
                rel_params: Tuple[float, ...] = (
                    (since,) if since is not None else tuple()
                )
                if rel_query is None:
                    rel_query = self._config.sql_templates["fetch_all_relationships"]
                    rel_params = tuple()
                cursor.execute(rel_query, rel_params)
                relationships_raw = cursor.fetchall()
                lexical_relationships: List[GraphRelationship] = [
                    GraphRelationship(
                        word_id=int(row["word_id"]),
                        related_term=str(row["related_term"]),
                        related_normalized_term=str(row["related_normalized_term"]),
                        related_language=str(row["related_language"]),
                        relationship_type=str(row["relationship_type"]),
                        dimension="lexical",
                        valence=None,
                        arousal=None,
                        source=str(row["source"]),
                        confidence=float(row["confidence"]),
                    )
                    for row in relationships_raw
                    if row["word_id"] is not None
                    and row["related_term"] is not None
                    and row["relationship_type"] is not None
                ]
                relationships.extend(lexical_relationships)

                # Fetch emotional relationships when the table exists
                emotional_count = 0
                try:
                    emotional_query_key = (
                        "get_emotional_relationships_since"
                        if since is not None
                        else "get_all_emotional_relationships"
                    )
                    emotional_query = self._config.sql_templates.get(
                        emotional_query_key
                    )
                    if emotional_query is None:
                        emotional_query = self._config.sql_templates[
                            "get_all_emotional_relationships"
                        ]
                        emotional_params: Tuple[float, ...] = tuple()
                    else:
                        emotional_params = (since,) if since is not None else tuple()
                    cursor.execute(emotional_query, emotional_params)
                    emotional_rows = cursor.fetchall()
                    for row in emotional_rows:
                        try:
                            latest_refresh = max(
                                latest_refresh, float(row["last_updated"] or 0.0)
                            )
                        except (TypeError, ValueError):
                            continue
                    emotional_relationships: List[GraphRelationship] = [
                        GraphRelationship(
                            word_id=int(row["word_id"]),
                            related_term=str(row["related_term"]),
                            related_normalized_term=normalize_term(
                                str(row["related_term"])
                            ),
                            related_language=str(row["related_language"]),
                            relationship_type=str(row["relationship_type"]),
                            dimension="emotional",
                            valence=float(row["valence"]),
                            arousal=float(row["arousal"]),
                            source="emotion-derived",
                            confidence=1.0,
                        )
                        for row in emotional_rows
                        if row["word_id"] is not None
                        and row["related_term"] is not None
                        and row["relationship_type"] is not None
                    ]
                    relationships.extend(emotional_relationships)
                    emotional_count = len(emotional_relationships)
                except sqlite3.Error as emotional_err:
                    self.logger.debug(
                        "Emotional relationships unavailable: %s", emotional_err
                    )

                self.logger.debug(
                    "Fetched %d lexical and %d emotional relationship entries.",
                    len(lexical_relationships),
                    emotional_count,
                )

                return words, relationships, latest_refresh
        except sqlite3.Error as db_err:
            # Log specific DB error and re-raise as GraphDataError
            self.logger.error(f"Database query failed during data fetch: {db_err}")
            raise GraphDataError(
                f"Failed to fetch graph data: {db_err}", db_err
            ) from db_err
        except Exception as e:
            # Catch any other unexpected errors during fetch
            self.logger.error(f"Unexpected error during data fetch: {e}")
            raise GraphDataError(
                f"An unexpected error occurred while fetching graph data: {e}", e
            ) from e

    def ensure_sample_data(self) -> bool:
        """
        Ensure the database contains sample data if it's currently empty.

        Checks if the 'words' table has any entries. If not, it attempts to
        insert predefined sample words and relationships from the configuration.

        Returns:
            bool: True if sample data was added, False otherwise.

        Raises:
            GraphError: If adding sample data fails due to database issues.
        """
        self.logger.debug(
            "Checking for existing data before potentially adding samples."
        )
        try:
            # Check word count first as a proxy for existing data
            with self.manager._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM words")
                count_result = cursor.fetchone()
                # Ensure fetchone result is not None before accessing index
                count = count_result[0] if count_result else 0
                if count > 0:
                    self.logger.info(
                        f"Database already contains {count} words. Skipping sample data insertion."
                    )
                    return False
        except sqlite3.Error as e:
            # Log error but proceed to attempt sample data insertion, as table might be missing
            self.logger.warning(
                f"Could not check for existing data (table might be missing): {e}. Attempting sample data insertion."
            )
        except Exception as e:
            self.logger.error(
                f"Unexpected error checking for existing data: {e}", exc_info=True
            )
            # Decide whether to proceed or raise based on policy
            raise GraphError("Failed to verify existing data presence.", e) from e

        self.logger.info(
            "Database appears empty or 'words' table missing. Attempting to add sample data."
        )
        try:
            # DBManager.create_tables handles its own connection management
            self.manager.db_manager.create_tables()  # Call without passing conn

            with self.manager._db_connection() as conn:
                cursor = conn.cursor()
                # Ensure tables exist first (idempotent operation) - Moved outside the 'with' block

                inserted_word_ids: Dict[str, WordId] = {}
                # Use sample data from config - Ensure config has sample_words
                sample_words = getattr(self._config, "sample_words", [])
                sample_relationships = self._config.sample_relationships

                # --- Insert Sample Words ---
                self.logger.debug(f"Inserting {len(sample_words)} sample words.")
                for word_data in sample_words:
                    term = word_data.get("term")
                    if not term:
                        self.logger.warning(
                            f"Skipping sample word due to missing 'term': {word_data}"
                        )
                        continue

                    try:
                        cursor.execute(
                            self._config.sql_templates["insert_sample_word"],
                            (
                                term,
                                normalize_term(term),
                                word_data.get("definition", ""),
                                word_data.get("part_of_speech", ""),
                            ),
                        )
                        inserted_id = cursor.lastrowid
                        # Fetch ID if lastrowid is not reliable (e.g., certain SQLite versions/configs)
                        if (
                            inserted_id is None or inserted_id == 0
                        ):  # Check for 0 as well
                            cursor.execute(
                                "SELECT id FROM words WHERE term = ?", (term,)
                            )
                            row = cursor.fetchone()
                            if row:
                                inserted_id = row[0]
                        if inserted_id is not None:
                            inserted_word_ids[term.lower()] = inserted_id
                            self.logger.debug(
                                f"Inserted sample word '{term}' with ID {inserted_id}."
                            )
                        else:
                            self.logger.error(
                                f"Failed to retrieve ID for inserted sample word '{term}'."
                            )

                    except sqlite3.IntegrityError:
                        # Word likely already exists, fetch its ID
                        self.logger.debug(  # Changed to debug as it's expected if run multiple times
                            f"Sample word '{term}' already exists. Fetching ID."
                        )
                        cursor.execute("SELECT id FROM words WHERE term = ?", (term,))
                        row = cursor.fetchone()
                        if row:
                            inserted_word_ids[term.lower()] = row[0]
                        else:
                            # This case is problematic - log error
                            self.logger.error(
                                f"Sample word '{term}' reported as existing (IntegrityError) but failed to retrieve its ID."
                            )
                    except sqlite3.Error as insert_err:
                        self.logger.error(
                            f"Database error inserting sample word '{term}': {insert_err}"
                        )

                # --- Insert Sample Relationships ---
                self.logger.debug(
                    f"Inserting {len(sample_relationships)} sample relationships."
                )
                for rel_data in sample_relationships:
                    # Ensure rel_data is a tuple/list of expected length
                    if not isinstance(rel_data, (tuple, list)) or len(rel_data) != 3:
                        self.logger.warning(
                            f"Skipping malformed sample relationship data: {rel_data}"
                        )
                        continue

                    term1, term2, rel_type = rel_data
                    if not all([term1, term2, rel_type]):
                        self.logger.warning(
                            f"Skipping sample relationship due to missing data: {rel_data}"
                        )
                        continue

                    id1 = inserted_word_ids.get(term1.lower())
                    # We need the ID of term1, but relate to term2 text
                    # Check if term2 exists to avoid inserting relationships to non-existent sample words
                    id2_check = inserted_word_ids.get(term2.lower())

                    if (
                        id1 is not None and id2_check is not None
                    ):  # Check both terms were successfully added/found
                        try:
                            cursor.execute(
                                self._config.sql_templates[
                                    "insert_sample_relationship"
                                ],
                                (
                                    id1,
                                    term2,  # Insert using term2 text as per schema
                                    normalize_term(term2),
                                    rel_type,
                                ),
                            )
                            self.logger.debug(
                                f"Inserted sample relationship: {term1} -> {term2} ({rel_type})."
                            )
                        except sqlite3.IntegrityError:
                            self.logger.debug(  # Changed to debug
                                f"Sample relationship {term1}-{rel_type}-{term2} already exists."
                            )
                        except sqlite3.Error as rel_err:
                            self.logger.error(
                                f"Database error inserting sample relationship {term1}-{rel_type}-{term2}: {rel_err}"
                            )
                    else:
                        missing = []
                        if id1 is None:
                            missing.append(f"'{term1}' (source)")
                        if id2_check is None:
                            missing.append(f"'{term2}' (target)")
                        self.logger.warning(
                            f"Could not insert sample relationship {term1}-{rel_type}-{term2} due to missing ID(s) for: {', '.join(missing)}"
                        )

                conn.commit()
                self.logger.info("Successfully added sample data to the database.")
                return True
        except sqlite3.Error as e:
            self.logger.error(
                f"Failed to add sample data due to database error: {e}", exc_info=True
            )
            raise GraphError(f"Failed to add sample data: {e}", e) from e
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred while adding sample data: {e}",
                exc_info=True,
            )
            raise GraphError(
                f"An unexpected error occurred while adding sample data: {e}", e
            ) from e

    def verify_database_tables(self) -> bool:
        """
        Verify that required database tables ('words', 'relationships') exist.

        Returns:
            bool: True if both required tables exist, False otherwise.
        """
        self.logger.debug("Verifying essential database tables.")
        try:
            with self.manager._db_connection() as conn:
                cursor = conn.cursor()
                # Check for 'words' table
                cursor.execute(self._config.sql_templates["check_words_table"])
                words_exists = cursor.fetchone() is not None
                # Check for 'relationships' table
                cursor.execute(self._config.sql_templates["check_relationships_table"])
                relationships_exists = cursor.fetchone() is not None

                if words_exists and relationships_exists:
                    self.logger.debug(
                        "Database tables 'words' and 'relationships' verified."
                    )
                    return True
                elif words_exists:
                    self.logger.warning(
                        "Database table 'words' exists, but 'relationships' is missing."
                    )
                    return (
                        False  # Or True depending on whether relationships are optional
                    )
                else:
                    self.logger.error("Essential database table 'words' is missing.")
                    return False
        except sqlite3.Error as e:
            self.logger.error(f"Database error during table verification: {e}")
            return False
        except Exception as e:
            self.logger.error(
                f"Unexpected error during database table verification: {e}"
            )
            return False
