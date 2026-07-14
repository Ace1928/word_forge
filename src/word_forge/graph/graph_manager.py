"""
Central orchestrator for knowledge graph operations in Word Forge.

Manages the lifecycle, structure, analysis, and visualization of a
multidimensional knowledge graph representing lexical, emotional, and
affective relationships between terms. Integrates functionalities from
specialized sub-modules (Builder, Query, Layout, Visualizer, IO, Analysis).

Adheres to Eidosian principles: modularity, precision, recursive refinement,
and structural integrity. Ensures thread-safe operations where applicable.

Architecture:
    ┌────────────────────┐
    │    GraphManager    │ (Orchestrator)
    │ (State & Interface)│
    └─────────┬──────────┘
              │ Delegates To:
    ┌─────────┴───────────────────────────────────────────────┐
    │         │           │          │            │           │
┌───▼────┐ ┌──▼─────┐ ┌───▼────┐ ┌───▼──────┐ ┌───▼───┐ ┌───▼──────┐
│ Builder│ │ Query  │ │ Layout │ │Visualizer│ │  IO   │ │ Analysis │
│(DB→Graph)│(Info/Srch)│(Position)│ (Plotting) │(Files)│ (Insights) │
└────────┘ └────────┘ └────────┘ └──────────┘ └───────┘ └──────────┘
      │         │          │          │           │         │
      └─────────┴──────────┴──────────┴───────────┴─────────┘
                               │
                               ▼
                      ┌────────────────┐
                      │    NetworkX    │ (Core Graph Lib)
                      └────────────────┘
                      ┌────────────────┐
                      │   DBManager    │ (Persistence)
                      └────────────────┘
                      ┌────────────────┐
                      │ Optional Libs  │ (Pyvis, Plotly, etc.)
                      └────────────────┘
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections import Counter, defaultdict
from contextlib import contextmanager
from typing import (
    Any,
    DefaultDict,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import networkx as nx

# Core components
from word_forge.config import config
from word_forge.database.database_manager import DBManager
from word_forge.exceptions import GraphDataError, NodeNotFoundError
from word_forge.graph.graph_analysis import (
    ClusterResult,
    GraphAnalysis,
    MetaEmotionalResult,
    MultiDimResult,
    TransitionResult,
    ValenceDistResult,
)
from word_forge.graph.graph_builder import (
    AssertionMutation,
    GraphBuilder,
    GraphUpdateMetrics,
)
from word_forge.graph.graph_config import (
    PositionDict,  # Ensure PositionDict is imported
)
from word_forge.graph.graph_config import (
    GraphConfig,
    GraphInfoDict,
    LayoutAlgorithm,
    LexicalIdentity,
    RelationshipDimension,
    RelType,
    Term,
    WordId,
)
from word_forge.graph.graph_io import GraphIO
from word_forge.graph.graph_layout import GraphLayout
from word_forge.graph.graph_query import GraphQuery
from word_forge.graph.graph_visualizer import GraphVisualizer
from word_forge.parser.linguistics import (
    canonicalize_language_tag,
    infer_script,
    normalize_term,
)

# Make relationship_properties accessible if needed internally
from word_forge.relationships import RELATIONSHIP_TYPES as relationship_properties
from word_forge.relationships import RelationshipProperties

_NODE_IDENTITY_ATTRIBUTES = frozenset({"id", "language", "normalized_term", "term"})
_EDGE_ASSERTION_ATTRIBUTES = frozenset(
    {
        "assertion_count",
        "assertions_json",
        "dimension",
        "dimensions",
        "relationship",
        "relationship_types",
        "related_language",
        "sources",
    }
)


class GraphManager:
    """
    Orchestrates knowledge graph operations, managing state and sub-modules.

    Provides a unified interface for building, querying, analyzing, visualizing,
    and managing the lifecycle of the multidimensional knowledge graph.

    Attributes:
        db_manager: Instance of DBManager for database interactions.
        config: Graph configuration settings.
        g: The core NetworkX graph object (Graph or DiGraph).
        dimensions: Dimensionality for layout and visualization (2 or 3).
        builder: Instance of GraphBuilder.
        query: Instance of GraphQuery.
        layout: Instance of GraphLayout.
        visualizer: Instance of GraphVisualizer.
        io: Instance of GraphIO.
        analysis: Instance of GraphAnalysis.
        logger: Logger instance for this manager.
    """

    def __init__(
        self,
        db_manager: DBManager,
        graph_config: Optional[GraphConfig] = None,
        graph_type: type = nx.Graph,  # Default to undirected graph
        dimensions: int = 2,
    ) -> None:
        """
        Initialize the GraphManager.

        Args:
            db_manager: The database manager instance.
            graph_config: Optional graph configuration. Defaults to global config.
            graph_type: The type of NetworkX graph to use (nx.Graph or nx.DiGraph).
            dimensions: The dimensionality for layout/visualization (2 or 3).
        """
        self.db_manager = db_manager
        self.config = graph_config or config.graph
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing GraphManager...")

        # --- Core Graph State ---
        self.g: Union[nx.Graph, nx.DiGraph] = graph_type()
        self.dimensions: int = dimensions
        self._identity_to_id: Dict[LexicalIdentity, WordId] = {}
        self._term_to_ids: DefaultDict[str, Set[WordId]] = defaultdict(set)
        # Compatibility index containing only spellings that are unambiguous
        # across the currently loaded languages.
        self._term_to_id: Dict[str, WordId] = {}
        self._positions: PositionDict = {}  # node_id -> position tuple mapping
        self._relationship_counts: Counter[RelType] = Counter()
        self._emotional_contexts: Dict[str, Dict[str, float]] = {}  # Stored contexts

        # --- Lock for thread safety on graph modifications ---
        # RLock allows re-entrant locking within the same thread
        self._graph_lock = threading.RLock()

        # --- Initialize Sub-modules ---
        # Pass self (the manager instance) to each sub-module
        self.builder = GraphBuilder(self)
        self.query = GraphQuery(self)
        self.layout = GraphLayout(self)
        self.visualizer = GraphVisualizer(self)
        self.io = GraphIO(self)
        self.analysis = GraphAnalysis(self)

        self.logger.info("GraphManager initialized successfully.")

    # ==========================================
    # Context Manager for DB Connection
    # ==========================================
    @contextmanager
    def _db_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Provide a managed database connection by using the DBManager's context.
        Yields the actual connection object.
        """
        # Directly use the context manager from db_manager to get the connection
        with self.db_manager.get_connection() as conn:
            try:
                yield conn  # Yield the actual connection obtained from the inner context
            except sqlite3.Error as db_err:
                # Optional: Add specific error handling/logging here if needed
                self.logger.error(
                    f"Database operation failed within context: {db_err}", exc_info=True
                )
                raise  # Re-raise the original error
            # The 'finally' block for cleanup is handled by the 'with' statement
            # managing db_manager.get_connection()

    # ==========================================
    # Public Accessors for Internal State
    # ==========================================
    def get_positions(self) -> PositionDict:
        """
        Return a copy of the computed node positions. Thread-safe.

        Returns:
            PositionDict: A dictionary mapping node IDs to their positions.
        """
        with self._graph_lock:
            # Return a copy to prevent external modification
            return self._positions.copy()

    def _clear_node_indexes(self) -> None:
        """Clear every derived lexical node index together."""

        self._identity_to_id.clear()
        self._term_to_ids.clear()
        self._term_to_id.clear()

    def _index_node(
        self,
        word_id: WordId,
        term: str,
        language: str,
        normalized_term: Optional[str] = None,
    ) -> None:
        """Register a graph node under its normalized term and language."""

        canonical_language = canonicalize_language_tag(language)
        normalized = normalize_term(term)
        if (
            normalized_term is not None
            and normalize_term(normalized_term) != normalized
        ):
            raise GraphDataError(
                f"Node {word_id} normalized term does not match its display term"
            )
        identity = (normalized, canonical_language)
        existing_id = self._identity_to_id.get(identity)
        if existing_id is not None and existing_id != word_id:
            raise GraphDataError(
                f"Duplicate graph identity {term!r} ({canonical_language}) for "
                f"node IDs {existing_id} and {word_id}"
            )
        self._identity_to_id[identity] = word_id
        if word_id in self.g:
            self.g.nodes[word_id]["language"] = canonical_language
            self.g.nodes[word_id]["normalized_term"] = normalized
        matching_ids = self._term_to_ids[normalized]
        matching_ids.add(word_id)
        if len(matching_ids) == 1:
            self._term_to_id[normalized] = word_id
        else:
            self._term_to_id.pop(normalized, None)
        self._refresh_display_labels(normalized)

    def _refresh_display_labels(self, normalized_term: str) -> None:
        """Disambiguate labels only when one spelling spans languages."""

        node_ids = self._term_to_ids.get(normalized_term, set())
        ambiguous = len(node_ids) > 1
        for node_id in node_ids:
            if node_id not in self.g:
                continue
            attributes = self.g.nodes[node_id]
            term = str(attributes.get("term", node_id))
            language = str(attributes.get("language", "und"))
            attributes["label"] = f"{term} [{language}]" if ambiguous else term

    @staticmethod
    def _merge_node_attributes(
        attributes: Optional[Dict[str, Any]],
        *,
        word_id: WordId,
        term: str,
        normalized_term: str,
        language: str,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Merge custom metadata without allowing lexical identity drift."""

        merged = dict(defaults or {})
        if attributes:
            conflicting = {
                "id": word_id,
                "language": language,
                "normalized_term": normalized_term,
                "term": term,
            }
            for name in _NODE_IDENTITY_ATTRIBUTES.intersection(attributes):
                supplied = attributes[name]
                expected = conflicting[name]
                if name == "language":
                    matches = canonicalize_language_tag(str(supplied)) == expected
                elif name == "normalized_term":
                    matches = normalize_term(str(supplied)) == expected
                else:
                    matches = supplied == expected
                if not matches:
                    raise ValueError(
                        f"attributes.{name} cannot change an existing lexical identity"
                    )
            merged.update(attributes)
        merged.update(
            {
                "id": word_id,
                "language": language,
                "normalized_term": normalized_term,
                "term": term,
            }
        )
        return merged

    # ==========================================
    # Graph Building & Modification Methods (via Builder)
    # ==========================================
    def build_graph(self) -> None:
        """
        Build the graph from the database, replacing the existing graph.

        Delegates to GraphBuilder.build_graph. Thread-safe.

        Raises:
            GraphDataError: If fetching data fails.
            GraphError: For other construction issues.
        """
        with self._graph_lock:
            self.builder.build_graph()

    def update_graph(self) -> int:
        """
        Incrementally update the graph with new data from the database.

        Delegates to GraphBuilder.update_graph. Thread-safe.

        Returns:
            int: Number of new nodes added.

        Raises:
            GraphDataError: If fetching data fails.
            GraphError: For other update issues.
        """
        with self._graph_lock:
            return self.builder.update_graph()

    def get_last_update_metrics(self) -> GraphUpdateMetrics:
        """Expose metrics from the most recent graph build or update cycle."""

        with self._graph_lock:
            return self.builder.last_update_metrics

    def ensure_sample_data(self) -> bool:
        """
        Ensure sample data exists in the database if it's empty.

        Delegates to GraphBuilder.ensure_sample_data.

        Returns:
            bool: True if sample data was added, False otherwise.

        Raises:
            GraphError: If adding sample data fails.
        """
        # This modifies the DB, potentially read by builder, lock if needed
        # Although builder methods lock graph access, DB access might need coordination
        # For simplicity here, assuming DB operations are safe or handled by DBManager
        return self.builder.ensure_sample_data()

    def verify_database_tables(self) -> bool:
        """
        Verify that required database tables exist.

        Delegates to GraphBuilder.verify_database_tables.

        Returns:
            bool: True if required tables exist, False otherwise.
        """
        return self.builder.verify_database_tables()

    def add_word_node(
        self,
        term: Term,
        attributes: Optional[Dict[str, Any]] = None,
        *,
        language: str = "en",
    ) -> WordId:
        """
        Add a single word node to the graph if it doesn't exist.

        Handles case-insensitive term checking and updates internal mappings.
        Triggers incremental layout update. Thread-safe.

        Args:
            term: The word or phrase to add.
            attributes: Optional dictionary of node attributes (e.g., {'valence': 0.5}).

        Returns:
            WordId: The ID of the added or existing node.

        Raises:
            ValueError: If the term is empty or invalid.
        """
        if not term or not isinstance(term, str):
            raise ValueError("Term must be a non-empty string.")

        canonical_language = canonicalize_language_tag(language)
        normalized = normalize_term(term)
        identity = (normalized, canonical_language)
        with self._graph_lock:
            existing_id = self._identity_to_id.get(identity)
            if existing_id is not None:
                # Node exists, potentially update attributes
                if attributes:
                    current = self.g.nodes[existing_id]
                    merged_attributes = self._merge_node_attributes(
                        attributes,
                        word_id=existing_id,
                        term=str(current.get("term", term)),
                        normalized_term=normalized,
                        language=canonical_language,
                        defaults=dict(current),
                    )
                    nx.set_node_attributes(self.g, {existing_id: merged_attributes})
                    self.logger.debug(
                        f"Updated attributes for existing node '{term}' (ID: {existing_id})."
                    )
                return existing_id
            else:
                # Node doesn't exist, create new ID and add
                # Simple ID strategy: max_id + 1 (ensure graph isn't empty)
                integer_nodes = [
                    node for node in self.g.nodes() if isinstance(node, int)
                ]
                new_id = max(integer_nodes, default=0) + 1

                default_attributes = {
                    "term": term,
                    "normalized_term": normalized,
                    "language": canonical_language,
                    "script": infer_script(term),
                    "source": "graph-api",
                    "is_stub": False,
                    "id": new_id,
                }
                node_attrs = self._merge_node_attributes(
                    attributes,
                    word_id=new_id,
                    term=term,
                    normalized_term=normalized,
                    language=canonical_language,
                    defaults=default_attributes,
                )

                self.g.add_node(new_id, **node_attrs)
                self._index_node(new_id, term, canonical_language, normalized)
                self.logger.info(f"Added new node '{term}' with ID {new_id}.")

                # Trigger incremental layout update for the new node
                self.layout.update_layout_incrementally([new_id])

                return new_id

    def add_relationship(
        self,
        source_term_or_id: Union[Term, WordId],
        target_term_or_id: Union[Term, WordId],
        relationship: RelType,
        dimension: Optional[RelationshipDimension] = None,
        weight: Optional[float] = None,
        color: Optional[str] = None,
        bidirectional: Optional[bool] = None,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Add a relationship (edge) between two terms/nodes.

        Handles resolving terms to IDs, determining relationship properties
        (dimension, weight, color, etc.), and adding the edge with attributes.
        Thread-safe.

        Args:
            source_term_or_id: The source term (str) or node ID (int).
            target_term_or_id: The target term (str) or node ID (int).
            relationship: The type of relationship (e.g., 'synonym').
            dimension: Optional dimension override ('lexical', 'emotional', etc.).
                       If None, determined automatically from relationship type.
            weight: Optional weight override for the edge.
            color: Optional color override for the edge.
            bidirectional: Optional override for edge directionality.
            **kwargs: Additional attributes to add to the edge.

        Returns:
            bool: True if the relationship was added successfully, False otherwise
                  (e.g., if nodes don't exist and cannot be added).

        Raises:
            NodeNotFoundError: If a term is provided but cannot be found or added.
            ValueError: If relationship type is invalid.
        """
        if not relationship or not isinstance(relationship, str):
            raise ValueError("Relationship type must be a non-empty string.")

        # Placeholder for future incremental layout support
        # new_nodes_added: List[WordId] = []

        with self._graph_lock:
            # --- Resolve Source Node ---
            if isinstance(source_term_or_id, str):
                source_id = self.query.get_node_id(
                    source_term_or_id, language=source_language
                )
                if source_id is None:
                    # Option: Add node implicitly or raise error
                    # self.logger.warning(f"Source term '{source_term_or_id}' not found, adding implicitly.")
                    # source_id = self.add_word_node(source_term_or_id)  # add_word_node handles locking
                    raise NodeNotFoundError(
                        f"Source term '{source_term_or_id}' not found."
                    )
            elif isinstance(source_term_or_id, int):
                source_id = source_term_or_id
                if source_id not in self.g.nodes():
                    raise NodeNotFoundError(f"Source node ID {source_id} not found.")
            else:
                raise TypeError("source_term_or_id must be str or int.")

            # --- Resolve Target Node ---
            if isinstance(target_term_or_id, str):
                target_id = self.query.get_node_id(
                    target_term_or_id, language=target_language
                )
                if target_id is None:
                    # Option: Add node implicitly or raise error
                    # self.logger.warning(f"Target term '{target_term_or_id}' not found, adding implicitly.")
                    # target_id = self.add_word_node(target_term_or_id)  # add_word_node handles locking
                    raise NodeNotFoundError(
                        f"Target term '{target_term_or_id}' not found."
                    )
            elif isinstance(target_term_or_id, int):
                target_id = target_term_or_id
                if target_id not in self.g.nodes():
                    raise NodeNotFoundError(f"Target node ID {target_id} not found.")
            else:
                raise TypeError("target_term_or_id must be str or int.")

            # Prevent self-loops
            if source_id == target_id:
                self.logger.warning(
                    f"Attempted to add self-loop for node {source_id}. Skipped."
                )
                return False

            # --- Determine Edge Properties ---
            edge_dimension = dimension or self._determine_dimension(relationship)
            assertion_source = str(kwargs.pop("source", "graph-api"))
            confidence = kwargs.pop("confidence", 1.0)
            valence = kwargs.pop("valence", None)
            arousal = kwargs.pop("arousal", None)
            reserved_attributes = _EDGE_ASSERTION_ATTRIBUTES.intersection(kwargs)
            if reserved_attributes:
                names = ", ".join(sorted(reserved_attributes))
                raise ValueError(
                    f"Reserved edge assertion attributes cannot be overridden: {names}"
                )

            target_node_language = str(self.g.nodes[target_id].get("language", "und"))
            mutation = self.builder._add_relationship_edge(
                source_id,
                target_id,
                relationship,
                dimension=edge_dimension,
                valence=valence,
                arousal=arousal,
                source=assertion_source,
                confidence=confidence,
                related_language=target_node_language,
            )

            edge_overrides = dict(kwargs)
            if weight is not None:
                edge_overrides["weight"] = weight
            if color is not None:
                edge_overrides["color"] = color
            if bidirectional is not None:
                edge_overrides["bidirectional"] = bidirectional
            if edge_overrides:
                nx.set_edge_attributes(self.g, {(source_id, target_id): edge_overrides})

            log_method = (
                self.logger.info
                if mutation is not AssertionMutation.UNCHANGED
                else self.logger.debug
            )
            log_method(
                "%s relationship assertion %r between %s and %s.",
                mutation.value.capitalize(),
                relationship,
                source_id,
                target_id,
            )

            return True

    # ==========================================
    # Query Methods (via Query)
    # ==========================================
    def get_node_id(
        self, term: Term, language: Optional[str] = None
    ) -> Optional[WordId]:
        """Retrieve node ID for a term. Delegates to GraphQuery."""
        return self.query.get_node_id(term, language=language)

    def get_related_terms(
        self,
        term: Term,
        rel_type: Optional[RelType] = None,
        *,
        language: Optional[str] = None,
    ) -> List[Term]:
        """Find related terms. Delegates to GraphQuery."""
        return self.query.get_related_terms(term, rel_type, language=language)

    def get_node_count(self) -> int:
        """Get node count. Delegates to GraphQuery."""
        return self.query.get_node_count()

    def get_edge_count(self) -> int:
        """Get edge count. Delegates to GraphQuery."""
        return self.query.get_edge_count()

    def get_term_by_id(self, word_id: WordId) -> Optional[Term]:
        """Get term by ID. Delegates to GraphQuery."""
        return self.query.get_term_by_id(word_id)

    def get_graph_info(self) -> GraphInfoDict:
        """Get graph summary info. Delegates to GraphQuery."""
        return self.query.get_graph_info()

    def display_graph_summary(self) -> None:
        """Display graph summary. Delegates to GraphQuery."""
        self.query.display_graph_summary()

    def get_subgraph(
        self, term: Term, depth: int = 1, *, language: Optional[str] = None
    ) -> nx.Graph:
        """Extract a subgraph. Delegates to GraphQuery."""
        return self.query.get_subgraph(term, depth, language=language)

    def get_relationships_by_dimension(
        self,
        dimension: RelationshipDimension = "lexical",
        rel_type: Optional[RelType] = None,
        valence_range: Optional[Tuple[float, float]] = None,
    ) -> List[Tuple[Term, Term, RelType, Dict[str, Any]]]:
        """Get relationships filtered by dimension. Delegates to GraphQuery."""
        return self.query.get_relationships_by_dimension(
            dimension, rel_type, valence_range
        )

    # ==========================================
    # Layout Methods (via Layout)
    # ==========================================
    def compute_layout(self, algorithm: Optional[LayoutAlgorithm] = None) -> None:
        """Compute graph layout. Delegates to GraphLayout."""
        # Layout computation can be read-heavy but writes to _positions
        # Lock ensures position dictionary isn't modified during read by visualizer
        with self._graph_lock:
            self.layout.compute_layout(algorithm)

    # update_layout_incrementally is called internally by add_word_node

    # ==========================================
    # Visualization Methods (via Visualizer)
    # ==========================================
    def visualize(
        self,
        output_path: Optional[str] = None,
        height: Optional[str] = None,
        width: Optional[str] = None,
        use_3d: Optional[bool] = None,
        dimensions_filter: Optional[List[RelationshipDimension]] = None,
        open_in_browser: bool = False,
    ) -> None:
        """Generate graph visualization. Delegates to GraphVisualizer."""
        # Visualization reads graph structure and positions, lock ensures consistency
        with self._graph_lock:
            self.visualizer.visualize(
                output_path, height, width, use_3d, dimensions_filter, open_in_browser
            )

    def visualize_2d(
        self,
        output_path: Optional[str] = None,
        height: Optional[str] = None,
        width: Optional[str] = None,
        dimensions_filter: Optional[List[RelationshipDimension]] = None,
        open_in_browser: bool = False,
    ) -> None:
        """Generate 2D graph visualization. Delegates to GraphVisualizer."""
        with self._graph_lock:
            self.visualizer.visualize_2d(
                output_path, height, width, dimensions_filter, open_in_browser
            )

    def visualize_3d(
        self,
        output_path: Optional[str] = None,
        dimensions_filter: Optional[List[RelationshipDimension]] = None,
        open_in_browser: bool = False,
    ) -> None:
        """Generate 3D graph visualization. Delegates to GraphVisualizer."""
        with self._graph_lock:
            self.visualizer.visualize_3d(
                output_path, dimensions_filter, open_in_browser
            )

    # ==========================================
    # IO Methods (via IO)
    # ==========================================
    def save_to_gexf(self, path: Optional[str] = None) -> None:
        """Save graph to GEXF. Delegates to GraphIO."""
        # Reads graph structure, lock ensures consistency
        with self._graph_lock:
            self.io.save_to_gexf(path)

    def load_from_gexf(self, path: Optional[str] = None) -> None:
        """Load graph from GEXF. Delegates to GraphIO."""
        # Replaces graph structure, requires exclusive access
        with self._graph_lock:
            self.io.load_from_gexf(path)

    def export_subgraph(
        self,
        term: Term,
        depth: int = 1,
        output_path: Optional[str] = None,
        *,
        language: Optional[str] = None,
    ) -> str:
        """Export subgraph to GEXF. Delegates to GraphIO."""
        # Reads graph structure, lock ensures consistency
        with self._graph_lock:
            return str(
                self.io.export_subgraph(term, depth, output_path, language=language)
            )

    # ==========================================
    # Analysis Methods (via Analysis)
    # ==========================================
    def analyze_semantic_clusters(
        self,
        min_community_size: int = 3,
        weight_attribute: Optional[str] = "weight",
        resolution: float = 1.0,
        random_state: Optional[int] = None,
    ) -> ClusterResult:
        """Analyze semantic clusters. Delegates to GraphAnalysis."""
        # Reads graph structure, lock ensures consistency
        with self._graph_lock:
            return self.analysis.analyze_semantic_clusters(
                min_community_size, weight_attribute, resolution, random_state
            )

    def analyze_multidimensional_relationships(self) -> MultiDimResult:
        """Analyze multidimensional relationships. Delegates to GraphAnalysis."""
        with self._graph_lock:
            return self.analysis.analyze_multidimensional_relationships()

    def extract_meta_emotional_patterns(self) -> MetaEmotionalResult:
        """Extract meta-emotional patterns. Delegates to GraphAnalysis."""
        with self._graph_lock:
            return self.analysis.extract_meta_emotional_patterns()

    def analyze_emotional_valence_distribution(
        self, dimension: RelationshipDimension = "emotional"
    ) -> ValenceDistResult:
        """Analyze emotional valence distribution. Delegates to GraphAnalysis."""
        with self._graph_lock:
            return self.analysis.analyze_emotional_valence_distribution(dimension)

    def integrate_emotional_context(
        self, context_name: str, context_weights: Dict[str, float]
    ) -> int:
        """Integrate emotional context. Delegates to GraphAnalysis."""
        # Potentially modifies graph state (_emotional_contexts), lock needed
        with self._graph_lock:
            return int(
                self.analysis.integrate_emotional_context(context_name, context_weights)
            )

    def analyze_emotional_transitions(
        self, path_length: int = 2, min_transition_strength: float = 0.1
    ) -> TransitionResult:
        """Analyze emotional transitions. Delegates to GraphAnalysis."""
        with self._graph_lock:
            return self.analysis.analyze_emotional_transitions(
                path_length, min_transition_strength
            )

    def get_emotional_subgraph(
        self,
        term: Term,
        depth: int = 1,
        context: Optional[Union[str, Dict[str, float]]] = None,
        emotional_types: Optional[List[RelType]] = None,
        min_intensity: float = 0.0,
    ) -> nx.Graph:
        """Get emotional subgraph. Delegates to GraphAnalysis."""
        with self._graph_lock:
            return cast(
                nx.Graph,
                self.analysis.get_emotional_subgraph(
                    term, depth, context, emotional_types, min_intensity
                ),
            )

    # ==========================================
    # Internal Helper Methods
    # ==========================================
    def _get_relationship_properties(self, rel_type: RelType) -> RelationshipProperties:
        """Retrieve properties for a given relationship type."""
        # Access the global relationship_properties dictionary
        # Provide a default empty dict if type is unknown
        # Use .get with a default value for safety
        default_props = relationship_properties.get(
            "default", {"weight": 0.3, "color": "#aaaaaa", "bidirectional": True}
        )
        props = relationship_properties.get(rel_type.lower(), default_props)
        # Ensure type correctness using cast or TypedDict validation if needed
        # Make sure the returned dict conforms to RelationshipProperties structure
        return cast(
            RelationshipProperties,
            {
                "weight": props.get("weight", default_props["weight"]),
                "color": props.get("color", default_props["color"]),
                "bidirectional": props.get(
                    "bidirectional", default_props["bidirectional"]
                ),
            },
        )

    def _determine_dimension(self, rel_type: RelType) -> RelationshipDimension:
        """Determine the primary dimension for a relationship type."""
        # Simple logic: Check known emotional/affective types first, else lexical
        # This could be made more sophisticated based on config or properties
        rel_type_lower = rel_type.lower()
        if rel_type_lower in self.config.emotional_relationship_colors:
            return "emotional"
        if rel_type_lower in self.config.affective_relationship_colors:
            return "affective"
        # Add checks for other dimensions (connotative, contextual) if defined
        # Default to lexical
        return "lexical"

    def __del__(self) -> None:
        """Ensure database connection is closed when manager is destroyed."""
        self.logger.info("GraphManager shutting down. Closing database connection.")
        self.db_manager.close()
