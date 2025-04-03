"""
Multidimensional knowledge graph management system for lexical and semantic relationships.

This module provides a comprehensive framework for building, maintaining, and visualizing
knowledge graphs where nodes represent terms and edges represent semantic relationships.
The graph structure supports both 2D and 3D visualizations with rich semantic encoding
of relationship types through colors, weights, and directional properties.

Architecture:
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │   GraphManager  │────>│  NetworkX Graph │────>│ Visualizations  │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
           │                        │                       │
           │                        │                       │
           ▼                        ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │   Database      │     │  Layout Engines │     │ Export Formats  │
    └─────────────────┘     └─────────────────┘     └─────────────────┘

Key Components:
    - GraphManager: Core class providing graph operations
    - Custom exceptions for precise error handling
    - Type definitions for structural integrity
    - Layout algorithms for 2D and 3D visualization
    - Semantic analysis capabilities
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, TypedDict, Union, cast

import networkx as nx
import numpy as np
from pyvis.network import Network

from word_forge.config import config
from word_forge.database.db_manager import DatabaseError, DBManager
from word_forge.relationships import RELATIONSHIP_TYPES


class GraphError(DatabaseError):
    """
    Base exception for graph operations providing context on failures.

    Inherits from DatabaseError to maintain error hierarchy while
    adding graph-specific context information.
    """

    pass


class NodeNotFoundError(GraphError):
    """
    Raised when a term lookup fails within the graph.

    This occurs when attempting to access a node that doesn't exist,
    typically during relationship or subgraph operations.
    """

    pass


class GraphDataError(GraphError):
    """
    Raised when graph data structure contains inconsistencies.

    This indicates a structural problem with the graph data itself,
    such as missing required node attributes or invalid edge structures.
    """

    pass


class GraphVisualizationError(GraphError):
    """
    Raised when graph visualization generation fails.

    This typically occurs during rendering operations, HTML generation,
    or when visualization libraries encounter errors.
    """

    pass


class GraphDimensionError(GraphError):
    """
    Raised when graph dimensional operations fail.

    This occurs when attempting to set invalid dimensions or
    when dimensional operations (like projection) fail.
    """

    pass


class WordTupleDict(TypedDict):
    """
    Type definition for word node information in the graph.

    Attributes:
        id: Unique identifier for the word
        term: The actual word or phrase text
    """

    id: int
    term: str


class RelationshipTupleDict(TypedDict):
    """
    Type definition for relationship information between words.

    Attributes:
        word_id: ID of the source word
        related_term: Text of the target word
        relationship_type: Type of relationship (e.g., synonym, antonym)
    """

    word_id: int
    related_term: str
    relationship_type: str


class GraphInfoDict(TypedDict):
    """
    Type definition for aggregated graph information.

    Attributes:
        nodes: Total number of nodes in the graph
        edges: Total number of edges in the graph
        dimensions: Dimensionality of the graph (2D or 3D)
        sample_nodes: Representative sample of nodes
        sample_relationships: Representative sample of edges
        relationship_types: List of all relationship types in the graph
    """

    nodes: int
    edges: int
    dimensions: int
    sample_nodes: List[WordTupleDict]
    sample_relationships: List[Dict[str, str]]
    relationship_types: List[str]


# Type aliases for improved readability and type safety
WordId = int
Term = str
RelType = str
WordTuple = Tuple[WordId, Term]
RelationshipTuple = Tuple[WordId, Term, RelType]
GraphData = Tuple[List[WordTuple], List[RelationshipTuple]]
Position = Union[Tuple[float, float], Tuple[float, float, float]]
PositionDict = Dict[int, Position]


# SQL query constants from centralized config
SQL_CHECK_WORDS_TABLE = config.graph.sql_templates["check_words_table"]
SQL_CHECK_RELATIONSHIPS_TABLE = config.graph.sql_templates["check_relationships_table"]
SQL_FETCH_ALL_WORDS = config.graph.sql_templates["fetch_all_words"]
SQL_FETCH_ALL_RELATIONSHIPS = config.graph.sql_templates["fetch_all_relationships"]
SQL_INSERT_SAMPLE_WORD = config.graph.sql_templates["insert_sample_word"]
SQL_INSERT_SAMPLE_RELATIONSHIP = config.graph.sql_templates[
    "insert_sample_relationship"
]


class GraphManager:
    """
    Builds and maintains a multidimensional lexical/semantic graph from database records.

    This class creates a networkx graph representation where:
    - Nodes represent words (with word_id as node identifier)
    - Edges represent relationships between words (synonym, antonym, etc.)
    - Node attributes include the term text and multidimensional positioning
    - Edge attributes include the relationship type, weight, and color
    - The graph can be visualized in 2D or 3D space with rich semantic encoding

    Attributes:
        db_manager: Database manager providing access to word data
        g: The NetworkX graph instance containing nodes and edges
        _term_to_id: Mapping from lowercase terms to their IDs for fast lookup
        _dimensions: Number of dimensions for layout (2 or 3)
        _positions: Dictionary mapping node IDs to spatial coordinates
        _relationship_counts: Counter tracking relationship type frequencies

    Usage Examples:
        ```python
        # Initialize with database manager
        db_manager = DBManager()
        graph_manager = GraphManager(db_manager)

        # Build graph from database
        graph_manager.build_graph()

        # Get information about the graph
        info = graph_manager.get_graph_info()
        print(f"Graph has {info['nodes']} nodes and {info['edges']} edges")

        # Find related terms
        related_terms = graph_manager.get_related_terms("algorithm")

        # Create visualization
        graph_manager.visualize()
        ```
    """

    def __init__(self, db_manager: DBManager) -> None:
        """
        Initialize the graph manager with a database connection.

        Sets up the graph structure, term mappings, and dimensional parameters.
        Ensures the database directory exists to prevent file operation errors.

        Args:
            db_manager: Database manager providing access to word data
        """
        self.db_manager = db_manager
        self.g = nx.Graph()
        self._term_to_id: Dict[str, WordId] = {}
        self._dimensions: int = 2  # Default is 2D for backward compatibility
        self._positions: Dict[WordId, Position] = {}
        self._relationship_counts: Dict[str, int] = {}

        # Ensure the database parent directory exists
        db_path = Path(self.db_manager.db_path)
        if not db_path.exists() and not db_path.parent.exists():
            db_path.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory for database: {db_path.parent}")

    @property
    def dimensions(self) -> int:
        """
        Get the number of dimensions for the graph layout.

        Returns:
            The current number of dimensions (2 or 3)
        """
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dims: int) -> None:
        """
        Set the number of dimensions for the graph layout.

        Updates the dimensionality parameter which affects layout computation
        and visualization. When changed, requires recomputing layout.

        Args:
            dims: Number of dimensions (2 or 3)

        Raises:
            GraphDimensionError: If dimensions are not 2 or 3

        Example:
            ```python
            # Set to 3D for spatial visualization
            graph_manager.dimensions = 3
            graph_manager._compute_layout()  # Recompute layout
            ```
        """
        if dims not in (2, 3):
            raise GraphDimensionError("Graph dimensions must be either 2 or 3")
        self._dimensions = dims

    @contextmanager
    def _db_connection(self):
        """
        Create a database connection using the DBManager's path.

        Context manager that provides a database connection with proper
        setup and guaranteed cleanup regardless of exceptions.
        Ensures the parent directory exists before connecting.

        Yields:
            sqlite3.Connection: An active database connection with row factory

        Raises:
            GraphError: If database connection fails

        Example:
            ```python
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
            # Connection automatically closed here
            ```
        """
        try:
            # Ensure parent directory exists
            db_path = Path(self.db_manager.db_path)
            if not db_path.parent.exists():
                db_path.parent.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created directory for database: {db_path.parent}")

            conn = sqlite3.connect(self.db_manager.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
        except sqlite3.Error as e:
            error_msg = f"Database connection error in graph manager: {e}"
            logging.error(error_msg)
            raise GraphError(error_msg) from e

    def build_graph(self) -> None:
        """
        Clear existing in-memory graph and rebuild from database.

        This method:
        1. Clears any existing graph data
        2. Fetches words and relationships from the database
        3. Adds each word as a node with its term as an attribute
        4. Adds edges between words based on their relationships
        5. Computes layout positions in 2D or 3D space

        Raises:
            GraphError: If a database error occurs during fetching

        Example:
            ```python
            # Complete rebuild from database
            graph_manager.build_graph()
            ```
        """
        self.g.clear()
        self._term_to_id.clear()
        self._positions.clear()
        self._relationship_counts = {}

        words, relationships = self._fetch_data()

        # Add nodes and build term->id mapping
        for word_id, term in words:
            self.g.add_node(word_id, term=term, label=term)
            if term:  # Ensure term is not None or empty
                self._term_to_id[term.lower()] = word_id

        # Add edges between related words with rich attributes
        for word_id, related_term, rel_type in relationships:
            # Skip if either node doesn't exist
            if word_id not in self.g.nodes():
                continue

            # Find ID for related term (skip if not found)
            related_id = self._term_to_id.get(related_term.lower())
            if related_id is None or related_id not in self.g.nodes():
                continue

            # Get relationship properties
            rel_props = self._get_relationship_properties(rel_type)
            weight = rel_props["weight"]
            color = rel_props["color"]
            bidirectional = rel_props["bidirectional"]

            # Determine relationship dimension
            dimension = "lexical"  # Default dimension
            if any(
                rel_type.startswith(prefix)
                for prefix in [
                    "emotional_",
                    "evokes",
                    "responds_to",
                    "valence_",
                    "arousal_",
                ]
            ):
                dimension = "emotional"
            elif any(
                rel_type.startswith(prefix)
                for prefix in ["positive_", "negative_", "high_", "low_"]
            ):
                dimension = "affective"

            # Add edge with rich attributes
            self.g.add_edge(
                word_id,
                related_id,
                relationship=rel_type,
                weight=weight,
                color=color,
                bidirectional=bidirectional,
                dimension=dimension,
                title=f"{rel_type}: {self.g.nodes[word_id]['term']} → {self.g.nodes[related_id]['term']}",
            )

            # Track relationship counts for statistics
            self._relationship_counts[rel_type] = (
                self._relationship_counts.get(rel_type, 0) + 1
            )

        # Generate layout positions
        self._compute_layout()

    def _get_relationship_properties(
        self, rel_type: str
    ) -> Dict[str, Union[float, str, bool]]:
        """
        Get the properties for a given relationship type.

        Args:
            rel_type: The relationship type

        Returns:
            Dictionary with weight, color, and bidirectional properties
        """
        return RELATIONSHIP_TYPES.get(rel_type, RELATIONSHIP_TYPES["default"])

    def _compute_layout(self, algorithm: Optional[str] = None) -> None:
        """
        Compute node positions for the graph using the specified algorithm.

        Calculates spatial positions for all nodes using the selected layout
        algorithm, with different options available for 2D and 3D visualization.
        Results are stored in both the _positions dictionary and as node attributes.

        Args:
            algorithm: Layout algorithm to use (defaults to config.graph.default_layout)
                Options include: "force_directed", "spectral", "circular"

        Raises:
            GraphError: If layout computation fails

        Example:
            ```python
            # Compute with force-directed layout
            graph_manager._compute_layout("force_directed")

            # Compute with default layout algorithm
            graph_manager._compute_layout()
            ```
        """
        if self.g.number_of_nodes() == 0:
            return

        try:
            # Choose layout algorithm based on dimensions
            if not algorithm:
                algorithm = config.graph.default_layout

            pos = self._calculate_layout_positions(algorithm)

            # Store positions
            self._positions = pos

            # Add positions as node attributes
            for node_id, position in pos.items():
                if self._dimensions == 3:
                    self.g.nodes[node_id]["x"] = float(position[0])
                    self.g.nodes[node_id]["y"] = float(position[1])
                    self.g.nodes[node_id]["z"] = float(position[2])
                else:
                    self.g.nodes[node_id]["x"] = float(position[0])
                    self.g.nodes[node_id]["y"] = float(position[1])

        except Exception as e:
            raise GraphError(f"Failed to compute graph layout: {e}") from e

    def _calculate_layout_positions(self, algorithm: str) -> PositionDict:
        """
        Calculate node positions using the specified layout algorithm.

        Separated from _compute_layout for clarity and testing purposes,
        this method handles the actual layout algorithm selection and execution
        based on the graph's dimensions.

        Args:
            algorithm: Layout algorithm name (force_directed, spectral, circular)

        Returns:
            Dictionary mapping node IDs to position tuples

        Raises:
            GraphError: If the layout algorithm fails
        """
        if self._dimensions == 3:
            # 3D layout algorithms
            if algorithm == "force_directed":
                return nx.spring_layout(self.g, dim=3, weight="weight", scale=2.0)
            elif algorithm == "spectral":
                # Spectral layout extended to 3D
                pos_2d = nx.spectral_layout(self.g, weight="weight")
                # Add z-coordinate based on betweenness centrality
                bc = nx.betweenness_centrality(self.g, weight="weight")
                return {n: (*pos_2d[n], bc.get(n, 0)) for n in self.g.nodes()}
            else:
                # Default to 3D spring layout
                return nx.spring_layout(self.g, dim=3, weight="weight", scale=2.0)
        else:
            # 2D layout algorithms
            if algorithm == "force_directed":
                return nx.spring_layout(self.g, weight="weight")
            elif algorithm == "spectral":
                return nx.spectral_layout(self.g, weight="weight")
            elif algorithm == "circular":
                return nx.circular_layout(self.g)
            else:
                return nx.spring_layout(self.g, weight="weight")

    def save_to_gexf(self, path: Optional[str] = None) -> None:
        """
        Save the current graph to a GEXF format file.

        Exports the graph to the GEXF format for use with external tools
        like Gephi or for later reloading. Creates parent directories
        if they don't exist.

        Args:
            path: Destination file path (defaults to config value)

        Raises:
            ValueError: If the graph is empty
            GraphError: If writing to the file fails

        Example:
            ```python
            # Save to default path
            graph_manager.save_to_gexf()

            # Save to custom path
            graph_manager.save_to_gexf("data/my_graph.gexf")
            ```
        """
        if self.g.number_of_nodes() == 0:
            raise ValueError("Cannot save an empty graph")

        gexf_path = path or str(config.graph.get_export_path())
        try:
            # Ensure directory exists
            Path(gexf_path).parent.mkdir(parents=True, exist_ok=True)
            nx.write_gexf(self.g, gexf_path)
        except Exception as e:
            raise GraphError(f"Failed to save graph to {gexf_path}: {e}") from e

    def load_from_gexf(self, path: Optional[str] = None) -> None:
        """
        Load a graph from a GEXF format file.

        Imports a previously saved graph or one created by external tools.
        Reconstructs all internal mappings (_term_to_id, _positions) and
        detects dimensionality from the file.

        Args:
            path: Source file path (defaults to config value)

        Raises:
            FileNotFoundError: If the file doesn't exist
            GraphError: If the file format is invalid

        Example:
            ```python
            # Load from default path
            graph_manager.load_from_gexf()

            # Load from custom path
            graph_manager.load_from_gexf("data/my_graph.gexf")
            ```
        """
        gexf_path = path or str(config.graph.get_export_path())
        if not Path(gexf_path).exists():
            raise FileNotFoundError(f"Graph file not found: {gexf_path}")

        try:
            self.g = nx.read_gexf(gexf_path)

            # Rebuild term_to_id mapping from loaded graph
            self._term_to_id.clear()
            for node_id, attrs in self.g.nodes(data=True):
                if "term" in attrs:
                    self._term_to_id[attrs["term"].lower()] = cast(WordId, node_id)

            # Detect dimensionality of loaded graph
            if "z" in next(iter(self.g.nodes(data=True)))[1]:
                self._dimensions = 3
            else:
                self._dimensions = 2

            # Rebuild positions from node attributes
            self._positions = {}
            for node_id, attrs in self.g.nodes(data=True):
                if self._dimensions == 3 and all(k in attrs for k in ("x", "y", "z")):
                    self._positions[node_id] = (attrs["x"], attrs["y"], attrs["z"])
                elif all(k in attrs for k in ("x", "y")):
                    self._positions[node_id] = (attrs["x"], attrs["y"])

            # Rebuild relationship counts
            self._relationship_counts = {}
            for _, _, data in self.g.edges(data=True):
                rel_type = data.get("relationship_type", "default")
                self._relationship_counts[rel_type] = (
                    self._relationship_counts.get(rel_type, 0) + 1
                )

        except Exception as e:
            raise GraphError(f"Failed to load graph from {gexf_path}: {e}") from e

    def update_graph(self) -> int:
        """
        Update existing graph with new words from the database.

        Instead of clearing the graph, this method:
        1. Identifies words in the database not yet in the graph
        2. Adds them as new nodes
        3. Adds relationships involving the new words
        4. Updates layout positions preserving existing node positions

        Returns:
            Number of new words added to the graph

        Raises:
            GraphError: If database access fails

        Example:
            ```python
            # Add any new words from the database
            new_words = graph_manager.update_graph()
            print(f"Added {new_words} new terms to the graph")
            ```
        """
        if self.g.number_of_nodes() == 0:
            # If graph is empty, just build it from scratch
            self.build_graph()
            return self.g.number_of_nodes()

        # Get current words in the graph
        current_ids: Set[WordId] = set(self.g.nodes())

        # Fetch all words and relationships from database
        all_words, all_relationships = self._fetch_data()

        # Find words not yet in the graph
        new_words = [
            (word_id, term) for word_id, term in all_words if word_id not in current_ids
        ]
        new_word_count = len(new_words)

        # Add new words to the graph
        for word_id, term in new_words:
            self.g.add_node(word_id, term=term)
            self._term_to_id[term.lower()] = word_id

        # Add relationships for new words with rich attributes
        new_edges = 0
        for word_id, related_term, rel_type in all_relationships:
            # Get related id from term
            related_id = self._term_to_id.get(related_term.lower())

            # Add edge if it involves a new word and doesn't exist yet
            if (
                related_id
                and (word_id not in current_ids or related_id not in current_ids)
                and not self.g.has_edge(word_id, related_id)
            ):

                # Get relationship attributes
                rel_attrs = RELATIONSHIP_TYPES.get(
                    rel_type, RELATIONSHIP_TYPES["default"]
                )

                # Add edge with attributes
                self.g.add_edge(
                    word_id,
                    related_id,
                    relationship_type=rel_type,
                    weight=rel_attrs["weight"],
                    color=rel_attrs["color"],
                    bidirectional=rel_attrs["bidirectional"],
                )
                new_edges += 1

                # Update relationship counts
                self._relationship_counts[rel_type] = (
                    self._relationship_counts.get(rel_type, 0) + 1
                )

        # Update layout only if we added new nodes/edges
        if new_word_count > 0 or new_edges > 0:
            self._update_layout_incrementally()

        return new_word_count

    def _update_layout_incrementally(self) -> None:
        """
        Update layout positions incrementally, preserving existing node positions.

        This method:
        1. Keeps positions of existing nodes
        2. Places new nodes based on their connections
        3. Fine-tunes all positions with a few iterations

        Raises:
            GraphError: If layout update fails
        """
        try:
            # Get nodes without positions (new nodes)
            nodes_without_pos = [n for n in self.g.nodes() if n not in self._positions]

            if not nodes_without_pos:
                return

            # Create initial positions for new nodes based on dimensions
            self._initialize_new_node_positions(nodes_without_pos)

            # Fine-tune positions with a few iterations of force-directed layout
            # but only move new nodes significantly
            pos = self._positions.copy()

            # Run a few iterations of force-directed layout
            fixed_nodes = [n for n in self.g.nodes() if n not in nodes_without_pos]
            pos = nx.spring_layout(
                self.g,
                pos=pos,
                fixed=fixed_nodes,
                weight="weight",
                iterations=50,
                dim=self._dimensions,
            )

            # Update positions
            self._positions = pos

            # Update node attributes with new positions
            for node_id, position in pos.items():
                if self._dimensions == 3:
                    self.g.nodes[node_id]["x"] = float(position[0])
                    self.g.nodes[node_id]["y"] = float(position[1])
                    self.g.nodes[node_id]["z"] = float(position[2])
                else:
                    self.g.nodes[node_id]["x"] = float(position[0])
                    self.g.nodes[node_id]["y"] = float(position[1])

        except Exception as e:
            raise GraphError(f"Failed to update graph layout incrementally: {e}") from e

    def _initialize_new_node_positions(self, nodes_without_pos: List[int]) -> None:
        """
        Initialize positions for newly added nodes based on graph topology.

        Positions new nodes near their neighbors when possible, or randomly
        when no connected neighbors have positions. Handles both 2D and 3D cases.

        Args:
            nodes_without_pos: List of node IDs that need positioning
        """
        for node in nodes_without_pos:
            neighbors = list(self.g.neighbors(node))

            if not neighbors:
                # No neighbors, place randomly
                self._assign_random_position(node)
                continue

            # Position near the average of its neighbors
            neighbor_positions = [
                self._positions.get(n, self._get_zero_position())
                for n in neighbors
                if n in self._positions
            ]

            if not neighbor_positions:
                # No positioned neighbors, place randomly
                self._assign_random_position(node)
                continue

            # Calculate average position of neighbors
            avg_pos = self._calculate_average_position(neighbor_positions)

            # Add some randomness to avoid overlap
            if self._dimensions == 3:
                self._positions[node] = (
                    avg_pos[0] + 0.1 * np.random.randn(),
                    avg_pos[1] + 0.1 * np.random.randn(),
                    avg_pos[2] + 0.1 * np.random.randn(),
                )
            else:
                self._positions[node] = (
                    avg_pos[0] + 0.1 * np.random.randn(),
                    avg_pos[1] + 0.1 * np.random.randn(),
                )

    def _assign_random_position(self, node: int) -> None:
        """
        Assign a random position to a node based on current dimensions.

        Args:
            node: ID of the node to position
        """
        if self._dimensions == 3:
            self._positions[node] = (
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1),
            )
        else:
            self._positions[node] = (
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1),
            )

    def _get_zero_position(self) -> Position:
        """
        Get a zero position tuple based on current dimensions.

        Returns:
            A 2D or 3D tuple of zeros
        """
        return (0, 0, 0) if self._dimensions == 3 else (0, 0)

    def _calculate_average_position(self, positions: List[Position]) -> Position:
        """
        Calculate the average position from a list of positions.

        Args:
            positions: List of position tuples

        Returns:
            Average position as a tuple
        """
        if self._dimensions == 3:
            avg_x = sum(p[0] for p in positions) / len(positions)
            avg_y = sum(p[1] for p in positions) / len(positions)
            avg_z = sum(p[2] for p in positions) / len(positions)
            return (avg_x, avg_y, avg_z)
        else:
            avg_x = sum(p[0] for p in positions) / len(positions)
            avg_y = sum(p[1] for p in positions) / len(positions)
            return (avg_x, avg_y)

    def ensure_sample_data(self) -> bool:
        """
        Ensure the database contains sample data if it's empty.

        Checks if the database has any words, and if not, populates it
        with predefined sample data from configuration. Adds both sample
        words and their relationships.

        Returns:
            True if sample data was added, False if database already had data

        Raises:
            GraphError: If adding sample data fails

        Example:
            ```python
            # Ensure there's at least sample data
            if graph_manager.ensure_sample_data():
                print("Added sample data to empty database")
            ```
        """
        # Check if the database has any words
        words, _ = self._fetch_data()
        if words:
            return False

        # Database is empty, add sample data
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()

                # Add sample words if not present
                for word_data in config.graph.sample_words:
                    term = word_data.get("term", "")
                    definition = word_data.get("definition", "")
                    pos = word_data.get("part_of_speech", "")

                    if term:
                        cursor.execute(SQL_INSERT_SAMPLE_WORD, (term, definition, pos))

                # Get the inserted word IDs
                cursor.execute(SQL_FETCH_ALL_WORDS)
                words = cursor.fetchall()

                # Create a mapping of term to ID
                word_id_map = {term.lower(): id for id, term in words}

                # Add sample relationships from config
                for term1, term2, rel_type in config.graph.sample_relationships:
                    if term1.lower() in word_id_map:
                        cursor.execute(
                            SQL_INSERT_SAMPLE_RELATIONSHIP,
                            (word_id_map[term1.lower()], term2, rel_type),
                        )

                conn.commit()
                return True

        except sqlite3.Error as e:
            error_msg = f"Failed to add sample data: {e}"
            logging.error(error_msg)
            raise GraphError(error_msg) from e

    def _fetch_data(self) -> GraphData:
        """
        Fetch words and relationships from the database.

        Retrieves all words and their relationships including lexical, emotional,
        and affective dimensions from the database tables.

        Returns:
            Tuple containing (word_tuples, relationship_tuples)

        Raises:
            GraphError: If database access fails
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                # Check if tables exist
                cursor.execute(SQL_CHECK_WORDS_TABLE)
                words_table_exists = bool(cursor.fetchone())

                cursor.execute(SQL_CHECK_RELATIONSHIPS_TABLE)
                relationships_table_exists = bool(cursor.fetchone())

                if not words_table_exists or not relationships_table_exists:
                    # If tables don't exist, create them and add sample data
                    if self.ensure_sample_data():
                        return (
                            self._fetch_data()
                        )  # Recursive call after creating sample data

                # Fetch words
                cursor.execute(SQL_FETCH_ALL_WORDS)
                words = [(word_id, term) for word_id, term, _ in cursor.fetchall()]

                # Fetch standard lexical relationships
                cursor.execute(SQL_FETCH_ALL_RELATIONSHIPS)
                relationships = [
                    (word_id, related_term, rel_type)
                    for word_id, related_term, rel_type in cursor.fetchall()
                ]

                # Fetch emotional relationships if table exists
                try:
                    cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='emotional_relationships'"
                    )
                    if cursor.fetchone():
                        cursor.execute(
                            config.graph.sql_templates.get(
                                "get_all_emotional_relationships",
                                "SELECT word_id, related_term, relationship_type FROM emotional_relationships",
                            )
                        )
                        emotional_rels = cursor.fetchall()
                        relationships.extend(
                            [
                                (word_id, related_term, f"{rel_type}")
                                for word_id, related_term, rel_type, *_ in emotional_rels
                            ]
                        )
                except Exception as e:
                    # Log but continue if emotional relationships table doesn't exist
                    print(f"Note: Could not fetch emotional relationships: {e}")

                # Fetch affective relationships if table exists
                try:
                    cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='affective_relationships'"
                    )
                    if cursor.fetchone():
                        cursor.execute(
                            "SELECT word_id, related_term, relationship_type FROM affective_relationships"
                        )
                        affective_rels = cursor.fetchall()
                        relationships.extend(
                            [
                                (word_id, related_term, f"{rel_type}")
                                for word_id, related_term, rel_type in affective_rels
                            ]
                        )
                except Exception as e:
                    # Log but continue if affective relationships table doesn't exist
                    print(f"Note: Could not fetch affective relationships: {e}")

                return words, relationships

        except Exception as e:
            raise GraphError(f"Failed to fetch graph data: {e}") from e

    def get_related_terms(self, term: str, rel_type: Optional[str] = None) -> List[str]:
        """
        Find terms related to the given term, optionally filtered by relationship type.

        Traverses the graph to find connected terms, with optional filtering
        by relationship type (e.g., only synonyms).

        Args:
            term: The term to find relationships for
            rel_type: Optional relationship type filter (e.g., 'synonym', 'antonym')

        Returns:
            List of related terms

        Raises:
            NodeNotFoundError: If the term is not found in the graph

        Example:
            ```python
            # Get all related terms
            all_related = graph_manager.get_related_terms("algorithm")

            # Get only synonyms
            synonyms = graph_manager.get_related_terms("algorithm", rel_type="synonym")
            ```
        """
        term_lower = term.lower()
        if term_lower not in self._term_to_id:
            raise NodeNotFoundError(f"Term not found in graph: {term}")

        term_id = self._term_to_id[term_lower]
        related_terms = []

        for neighbor_id in self.g.neighbors(term_id):
            edge_data = self.g.get_edge_data(term_id, neighbor_id)
            neighbor_rel_type = edge_data.get("relationship_type")

            # Filter by relationship type if specified
            if rel_type and neighbor_rel_type != rel_type:
                continue

            neighbor_term = self.g.nodes[neighbor_id].get("term")
            if neighbor_term:
                related_terms.append(neighbor_term)

        return related_terms

    def get_node_count(self) -> int:
        """
        Get the number of word nodes in the graph.

        Returns:
            Integer count of nodes

        Example:
            ```python
            count = graph_manager.get_node_count()
            print(f"Graph contains {count} terms")
            ```
        """
        return self.g.number_of_nodes()

    def get_edge_count(self) -> int:
        """
        Get the number of relationship edges in the graph.

        Returns:
            Integer count of edges

        Example:
            ```python
            count = graph_manager.get_edge_count()
            print(f"Graph contains {count} relationships")
            ```
        """
        return self.g.number_of_edges()

    def get_term_by_id(self, word_id: WordId) -> Optional[str]:
        """
        Retrieve the term associated with a given word ID.

        Args:
            word_id: The ID of the word to look up

        Returns:
            The term string or None if not found

        Example:
            ```python
            term = graph_manager.get_term_by_id(42)
            if term:
                print(f"Word ID 42 is the term '{term}'")
            ```
        """
        if word_id in self.g:
            return self.g.nodes[word_id].get("term")
        return None

    def get_graph_info(self) -> GraphInfoDict:
        """
        Get detailed information about the graph structure.

        Collects statistics and sample data to provide an overview
        of the graph's content and structure.

        Returns:
            Dictionary with graph statistics and sample data

        Raises:
            GraphDataError: If the graph structure is invalid

        Example:
            ```python
            info = graph_manager.get_graph_info()
            print(f"Graph has {info['nodes']} nodes and {info['edges']} edges")
            print(f"Relationship types: {info['relationship_types']}")
            ```
        """
        try:
            # Sample nodes (up to 5)
            sample_nodes = []
            for node_id, attrs in list(self.g.nodes(data=True))[:5]:
                term = attrs.get("term", "Unknown")
                sample_nodes.append({"id": node_id, "term": term})

            # Sample relationships (up to 5)
            sample_relationships = []
            for n1, n2, attrs in list(self.g.edges(data=True))[:5]:
                term1 = self.g.nodes[n1].get("term", "Unknown")
                term2 = self.g.nodes[n2].get("term", "Unknown")
                rel_type = attrs.get("relationship_type", "related")
                sample_relationships.append(
                    {"source": term1, "target": term2, "type": rel_type}
                )

            # Get list of relationship types used in the graph
            relationship_types = list(self._relationship_counts.keys())

            return {
                "nodes": self.g.number_of_nodes(),
                "edges": self.g.number_of_edges(),
                "dimensions": self._dimensions,
                "sample_nodes": sample_nodes,
                "sample_relationships": sample_relationships,
                "relationship_types": relationship_types,
            }
        except Exception as e:
            raise GraphDataError(f"Error generating graph information: {e}") from e

    def display_graph_summary(self) -> None:
        """
        Display a summary of the graph structure.

        Prints key statistics and sample nodes/relationships to the console.
        Provides a human-readable overview of the graph contents.

        Example:
            ```python
            graph_manager.display_graph_summary()
            ```
        """
        if self.g.number_of_nodes() == 0:
            print("Graph is empty")
            return

        info = self.get_graph_info()

        print(
            f"Graph contains {info['nodes']} nodes and {info['edges']} edges in {info['dimensions']}D space"
        )

        # Show relationship type distribution
        if self._relationship_counts:
            print("\nRelationship types:")
            for rel_type, count in sorted(
                self._relationship_counts.items(), key=lambda x: x[1], reverse=True
            ):
                color = RELATIONSHIP_TYPES.get(rel_type, RELATIONSHIP_TYPES["default"])[
                    "color"
                ]
                print(f"  - {rel_type}: {count} instances (color: {color})")

        # Show sample of nodes
        print("\nSample nodes:")
        for node in info["sample_nodes"]:
            print(f"  - Node {node['id']}: {node['term']}")

        # Show sample of relationships
        if info["sample_relationships"]:
            print("\nSample relationships:")
            for rel in info["sample_relationships"]:
                print(f"  - {rel['source']} is {rel['type']} to {rel['target']}")

    def visualize(
        self,
        output_path: Optional[str] = None,
        height: Optional[str] = None,
        width: Optional[str] = None,
        use_3d: Optional[bool] = None,
        dimensions: Optional[List[str]] = None,
    ) -> None:
        """
        Generate an interactive HTML visualization of the graph.

        Creates an interactive visualization using pyvis, with options for
        2D or 3D rendering and dimensional filtering.

        Args:
            output_path: Path to save the HTML file (defaults to config value)
            height: Height of the visualization (default: config.graph.vis_height)
            width: Width of the visualization (default: config.graph.vis_width)
            use_3d: Whether to use 3D visualization (defaults to self._dimensions == 3)
            dimensions: List of dimensions to include (default: all dimensions)
                        Options: ["lexical", "emotional", "affective"]

        Raises:
            GraphVisualizationError: If visualization generation fails
            ValueError: If the graph is empty

        Example:
            ```python
            # Default visualization
            graph_manager.visualize()

            # Only show emotional relationships
            graph_manager.visualize(dimensions=["emotional"])

            # Show lexical and affective but not emotional
            graph_manager.visualize(dimensions=["lexical", "affective"])
            ```
        """
        if self.g.number_of_nodes() == 0:
            raise ValueError("Cannot visualize an empty graph")

        use_3d = use_3d if use_3d is not None else (self._dimensions == 3)

        # Create network
        net = Network(
            height=height or f"{config.graph.vis_height}px",
            width=width or f"{config.graph.vis_width}px",
            directed=True,
            notebook=False,
        )

        # Filter graph by dimensions if specified
        filtered_graph = self.g
        if dimensions:
            # Create a subgraph with only the specified dimensions
            edge_list = [
                (s, t)
                for s, t, d in self.g.edges(data=True)
                if d.get("dimension", "lexical") in dimensions
            ]
            filtered_graph = self.g.edge_subgraph(edge_list).copy()

        # Configure visualization options
        self._configure_visualization_options(net, use_3d)

        # Add nodes with filtered graph
        for node_id in filtered_graph.nodes():
            term = filtered_graph.nodes[node_id].get("term", "")
            label = term

            if use_3d and node_id in self._positions:
                x, y, z = self._positions[node_id]
            elif node_id in self._positions:
                x, y = self._positions[node_id][:2]  # Take first two coordinates
                z = 0
            else:
                # Fallback: random position
                x, y, z = (random.uniform(-50, 50) for _ in range(3))

            # Calculate node size based on connectivity
            size = config.graph.min_node_size + (
                filtered_graph.degree(node_id)
                * (config.graph.max_node_size - config.graph.min_node_size)
                / max(max(dict(filtered_graph.degree()).values(), default=1), 1)
            )

            # Add to visualization
            net.add_node(
                node_id,
                label=label,
                title=f"Term: {term}<br>Connections: {filtered_graph.degree(node_id)}",
                x=x * 100,
                y=y * 100,
                z=z * 100 if use_3d else None,
                size=size,
                color={"border": "#023047", "background": "#219ebc"},
            )

        # Add edges with filtered graph
        for source, target, data in filtered_graph.edges(data=True):
            rel_type = data.get("relationship", "related")
            color = data.get("color", "#aaaaaa")
            weight = data.get("weight", 1.0)
            bidirectional = data.get("bidirectional", True)

            # Add to visualization
            width = config.graph.min_edge_width + (
                weight * (config.graph.max_edge_width - config.graph.min_edge_width)
            )

            net.add_edge(
                source,
                target,
                title=f"{rel_type}",
                width=width,
                color=color,
                arrows={"to": not bidirectional},
            )

        # Save to file
        viz_path = output_path or str(
            config.graph.get_visualization_path() / "graph.html"
        )
        Path(viz_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            net.save_graph(viz_path)
            print(f"Visualization saved to: {viz_path}")
        except Exception as e:
            raise GraphVisualizationError(f"Failed to save visualization: {e}") from e

    def _configure_visualization_options(self, net: "Network", use_3d: bool) -> None:
        """
        Configure visualization options for the network.

        Sets up physics, node styling, edge styling, and interaction options
        based on whether 3D visualization is enabled.

        Args:
            net: The pyvis Network instance to configure
            use_3d: Whether to enable 3D visualization options
        """
        if use_3d:
            net.set_options(
                """
            {
                "physics": {
                    "enabled": true,
                    "stabilization": {
                        "iterations": 100
                    },
                    "barnesHut": {
                        "gravitationalConstant": -2000,
                        "springLength": 120,
                        "springConstant": 0.01
                    }
                },
                "nodes": {
                    "shape": "dot",
                    "scaling": {
                        "min": 10,
                        "max": 30,
                        "label": {
                            "enabled": true
                        }
                    },
                    "font": {
                        "size": 12,
                        "face": "Tahoma"
                    }
                },
                "edges": {
                    "smooth": {
                        "enabled": true,
                        "type": "continuous"
                    },
                    "arrows": {
                        "to": {
                            "enabled": true,
                            "scaleFactor": 0.5
                        }
                    },
                    "font": {
                        "size": 10
                    }
                },
                "interaction": {
                    "navigationButtons": true,
                    "keyboard": true
                }
            }
            """
            )

    def _add_nodes_to_visualization(self, net: "Network", use_3d: bool) -> None:
        """
        Add nodes to the visualization network.

        Configures each node with appropriate attributes including:
        - Label (term text)
        - Size (based on connectivity)
        - Position (from layout calculations)
        - Title (hover information)

        Args:
            net: The pyvis Network instance
            use_3d: Whether to add 3D coordinates
        """
        for node_id, node_attrs in self.g.nodes(data=True):
            label = node_attrs.get("term", f"Node {node_id}")
            title = f"ID: {node_id}<br>Term: {label}"

            # Size node based on degree (connectivity)
            degree = self.g.degree(node_id)
            size = 10 + min(20, degree * 2)

            # Position in 3D or 2D space if available
            pos = self._positions.get(node_id)

            if use_3d and self._dimensions == 3 and pos is not None and len(pos) == 3:
                x, y, z = pos
                net.add_node(
                    node_id,
                    label=label,
                    title=title,
                    size=size,
                    x=float(x) * 500,  # Scale to pyvis coordinates
                    y=float(y) * 500,
                    z=float(z) * 500,
                )
            elif pos is not None and len(pos) >= 2:
                x, y = pos[0], pos[1]
                net.add_node(
                    node_id,
                    label=label,
                    title=title,
                    size=size,
                    x=float(x) * 500,
                    y=float(y) * 500,
                )
            else:
                net.add_node(node_id, label=label, title=title, size=size)

    def _add_edges_to_visualization(self, net: "Network") -> None:
        """
        Add all edges to network visualization with proper styling.

        Configures edge appearance based on relationship type and dimension,
        applying appropriate colors, weights, and styles.

        Args:
            net: The pyvis Network instance to add edges to
        """
        # Track unique relationship types for legend
        relationship_legend = {}

        for source, target, data in self.g.edges(data=True):
            # Get relationship properties
            rel_type = data.get("relationship", "related")
            weight = data.get("weight", 1.0)
            color = data.get("color", "#aaaaaa")
            bidirectional = data.get("bidirectional", True)
            dimension = data.get("dimension", "lexical")
            title = data.get("title", "")

            # Set width based on weight (scaled)
            width = config.graph.min_edge_width + (
                weight * (config.graph.max_edge_width - config.graph.min_edge_width)
            )

            # Set edge style based on dimension
            style = "solid"  # Default
            if dimension == "emotional":
                style = "dashed"
            elif dimension == "affective":
                style = "dotted"

            # Only show arrows for directional relationships
            arrows = not bidirectional

            # Add to visualization
            net.add_edge(
                source,
                target,
                width=width,
                color=color,
                title=title,
                arrows={"to": arrows},
                dashes=(style != "solid"),
                smooth={"type": "continuous"},
            )

            # Track for legend
            if rel_type not in relationship_legend:
                relationship_legend[rel_type] = {
                    "color": color,
                    "style": style,
                    "dimension": dimension,
                }

        # Store legend data on the network for later use
        net.relationship_legend = relationship_legend

    def visualize_3d(self, output_path: Optional[str] = None) -> None:
        """
        Generate a 3D interactive visualization of the graph.

        This is a convenience method that ensures 3D layout and visualization.
        It temporarily sets dimensions to 3 if not already, recomputes the layout
        if needed, and then calls the standard visualization method.

        Args:
            output_path: Path where the HTML visualization will be saved (defaults to config)

        Raises:
            GraphVisualizationError: If visualization creation fails

        Example:
            ```python
            # Create 3D visualization
            graph_manager.visualize_3d()

            # Create 3D visualization at custom path
            graph_manager.visualize_3d("data/my_3d_graph.html")
            ```
        """
        # Set dimensions to 3D if not already
        original_dimensions = self._dimensions
        if self._dimensions != 3:
            self._dimensions = 3
            self._compute_layout()  # Recompute layout in 3D

        try:
            # Call the main visualization method with 3D flag
            self.visualize(output_path=output_path, use_3d=True)
        finally:
            # Restore original dimensions if changed
            if original_dimensions != self._dimensions:
                self._dimensions = original_dimensions

    def get_subgraph(self, term: str, depth: int = 1) -> nx.Graph:
        """
        Extract a subgraph centered on the given term with specified radius.

        Creates a new graph containing the specified term and all related terms
        within the given number of hops. Useful for focused analysis or visualization
        of a term's local neighborhood.

        Args:
            term: The central term for the subgraph
            depth: How many hops to include (1 = immediate neighbors only)

        Returns:
            NetworkX graph object containing the subgraph

        Raises:
            NodeNotFoundError: If the term is not found in the graph

        Example:
            ```python
            # Get immediate neighborhood of "algorithm"
            local_graph = graph_manager.get_subgraph("algorithm", depth=1)

            # Get extended neighborhood (2 hops)
            extended_graph = graph_manager.get_subgraph("algorithm", depth=2)
            ```
        """
        term_lower = term.lower()
        if term_lower not in self._term_to_id:
            raise NodeNotFoundError(f"Term not found in graph: {term}")

        center_id = self._term_to_id[term_lower]

        # Get nodes within specified depth
        nodes_to_include = {center_id}
        current_depth = 0
        frontier = {center_id}

        while current_depth < depth:
            next_frontier = set()
            for node in frontier:
                for neighbor in self.g.neighbors(node):
                    if neighbor not in nodes_to_include:
                        nodes_to_include.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
            current_depth += 1

        # Create subgraph with the selected nodes
        subgraph = self.g.subgraph(nodes_to_include).copy()
        return subgraph

    def export_subgraph(
        self, term: str, depth: int = 1, output_path: Optional[str] = None
    ) -> str:
        """
        Extract and save a subgraph centered on the given term.

        Extracts a local neighborhood of the specified term and saves it to a file
        in GEXF format. Combines the functionality of get_subgraph() and save_to_gexf().

        Args:
            term: The central term for the subgraph
            depth: How many hops to include (1 = immediate neighbors only)
            output_path: Path where to save the subgraph (defaults to a generated path)

        Returns:
            Path where the subgraph was saved

        Raises:
            NodeNotFoundError: If the term is not found in the graph
            GraphError: If the export fails

        Example:
            ```python
            # Export immediate neighborhood of "algorithm"
            subgraph_path = graph_manager.export_subgraph("algorithm", depth=1)
            print(f"Subgraph saved to {subgraph_path}")
            ```
        """
        # Get the subgraph
        subgraph = self.get_subgraph(term, depth)

        # Generate a default path if none provided
        if not output_path:
            safe_term = term.replace(" ", "_").lower()
            # Handle the export_path correctly whether it's a property or method
            export_path = config.graph.get_export_path
            if callable(export_path):
                export_path = export_path()
            output_path = str(export_path / f"subgraph_{safe_term}_d{depth}.gexf")

        try:
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            nx.write_gexf(subgraph, output_path)
            return output_path
        except Exception as e:
            raise GraphError(f"Failed to export subgraph: {e}") from e

    def analyze_semantic_clusters(
        self, min_community_size: int = 3
    ) -> Dict[int, List[str]]:
        """
        Identify semantic clusters (communities) in the graph.

        Uses community detection algorithms to find clusters of related terms.
        Helps identify semantic groupings within the knowledge graph.

        Args:
            min_community_size: Minimum number of terms to be considered a community

        Returns:
            Dictionary mapping community IDs to lists of terms

        Raises:
            GraphError: If community detection fails

        Example:
            ```python
            # Find semantic clusters with at least 3 terms
            clusters = graph_manager.analyze_semantic_clusters(min_community_size=3)

            # Print each cluster
            for cluster_id, terms in clusters.items():
                print(f"Cluster {cluster_id}: {', '.join(terms)}")
            ```
        """
        if self.g.number_of_nodes() < 3:
            return {}

        try:
            # Use networkx's community detection
            import community as community_louvain

            # Generate communities
            partition = community_louvain.best_partition(self.g)

            # Group terms by community
            communities: Dict[int, List[str]] = {}
            for node_id, community_id in partition.items():
                term = self.g.nodes[node_id].get("term", f"Unknown {node_id}")
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(term)

            # Filter small communities
            return {
                k: v for k, v in communities.items() if len(v) >= min_community_size
            }
        except Exception as e:
            raise GraphError(f"Failed to analyze semantic clusters: {e}") from e

    def get_relationships_by_dimension(
        self, dimension: str = "lexical"
    ) -> List[Tuple[str, str, str]]:
        """
        Get all relationships of a specific dimension.

        Retrieves all relationships in the graph belonging to the specified dimension
        (lexical, emotional, affective, etc.)

        Args:
            dimension: Relationship dimension to filter by
                      Options: "lexical", "emotional", "affective"

        Returns:
            List of tuples containing (source_term, target_term, relationship_type)

        Example:
            ```python
            # Get all emotional relationships
            emotional_relationships = graph_manager.get_relationships_by_dimension("emotional")
            for source, target, rel_type in emotional_relationships:
                print(f"{source} {rel_type} {target}")
            ```
        """
        result = []

        for source, target, data in self.g.edges(data=True):
            edge_dimension = data.get("dimension", "lexical")  # Default to lexical
            if edge_dimension == dimension:
                source_term = self.g.nodes[source].get("term", "")
                target_term = self.g.nodes[target].get("term", "")
                rel_type = data.get("relationship", "related")
                result.append((source_term, target_term, rel_type))

        return result

    def get_emotional_subgraph(self, term: str, depth: int = 1) -> nx.Graph:
        """
        Extract a subgraph of emotional relationships for a specific term.

        Creates a subgraph centered on the given term, including only
        emotional dimension relationships up to the specified depth.

        Args:
            term: The central term to build the subgraph around
            depth: The number of relationship steps to include (default: 1)

        Returns:
            A NetworkX graph containing the emotional subgraph

        Raises:
            NodeNotFoundError: If the term is not found in the graph

        Example:
            ```python
            # Get emotional relationships around "happiness"
            emotional_graph = graph_manager.get_emotional_subgraph("happiness", depth=2)
            print(f"Found {emotional_graph.number_of_nodes()} emotionally connected terms")
            ```
        """
        term_lower = term.lower()
        if term_lower not in self._term_to_id:
            raise NodeNotFoundError(f"Term '{term}' not found in the graph")

        node_id = self._term_to_id[term_lower]

        # Initialize with the central node
        nodes_to_include = {node_id}
        current_nodes = {node_id}

        # Breadth-first search up to specified depth
        for _ in range(depth):
            next_nodes = set()
            for node in current_nodes:
                for neighbor, _, data in self.g.edges(node, data=True):
                    # Only include emotional dimension edges
                    if data.get("dimension") == "emotional":
                        next_nodes.add(neighbor)

            nodes_to_include.update(next_nodes)
            current_nodes = next_nodes

        # Create the subgraph
        emotional_subgraph = self.g.subgraph(nodes_to_include).copy()

        # Filter out non-emotional edges
        edges_to_remove = []
        for source, target, data in emotional_subgraph.edges(data=True):
            if data.get("dimension") != "emotional":
                edges_to_remove.append((source, target))

        for edge in edges_to_remove:
            emotional_subgraph.remove_edge(*edge)

        return emotional_subgraph

    def analyze_multidimensional_relationships(self) -> Dict[str, Dict[str, int]]:
        """
        Analyze relationships across different dimensions.

        Provides statistics about relationship types across lexical, emotional,
        and affective dimensions, identifying patterns and correlations.

        Returns:
            Dictionary with dimension statistics and co-occurrence patterns

        Example:
            ```python
            # Analyze multidimensional patterns
            analysis = graph_manager.analyze_multidimensional_relationships()
            for dimension, stats in analysis.items():
                print(f"{dimension}: {stats}")
            ```
        """
        # Initialize results structure
        results = {"dimensions": {}, "co_occurrences": {}, "most_common": {}}

        # Count relationships by dimension
        dimension_counts = {}
        for _, _, data in self.g.edges(data=True):
            dimension = data.get("dimension", "lexical")
            dimension_counts[dimension] = dimension_counts.get(dimension, 0) + 1

        results["dimensions"] = dimension_counts

        # Find nodes with multiple dimension connections
        nodes_with_multi_dimensions = {}
        for node in self.g.nodes():
            dimensions = set()
            for _, _, data in self.g.edges(node, data=True):
                dimensions.add(data.get("dimension", "lexical"))

            if len(dimensions) > 1:
                term = self.g.nodes[node].get("term", "")
                nodes_with_multi_dimensions[term] = list(dimensions)

        results["multi_dimensional_nodes"] = nodes_with_multi_dimensions

        # Most common relationship types by dimension
        relationship_by_dimension = {}
        for _, _, data in self.g.edges(data=True):
            dimension = data.get("dimension", "lexical")
            rel_type = data.get("relationship", "related")

            if dimension not in relationship_by_dimension:
                relationship_by_dimension[dimension] = {}

            relationship_by_dimension[dimension][rel_type] = (
                relationship_by_dimension[dimension].get(rel_type, 0) + 1
            )

        # Find most common for each dimension
        for dimension, counts in relationship_by_dimension.items():
            sorted_rels = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            results["most_common"][dimension] = sorted_rels[:5]  # Top 5

        return results


def main() -> None:
    """
    Demonstrate key functionality of the GraphManager class.

    This function serves as both a demonstration and basic test suite,
    showing how to use the core features of the GraphManager.

    - Initializes the database and graph manager
    - Builds the graph from database or sample data
    - Displays graph statistics and relationships
    - Generates visualizations in both 2D and 3D
    - Exports and analyzes subgraphs

    Raises:
        GraphError: If demonstration operations fail

    Example:
        ```python
        # Run the demonstration
        main()
        ```
    """
    from word_forge.database.db_manager import DBManager

    # Initialize database and graph managers
    db_manager = DBManager()
    graph_manager = GraphManager(db_manager)

    try:
        # Create database tables if they don't exist
        db_manager.create_tables()

        # Check if DB has data, add sample data if empty
        if graph_manager.ensure_sample_data():
            print("Added sample data to empty database")

        # Build the graph from database
        print("Building lexical graph from database...")
        graph_manager.build_graph()

        # Display graph information
        nodes_count = graph_manager.get_node_count()
        edges_count = graph_manager.get_edge_count()
        print(f"Graph built with {nodes_count} words and {edges_count} relationships")

        if nodes_count == 0:
            print("No words found in the database. Please add some data first.")
            return

        # Display detailed graph summary
        graph_manager.display_graph_summary()

        # Get related terms example
        example_term = "algorithm"  # Changed to a sample term likely to exist
        try:
            related_terms = graph_manager.get_related_terms(example_term)
            print(f"\nTerms related to '{example_term}': {related_terms}")

            # Filter by relationship type
            synonyms = graph_manager.get_related_terms(example_term, rel_type="synonym")
            print(f"Synonyms of '{example_term}': {synonyms}")
        except NodeNotFoundError as e:
            print(f"Warning: {e}")

        # Generate and save both 2D and 3D visualizations
        try:
            # Create standard 2D visualization
            vis_path_2d = "data/graph_visualization_2d.html"
            print("\nGenerating 2D interactive visualization...")
            graph_manager.visualize(output_path=vis_path_2d)

            # Create enhanced 3D visualization
            vis_path_3d = "data/graph_visualization_3d.html"
            print("\nGenerating 3D interactive visualization...")
            graph_manager.visualize_3d(output_path=vis_path_3d)

            print("\nVisualizations saved:")
            print(f"  - 2D: {vis_path_2d}")
            print(f"  - 3D: {vis_path_3d}")
            print("Open these files in a web browser to explore the graph")

        except ImportError as e:
            print(f"Note: {e}")
        except Exception as e:
            print(f"Warning: Could not generate visualization: {e}")

        # Save the graph to a file
        output_path = "data/lexical_graph.gexf"
        print(f"\nSaving graph to {output_path}")
        graph_manager.save_to_gexf(output_path)
        print(f"Graph saved successfully to {output_path}")

        # Try to extract a subgraph for a term (e.g., "algorithm")
        try:
            print(f"\nExtracting subgraph for '{example_term}'...")
            subgraph_path = graph_manager.export_subgraph(example_term, depth=2)
            print(f"Subgraph exported to {subgraph_path}")
        except NodeNotFoundError:
            print(
                f"Warning: Could not extract subgraph for '{example_term}' (term not found)"
            )

        # Analyze multidimensional relationships
        print("\nAnalyzing multidimensional relationship patterns...")
        relationship_analysis = graph_manager.analyze_multidimensional_relationships()

        # Display dimension statistics
        print("Relationship dimensions:")
        for dimension, count in relationship_analysis.get("dimensions", {}).items():
            print(f"  - {dimension}: {count} relationships")

        # Display multi-dimensional nodes
        multi_dim_nodes = relationship_analysis.get("multi_dimensional_nodes", {})
        if multi_dim_nodes:
            print("\nTerms with multiple relationship dimensions:")
            for term, dimensions in list(multi_dim_nodes.items())[:5]:  # Show first 5
                print(f"  - {term}: {', '.join(dimensions)}")

        # Display most common relationship types
        most_common = relationship_analysis.get("most_common", {})
        if most_common:
            print("\nMost common relationship types by dimension:")
            for dimension, types in most_common.items():
                if types:
                    print(f"  - {dimension}: {types[0][0]} ({types[0][1]} occurrences)")

    except GraphError as e:
        print(f"Graph error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Ensure connections are properly closed
        db_manager.close()


if __name__ == "__main__":
    main()
