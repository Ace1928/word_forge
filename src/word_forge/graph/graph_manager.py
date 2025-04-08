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
import random
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

import networkx as nx
import numpy as np

# Import pyvis with a type ignore comment to fix the missing stub issue
from pyvis.network import Network  # type: ignore

from word_forge.database.db_manager import DBManager
from word_forge.exceptions import (
    GraphDimensionError,
    GraphError,
    GraphVisualizationError,
    NodeNotFoundError,
)
from word_forge.graph.graph_config import (
    GraphConfig,
    GraphInfoDict,
    LayoutAlgorithm,
    Position,
    PositionDict,
    WordId,
)
from word_forge.relationships import RELATIONSHIP_TYPES, RelationshipProperties

config = GraphConfig()  # Create instance first
get_export_path = config.get_export_filepath("default_gexf")


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
        self.g: nx.Graph = nx.Graph()  # Add proper type annotation
        self._term_to_id: Dict[str, int] = {}
        self._dimensions: int = 2  # Default is 2D for backward compatibility
        self._positions: Dict[int, Tuple[float, ...]] = {}
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
            raise GraphError(error_msg, e) from e

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
        self.g.clear()  # type: ignore
        self._term_to_id.clear()
        self._positions.clear()
        self._relationship_counts = {}

        words, relationships = self._fetch_data()

        # Add nodes and build term->id mapping
        for word_id, term in words:
            self.g.add_node(node_for_adding=word_id, term=term)
            if term:  # Ensure term is not None or empty
                self._term_to_id[term.lower()] = word_id
                self.g.nodes[word_id]["id"] = word_id
                self.g.nodes[word_id]["term"] = term

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

    def _get_relationship_properties(self, rel_type: str) -> RelationshipProperties:
        """
        Get the properties for a given relationship type.

        Args:
            rel_type: The relationship type

        Returns:
            Dictionary with weight, color, and bidirectional properties
        """
        result = RELATIONSHIP_TYPES.get(rel_type, RELATIONSHIP_TYPES["default"])
        return result

    def _compute_layout(self, algorithm: Optional[LayoutAlgorithm] = None) -> None:
        """
        Compute node positions for the graph using the specified algorithm.

        Calculates spatial positions for all nodes using the selected layout
        algorithm, with different options available for 2D and 3D visualization.
        Results are stored in both the _positions dictionary and as node attributes.

        Args:
            algorithm: Layout algorithm to use (defaults to config.default_layout)
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
            algorithm_to_use: LayoutAlgorithm
            if not algorithm:
                # Cast to ensure type safety
                algorithm_to_use = "force_directed"
            else:
                algorithm_to_use = algorithm

            pos: PositionDict = self._calculate_layout_positions(algorithm_to_use)

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
            error_msg = f"Failed to compute layout: {e}"
            raise GraphError(error_msg, e) from e

    def _calculate_layout_positions(self, algorithm: LayoutAlgorithm) -> PositionDict:
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

        gexf_path = path or str(get_export_path)
        try:
            # Ensure directory exists
            Path(gexf_path).parent.mkdir(parents=True, exist_ok=True)
            nx.write_gexf(self.g, gexf_path)
        except Exception as e:
            error_msg = f"Failed to save graph to {gexf_path}: {e}"
            raise GraphError(error_msg, e) from e

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
        gexf_path = path or str(get_export_path)
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
            error_msg = f"Failed to load graph from {gexf_path}: {e}"
            raise GraphError(error_msg, e) from e

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
            error_msg = f"Failed to update graph layout incrementally: {e}"
            raise GraphError(error_msg, e) from e

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
                for word_data in GraphConfig.sample_relationships:
                    term = word_data.get("term", "")
                    definition = word_data.get("definition", "")
                    pos = word_data.get("part_of_speech", "")

                    if term:
                        cursor.execute(SQL_INSERT_SAMPLE_WORD, (term, definition, pos))

                # Get the inserted word IDs
                cursor.execute(SQL_FETCH_ALL_WORDS)
                words = cursor.fetchall()

                # Create a mapping of term to ID
                word_id_map = {term.lower(): id for id, term, _ in words}

                # Add sample relationships from config
                for term1, term2, rel_type in GraphConfig.sample_relationships:
                    if term1.lower() in word_id_map:
                        cursor.execute(
                            GraphConfig.sql_templates.get(
                                "SQL_INSERT_SAMPLE_RELATIONSHIP"
                            ),
                            (
                                word_id_map[term1.lower()],
                                word_id_map[term2.lower()],
                                rel_type,
                            ),
                        )

                conn.commit()
                return True

        except sqlite3.Error as e:
            error_msg = f"Failed to add sample data: {e}"
            raise GraphError(error_msg, e) from e

    def _fetch_data(self) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str, str]]]:
        """
        Fetch words and relationships from the database.

        Returns:
            Tuple containing (word_tuples, relationship_tuples)

        Raises:
            GraphError: If database access fails
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()

                # Check if tables exist before querying
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='words'"
                )
                if not cursor.fetchone():
                    # Tables don't exist, return empty data
                    return [], []

                # Get all words - use id instead of word_id to match actual schema
                cursor.execute("SELECT id, term FROM words")
                words = cursor.fetchall()

                # Check if relationships table exists
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='relationships'"
                )
                if not cursor.fetchone():
                    # Relationships table doesn't exist, return words without relationships
                    return words, []

                # Get relationships - ensure column names match the actual schema
                cursor.execute(
                    """
                    SELECT word_id, related_term, relationship_type
                    FROM relationships
                """
                )
                relationships = cursor.fetchall()

                return words, relationships
        except Exception as e:
            # Ensure proper error propagation with cause parameter
            raise GraphError(f"Failed to fetch graph data: {e}", e)

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
            error_msg = f"Error generating graph information: {e}"
            raise GraphDataError(error_msg, e) from e

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
            height: Height of the visualization (default: config.vis_height)
            width: Width of the visualization (default: config.vis_width)
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
            height=height or f"{config.vis_height}px",
            width=width or f"{config.vis_width}px",
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
            size = config.min_node_size + (
                filtered_graph.degree(node_id)
                * (config.max_node_size - config.min_node_size)
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
            width = config.min_edge_width + (
                weight * (config.max_edge_width - config.min_edge_width)
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
        viz_path = output_path or str(config.get_visualization_path() / "graph.html")
        Path(viz_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            net.save_graph(viz_path)
            print(f"Visualization saved to: {viz_path}")
        except Exception as e:
            error_msg = f"Failed to save visualization: {e}"
            raise GraphVisualizationError(error_msg, e) from e

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
            width = config.min_edge_width + (
                weight * (config.max_edge_width - config.min_edge_width)
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
        Export a subgraph centered on a specific term to a GEXF file.

        Args:
            term: The central term for the subgraph
            depth: How many relationship hops to include
            output_path: Optional custom output path

        Returns:
            Path to the exported file

        Raises:
            NodeNotFoundError: If the term is not found in the graph
        """
        subgraph = self.get_subgraph(term, depth)

        # Fix: Use global get_export_path or config's get_export_filepath directly
        export_path = output_path or str(get_export_path)

        # Create parent directory if it doesn't exist
        Path(export_path).parent.mkdir(parents=True, exist_ok=True)

        # Save the subgraph
        nx.write_gexf(subgraph, export_path)

        return export_path

    def analyze_semantic_clusters(
        self,
        min_community_size: int = 3,
        weight_emotional: float = 1.0,
        emotion_only: bool = False,
    ) -> Dict[int, List[Dict[str, Union[str, float, List[str], bool]]]]:
        """
        Identify semantic and emotional clusters (communities) in the graph.

        Uses community detection algorithms to find clusters of related terms based on their
        semantic and emotional relationships. Enhances clustering by incorporating emotional
        dimensions (valence, arousal) and meta-emotional relationships into the analysis.

        Args:
            min_community_size: Minimum number of terms to be considered a community
            weight_emotional: Weight multiplier for emotional relationships (1.0 = equal weight)
            emotion_only: If True, only considers emotional dimension relationships

        Returns:
            Dictionary mapping community IDs to lists of term data including:
            - term: The term text
            - valence: Emotional valence if available (-1.0 to 1.0)
            - arousal: Emotional arousal if available (0.0 to 1.0)
            - central: Whether this is a central term in the cluster
            - related_dimensions: List of relationship dimensions for this term
            - meta_emotions: Any meta-emotional relationships detected

        Raises:
            GraphError: If community detection fails
            ImportError: If required community detection library is not installed

        Example:
            ```python
            # Find semantic clusters with at least 3 terms
            clusters = graph_manager.analyze_semantic_clusters(min_community_size=3)

            # Print each cluster with emotional data
            for cluster_id, terms in clusters.items():
                print(f"Cluster {cluster_id}:")
                for term_data in terms:
                    print(f"  - {term_data['term']} "
                          f"(valence: {term_data.get('valence', 'N/A')}, "
                          f"arousal: {term_data.get('arousal', 'N/A')})")

            # Find emotion-weighted clusters
            emotional_clusters = graph_manager.analyze_semantic_clusters(
                min_community_size=3,
                weight_emotional=2.0  # Double weight for emotional connections
            )

            # Find pure emotional clusters
            pure_emotional = graph_manager.analyze_semantic_clusters(
                min_community_size=2,
                emotion_only=True
            )
            ```
        """
        # Early return if graph is too small
        if self.g is None or self.g.number_of_nodes() < 3:
            return {}

        try:
            # Type definition for the partition function
            PartitionFunction = Callable[[nx.Graph], Dict[Any, int]]
            best_partition_func: PartitionFunction

            # Use networkx's community detection with robust import handling
            try:
                # Try older versions first (with type ignore for missing stubs)
                import community as community_louvain  # type: ignore

                best_partition_func = community_louvain.best_partition  # type: ignore
            except (ImportError, AttributeError):
                try:
                    # Try newer versions (with type ignore for missing stubs)
                    from community import community_louvain  # type: ignore

                    best_partition_func = community_louvain.best_partition  # type: ignore
                except (ImportError, AttributeError):
                    # Fall back to NetworkX's implementation if available
                    try:
                        # NetworkX should already be imported at module level
                        if hasattr(nx.algorithms.community, "louvain_communities"):
                            # Create a type-safe wrapper for NetworkX's function
                            def best_partition_func(graph: nx.Graph) -> Dict[Any, int]:
                                communities = list(
                                    nx.algorithms.community.louvain_communities(graph)
                                )
                                return {
                                    node: i
                                    for i, community in enumerate(communities)
                                    for node in community
                                }

                        else:
                            raise AttributeError(
                                "NetworkX lacks louvain_communities function"
                            )
                    except (ImportError, AttributeError):
                        raise ImportError(
                            "Community detection requires either python-louvain package or NetworkX 2.7+. "
                            "Install with: pip install python-louvain or upgrade networkx."
                        )

            # Create a weighted copy of the graph for analysis
            weighted_graph = cast(nx.Graph, self.g.copy())

            # Apply emotional weighting or filtering
            if emotion_only or weight_emotional != 1.0:
                for u, v, data in list(weighted_graph.edges(data=True)):
                    dimension = data.get("dimension", "lexical")
                    if emotion_only and dimension != "emotional":
                        weighted_graph.remove_edge(u, v)
                    elif dimension == "emotional" and weight_emotional != 1.0:
                        # Adjust weight for emotional relationships
                        weighted_graph[u][v]["weight"] = (
                            float(data.get("weight", 1.0)) * weight_emotional
                        )

            # Skip if we filtered out all edges in emotion_only mode
            if weighted_graph.number_of_edges() == 0:
                return {}

            # Generate communities using the function we found
            partition = best_partition_func(weighted_graph)

            # Group terms by community
            communities: Dict[
                int, List[Dict[str, Union[str, float, List[str], bool]]]
            ] = {}

            # Calculate node centrality for identifying central terms
            centrality = nx.betweenness_centrality(weighted_graph, weight="weight")

            # First pass: collect all terms and their emotional data
            for node_id, community_id in partition.items():
                if node_id not in weighted_graph:
                    continue

                # Get the term text for this node
                term_value = self.g.nodes[node_id].get("term", f"Unknown {node_id}")
                term = (
                    str(term_value) if term_value is not None else f"Unknown {node_id}"
                )

                # Initialize community if needed
                if community_id not in communities:
                    communities[community_id] = []

                # Get related dimensions for this term
                dimensions: Set[str] = set()
                meta_emotions: List[str] = []

                for _, _, data in self.g.edges(node_id, data=True):
                    # Get dimension and add to set
                    dim_value = data.get("dimension", "lexical")
                    dim = str(dim_value) if dim_value is not None else "lexical"
                    dimensions.add(dim)

                    # Check for meta-emotional relationships
                    rel_type_value = data.get("relationship", "")
                    rel_type = str(rel_type_value) if rel_type_value is not None else ""

                    if rel_type.startswith("meta_emotion") or rel_type == "evokes":
                        neighbor_id = None
                        # Find neighbors with emotional dimension
                        for n in self.g.neighbors(node_id):
                            edge_data = self.g[node_id][n]
                            if "emotional" in str(edge_data.get("dimension", "")):
                                neighbor_id = n
                                break

                        if neighbor_id is not None:
                            neighbor_term_value = self.g.nodes[neighbor_id].get(
                                "term", ""
                            )
                            neighbor_term = (
                                str(neighbor_term_value)
                                if neighbor_term_value is not None
                                else ""
                            )
                            meta_emotions.append(neighbor_term)

                # Try to get valence/arousal from node attributes
                valence_value = self.g.nodes[node_id].get("valence", None)
                valence = float(valence_value) if valence_value is not None else None

                arousal_value = self.g.nodes[node_id].get("arousal", None)
                arousal = float(arousal_value) if arousal_value is not None else None

                # Calculate whether this is a central term in the cluster
                centrality_value = centrality.get(node_id, 0)
                median_value = np.median([float(v) for v in centrality.values()])
                is_central = bool(centrality_value > median_value)

                # Add term data to the community
                communities[community_id].append(
                    {
                        "term": term,
                        "valence": valence,
                        "arousal": arousal,
                        "central": is_central,
                        "related_dimensions": list(dimensions),
                        "meta_emotions": meta_emotions,
                    }
                )

            # Filter small communities
            return {
                k: v for k, v in communities.items() if len(v) >= min_community_size
            }
        except Exception as e:
            error_msg = f"Failed to analyze semantic clusters: {e}"
            raise GraphError(error_msg, e) from e

    def get_relationships_by_dimension(
        self,
        dimension: str = "lexical",
        rel_type: Optional[str] = None,
        include_meta: bool = False,
        valence_range: Optional[Tuple[float, float]] = None,
    ) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """
        Get all relationships of a specific dimension with detailed filtering.

        Retrieves relationships in the graph belonging to the specified dimension
        (lexical, emotional, affective) with optional filtering by relationship type,
        meta-emotional connections, and valence range for emotional terms.

        Args:
            dimension: Relationship dimension to filter by
                      Options: "lexical", "emotional", "affective"
            rel_type: Optional specific relationship type to filter by
                     (e.g., "synonym", "emotional_synonym", "intensifies")
            include_meta: Whether to include meta-emotional relationships
            valence_range: Optional tuple of (min_valence, max_valence) to filter by
                          emotional valence (for emotional dimension only)

        Returns:
            List of tuples containing (source_term, target_term, relationship_type, attributes)
            where attributes is a dictionary with additional data about the relationship

        Example:
            ```python
            # Get all emotional relationships
            emotional_relationships = graph_manager.get_relationships_by_dimension("emotional")
            for source, target, rel_type, attrs in emotional_relationships:
                print(f"{source} {rel_type} {target} (weight: {attrs.get('weight')})")

            # Get only intensifying emotional relationships
            intensifiers = graph_manager.get_relationships_by_dimension(
                "emotional", rel_type="intensifies"
            )

            # Get high-valence emotional relationships (0.5 to 1.0)
            positive_emotions = graph_manager.get_relationships_by_dimension(
                "emotional", valence_range=(0.5, 1.0)
            )

            # Include meta-emotional relationships
            with_meta = graph_manager.get_relationships_by_dimension(
                "emotional", include_meta=True
            )
            ```
        """
        result = []

        for source, target, data in self.g.edges(data=True):
            edge_dimension = data.get("dimension", "lexical")

            # Check dimension filter
            if edge_dimension != dimension:
                # Check for meta-emotional relationships if requested
                if (
                    include_meta
                    and edge_dimension == "emotional"
                    and dimension == "lexical"
                ):
                    pass  # Include this edge as it's a meta-relationship
                else:
                    continue

            # Check relationship type filter
            if rel_type and data.get("relationship", "") != rel_type:
                continue

            # Get source and target terms
            source_term = self.g.nodes[source].get("term", "")
            target_term = self.g.nodes[target].get("term", "")
            rel_type = data.get("relationship", "related")

            # Check valence range if specified and available (for emotional dimension)
            if valence_range and edge_dimension == "emotional":
                # Try to get source valence first from edge, then from node
                source_valence = data.get(
                    "source_valence", self.g.nodes[source].get("valence", None)
                )

                if source_valence is not None:
                    min_val, max_val = valence_range
                    if not (min_val <= source_valence <= max_val):
                        continue

            # Collect additional attributes for analysis
            attrs = {
                "weight": data.get("weight", 1.0),
                "bidirectional": data.get("bidirectional", True),
                "color": data.get("color", "#aaaaaa"),
            }

            # Add any emotional attributes if available
            if edge_dimension == "emotional":
                for attr in [
                    "valence",
                    "arousal",
                    "source_valence",
                    "target_valence",
                    "confidence",
                    "intensity",
                ]:
                    if attr in data:
                        attrs[attr] = data[attr]

            result.append((source_term, target_term, rel_type, attrs))

        return result

    def get_emotional_subgraph(
        self,
        term: str,
        depth: int = 1,
        context: Optional[Union[str, Dict[str, float]]] = None,
        emotional_types: Optional[List[str]] = None,
        min_intensity: float = 0.0,
    ) -> nx.Graph:
        """
        Extract a subgraph of emotional relationships for a specific term.

        Creates a subgraph centered on the given term, including only
        emotional dimension relationships up to the specified depth.
        Supports filtering by emotional context, relationship types,
        and minimum emotional intensity.

        Args:
            term: The central term to build the subgraph around
            depth: The number of relationship steps to include (default: 1)
            context: Optional emotional context as string name or dictionary of
                   emotional factors with weights (e.g., {"clinical": 0.8})
            emotional_types: Optional list of specific emotional relationship types
                           to include (e.g., ["intensifies", "evokes"])
            min_intensity: Minimum intensity/weight for relationships to include

        Returns:
            A NetworkX graph containing the emotional subgraph with enhanced
            node and edge attributes for emotional analysis

        Raises:
            NodeNotFoundError: If the term is not found in the graph

        Example:
            ```python
            # Get emotional relationships around "happiness"
            emotional_graph = graph_manager.get_emotional_subgraph("happiness", depth=2)
            print(f"Found {emotional_graph.number_of_nodes()} emotionally connected terms")

            # Get only intense emotional relationships
            intense_graph = graph_manager.get_emotional_subgraph(
                "fear", min_intensity=0.7
            )

            # Get specific emotional relationship types
            specific_graph = graph_manager.get_emotional_subgraph(
                "anxiety", emotional_types=["intensifies", "evokes"]
            )

            # Apply emotional context to focus analysis
            clinical_graph = graph_manager.get_emotional_subgraph(
                "pain", context={"clinical": 0.9, "emergency": 0.1}
            )
            ```
        """
        term_lower = term.lower()
        if term_lower not in self._term_to_id:
            raise NodeNotFoundError(f"Term '{term}' not found in the graph")

        node_id = self._term_to_id[term_lower]

        # Initialize with the central node
        nodes_to_include = {node_id}
        current_nodes = {node_id}

        # Apply context weights if provided
        context_weights = {}
        if isinstance(context, dict):
            context_weights = context
        elif isinstance(context, str):
            # Try to find named context in relationships
            # This is a placeholder - in a full implementation, you would retrieve
            # the actual context weights from a context registry
            context_weights = {"context": context}  # Simplified version

        # Breadth-first search up to specified depth
        for _ in range(depth):
            next_nodes = set()
            for node in current_nodes:
                for neighbor in self.g.neighbors(node):
                    # Get edge data
                    edge_data = self.g.get_edge_data(node, neighbor)

                    # Skip if not an emotional dimension
                    if edge_data.get("dimension") != "emotional":
                        continue

                    # Check emotional relationship type if filter is active
                    rel_type = edge_data.get("relationship", "")
                    if emotional_types and rel_type not in emotional_types:
                        continue

                    # Check minimum intensity/weight
                    weight = edge_data.get("weight", 0.0)
                    intensity = edge_data.get("intensity", weight)
                    if intensity < min_intensity:
                        continue

                    # Add to next frontier
                    if neighbor not in nodes_to_include:
                        next_nodes.add(neighbor)

            nodes_to_include.update(next_nodes)
            current_nodes = next_nodes

        # Create the subgraph
        emotional_subgraph = self.g.subgraph(nodes_to_include).copy()

        # Filter out non-emotional edges
        edges_to_remove = []
        for source, target, data in emotional_subgraph.edges(data=True):
            # Check if it's an emotional edge
            if data.get("dimension") != "emotional":
                edges_to_remove.append((source, target))
                continue

            # Apply additional filters
            if emotional_types and data.get("relationship", "") not in emotional_types:
                edges_to_remove.append((source, target))
                continue

            # Check minimum intensity
            weight = data.get("weight", 0.0)
            intensity = data.get("intensity", weight)
            if intensity < min_intensity:
                edges_to_remove.append((source, target))
                continue

            # Apply context weighting to edge attributes
            if context_weights:
                # Store original weight for reference
                data["original_weight"] = weight

                # Apply context weighting - this is a simplified version
                # A full implementation would use more sophisticated context application
                if (
                    "context" in context_weights
                    and data.get("context") == context_weights["context"]
                ):
                    data["weight"] = weight * 1.5  # Emphasize matching context

                # Here you would apply more detailed contextual adjustments based on the
                # specific emotional factors in the context_weights dictionary

        # Remove filtered edges
        for edge in edges_to_remove:
            emotional_subgraph.remove_edge(*edge)

        # Add emotional metadata to nodes
        for node in emotional_subgraph.nodes():
            # Get the term text
            term = emotional_subgraph.nodes[node].get("term", "")

            # Add emotional metadata if available (from emotion_manager data)
            # In a real implementation, you might query an emotion_manager here
            emotional_subgraph.nodes[node]["valence"] = emotional_subgraph.nodes[
                node
            ].get("valence", 0.0)
            emotional_subgraph.nodes[node]["arousal"] = emotional_subgraph.nodes[
                node
            ].get("arousal", 0.0)

            # Add degree centrality as a measure of emotional connectivity
            emotional_subgraph.nodes[node]["emotional_centrality"] = (
                emotional_subgraph.degree(node)
            )

        return emotional_subgraph

    def analyze_multidimensional_relationships(self) -> Dict[str, Any]:
        """
        Analyze relationships across different dimensions with emotional intelligence.

        Provides comprehensive statistics about relationship types across lexical,
        emotional, and affective dimensions, identifying patterns, correlations,
        emotional clusters, meta-emotional relationships, and emotional transitions.

        Returns:
            Dictionary with detailed analysis including:
            - dimensions: Counts of relationships by dimension
            - co_occurrences: Co-occurrence patterns between dimensions
            - most_common: Most common relationship types by dimension
            - multi_dimensional_nodes: Terms with connections in multiple dimensions
            - emotional_valence_distribution: Statistical distribution of emotional valence
            - meta_emotional_patterns: Patterns of emotions about emotions
            - emotional_clusters: Clusters of emotionally related terms
            - affective_transitions: Common pathways between emotional states

        Example:
            ```python
            # Get comprehensive multidimensional analysis
            analysis = graph_manager.analyze_multidimensional_relationships()

            # Examine emotional dimensions
            emotional_stats = analysis.get("dimensions", {}).get("emotional", 0)
            print(f"Emotional relationships: {emotional_stats}")

            # Examine meta-emotional patterns
            meta_patterns = analysis.get("meta_emotional_patterns", {})
            for source, targets in meta_patterns.items():
                print(f"Meta-emotion: {source} → {', '.join(targets)}")

            # Examine emotional transitions
            transitions = analysis.get("affective_transitions", [])
            for source, target, strength in transitions[:5]:  # Show top 5
                print(f"Emotional transition: {source} → {target} (strength: {strength:.2f})")
            ```
        """
        # Initialize results structure with extended emotional analysis
        results = {
            "dimensions": {},  # Relationship counts by dimension
            "co_occurrences": {},  # Cross-dimensional patterns
            "most_common": {},  # Most common relationship types
            "multi_dimensional_nodes": {},  # Terms with connections across dimensions
            "emotional_valence_distribution": {},  # Statistical distribution of emotional valence
            "meta_emotional_patterns": {},  # Patterns of emotions about emotions
            "emotional_clusters": [],  # Clusters of emotionally related terms
            "affective_transitions": [],  # Common pathways between emotional states
        }

        # Count relationships by dimension
        dimension_counts = {}
        for _, _, data in self.g.edges(data=True):
            dimension = data.get("dimension", "lexical")
            dimension_counts[dimension] = dimension_counts.get(dimension, 0) + 1

        results["dimensions"] = dimension_counts

        # Find nodes with multiple dimension connections and track emotional valences
        nodes_with_multi_dimensions = {}
        valence_values = []
        meta_emotional_sources = {}

        for node in self.g.nodes():
            term = self.g.nodes[node].get("term", "")
            if not term:
                continue

            # Track dimensions for this node
            dimensions = set()
            emotional_neighbors = []

            # Valence for emotional terms
            valence = self.g.nodes[node].get("valence")
            if valence is not None:
                valence_values.append(valence)

            # Examine node's edges
            for neighbor in self.g.neighbors(node):
                edge_data = self.g.get_edge_data(node, neighbor)
                dimension = edge_data.get("dimension", "lexical")
                dimensions.add(dimension)

                # Track emotional connections
                if dimension == "emotional":
                    neighbor_term = self.g.nodes[neighbor].get("term", "")
                    rel_type = edge_data.get("relationship", "")
                    emotional_neighbors.append((neighbor_term, rel_type))

                    # Check for meta-emotional relationships
                    if rel_type in ["meta_emotion", "evokes", "emotional_component"]:
                        if term not in meta_emotional_sources:
                            meta_emotional_sources[term] = []
                        meta_emotional_sources[term].append(neighbor_term)

            # Record nodes with multiple dimensions
            if len(dimensions) > 1:
                nodes_with_multi_dimensions[term] = {
                    "dimensions": list(dimensions),
                    "emotional_connections": emotional_neighbors,
                }

        results["multi_dimensional_nodes"] = nodes_with_multi_dimensions
        results["meta_emotional_patterns"] = meta_emotional_sources

        # Analyze valence distribution for emotional terms
        if valence_values:
            results["emotional_valence_distribution"] = {
                "count": len(valence_values),
                "mean": np.mean(valence_values),
                "median": np.median(valence_values),
                "std": np.std(valence_values),
                "min": min(valence_values),
                "max": max(valence_values),
                "positive_ratio": sum(1 for v in valence_values if v > 0)
                / len(valence_values),
                "negative_ratio": sum(1 for v in valence_values if v < 0)
                / len(valence_values),
                "neutral_ratio": sum(1 for v in valence_values if v == 0)
                / len(valence_values),
            }

        # Find most common relationship types by dimension
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

        # Analyze co-occurrences between dimensions
        dimension_pairs = {}
        for node in self.g.nodes():
            # Count dimensions for this node's edges
            node_dimensions = {}
            for _, _, data in self.g.edges(node, data=True):
                dimension = data.get("dimension", "lexical")
                node_dimensions[dimension] = node_dimensions.get(dimension, 0) + 1

            # Record co-occurrences for each dimension pair
            dimensions = list(node_dimensions.keys())
            for i in range(len(dimensions)):
                for j in range(i + 1, len(dimensions)):
                    dim1, dim2 = dimensions[i], dimensions[j]
                    key = f"{dim1}_{dim2}"
                    dimension_pairs[key] = dimension_pairs.get(key, 0) + 1

        results["co_occurrences"] = dimension_pairs

        # Identify emotional clusters using basic community detection on emotional subgraph
        try:
            # Extract emotional edges
            emotional_edges = [
                (u, v)
                for u, v, d in self.g.edges(data=True)
                if d.get("dimension") == "emotional"
            ]

            if emotional_edges:
                emotional_graph = self.g.edge_subgraph(emotional_edges).copy()

                # Simple clustering - in a full implementation, use more sophisticated methods
                from networkx.algorithms import community

                communities = community.greedy_modularity_communities(emotional_graph)

                # Convert to list of emotional clusters
                for i, comm in enumerate(communities):
                    cluster = []
                    for node in comm:
                        term = self.g.nodes[node].get("term", "")
                        valence = self.g.nodes[node].get("valence")
                        arousal = self.g.nodes[node].get("arousal")
                        cluster.append(
                            {"term": term, "valence": valence, "arousal": arousal}
                        )

                    if len(cluster) >= 2:  # Only include non-trivial clusters
                        results["emotional_clusters"].append(
                            {"id": i, "terms": cluster, "size": len(cluster)}
                        )
        except Exception:
            # Silently fail, leaving emotional_clusters empty
            pass

        # Identify common affective transitions
        # This represents pathways between emotional states
        transitions = []

        emotional_nodes = [
            n
            for n, d in self.g.nodes(data=True)
            if d.get("valence") is not None or d.get("arousal") is not None
        ]

        for source in emotional_nodes:
            source_term = self.g.nodes[source].get("term", "")
            source_valence = self.g.nodes[source].get("valence", 0)

            for target in self.g.neighbors(source):
                if target in emotional_nodes:
                    edge_data = self.g.get_edge_data(source, target)
                    if edge_data.get("dimension") == "emotional":
                        target_term = self.g.nodes[target].get("term", "")
                        target_valence = self.g.nodes[target].get("valence", 0)

                        # Skip if same valence (no transition)
                        if source_valence == target_valence:
                            continue

                        # Calculate transition strength
                        weight = edge_data.get("weight", 1.0)
                        transition_strength = weight * abs(
                            source_valence - target_valence
                        )

                        transitions.append(
                            (source_term, target_term, transition_strength)
                        )

        # Sort transitions by strength
        if transitions:
            results["affective_transitions"] = sorted(
                transitions, key=lambda x: x[2], reverse=True
            )[
                :10
            ]  # Top 10 transitions

        return results

    def extract_meta_emotional_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract patterns of meta-emotions (emotions about emotions) from the graph.

        Identifies and analyzes meta-emotional relationships, where one emotional term
        relates to another emotional term through relationships like "evokes", "meta_emotion",
        or "emotional_component". These patterns reveal how emotions relate to each other
        in complex hierarchies and networks.

        Returns:
            Dictionary mapping source emotions to lists of target emotion dictionaries,
            each containing the target term, relationship type, and relationship strength

        Example:
            ```python
            # Extract meta-emotional patterns
            patterns = graph_manager.extract_meta_emotional_patterns()

            # Examine patterns
            for source_emotion, targets in patterns.items():
                print(f"Emotion: {source_emotion}")
                for target in targets:
                    print(f"  → {target['term']} ({target['relationship']})"
                          f" strength: {target['strength']:.2f}")
            ```
        """
        patterns = {}

        # Find all emotional terms (nodes with "emotion" in their description or tagged as emotional)
        emotional_nodes = []
        for node, attrs in self.g.nodes(data=True):
            term = attrs.get("term", "")
            if not term:
                continue

            # Check if node has valence/arousal or is marked as emotional
            has_emotional_attrs = (
                attrs.get("valence") is not None or attrs.get("arousal") is not None
            )

            # Also check node's edges for emotional connections
            has_emotional_edges = any(
                self.g.get_edge_data(node, neighbor).get("dimension") == "emotional"
                for neighbor in self.g.neighbors(node)
            )

            if has_emotional_attrs or has_emotional_edges:
                emotional_nodes.append(node)

        # Analyze connections between emotional nodes
        meta_relationships = [
            "meta_emotion",
            "evokes",
            "emotional_component",
            "intensifies",
            "diminishes",
        ]

        for source in emotional_nodes:
            source_term = self.g.nodes[source].get("term", "")

            # Find meta-emotional connections
            meta_targets = []

            for target in self.g.neighbors(source):
                if target in emotional_nodes:
                    edge_data = self.g.get_edge_data(source, target)
                    rel_type = edge_data.get("relationship", "")

                    # Check if this is a meta-emotional relationship
                    if rel_type in meta_relationships:
                        target_term = self.g.nodes[target].get("term", "")
                        weight = edge_data.get("weight", 1.0)

                        meta_targets.append(
                            {
                                "term": target_term,
                                "relationship": rel_type,
                                "strength": weight,
                                "bidirectional": edge_data.get("bidirectional", False),
                                "source_valence": self.g.nodes[source].get("valence"),
                                "target_valence": self.g.nodes[target].get("valence"),
                            }
                        )

            # Only include emotions with meta-connections
            if meta_targets:
                patterns[source_term] = meta_targets

        return patterns

    def analyze_emotional_valence_distribution(
        self, dimension: str = "emotional"
    ) -> Dict[str, Union[float, Dict[str, int]]]:
        """
        Analyze the distribution of emotional valence across the graph.

        Computes statistical metrics about the emotional valence (positivity/negativity)
        of terms in the graph, provides distribution histograms, and identifies terms at
        the extremes of the valence spectrum.

        Args:
            dimension: Dimension to analyze, typically "emotional" (default) or "affective"

        Returns:
            Dictionary with statistical analysis including:
            - count: Number of terms with valence values
            - mean/median/std: Statistical measures of central tendency and spread
            - range: Minimum and maximum valence values
            - distribution: Histogram of valence value frequencies
            - top_positive/top_negative: Terms with highest/lowest valence scores
            - clusters: Valence-based term clusters

        Example:
            ```python
            # Analyze emotional valence distribution
            valence_stats = graph_manager.analyze_emotional_valence_distribution()

            # Print key statistics
            print(f"Emotional terms: {valence_stats['count']}")
            print(f"Mean valence: {valence_stats['mean']:.2f}")
            print(f"Valence range: {valence_stats['range'][0]:.2f} to {valence_stats['range'][1]:.2f}")

            # Print most positive and negative terms
            print("Most positive terms:")
            for term, val in valence_stats['top_positive']:
                print(f"  - {term}: {val:.2f}")

            print("Most negative terms:")
            for term, val in valence_stats['top_negative']:
                print(f"  - {term}: {val:.2f}")
            ```
        """
        # Collection stage
        valence_data = []
        term_to_valence = {}

        for node, attrs in self.g.nodes(data=True):
            term = attrs.get("term", "")
            if not term:
                continue

            # Try to get valence from node attributes
            valence = attrs.get("valence")

            # If not found directly, check if this node has emotional edges
            if valence is None:
                # Find average valence from emotional edges
                emotional_edges = [
                    self.g.get_edge_data(node, neighbor)
                    for neighbor in self.g.neighbors(node)
                    if self.g.get_edge_data(node, neighbor).get("dimension")
                    == dimension
                ]

                if emotional_edges:
                    # Extract valence values, default to 0 if not specified
                    edge_valences = [
                        edge.get("valence", 0)
                        for edge in emotional_edges
                        if "valence" in edge
                    ]

                    if edge_valences:
                        valence = sum(edge_valences) / len(edge_valences)

            # If we found a valence value, record it
            if valence is not None:
                valence_data.append(valence)
                term_to_valence[term] = valence

        # Statistical analysis
        if not valence_data:
            return {
                "count": 0,
                "mean": 0,
                "median": 0,
                "std": 0,
                "range": [0, 0],
                "distribution": {},
                "top_positive": [],
                "top_negative": [],
                "clusters": {},
            }

        # Calculate basic statistics
        count = len(valence_data)
        mean_valence = np.mean(valence_data)
        median_valence = np.median(valence_data)
        std_valence = np.std(valence_data)
        min_valence = min(valence_data)
        max_valence = max(valence_data)

        # Generate histogram (distribution)
        hist_bins = 7  # -1.0 to 1.0 in increments of ~0.3
        hist, bin_edges = np.histogram(valence_data, bins=hist_bins, range=(-1.0, 1.0))

        # Convert histogram to dictionary
        distribution = {}
        for i, count in enumerate(hist):
            bin_name = f"{bin_edges[i]:.1f} to {bin_edges[i+1]:.1f}"
            distribution[bin_name] = int(count)

        # Find top positive and negative terms
        sorted_terms = sorted(term_to_valence.items(), key=lambda x: x[1])
        top_negative = sorted_terms[
            : min(5, len(sorted_terms) // 5)
        ]  # Bottom 20% up to 5 terms
        top_positive = sorted_terms[
            -min(5, len(sorted_terms) // 5) :
        ]  # Top 20% up to 5 terms

        # Group terms into clusters by valence range
        clusters = {
            "very_negative": [],  # -1.0 to -0.6
            "negative": [],  # -0.6 to -0.2
            "slightly_negative": [],  # -0.2 to -0.0
            "neutral": [],  # 0.0
            "slightly_positive": [],  # 0.0 to 0.2
            "positive": [],  # 0.2 to 0.6
            "very_positive": [],  # 0.6 to 1.0
        }

        for term, val in term_to_valence.items():
            if val < -0.6:
                clusters["very_negative"].append(term)
            elif val < -0.2:
                clusters["negative"].append(term)
            elif val < 0:
                clusters["slightly_negative"].append(term)
            elif val == 0:
                clusters["neutral"].append(term)
            elif val <= 0.2:
                clusters["slightly_positive"].append(term)
            elif val <= 0.6:
                clusters["positive"].append(term)
            else:
                clusters["very_positive"].append(term)

        # Final result structure
        return {
            "count": count,
            "mean": mean_valence,
            "median": median_valence,
            "std": std_valence,
            "range": [min_valence, max_valence],
            "distribution": distribution,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "clusters": {
                k: v for k, v in clusters.items() if v
            },  # Only include non-empty clusters
        }

    def integrate_emotional_context(
        self, context_name: str, context_weights: Dict[str, float]
    ) -> None:
        """
        Integrate an emotional context into the graph for contextual analysis.

        Registers an emotional context definition that can be used to filter
        and weight relationships during emotional analysis. This allows for
        domain-specific or situation-specific emotional interpretations.

        Args:
            context_name: Name to assign to this emotional context
            context_weights: Dictionary of emotional factors and their weights
                           in this context (e.g., {'clinical': 0.8, 'urgency': 0.6})

        Example:
            ```python
            # Define a clinical medical context
            graph_manager.integrate_emotional_context(
                "clinical",
                {
                    "professional": 0.9,
                    "detached": 0.7,
                    "analytical": 0.8,
                    "urgency": 0.6,
                    "empathy": 0.4
                }
            )

            # Define a literary context
            graph_manager.integrate_emotional_context(
                "literary",
                {
                    "expressive": 0.9,
                    "narrative": 0.8,
                    "descriptive": 0.7,
                    "dramatic": 0.6
                }
            )

            # Later use the context in analysis
            subgraph = graph_manager.get_emotional_subgraph(
                "anxiety", context="clinical"
            )
            ```
        """
        # Store the context in graph metadata
        if not hasattr(self, "_emotional_contexts"):
            self._emotional_contexts = {}

        # Validate weights (should be between 0 and 1)
        for factor, weight in context_weights.items():
            if not 0 <= weight <= 1:
                raise ValueError(
                    f"Context weight for '{factor}' must be between 0 and 1"
                )

        # Store the context
        self._emotional_contexts[context_name] = context_weights

        # Add context as a graph attribute for persistence
        self.g.graph[f"emotional_context_{context_name}"] = context_weights

        # Apply context tags to relevant emotional edges
        # This is a simplified implementation - in a real system, you would use
        # more sophisticated matching to determine which edges relate to this context
        updated_count = 0

        for source, target, data in self.g.edges(data=True):
            if data.get("dimension") == "emotional":
                # Check if any context factors match this relationship
                for factor in context_weights.keys():
                    # Very simple matching - check if factor appears in relationship type
                    rel_type = data.get("relationship", "")
                    source_term = self.g.nodes[source].get("term", "").lower()
                    target_term = self.g.nodes[target].get("term", "").lower()

                    # Simple heuristic - if factor is in the relationship or terms
                    if (
                        factor.lower() in rel_type.lower()
                        or factor.lower() in source_term
                        or factor.lower() in target_term
                    ):

                        # Tag edge with context
                        data["context"] = context_name
                        # Adjust weight based on context importance
                        data["context_weight"] = context_weights[factor]
                        updated_count += 1
                        break

        return updated_count

    def analyze_emotional_transitions(
        self, path_length: int = 2, min_transition_strength: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Analyze emotional transition pathways in the emotion graph.

        Identifies common paths of emotional change/transition, showing how
        emotions flow from one state to another. These transitions reveal
        emotional narratives, progression patterns, and psychological pathways.

        Args:
            path_length: Maximum path length to consider (default: 2)
            min_transition_strength: Minimum strength threshold for transitions

        Returns:
            List of dictionaries representing transition paths, each containing:
            - path: List of terms in the transition path
            - strength: Overall transition strength
            - valence_shift: Net change in valence
            - arousal_shift: Net change in arousal
            - relationship_types: Types of relationships in the path

        Example:
            ```python
            # Analyze emotional transitions
            transitions = graph_manager.analyze_emotional_transitions()

            # Print top transitions
            for t in transitions[:5]:  # Top 5
                path_str = " → ".join(t["path"])
                print(f"Transition: {path_str}")
                print(f"  Strength: {t['strength']:.2f}")
                print(f"  Valence shift: {t['valence_shift']:.2f}")
                print(f"  Relationship types: {', '.join(t['relationship_types'])}")
            ```
        """
        # Find emotional nodes
        emotional_nodes = []
        for node, attrs in self.g.nodes(data=True):
            valence = attrs.get("valence")
            arousal = attrs.get("arousal")

            # Include node if it has emotional attributes
            if valence is not None or arousal is not None:
                emotional_nodes.append(node)

        # If too few emotional nodes, return empty result
        if len(emotional_nodes) < 2:
            return []

        transitions = []

        # Analyze paths between emotional nodes
        for source in emotional_nodes:
            source_term = self.g.nodes[source].get("term", "")
            source_valence = self.g.nodes[source].get("valence", 0)
            source_arousal = self.g.nodes[source].get("arousal", 0)

            # Use simple BFS to find paths up to path_length
            paths = self._find_emotional_paths(source, path_length, emotional_nodes)

            for path in paths:
                # Skip too short paths
                if len(path) < 2:
                    continue

                # Calculate transition properties
                path_terms = [self.g.nodes[n].get("term", "") for n in path]
                relationship_types = []

                # Calculate cumulative strength and changes
                strength = 1.0  # Start with full strength

                # Get initial and final emotional states
                final_node = path[-1]
                final_valence = self.g.nodes[final_node].get("valence", 0)
                final_arousal = self.g.nodes[final_node].get("arousal", 0)

                # Calculate shifts
                valence_shift = final_valence - source_valence
                arousal_shift = final_arousal - source_arousal

                # Extract relationship types along the path
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge_data = self.g.get_edge_data(u, v)
                    rel_type = edge_data.get("relationship", "related")
                    relationship_types.append(rel_type)

                    # Multiply by edge weight to get path strength
                    edge_weight = edge_data.get("weight", 1.0)
                    strength *= edge_weight

                # Only include paths above threshold
                if strength >= min_transition_strength:
                    transitions.append(
                        {
                            "path": path_terms,
                            "strength": strength,
                            "valence_shift": valence_shift,
                            "arousal_shift": arousal_shift,
                            "relationship_types": relationship_types,
                        }
                    )

        # Sort transitions by strength
        transitions.sort(key=lambda x: x["strength"], reverse=True)

        return transitions

    def _find_emotional_paths(
        self, start_node: WordId, max_length: int, valid_nodes: List[WordId]
    ) -> List[List[WordId]]:
        """
        Find paths from start_node to other emotional nodes up to max_length.

        Helper method for analyze_emotional_transitions().

        Args:
            start_node: Starting node ID
            max_length: Maximum path length
            valid_nodes: List of valid destination nodes (emotional nodes)

        Returns:
            List of node ID paths
        """
        # Simple BFS path finding
        paths: List[List[WordId]] = []
        queue: List[Tuple[WordId, List[WordId]]] = [(start_node, [start_node])]

        while queue:
            node, path = queue.pop(0)

            # If path is already at max length, don't expand further
            if len(path) >= max_length:
                continue

            for neighbor in self.g.neighbors(node):
                # Cast neighbor to WordId to satisfy type checker
                neighbor_id: WordId = cast(WordId, neighbor)

                # Skip already visited nodes in this path
                if neighbor_id in path:
                    continue

                # Create new path with this neighbor
                new_path: List[WordId] = path + [neighbor_id]

                # If neighbor is a valid destination, add the path
                if neighbor_id in valid_nodes and neighbor_id != start_node:
                    paths.append(new_path)

                # Continue BFS with this new path
                queue.append((neighbor_id, new_path))

        return paths

    def verify_database_tables(self) -> bool:
        """Verify that required database tables exist.

        Returns:
            bool: True if tables exist, False otherwise
        """
        try:
            # Use _db_connection to verify tables
            with self._db_connection() as conn:
                cursor = conn.cursor()
                # Check if words table exists
                cursor.execute(SQL_CHECK_WORDS_TABLE)
                if not cursor.fetchone():
                    return False

                # Check if relationships table exists
                cursor.execute(SQL_CHECK_RELATIONSHIPS_TABLE)
                if not cursor.fetchone():
                    return False

                return True
        except Exception as e:
            # Log the specific database error for debugging
            logging.error(f"Database verification failed: {e}")
            return False


def graph_demo() -> None:
    """
    Demonstrate key functionality of the GraphManager class.

    This function showcases the full capabilities of the GraphManager including:
    - Basic graph operations (building, querying, visualization)
    - Emotional and affective relationship analysis
    - Meta-emotional patterns and transitions
    - Semantic clustering and multidimensional analysis
    - Context-based emotional analysis
    - Advanced visualization techniques

    This serves as both a demonstration and comprehensive test suite.

    Raises:
        GraphError: If demonstration operations fail

    Example:
        ```python
        # Run the comprehensive demonstration
        main()
        ```
    """
    import time

    start_time = time.time()
    print("Starting GraphManager demonstration...\n")

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

        # Phase 1: Basic Relationship Analysis
        print("\n=== PHASE 1: BASIC RELATIONSHIP ANALYSIS ===")

        # Get related terms example
        example_term = "algorithm"  # Changed to a sample term likely to exist
        try:
            related_terms = graph_manager.get_related_terms(example_term)
            print(f"\nTerms related to '{example_term}': {related_terms}")

            # Filter by relationship type
            synonyms = graph_manager.get_related_terms(example_term, rel_type="synonym")
            print(f"Synonyms of '{example_term}': {synonyms}")

            # Get other relationship types if available
            for rel_type in ["antonym", "hypernym", "hyponym"]:
                try:
                    terms = graph_manager.get_related_terms(
                        example_term, rel_type=rel_type
                    )
                    if terms:
                        print(f"{rel_type.capitalize()}s of '{example_term}': {terms}")
                except Exception:
                    pass

        except NodeNotFoundError as e:
            print(f"Warning: {e}")
            # Try an alternative term from the sample data
            alternative_terms = ["data", "computer", "software", "function"]
            for alt_term in alternative_terms:
                try:
                    related_terms = graph_manager.get_related_terms(alt_term)
                    print(f"\nTerms related to '{alt_term}': {related_terms}")
                    example_term = alt_term  # Update for later use
                    break
                except NodeNotFoundError:
                    continue

        # Phase 2: Multidimensional Relationship Analysis
        print("\n=== PHASE 2: MULTIDIMENSIONAL RELATIONSHIP ANALYSIS ===")

        # Analyze multidimensional relationships
        print("Analyzing multidimensional relationship patterns...")
        relationship_analysis = graph_manager.analyze_multidimensional_relationships()

        # Display dimension statistics
        print("Relationship dimensions:")
        for dimension, count in relationship_analysis.get("dimensions", {}).items():
            print(f"  - {dimension}: {count} relationships")

        # Display multi-dimensional nodes
        multi_dim_nodes = relationship_analysis.get("multi_dimensional_nodes", {})
        if multi_dim_nodes:
            print("\nTerms with multiple relationship dimensions:")
            for term, data in list(multi_dim_nodes.items())[:5]:  # Show first 5
                dimensions = data.get("dimensions", [])
                print(f"  - {term}: {', '.join(dimensions)}")

        # Display most common relationship types
        most_common = relationship_analysis.get("most_common", {})
        if most_common:
            print("\nMost common relationship types by dimension:")
            for dimension, types in most_common.items():
                if types:
                    print(f"  - {dimension}: {types[0][0]} ({types[0][1]} occurrences)")

        # Phase 3: Emotional Relationship Analysis
        print("\n=== PHASE 3: EMOTIONAL RELATIONSHIP ANALYSIS ===")

        # Analyze emotional valence distribution
        print("Analyzing emotional valence distribution...")
        valence_analysis = graph_manager.analyze_emotional_valence_distribution()

        if valence_analysis["count"] > 0:
            print(f"Found {valence_analysis['count']} terms with emotional valence")
            print(
                f"Average valence: {valence_analysis['mean']:.2f} (range: {valence_analysis['range'][0]:.2f} to {valence_analysis['range'][1]:.2f})"
            )

            # Show positive and negative examples
            if valence_analysis.get("top_positive"):
                print("\nMost positive terms:")
                for term, val in valence_analysis["top_positive"]:
                    print(f"  - {term}: {val:.2f}")

            if valence_analysis.get("top_negative"):
                print("\nMost negative terms:")
                for term, val in valence_analysis["top_negative"]:
                    print(f"  - {term}: {val:.2f}")
        else:
            print("No emotional valence data found in the graph")

            # Add some sample emotional relationships for demonstration
            print("\nAdding sample emotional relationships for demonstration...")
            sample_emotional_relations = [
                ("joy", "happiness", "emotional_synonym", 0.9),
                ("sadness", "grief", "emotional_synonym", 0.8),
                ("anger", "rage", "intensifies", 0.7),
                ("fear", "anxiety", "related_emotion", 0.6),
                ("surprise", "shock", "emotional_spectrum", 0.5),
            ]

            # Add these to the graph (simplified for demonstration)
            for source, target, rel_type, weight in sample_emotional_relations:
                # First ensure the nodes exist (simplified)
                if source.lower() not in graph_manager._term_to_id:
                    source_id = len(graph_manager._term_to_id) + 1
                    graph_manager.g.add_node(
                        source_id,
                        term=source,
                        valence=(0.7 if source in ["joy", "happiness"] else -0.7),
                    )
                    graph_manager._term_to_id[source.lower()] = source_id
                else:
                    source_id = graph_manager._term_to_id[source.lower()]

                if target.lower() not in graph_manager._term_to_id:
                    target_id = len(graph_manager._term_to_id) + 1
                    graph_manager.g.add_node(
                        target_id,
                        term=target,
                        valence=(0.8 if target in ["happiness"] else -0.8),
                    )
                    graph_manager._term_to_id[target.lower()] = target_id
                else:
                    target_id = graph_manager._term_to_id[target.lower()]

                # Add the emotional edge
                graph_manager.g.add_edge(
                    source_id,
                    target_id,
                    relationship=rel_type,
                    dimension="emotional",
                    weight=weight,
                    color="#ff0000",  # Red for emotional relationships
                )

            print("Sample emotional relationships added")

        # Phase 4: Meta-Emotional Patterns
        print("\n=== PHASE 4: META-EMOTIONAL PATTERNS ===")

        # Extract meta-emotional patterns
        meta_patterns = graph_manager.extract_meta_emotional_patterns()

        if meta_patterns:
            print(f"Found {len(meta_patterns)} meta-emotional patterns")
            print("\nSample meta-emotional patterns:")
            for source, targets in list(meta_patterns.items())[:3]:  # Show first 3
                target_str = ", ".join(
                    [f"{t['term']} ({t['relationship']})" for t in targets[:2]]
                )
                print(f"  - {source} → {target_str}")
        else:
            print("No meta-emotional patterns found")

            # Add sample meta-emotional patterns for demonstration
            print("\nAdding sample meta-emotional patterns for demonstration...")
            sample_meta_relations = [
                ("anxiety", "fear", "meta_emotion", 0.8),
                ("regret", "sadness", "evokes", 0.7),
                ("awe", "surprise", "emotional_component", 0.9),
            ]

            # Add these to the graph (simplified)
            for source, target, rel_type, weight in sample_meta_relations:
                # First ensure the nodes exist (simplified)
                if source.lower() not in graph_manager._term_to_id:
                    source_id = len(graph_manager._term_to_id) + 1
                    graph_manager.g.add_node(source_id, term=source, valence=-0.3)
                    graph_manager._term_to_id[source.lower()] = source_id
                else:
                    source_id = graph_manager._term_to_id[source.lower()]

                if target.lower() not in graph_manager._term_to_id:
                    target_id = len(graph_manager._term_to_id) + 1
                    graph_manager.g.add_node(target_id, term=target, valence=-0.5)
                    graph_manager._term_to_id[target.lower()] = target_id
                else:
                    target_id = graph_manager._term_to_id[target.lower()]

                # Add the meta-emotional edge
                graph_manager.g.add_edge(
                    source_id,
                    target_id,
                    relationship=rel_type,
                    dimension="emotional",
                    weight=weight,
                    color="#800080",  # Purple for meta-emotional
                )

            print("Sample meta-emotional patterns added")

        # Phase 5: Emotional Transitions
        print("\n=== PHASE 5: EMOTIONAL TRANSITIONS ===")

        # Analyze emotional transitions
        transitions = graph_manager.analyze_emotional_transitions()

        if transitions:
            print(f"Found {len(transitions)} emotional transition pathways")
            print("\nTop emotional transitions:")
            for t in transitions[:3]:  # Show top 3
                path_str = " → ".join(t["path"])
                print(f"  - {path_str}")
                print(
                    f"    Strength: {t['strength']:.2f}, Valence shift: {t['valence_shift']:.2f}"
                )
        else:
            print("No emotional transitions found in the graph")

        # Phase 6: Semantic Clusters
        print("\n=== PHASE 6: SEMANTIC CLUSTERS ===")

        # Analyze semantic clusters
        try:
            print("Identifying semantic and emotional clusters...")
            clusters = graph_manager.analyze_semantic_clusters(min_community_size=2)

            if clusters:
                print(f"Found {len(clusters)} semantic clusters")
                print("\nSample clusters:")
                for cluster_id, terms in list(clusters.items())[:3]:  # Show first 3
                    print(f"  Cluster {cluster_id}:")
                    for term_data in terms[:3]:  # Show first 3 terms per cluster
                        term = term_data["term"]
                        valence = term_data.get("valence")
                        valence_str = (
                            f", valence: {valence:.2f}" if valence is not None else ""
                        )
                        print(f"    - {term}{valence_str}")
            else:
                print("No significant semantic clusters found")
        except ImportError:
            print("Note: Semantic clustering requires python-louvain package")
            print("Install with: pip install python-louvain")

        # Phase 7: Context Integration
        print("\n=== PHASE 7: CONTEXT INTEGRATION ===")

        # Define and integrate emotional contexts
        print("Integrating emotional contexts...")

        # Define a clinical/medical context
        clinical_context = {
            "professional": 0.9,
            "analytical": 0.8,
            "detached": 0.6,
            "compassionate": 0.5,
        }

        # Define a literary/narrative context
        literary_context = {
            "expressive": 0.9,
            "narrative": 0.8,
            "dramatic": 0.7,
            "metaphorical": 0.6,
        }

        # Integrate contexts
        try:
            updated_clinical = graph_manager.integrate_emotional_context(
                "clinical", clinical_context
            )
            updated_literary = graph_manager.integrate_emotional_context(
                "literary", literary_context
            )

            print(
                f"Integrated clinical context (affected {updated_clinical} relationships)"
            )
            print(
                f"Integrated literary context (affected {updated_literary} relationships)"
            )

            # Apply context to emotional subgraph
            # Try with an emotional term if present
            emotional_terms = [
                t["term"]
                for t in valence_analysis.get("top_positive", [])
                + valence_analysis.get("top_negative", [])
            ]

            if emotional_terms:
                context_term = emotional_terms[0]
            else:
                # Fallback to one we might have added
                context_term = "anxiety"

            print(
                f"\nExtracting emotional subgraph for '{context_term}' with clinical context..."
            )
            emotional_subgraph = graph_manager.get_emotional_subgraph(
                context_term, depth=2, context="clinical"
            )

            print(
                f"Extracted emotional subgraph with {emotional_subgraph.number_of_nodes()} nodes "
                f"and {emotional_subgraph.number_of_edges()} emotional relationships"
            )
        except Exception as e:
            print(f"Note: Context integration skipped: {e}")

        # Phase 8: Visualization
        print("\n=== PHASE 8: VISUALIZATION ===")

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

            # Create dimension-specific visualizations
            if "emotional" in relationship_analysis.get("dimensions", {}):
                vis_path_emotional = "data/emotional_graph.html"
                print("\nGenerating emotional relationships visualization...")
                graph_manager.visualize(
                    output_path=vis_path_emotional, dimensions=["emotional"]
                )

            print("\nVisualizations saved:")
            print(f"  - 2D: {vis_path_2d}")
            print(f"  - 3D: {vis_path_3d}")
            if "emotional" in relationship_analysis.get("dimensions", {}):
                print(f"  - Emotional: {vis_path_emotional}")
            print("Open these files in a web browser to explore the graph")

        except ImportError as e:
            print(f"Note: {e}")
        except Exception as e:
            print(f"Warning: Could not generate visualization: {e}")

        # Save the complete graph to a file
        output_path = "data/lexical_graph.gexf"
        print(f"\nSaving complete graph to {output_path}")
        graph_manager.save_to_gexf(output_path)
        print(f"Graph saved successfully to {output_path}")

        # Export subgraphs
        try:
            print(f"\nExtracting subgraph for '{example_term}'...")
            subgraph_path = graph_manager.export_subgraph(example_term, depth=2)
            print(f"Subgraph exported to {subgraph_path}")
        except NodeNotFoundError:
            print(
                f"Warning: Could not extract subgraph for '{example_term}' (term not found)"
            )

        # Display execution time
        elapsed_time = time.time() - start_time
        print(f"\nDemonstration completed in {elapsed_time:.2f} seconds")

    except GraphError as e:
        print(f"Graph error: {e}")
    except Exception as e:
        import traceback

        print(f"Unexpected error: {e}")
        traceback.print_exc()
    finally:
        # Ensure connections are properly closed
        db_manager.close()


if __name__ == "__main__":
    graph_demo()
