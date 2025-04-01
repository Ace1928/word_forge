from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, TypedDict, Union, cast

import networkx as nx
import numpy as np  # For dimensional calculations

from word_forge.config import config
from word_forge.database.db_manager import DatabaseError, DBManager


class GraphError(DatabaseError):
    """Base exception for graph operations."""

    pass


class NodeNotFoundError(GraphError):
    """Raised when a node cannot be found in the graph."""

    pass


class GraphDataError(GraphError):
    """Raised when there's an issue with graph data structure."""

    pass


class GraphVisualizationError(GraphError):
    """Raised when graph visualization fails."""

    pass


class GraphDimensionError(GraphError):
    """Raised when there's an issue with graph dimensions."""

    pass


class WordTupleDict(TypedDict):
    """Type definition for word node in the graph."""

    id: int
    term: str


class RelationshipTupleDict(TypedDict):
    """Type definition for relationship between words."""

    word_id: int
    related_term: str
    relationship_type: str


class GraphInfoDict(TypedDict):
    """Type definition for graph information."""

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

# Expanded relationship types to include all relationships from ParserRefiner
RELATIONSHIP_TYPES = {
    # Core relationships
    "synonym": {"weight": 1.0, "color": "#4287f5", "bidirectional": True},
    "antonym": {"weight": 0.9, "color": "#f54242", "bidirectional": True},
    # Hierarchical relationships
    "hypernym": {"weight": 0.7, "color": "#42f584", "bidirectional": False},
    "hyponym": {"weight": 0.7, "color": "#a142f5", "bidirectional": False},
    # Part-whole relationships
    "holonym": {"weight": 0.6, "color": "#f5a142", "bidirectional": False},
    "meronym": {"weight": 0.6, "color": "#42f5f5", "bidirectional": False},
    # Translation relationships
    "translation": {"weight": 0.8, "color": "#42d4f5", "bidirectional": True},
    # Semantic field relationships
    "domain": {"weight": 0.5, "color": "#7a42f5", "bidirectional": False},
    "function": {"weight": 0.5, "color": "#f542a7", "bidirectional": False},
    # General semantic relationships
    "related": {"weight": 0.4, "color": "#42f5a1", "bidirectional": True},
    # Derivational relationships
    "derived_from": {"weight": 0.5, "color": "#8c42f5", "bidirectional": False},
    "etymological_source": {"weight": 0.4, "color": "#f5b942", "bidirectional": False},
    # Usage relationships
    "context": {"weight": 0.3, "color": "#42d4f5", "bidirectional": False},
    "register": {"weight": 0.3, "color": "#f542d4", "bidirectional": False},
    # Example-based relationships
    "example_of": {"weight": 0.3, "color": "#7adbf5", "bidirectional": False},
    "instance": {"weight": 0.4, "color": "#e642f5", "bidirectional": False},
    # Default for any other relationship
    "default": {"weight": 0.3, "color": "#aaaaaa", "bidirectional": True},
}


class GraphManager:
    """
    Builds and maintains a multidimensional lexical/semantic graph from database records.

    This class creates a networkx graph representation where:
    - Nodes represent words (with word_id as node identifier)
    - Edges represent relationships between words (synonym, antonym, etc.)
    - Node attributes include the term text and multidimensional positioning
    - Edge attributes include the relationship type, weight, and color
    - The graph can be visualized in 2D or 3D space with rich semantic encoding
    """

    def __init__(self, db_manager: DBManager) -> None:
        """
        Initialize the graph manager with a database connection.

        Args:
            db_manager: Database manager providing access to word data
        """
        self.db_manager = db_manager
        self.g = nx.Graph()
        self._term_to_id: Dict[str, WordId] = {}
        self._dimensions: int = 2  # Default is 2D for backward compatibility
        self._positions: Dict[WordId, Position] = {}
        self._relationship_counts: Dict[str, int] = {}

    @property
    def dimensions(self) -> int:
        """Get the number of dimensions for the graph."""
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dims: int) -> None:
        """
        Set the number of dimensions for the graph layout.

        Args:
            dims: Number of dimensions (2 or 3)

        Raises:
            GraphDimensionError: If dimensions are not 2 or 3
        """
        if dims not in (2, 3):
            raise GraphDimensionError("Graph dimensions must be either 2 or 3")
        self._dimensions = dims

    @contextmanager
    def _db_connection(self):
        """Create a database connection using the DBManager's path.

        Yields:
            sqlite3.Connection: An active database connection
        """
        conn = sqlite3.connect(self.db_manager.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

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
        """
        self.g.clear()
        self._term_to_id.clear()
        self._positions.clear()
        self._relationship_counts = {}

        words, relationships = self._fetch_data()

        # Add nodes and build term->id mapping
        for word_id, term in words:
            self.g.add_node(word_id, term=term)
            self._term_to_id[term.lower()] = word_id

        # Add edges between related words with rich attributes
        for word_id, related_term, rel_type in relationships:
            # Track relationship types for analytics
            self._relationship_counts[rel_type] = (
                self._relationship_counts.get(rel_type, 0) + 1
            )

            related_id = self._term_to_id.get(related_term.lower())
            if related_id:
                # Get relationship styling properties or use defaults if not recognized
                rel_props = RELATIONSHIP_TYPES.get(
                    rel_type, RELATIONSHIP_TYPES["default"]
                )

                # Add the edge with appropriate attributes
                self.g.add_edge(
                    word_id,
                    related_id,
                    relationship_type=rel_type,
                    weight=rel_props["weight"],
                    color=rel_props["color"],
                    bidirectional=rel_props["bidirectional"],
                )

        # Generate layout positions
        self._compute_layout()

    def _compute_layout(self, algorithm: Optional[str] = None) -> None:
        """
        Compute node positions for the graph using the specified algorithm.

        Args:
            algorithm: Layout algorithm to use (defaults to force-directed)

        Raises:
            GraphError: If layout computation fails
        """
        if self.g.number_of_nodes() == 0:
            return

        try:
            # Choose layout algorithm based on dimensions
            if not algorithm:
                algorithm = config.graph.default_layout

            if self._dimensions == 3:
                # 3D layout algorithms
                if algorithm == "force_directed":
                    pos = nx.spring_layout(self.g, dim=3, weight="weight", scale=2.0)
                elif algorithm == "spectral":
                    # Spectral layout extended to 3D
                    pos_2d = nx.spectral_layout(self.g, weight="weight")
                    # Add z-coordinate based on betweenness centrality
                    bc = nx.betweenness_centrality(self.g, weight="weight")
                    pos = {n: (*pos_2d[n], bc.get(n, 0)) for n in self.g.nodes()}
                else:
                    # Default to 3D spring layout
                    pos = nx.spring_layout(self.g, dim=3, weight="weight", scale=2.0)
            else:
                # 2D layout algorithms (for backward compatibility)
                if algorithm == "force_directed":
                    pos = nx.spring_layout(self.g, weight="weight")
                elif algorithm == "spectral":
                    pos = nx.spectral_layout(self.g, weight="weight")
                elif algorithm == "circular":
                    pos = nx.circular_layout(self.g)
                else:
                    pos = nx.spring_layout(self.g, weight="weight")

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

    def save_to_gexf(self, path: Optional[str] = None) -> None:
        """
        Save the current graph to a GEXF format file.

        Args:
            path: Destination file path (defaults to config value)

        Raises:
            ValueError: If the graph is empty
            GraphError: If writing to the file fails
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

        Args:
            path: Source file path (defaults to config value)

        Raises:
            FileNotFoundError: If the file doesn't exist
            GraphError: If the file format is invalid
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

            # Create initial positions for new nodes
            if self._dimensions == 3:
                # Place new nodes near their neighbors or randomly
                for node in nodes_without_pos:
                    neighbors = list(self.g.neighbors(node))
                    if neighbors:
                        # Position near the average of its neighbors
                        neighbor_positions = [
                            self._positions.get(n, (0, 0, 0))
                            for n in neighbors
                            if n in self._positions
                        ]
                        if neighbor_positions:
                            avg_x = sum(p[0] for p in neighbor_positions) / len(
                                neighbor_positions
                            )
                            avg_y = sum(p[1] for p in neighbor_positions) / len(
                                neighbor_positions
                            )
                            avg_z = sum(p[2] for p in neighbor_positions) / len(
                                neighbor_positions
                            )
                            # Add some randomness to avoid overlap
                            self._positions[node] = (
                                avg_x + 0.1 * np.random.randn(),
                                avg_y + 0.1 * np.random.randn(),
                                avg_z + 0.1 * np.random.randn(),
                            )
                        else:
                            # No positioned neighbors, place randomly
                            self._positions[node] = (
                                np.random.uniform(-1, 1),
                                np.random.uniform(-1, 1),
                                np.random.uniform(-1, 1),
                            )
                    else:
                        # No neighbors, place randomly
                        self._positions[node] = (
                            np.random.uniform(-1, 1),
                            np.random.uniform(-1, 1),
                            np.random.uniform(-1, 1),
                        )
            else:
                # 2D placement logic
                for node in nodes_without_pos:
                    neighbors = list(self.g.neighbors(node))
                    if neighbors:
                        # Position near the average of its neighbors
                        neighbor_positions = [
                            self._positions.get(n, (0, 0))
                            for n in neighbors
                            if n in self._positions
                        ]
                        if neighbor_positions:
                            avg_x = sum(p[0] for p in neighbor_positions) / len(
                                neighbor_positions
                            )
                            avg_y = sum(p[1] for p in neighbor_positions) / len(
                                neighbor_positions
                            )
                            # Add some randomness to avoid overlap
                            self._positions[node] = (
                                avg_x + 0.1 * np.random.randn(),
                                avg_y + 0.1 * np.random.randn(),
                            )
                        else:
                            # No positioned neighbors, place randomly
                            self._positions[node] = (
                                np.random.uniform(-1, 1),
                                np.random.uniform(-1, 1),
                            )
                    else:
                        # No neighbors, place randomly
                        self._positions[node] = (
                            np.random.uniform(-1, 1),
                            np.random.uniform(-1, 1),
                        )

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

    def ensure_sample_data(self) -> bool:
        """
        Ensure the database contains sample data if it's empty.

        Returns:
            True if sample data was added, False if database already had data

        Raises:
            GraphError: If adding sample data fails
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
        Retrieve words and their relationships from the database.

        Returns:
            Tuple containing:
            - List of (word_id, term) tuples
            - List of (word_id, related_term, relationship_type) tuples

        Raises:
            GraphError: If database access fails
        """
        words: List[WordTuple] = []
        relationships: List[RelationshipTuple] = []

        try:
            db_path = Path(self.db_manager.db_path)
            if not db_path.exists():
                logging.warning(f"Database file {db_path} does not exist")
                return words, relationships

            with self._db_connection() as conn:
                cursor = conn.cursor()

                # Check if words table exists
                cursor.execute(SQL_CHECK_WORDS_TABLE)
                if not cursor.fetchone():
                    logging.warning("Words table does not exist in the database")
                    return words, relationships

                # Fetch all words
                cursor.execute(SQL_FETCH_ALL_WORDS)
                words = [(row["id"], row["term"]) for row in cursor.fetchall()]

                # Check if relationships table exists
                cursor.execute(SQL_CHECK_RELATIONSHIPS_TABLE)
                if not cursor.fetchone():
                    logging.warning(
                        "Relationships table does not exist in the database"
                    )
                    return words, relationships

                # Fetch all relationships
                cursor.execute(SQL_FETCH_ALL_RELATIONSHIPS)
                relationships = [
                    (row["word_id"], row["related_term"], row["relationship_type"])
                    for row in cursor.fetchall()
                ]

        except sqlite3.Error as e:
            error_msg = f"Database error while fetching graph data: {e}"
            logging.error(error_msg)
            raise GraphError(error_msg) from e

        return words, relationships

    def get_related_terms(self, term: str, rel_type: Optional[str] = None) -> List[str]:
        """
        Find terms related to the given term, optionally filtered by relationship type.

        Args:
            term: The term to find relationships for
            rel_type: Optional relationship type filter (e.g., 'synonym', 'antonym')

        Returns:
            List of related terms

        Raises:
            NodeNotFoundError: If the term is not found in the graph
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
        """
        return self.g.number_of_nodes()

    def get_edge_count(self) -> int:
        """
        Get the number of relationship edges in the graph.

        Returns:
            Integer count of edges
        """
        return self.g.number_of_edges()

    def get_term_by_id(self, word_id: WordId) -> Optional[str]:
        """
        Retrieve the term associated with a given word ID.

        Args:
            word_id: The ID of the word to look up

        Returns:
            The term string or None if not found
        """
        if word_id in self.g:
            return self.g.nodes[word_id].get("term")
        return None

    def get_graph_info(self) -> GraphInfoDict:
        """
        Get detailed information about the graph structure.

        Returns:
            Dictionary with graph statistics and sample data

        Raises:
            GraphDataError: If the graph structure is invalid
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

        Prints key statistics and sample nodes/relationships.
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
    ) -> None:
        """
        Generate an interactive HTML visualization of the graph.

        Creates an HTML file with a fully interactive network graph visualization that can be
        viewed in any browser. Nodes can be dragged, zoomed, and explored interactively.

        Args:
            output_path: Path where the HTML visualization will be saved (defaults to config)
            height: Height of the visualization frame (CSS format, defaults to config)
            width: Width of the visualization frame (CSS format, defaults to config)
            use_3d: Force 3D visualization even if graph is 2D

        Raises:
            ImportError: If pyvis is not installed
            GraphVisualizationError: If visualization creation fails
            ValueError: If the graph is empty
        """
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError(
                "Pyvis is required for interactive visualization. "
                "Install it with: pip install pyvis"
            )

        if self.g.number_of_nodes() == 0:
            raise ValueError("Cannot visualize an empty graph")

        # Use config values as defaults
        vis_path = output_path or str(
            config.graph.get_visualization_path() / "graph_visualization.html"
        )
        vis_height = height or f"{config.graph.vis_height}px"
        vis_width = width or f"{config.graph.vis_width}px"

        # Determine if we should use 3D visualization
        use_3d_vis = use_3d if use_3d is not None else (self._dimensions == 3)

        try:
            # Create a pyvis network with appropriate dimensions
            notebook = False  # We're creating a standalone HTML file
            directed = False  # Undirected graph

            # Configure network
            net = Network(
                height=vis_height,
                width=vis_width,
                notebook=notebook,
                directed=directed,
                cdn_resources="remote",  # Use CDN for 3D support
            )

            # Enable 3D physics if requested
            if use_3d_vis:
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

            # Add nodes to the network with rich attributes
            for node_id, node_attrs in self.g.nodes(data=True):
                label = node_attrs.get("term", f"Node {node_id}")
                title = f"ID: {node_id}<br>Term: {label}"

                # Size node based on degree (connectivity)
                degree = self.g.degree(node_id)
                size = 10 + min(20, degree * 2)

                # Color node based on part of speech or other attributes
                # You could extract more node attributes from the database

                # Position in 3D space if available (fix the array comparison)
                pos = self._positions.get(node_id)
                if (
                    use_3d_vis
                    and self._dimensions == 3
                    and pos is not None
                    and len(pos) == 3
                ):
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

            # Add edges to the network with semantically rich styling
            for source, target, edge_attrs in self.g.edges(data=True):
                rel_type = edge_attrs.get("relationship_type", "related")

                # Get relationship styling from mapping or use default
                rel_props = RELATIONSHIP_TYPES.get(
                    rel_type, RELATIONSHIP_TYPES["default"]
                )

                # Create edge title with relationship information
                title = f"Relationship: {rel_type}"

                # Apply directional arrows based on relationship type
                arrows = {}
                if not rel_props["bidirectional"]:
                    arrows = {"to": {"enabled": True}}

                # Add the edge with relationship properties
                net.add_edge(
                    source,
                    target,
                    title=title,
                    width=rel_props["weight"] * 5,  # Scale weight for visibility
                    color=rel_props["color"],
                    label=rel_type if config.graph.enable_edge_labels else "",
                    arrows=arrows,
                )

            # Generate and save the visualization
            Path(vis_path).parent.mkdir(parents=True, exist_ok=True)
            net.save_graph(vis_path)

            print(
                f"Interactive {'3D' if use_3d_vis else '2D'} graph visualization saved to {vis_path}"
            )
            print("Open this file in a web browser to explore the graph")
        except Exception as e:
            raise GraphVisualizationError(f"Failed to create visualization: {e}") from e

    def visualize_3d(self, output_path: Optional[str] = None) -> None:
        """
        Generate a 3D interactive visualization of the graph.

        Args:
            output_path: Path where the HTML visualization will be saved (defaults to config)

        Raises:
            GraphVisualizationError: If visualization creation fails
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

        Args:
            term: The central term for the subgraph
            depth: How many hops to include (1 = immediate neighbors only)

        Returns:
            NetworkX graph object containing the subgraph

        Raises:
            NodeNotFoundError: If the term is not found in the graph
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

        Args:
            term: The central term for the subgraph
            depth: How many hops to include (1 = immediate neighbors only)
            output_path: Path where to save the subgraph (defaults to a generated path)

        Returns:
            Path where the subgraph was saved

        Raises:
            NodeNotFoundError: If the term is not found in the graph
            GraphError: If the export fails
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

        Args:
            min_community_size: Minimum number of terms to be considered a community

        Returns:
            Dictionary mapping community IDs to lists of terms

        Raises:
            GraphError: If community detection fails
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


def main() -> None:
    """
    Demonstrate key functionality of the GraphManager class.
    """
    from word_forge.database.db_manager import DBManager

    # Initialize database and graph managers
    db_manager = DBManager()
    graph_manager = GraphManager(db_manager)

    try:
        # Create database tables if they don't exist
        db_manager._create_tables()

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

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
