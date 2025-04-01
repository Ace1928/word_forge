from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, TypedDict, cast

import networkx as nx

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
    sample_nodes: List[WordTupleDict]
    sample_relationships: List[Dict[str, str]]


# Type aliases for improved readability and type safety
WordId = int
Term = str
RelType = str
WordTuple = Tuple[WordId, Term]
RelationshipTuple = Tuple[WordId, Term, RelType]
GraphData = Tuple[List[WordTuple], List[RelationshipTuple]]


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
    Builds and maintains a lexical/semantic graph from database records.

    This class creates a networkx graph representation where:
    - Nodes represent words (with word_id as node identifier)
    - Edges represent relationships between words (synonym, antonym, etc.)
    - Node attributes include the term text
    - Edge attributes include the relationship type
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

        Raises:
            GraphError: If a database error occurs during fetching
        """
        self.g.clear()
        self._term_to_id.clear()

        words, relationships = self._fetch_data()

        # Add nodes and build term->id mapping
        for word_id, term in words:
            self.g.add_node(word_id, term=term)
            self._term_to_id[term.lower()] = word_id

        # Add edges between related words
        for word_id, related_term, rel_type in relationships:
            related_id = self._term_to_id.get(related_term.lower())
            if related_id:
                self.g.add_edge(word_id, related_id, relationship_type=rel_type)

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
        except Exception as e:
            raise GraphError(f"Failed to load graph from {gexf_path}: {e}") from e

    def update_graph(self) -> int:
        """
        Update existing graph with new words from the database.

        Instead of clearing the graph, this method:
        1. Identifies words in the database not yet in the graph
        2. Adds them as new nodes
        3. Adds relationships involving the new words

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

        # Add relationships for new words
        for word_id, related_term, rel_type in all_relationships:
            related_id = self._term_to_id.get(related_term.lower())
            if (
                related_id
                and (word_id not in current_ids or related_id not in current_ids)
                and not self.g.has_edge(word_id, related_id)
            ):
                self.g.add_edge(word_id, related_id, relationship_type=rel_type)

        return new_word_count

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

                # Add sample words from config
                for term, definition, pos in config.graph.sample_words:
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

            return {
                "nodes": self.g.number_of_nodes(),
                "edges": self.g.number_of_edges(),
                "sample_nodes": sample_nodes,
                "sample_relationships": sample_relationships,
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

        print(f"Graph contains {info['nodes']} nodes and {info['edges']} edges")

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
    ) -> None:
        """
        Generate an interactive HTML visualization of the graph.

        Creates an HTML file with a fully interactive network graph visualization that can be
        viewed in any browser. Nodes can be dragged, zoomed, and explored interactively.

        Args:
            output_path: Path where the HTML visualization will be saved (defaults to config)
            height: Height of the visualization frame (CSS format, defaults to config)
            width: Width of the visualization frame (CSS format, defaults to config)

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
        vis_path = output_path or str(config.graph.get_vis_path())
        vis_height = height or config.graph.vis_height
        vis_width = width or config.graph.vis_width

        try:
            # Create a pyvis network with the same dimensions as our graph
            net = Network(
                height=vis_height, width=vis_width, notebook=False, directed=False
            )

            # Add nodes to the network
            for node_id, node_attrs in self.g.nodes(data=True):
                label = node_attrs.get("term", f"Node {node_id}")
                title = f"ID: {node_id}<br>Term: {label}"
                net.add_node(node_id, label=label, title=title)

            # Add edges to the network
            for source, target, edge_attrs in self.g.edges(data=True):
                rel_type = edge_attrs.get("relationship_type", "related")
                title = f"Relationship: {rel_type}"
                net.add_edge(source, target, title=title)

            # Generate and save the visualization
            Path(vis_path).parent.mkdir(parents=True, exist_ok=True)
            net.save_graph(vis_path)

            print(f"Interactive graph visualization saved to {vis_path}")
            print("Open this file in a web browser to explore the graph")
        except Exception as e:
            raise GraphVisualizationError(f"Failed to create visualization: {e}") from e


def main() -> None:
    """
    Demonstrate key functionality of the GraphManager class.
    """
    from word_forge.database.db_manager import DBManager

    # Initialize database and graph managers
    db_path = "word_forge.sqlite"
    print(f"Using database: {db_path}")
    db_manager = DBManager(db_path=db_path)
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
        example_term = "example"
        try:
            related_terms = graph_manager.get_related_terms(example_term)
            print(f"\nTerms related to '{example_term}': {related_terms}")

            # Filter by relationship type
            synonyms = graph_manager.get_related_terms(example_term, rel_type="synonym")
            print(f"Synonyms of '{example_term}': {synonyms}")
        except NodeNotFoundError as e:
            print(f"Warning: {e}")

        # Save the graph to a file
        output_path = "data/lexical_graph.gexf"
        print(f"\nSaving graph to {output_path}")
        graph_manager.save_to_gexf(output_path)
        print(f"Graph saved successfully to {output_path}")

        # Demonstrate graph update with new data
        print("\nUpdating graph with any new words from database...")
        new_words = graph_manager.update_graph()
        if new_words > 0:
            print(f"Added {new_words} new words to the graph")
        else:
            print("No new words found in the database")

        # Load the graph from file
        print(f"\nLoading graph from {output_path}")
        graph_manager.load_from_gexf(output_path)
        print("Graph loaded successfully")

        # Generate interactive visualization
        try:
            vis_path = "data/graph_visualization.html"
            print("\nGenerating interactive visualization...")
            graph_manager.visualize(output_path=vis_path)
        except ImportError as e:
            print(f"Note: {e}")
        except Exception as e:
            print(f"Warning: Could not generate visualization: {e}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
