from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, cast

import networkx as nx

from word_forge.database.db_manager import DBManager

# Type aliases for improved readability and type safety
WordId = int
Term = str
RelType = str
WordTuple = Tuple[WordId, Term]
RelationshipTuple = Tuple[WordId, Term, RelType]
GraphData = Tuple[List[WordTuple], List[RelationshipTuple]]


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

    def build_graph(self) -> None:
        """
        Clear existing in-memory graph and rebuild from database.

        This method:
        1. Clears any existing graph data
        2. Fetches words and relationships from the database
        3. Adds each word as a node with its term as an attribute
        4. Adds edges between words based on their relationships

        Raises:
            sqlite3.Error: If a database error occurs during fetching
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

    def save_to_gexf(self, path: str = "data/word_graph.gexf") -> None:
        """
        Save the current graph to a GEXF format file.

        Args:
            path: Destination file path

        Raises:
            ValueError: If the graph is empty
            IOError: If writing to the file fails
        """
        if self.g.number_of_nodes() == 0:
            raise ValueError("Cannot save an empty graph")

        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        nx.write_gexf(self.g, path)

    def load_from_gexf(self, path: str = "data/word_graph.gexf") -> None:
        """
        Load a graph from a GEXF format file.

        Args:
            path: Source file path

        Raises:
            FileNotFoundError: If the file doesn't exist
            nx.NetworkXError: If the file format is invalid
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Graph file not found: {path}")

        self.g = nx.read_gexf(path)

        # Rebuild term_to_id mapping from loaded graph
        self._term_to_id.clear()
        for node_id, attrs in self.g.nodes(data=True):
            if "term" in attrs:
                self._term_to_id[attrs["term"].lower()] = cast(WordId, node_id)

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
            sqlite3.Error: If database access fails
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
        """
        # Check if the database has any words
        words, _ = self._fetch_data()
        if words:
            return False

        # Database is empty, add sample data
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.cursor()

                # Add sample words
                sample_words = [
                    ("example", "a representative form or pattern", "noun"),
                    ("test", "a procedure for critical evaluation", "noun"),
                    (
                        "sample",
                        "a small part of something intended as representative of the whole",
                        "noun",
                    ),
                    ("word", "a unit of language", "noun"),
                    (
                        "graph",
                        "a diagram showing the relation between variable quantities",
                        "noun",
                    ),
                ]

                for term, definition, pos in sample_words:
                    cursor.execute(
                        "INSERT INTO words (term, definition, pos) VALUES (?, ?, ?)",
                        (term, definition, pos),
                    )

                # Get the inserted word IDs
                cursor.execute("SELECT id, term FROM words")
                words = cursor.fetchall()

                # Create a mapping of term to ID
                word_id_map = {term.lower(): id for id, term in words}

                # Add sample relationships
                sample_relationships = [
                    ("example", "sample", "synonym"),
                    ("example", "test", "related"),
                    ("test", "sample", "related"),
                    ("word", "term", "synonym"),
                    ("graph", "diagram", "synonym"),
                ]

                for term1, term2, rel_type in sample_relationships:
                    if term1.lower() in word_id_map:
                        cursor.execute(
                            "INSERT INTO relationships (word_id, related_term, relationship_type) VALUES (?, ?, ?)",
                            (word_id_map[term1.lower()], term2, rel_type),
                        )

                conn.commit()
                return True

        except sqlite3.Error as e:
            logging.error(f"Failed to add sample data: {e}")
            return False

    def _fetch_data(self) -> GraphData:
        """
        Retrieve words and their relationships from the database.

        Returns:
            Tuple containing:
            - List of (word_id, term) tuples
            - List of (word_id, related_term, relationship_type) tuples

        Raises:
            sqlite3.Error: If database access fails
        """
        words: List[WordTuple] = []
        relationships: List[RelationshipTuple] = []

        try:
            db_path = Path(self.db_manager.db_path)
            if not db_path.exists():
                logging.warning(f"Database file {db_path} does not exist")
                return words, relationships

            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()

                # Check if words table exists
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='words'"
                )
                if not cursor.fetchone():
                    logging.warning("Words table does not exist in the database")
                    return words, relationships

                # Fetch all words
                cursor.execute("SELECT id, term FROM words")
                words = cursor.fetchall()

                # Check if relationships table exists
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='relationships'"
                )
                if not cursor.fetchone():
                    logging.warning(
                        "Relationships table does not exist in the database"
                    )
                    return words, relationships

                # Fetch all relationships
                cursor.execute(
                    "SELECT word_id, related_term, relationship_type FROM relationships"
                )
                relationships = cursor.fetchall()

        except sqlite3.Error as e:
            error_msg = f"Database error while fetching graph data: {e}"
            logging.error(error_msg)
            raise sqlite3.Error(error_msg)

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
            ValueError: If the term is not found in the graph
        """
        term_lower = term.lower()
        if term_lower not in self._term_to_id:
            raise ValueError(f"Term not found in graph: {term}")

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

    def display_graph_summary(self) -> None:
        """
        Display a summary of the graph structure.

        Prints key statistics and sample nodes/relationships.
        """
        if self.g.number_of_nodes() == 0:
            print("Graph is empty")
            return

        print(
            f"Graph contains {self.g.number_of_nodes()} nodes and {self.g.number_of_edges()} edges"
        )

        # Show sample of nodes (up to 5)
        sample_nodes = list(self.g.nodes(data=True))[:5]
        print("\nSample nodes:")
        for node_id, attrs in sample_nodes:
            term = attrs.get("term", "Unknown")
            print(f"  - Node {node_id}: {term}")

        # Show sample of relationships (up to 5)
        sample_edges = list(self.g.edges(data=True))[:5]
        if sample_edges:
            print("\nSample relationships:")
            for n1, n2, attrs in sample_edges:
                term1 = self.g.nodes[n1].get("term", "Unknown")
                term2 = self.g.nodes[n2].get("term", "Unknown")
                rel_type = attrs.get("relationship_type", "related")
                print(f"  - {term1} is {rel_type} to {term2}")

    def visualize(
        self,
        output_path: str = "data/graph_vis.html",
        height: str = "600px",
        width: str = "100%",
    ) -> None:
        """
        Generate an interactive HTML visualization of the graph.

        Creates an HTML file with a fully interactive network graph visualization that can be
        viewed in any browser. Nodes can be dragged, zoomed, and explored interactively.

        Args:
            output_path: Path where the HTML visualization will be saved
            height: Height of the visualization frame (CSS format)
            width: Width of the visualization frame (CSS format)

        Raises:
            ImportError: If pyvis is not installed
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

        # Create a pyvis network with the same dimensions as our graph
        net = Network(height=height, width=width, notebook=False, directed=False)

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
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(output_path)

        print(f"Interactive graph visualization saved to {output_path}")
        print("Open this file in a web browser to explore the graph")


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
        except ValueError as e:
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
