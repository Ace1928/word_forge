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
from typing import TYPE_CHECKING, Dict, List, Set, Tuple, cast

# Import necessary components (adjust paths as needed)
from word_forge.exceptions import GraphDataError, GraphError
from word_forge.graph.graph_config import RelationshipTuple, RelType, WordId, WordTuple
from word_forge.relationships import RelationshipProperties

# Type hint for the main GraphManager to avoid circular imports
if TYPE_CHECKING:
    from .graph_manager import GraphManager


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

    def build_graph(self) -> None:
        """
        Construct the graph from the database, replacing any existing graph.

        Fetches all words and relationships, adds them as nodes and edges
        to the manager's graph object, builds the term-to-ID mapping,
        calculates relationship counts, and triggers a full layout computation.

        Raises:
            GraphDataError: If fetching data from the database fails.
            GraphError: For other graph construction issues.
        """
        self.logger.info("Initiating full graph build process.")
        # Clear existing graph state held by the manager
        self.manager.g.clear()
        self.manager._term_to_id.clear()
        self.manager._positions.clear()
        self.manager._relationship_counts.clear()  # Use clear() for consistency

        try:
            words, relationships = self._fetch_data()
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
        for word_id, term in words:
            # Ensure term is not None or empty before adding
            if term:
                # Add node with term and ID attributes for consistency
                self.manager.g.add_node(word_id, term=term, id=word_id)
                # Store lowercase term for case-insensitive lookup
                self.manager._term_to_id[term.lower()] = word_id
            else:
                self.logger.warning(
                    f"Skipping node with ID {word_id} due to missing term."
                )

        # --- Edge Addition ---
        for word_id, related_term, rel_type in relationships:
            # Validate source node exists
            if word_id not in self.manager.g:
                self.logger.debug(f"Skipping edge from non-existent node ID {word_id}.")
                continue

            # Find target node ID (case-insensitive)
            related_id = self.manager._term_to_id.get(related_term.lower())

            # Validate target node exists
            if related_id is None or related_id not in self.manager.g:
                self.logger.debug(
                    f"Skipping edge to non-existent term '{related_term}'."
                )
                continue

            # Prevent self-loops unless explicitly allowed by config (future)
            if word_id == related_id:
                self.logger.debug(f"Skipping self-loop for node ID {word_id}.")
                continue

            # Add edge with calculated properties
            self._add_relationship_edge(word_id, related_id, rel_type)

        self.logger.info(
            f"Graph built: {self.manager.g.number_of_nodes()} nodes, {self.manager.g.number_of_edges()} edges."
        )

        # Delegate layout computation via the manager
        if self.manager.g.number_of_nodes() > 0:
            self.logger.info("Triggering full graph layout computation.")
            self.manager.layout.compute_layout()
        else:
            self.logger.info("Graph is empty, skipping layout computation.")

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
            return self.manager.g.number_of_nodes()

        self.logger.info("Initiating incremental graph update.")
        try:
            all_words, all_relationships = self._fetch_data()
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
        new_edges_added_count = 0

        # --- Add New Nodes ---
        for word_id, term in all_words:
            if term and word_id not in current_node_ids:
                self.manager.g.add_node(word_id, term=term, id=word_id)
                self.manager._term_to_id[term.lower()] = word_id
                new_nodes_added.append(word_id)

        new_node_count = len(new_nodes_added)

        # --- Add New Edges ---
        for word_id, related_term, rel_type in all_relationships:
            related_id = self.manager._term_to_id.get(related_term.lower())
            # Ensure both nodes exist in the potentially updated graph
            if (
                word_id in self.manager.g
                and related_id is not None
                and related_id in self.manager.g
            ):
                # Check if the edge (or its reverse for undirected) already exists
                if not self.manager.g.has_edge(word_id, related_id):
                    # Prevent self-loops
                    if word_id == related_id:
                        continue
                    self._add_relationship_edge(word_id, related_id, rel_type)
                    new_edges_added_count += 1

        # --- Post-Update Actions ---
        if new_node_count > 0 or new_edges_added_count > 0:
            self.logger.info(
                f"Graph updated: Added {new_node_count} nodes and {new_edges_added_count} edges."
            )
            # Delegate incremental layout update only if nodes were added
            if new_node_count > 0:
                self.logger.info("Triggering incremental layout update.")
                self.manager.layout.update_layout_incrementally(new_nodes_added)
            else:
                self.logger.info(
                    "Only edges added, layout update may not be necessary depending on algorithm."
                )
                # Optionally trigger full re-layout if edge changes significantly impact structure
                # self.manager.layout.compute_layout()
        else:
            self.logger.info("Graph update: No new nodes or edges detected.")

        return new_node_count

    def _add_relationship_edge(
        self, source_id: WordId, target_id: WordId, rel_type: RelType
    ) -> None:
        """
        Adds a single relationship edge to the graph with calculated attributes.

        Internal helper method to encapsulate edge creation logic, including
        determining dimension, properties, and updating counts.

        Args:
            source_id: The ID of the source node.
            target_id: The ID of the target node.
            rel_type: The type of the relationship.
        """
        # Use manager's helper to get properties and determine dimension
        # Provide a default RelationshipProperties if type is unknown
        rel_props: RelationshipProperties = self.manager._get_relationship_properties(
            rel_type
        )
        dimension = self.manager._determine_dimension(rel_type)

        # Safely get term text for title, providing defaults
        source_term_text = self.manager.g.nodes[source_id].get(
            "term", f"ID:{source_id}"
        )
        target_term_text = self.manager.g.nodes[target_id].get(
            "term", f"ID:{target_id}"
        )

        # Construct edge attributes
        edge_attrs = {
            "relationship": rel_type,
            "weight": rel_props.get("weight", 1.0),  # Default weight if missing
            "color": rel_props.get(
                "color", self._config.relationship_colors.get("default", "#aaaaaa")
            ),  # Default color
            "bidirectional": rel_props.get(
                "bidirectional", False
            ),  # Default directionality
            "dimension": dimension,
            # Ensure title generation handles potential None terms gracefully
            "title": f"{rel_type}: {source_term_text or '?'} {'↔' if rel_props.get('bidirectional', False) else '→'} {target_term_text or '?'}",
        }

        # Add the edge to the manager's graph
        self.manager.g.add_edge(source_id, target_id, **edge_attrs)

        # Update relationship counts held by the manager
        # Use Counter's update method for clarity
        self.manager._relationship_counts.update([rel_type])

    def _fetch_data(self) -> Tuple[List[WordTuple], List[RelationshipTuple]]:
        """
        Fetch words and relationships from the database.

        Uses the manager's database connection context manager for safe access.

        Returns:
            Tuple containing a list of word tuples (id, term) and a list of
            relationship tuples (word_id, related_term, relationship_type).

        Raises:
            GraphDataError: If database tables are missing or query fails.
            sqlite3.Error: For underlying database connection or query errors.
        """
        self.logger.debug("Fetching graph data from database.")
        words: List[WordTuple] = []
        relationships: List[RelationshipTuple] = []

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
                cursor.execute(self._config.sql_templates["fetch_all_words"])
                words_raw = cursor.fetchall()
                words = [
                    cast(WordTuple, (row["id"], row["term"]))
                    for row in words_raw
                    if row["id"] is not None and row["term"] is not None
                ]
                self.logger.debug(f"Fetched {len(words)} valid word entries.")

                # Verify 'relationships' table exists
                cursor.execute(self._config.sql_templates["check_relationships_table"])
                if not cursor.fetchone():
                    self.logger.warning(
                        "Database table 'relationships' not found. Graph will have no edges."
                    )
                    # Return words only if relationships table is missing
                    return words, []

                # Fetch relationships: Ensure all parts are not NULL
                cursor.execute(self._config.sql_templates["fetch_all_relationships"])
                relationships_raw = cursor.fetchall()
                relationships = [
                    cast(
                        RelationshipTuple,
                        (row["word_id"], row["related_term"], row["relationship_type"]),
                    )
                    for row in relationships_raw
                    if row["word_id"] is not None
                    and row["related_term"] is not None
                    and row["relationship_type"] is not None
                ]
                self.logger.debug(
                    f"Fetched {len(relationships)} valid relationship entries."
                )

                return words, relationships
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
