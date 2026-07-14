"""
Provides methods for querying graph information and structure.

Encapsulates logic for retrieving node/edge counts, finding related terms,
extracting subgraphs, getting node IDs, and generating graph summaries.
Adheres to Eidosian principles of modularity, clarity, and precision.

Architecture:
    ┌──────────────────┐      ┌──────────────────┐
    │  GraphManager    │◄────►│    GraphQuery    │
    │ (Orchestrator)   │      │ (Information &   │
    └────────┬─────────┘      │ Structure Access)│
             │                └──────────────────┘
             ▼
    ┌──────────────────┐
    │    NetworkX      │
    │  (Graph Object)  │
    └──────────────────┘
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import networkx as nx

# Import necessary components
from word_forge.exceptions import AmbiguousTermError, NodeNotFoundError
from word_forge.graph.graph_assertions import decode_edge_assertions
from word_forge.graph.graph_config import (
    GraphInfoDict,
    RelationshipDimension,
    RelType,
    Term,
    WordId,
    WordTupleDict,
)
from word_forge.parser.linguistics import (
    canonicalize_language_tag,
    normalize_term,
)

# Type hint for the main GraphManager to avoid circular imports
if TYPE_CHECKING:
    from .graph_manager import GraphManager


class GraphQuery:
    """
    Provides methods to query and retrieve information from the graph.

    Offers functionalities like finding related terms, getting node/edge counts,
    looking up terms by ID, generating graph summaries, and extracting subgraphs.

    Attributes:
        manager: Reference to the main GraphManager for state access.
        logger: Logger instance for this module.
    """

    def __init__(self, manager: GraphManager) -> None:
        """
        Initialize the GraphQuery with a reference to the GraphManager.

        Args:
            manager: The orchestrating GraphManager instance.
        """
        self.manager: GraphManager = manager
        self.logger: logging.Logger = logging.getLogger(__name__)
        # Use config from manager for consistency
        self._config = self.manager.config

    def get_node_id(
        self, term: Term, language: Optional[str] = None
    ) -> Optional[WordId]:
        """
        Retrieve the node ID for a given term (case-insensitive).

        Args:
            term: The term (word or phrase) to look up.
            language: Optional BCP 47 tag required to disambiguate homographs.

        Returns:
            Optional[WordId]: The integer ID of the node if found, else None.
        """
        normalized = normalize_term(term)
        if language is not None:
            identity = (normalized, canonicalize_language_tag(language))
            return self.manager._identity_to_id.get(identity)

        candidate_ids = self.manager._term_to_ids.get(normalized, set())
        if len(candidate_ids) > 1:
            languages = sorted(
                str(self.manager.g.nodes[node_id].get("language", "und"))
                for node_id in candidate_ids
            )
            raise AmbiguousTermError(
                f"Term {term!r} exists in multiple languages "
                f"({', '.join(languages)}); provide language=..."
            )
        return next(iter(candidate_ids), None)

    def get_related_terms(
        self,
        term: Term,
        rel_type: Optional[RelType] = None,
        *,
        language: Optional[str] = None,
    ) -> List[Term]:
        """
        Find terms directly related to a given term in the graph.

        Args:
            term: The term for which to find related terms.
            rel_type: Optional filter to return only terms connected by a
                      specific relationship type (e.g., 'synonym').

        Returns:
            List[Term]: A list of related terms (strings).

        Raises:
            NodeNotFoundError: If the input term is not found in the graph.
        """
        start_node_id = self.get_node_id(term, language=language)
        if start_node_id is None:
            raise NodeNotFoundError(f"Term '{term}' not found in the graph.")

        related_terms_list: List[Term] = []
        # Iterate through neighbors of the start node
        for neighbor_id in self.manager.g.neighbors(start_node_id):
            # Get edge data between start_node and neighbor
            # Note: For MultiGraph, this might return multiple edges. Assuming Graph for now.
            edge_data = self.manager.g.get_edge_data(start_node_id, neighbor_id)
            if not edge_data:  # Should not happen for neighbors, but safety check
                continue

            current_rel_type = edge_data.get("relationship")
            relationship_types_value = edge_data.get(
                "relationship_types",
                [current_rel_type] if current_rel_type else [],
            )
            relationship_types = (
                relationship_types_value.split("|")
                if isinstance(relationship_types_value, str)
                else relationship_types_value
            )

            # Apply relationship type filter if provided
            if rel_type is None or (
                any(
                    str(candidate).lower() == rel_type.lower()
                    for candidate in relationship_types
                )
            ):
                # Get the term attribute from the neighbor node
                neighbor_term = self.manager.g.nodes[neighbor_id].get("term")
                if neighbor_term:
                    related_terms_list.append(str(neighbor_term))
                else:
                    self.logger.warning(
                        f"Neighbor node {neighbor_id} of '{term}' is missing 'term' attribute."
                    )

        self.logger.debug(
            f"Found {len(related_terms_list)} related terms for '{term}' (filter: {rel_type})."
        )
        return related_terms_list

    def get_node_count(self) -> int:
        """
        Get the total number of nodes (words) in the graph.

        Returns:
            int: The number of nodes.
        """
        return int(self.manager.g.number_of_nodes())

    def get_edge_count(self) -> int:
        """
        Get the total number of edges (relationships) in the graph.

        Returns:
            int: The number of edges.
        """
        return int(self.manager.g.number_of_edges())

    def get_term_by_id(self, word_id: WordId) -> Optional[Term]:
        """
        Retrieve the term associated with a given node ID.

        Args:
            word_id: The integer ID of the node.

        Returns:
            Optional[Term]: The term string if the node exists and has a 'term'
                            attribute, else None.
        """
        if word_id in self.manager.g.nodes():
            term_attr = self.manager.g.nodes[word_id].get("term")
            return str(term_attr) if term_attr is not None else None
        return None

    def get_graph_info(self) -> GraphInfoDict:
        """
        Retrieve detailed information about the graph structure and content.

        Provides counts, dimensions, sample nodes/relationships, and a list
        of unique relationship types present in the graph.

        Returns:
            GraphInfoDict: A dictionary containing graph metrics and samples.
        """
        nodes = self.get_node_count()
        edges = self.get_edge_count()

        # Sample nodes
        sample_node_ids = list(self.manager.g.nodes())[:5]
        sample_nodes_data: List[WordTupleDict] = []
        for node_id in sample_node_ids:
            node_data = self.manager.g.nodes[node_id]
            term = node_data.get(
                "term", f"ID:{node_id}"
            )  # Provide default if term missing
            sample_nodes_data.append(
                {
                    "id": node_id,
                    "term": str(term),
                    "language": str(node_data.get("language", "und")),
                    "script": str(node_data.get("script", "Zzzz")),
                }
            )

        # Sample relationships (more complex to get terms efficiently)
        sample_relationships_data: List[Dict[str, str]] = []
        edge_count = 0
        for u, v, data in self.manager.g.edges(data=True):
            if edge_count >= 5:
                break
            term_u = str(self.manager.g.nodes[u].get("term", f"ID:{u}"))
            term_v = str(self.manager.g.nodes[v].get("term", f"ID:{v}"))
            rel_type = str(data.get("relationship", "unknown"))
            sample_relationships_data.append(
                {"source": term_u, "target": term_v, "type": rel_type}
            )
            edge_count += 1

        # Get unique relationship types from stored counts
        relationship_types = sorted(list(self.manager._relationship_counts.keys()))

        info: GraphInfoDict = {
            "nodes": nodes,
            "edges": edges,
            "dimensions": self.manager.dimensions,
            "sample_nodes": sample_nodes_data,
            "sample_relationships": sample_relationships_data,
            "relationship_types": relationship_types,
        }
        return info

    def display_graph_summary(self) -> None:
        """
        Print a formatted summary of the graph's statistics to the console.

        Includes node/edge counts, dimensions, and counts of the most common
        relationship types.
        """
        info = self.get_graph_info()
        print("\n--- Graph Summary ---")
        print(f"Nodes: {info['nodes']}")
        print(f"Edges: {info['edges']}")
        print(f"Dimensions: {info['dimensions']}D")

        if self.manager._relationship_counts:
            print("\nRelationship Type Counts:")
            # Sort counts descending for display
            sorted_counts = sorted(
                self.manager._relationship_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            for rel_type, count in sorted_counts[:10]:  # Show top 10
                print(f"  - {rel_type}: {count}")
            if len(sorted_counts) > 10:
                print(f"  ... and {len(sorted_counts) - 10} more types.")
        else:
            print("\nNo relationship types found.")
        print("---------------------\n")

    def get_subgraph(
        self, term: Term, depth: int = 1, *, language: Optional[str] = None
    ) -> nx.Graph:
        """
        Extract a subgraph centered around a specific term up to a given depth.

        Performs a breadth-first search from the term's node.

        Args:
            term: The central term for the subgraph.
            depth: The maximum distance (number of hops) from the central term.

        Returns:
            nx.Graph: A NetworkX graph object representing the subgraph.

        Raises:
            NodeNotFoundError: If the input term is not found in the graph.
            ValueError: If depth is negative.
        """
        if depth < 0:
            raise ValueError("Subgraph depth cannot be negative.")

        start_node_id = self.get_node_id(term, language=language)
        if start_node_id is None:
            raise NodeNotFoundError(f"Term '{term}' not found for subgraph extraction.")

        # Use ego_graph to find all nodes within the specified radius (depth)
        subgraph_nodes = nx.ego_graph(
            self.manager.g, start_node_id, radius=depth
        ).nodes()

        # Create the subgraph from the identified nodes
        subgraph_view = self.manager.g.subgraph(subgraph_nodes)

        # Return a copy to prevent modifications affecting the main graph view
        self.logger.debug(
            f"Extracted subgraph for '{term}' (depth {depth}) with {subgraph_view.number_of_nodes()} nodes."
        )
        return subgraph_view.copy()

    def get_relationships_by_dimension(
        self,
        dimension: RelationshipDimension = "lexical",
        rel_type: Optional[RelType] = None,
        valence_range: Optional[Tuple[float, float]] = None,
    ) -> List[Tuple[Term, Term, RelType, Dict[str, Any]]]:
        """
        Retrieve relationships filtered by dimension and other optional criteria.

        Allows filtering edges based on their assigned dimension (e.g., 'lexical',
        'emotional'), relationship type, and potentially other attributes like
        emotional valence range (if applicable to the dimension).

        Args:
            dimension: The relationship dimension to filter by (default: 'lexical').
            rel_type: Optional specific relationship type to filter by.
            valence_range: Optional tuple (min_valence, max_valence) to filter
                           emotional relationships (applied only if dimension is
                           'emotional' or 'affective').

        Returns:
            List[Tuple[Term, Term, RelType, Dict[str, Any]]]: A list of tuples,
            each containing (source_term, target_term, relationship_type, edge_attributes).
        """
        self.logger.debug(
            f"Querying relationships for dimension '{dimension}' (type: {rel_type}, valence: {valence_range})."
        )
        filtered_relationships: List[Tuple[Term, Term, RelType, Dict[str, Any]]] = []

        for u, v, data in self.manager.g.edges(data=True):
            assertions = decode_edge_assertions(
                data,
                default_source_id=u,
                default_target_id=v,
            )
            for assertion in assertions:
                assertion_dimension = str(
                    assertion.get("dimension", data.get("dimension", "lexical"))
                )
                assertion_type = str(
                    assertion.get("relationship", data.get("relationship", "unknown"))
                )
                if assertion_dimension != dimension:
                    continue
                if rel_type is not None and assertion_type != rel_type:
                    continue

                if valence_range is not None and dimension in {
                    "emotional",
                    "affective",
                }:
                    min_valence, max_valence = valence_range
                    assertion_valence = assertion.get("valence")
                    candidate_valences = [assertion_valence]
                    if not isinstance(assertion_valence, (int, float)):
                        candidate_valences.extend(
                            [
                                self.manager.g.nodes[assertion["source_id"]].get(
                                    "valence"
                                ),
                                self.manager.g.nodes[assertion["target_id"]].get(
                                    "valence"
                                ),
                            ]
                        )
                    if not any(
                        isinstance(value, (int, float))
                        and min_valence <= float(value) <= max_valence
                        for value in candidate_valences
                    ):
                        continue

                assertion_attributes = dict(data)
                assertion_attributes.update(assertion)
                source_id = assertion["source_id"]
                target_id = assertion["target_id"]
                source_term = str(
                    self.manager.g.nodes[source_id].get("term", f"ID:{source_id}")
                )
                target_term = str(
                    self.manager.g.nodes[target_id].get("term", f"ID:{target_id}")
                )
                filtered_relationships.append(
                    (source_term, target_term, assertion_type, assertion_attributes)
                )

        self.logger.debug(
            f"Found {len(filtered_relationships)} relationships matching criteria."
        )
        return filtered_relationships
