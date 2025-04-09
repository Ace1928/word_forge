"""
Demonstration of GraphManager functionality.
"""

import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, cast  # Import Optional

import networkx as nx

from word_forge.database.database_manager import DBManager
from word_forge.exceptions import NodeNotFoundError
from word_forge.graph.graph_manager import GraphError, GraphManager

# Define type aliases for clarity
ValenceAnalysisResult = Dict[str, Union[float, int, List[Tuple[str, float]]]]


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
        graph_demo()
        ```
    """

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
        valence_analysis: ValenceAnalysisResult = cast(
            ValenceAnalysisResult,
            graph_manager.analyze_emotional_valence_distribution(),
        )

        if (
            isinstance(valence_analysis.get("count"), int)
            and cast(int, valence_analysis.get("count", 0))
            > 0  # Cast count to int before comparison
        ):
            mean_valence = cast(float, valence_analysis.get("mean", 0.0))
            valence_range = cast(List[float], valence_analysis.get("range", [0.0, 0.0]))
            print(f"Found {valence_analysis['count']} terms with emotional valence")
            print(
                f"Average valence: {mean_valence:.2f} (range: {valence_range[0]:.2f} to {valence_range[1]:.2f})"
            )

            top_positive = cast(
                List[Tuple[str, float]], valence_analysis.get("top_positive", [])
            )
            if top_positive:
                print("\nMost positive terms:")
                for term, val in top_positive:
                    print(f"  - {term}: {val:.2f}")

            top_negative = cast(
                List[Tuple[str, float]], valence_analysis.get("top_negative", [])
            )
            if top_negative:
                print("\nMost negative terms:")
                for term, val in top_negative:
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

            for source, target, rel_type, weight in sample_emotional_relations:
                # Use public methods or handle node creation safely
                source_id = graph_manager.get_node_id(source)
                if source_id is None:
                    source_id = graph_manager.add_word_node(
                        source,
                        attributes={
                            "valence": (0.7 if source in ["joy", "happiness"] else -0.7)
                        },
                    )

                target_id = graph_manager.get_node_id(target)
                if target_id is None:
                    target_id = graph_manager.add_word_node(
                        target,
                        attributes={
                            "valence": (0.8 if target in ["happiness"] else -0.8)
                        },
                    )

                if source_id is not None and target_id is not None:
                    graph_manager.add_relationship(
                        source_id,
                        target_id,
                        relationship=rel_type,
                        dimension="emotional",
                        weight=weight,
                        color="#ff0000",  # Red for emotional relationships
                    )
                else:
                    print(
                        f"Warning: Could not add relationship between {source} and {target} due to missing nodes."
                    )

            print("Sample emotional relationships added")

        # Phase 4: Meta-Emotional Patterns
        print("\n=== PHASE 4: META-EMOTIONAL PATTERNS ===")

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

            print("\nAdding sample meta-emotional patterns for demonstration...")
            sample_meta_relations = [
                ("anxiety", "fear", "meta_emotion", 0.8),
                ("regret", "sadness", "evokes", 0.7),
                ("awe", "surprise", "emotional_component", 0.9),
            ]

            for source, target, rel_type, weight in sample_meta_relations:
                # Use public methods or handle node creation safely
                source_id = graph_manager.get_node_id(source)
                if source_id is None:
                    source_id = graph_manager.add_word_node(
                        source, attributes={"valence": -0.3}
                    )

                target_id = graph_manager.get_node_id(target)
                if target_id is None:
                    target_id = graph_manager.add_word_node(
                        target, attributes={"valence": -0.5}
                    )

                if source_id is not None and target_id is not None:
                    graph_manager.add_relationship(
                        source_id,
                        target_id,
                        relationship=rel_type,
                        dimension="emotional",
                        weight=weight,
                        color="#800080",  # Purple for meta-emotional
                    )
                else:
                    print(
                        f"Warning: Could not add meta relationship between {source} and {target} due to missing nodes."
                    )

            print("Sample meta-emotional patterns added")

        # Phase 5: Emotional Transitions
        print("\n=== PHASE 5: EMOTIONAL TRANSITIONS ===")

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

        print("Integrating emotional contexts...")

        clinical_context = {
            "professional": 0.9,
            "analytical": 0.8,
            "detached": 0.6,
            "compassionate": 0.5,
        }

        literary_context = {
            "expressive": 0.9,
            "narrative": 0.8,
            "dramatic": 0.7,
            "metaphorical": 0.6,
        }

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

            top_positive_terms = cast(
                List[Tuple[str, float]], valence_analysis.get("top_positive", [])
            )
            top_negative_terms = cast(
                List[Tuple[str, float]], valence_analysis.get("top_negative", [])
            )
            emotional_terms: List[str] = [
                t[0] for t in top_positive_terms + top_negative_terms
            ]

            context_term: str
            if emotional_terms:
                context_term = emotional_terms[0]
            else:
                context_term = "anxiety"

            print(
                f"\nExtracting emotional subgraph for '{context_term}' with clinical context..."
            )
            emotional_subgraph: nx.Graph = graph_manager.get_emotional_subgraph(
                str(context_term), depth=2, context="clinical"
            )

            print(
                f"Extracted emotional subgraph with {emotional_subgraph.number_of_nodes()} nodes "
                f"and {emotional_subgraph.number_of_edges()} emotional relationships"  # Use number_of_edges() method
            )
        except Exception as e:
            print(f"Note: Context integration skipped: {e}")

        # Phase 8: Visualization
        print("\n=== PHASE 8: VISUALIZATION ===")
        vis_path_emotional: Optional[str] = None  # Define with Optional[str]

        try:
            vis_path_2d = "data/graph_visualization_2d.html"
            print("\nGenerating 2D interactive visualization...")
            graph_manager.visualize(output_path=vis_path_2d)

            vis_path_3d = "data/graph_visualization_3d.html"
            print("\nGenerating 3D interactive visualization...")
            graph_manager.visualize_3d(output_path=vis_path_3d)

            if "emotional" in relationship_analysis.get("dimensions", {}):
                vis_path_emotional = "data/emotional_graph.html"
                print("\nGenerating emotional relationships visualization...")
                # Ensure vis_path_emotional is passed correctly
                graph_manager.visualize(
                    output_path=vis_path_emotional, dimensions=["emotional"]
                )

            print("\nVisualizations saved:")
            print(f"  - 2D: {vis_path_2d}")
            print(f"  - 3D: {vis_path_3d}")
            if vis_path_emotional:
                print(f"  - Emotional: {vis_path_emotional}")
            print("Open these files in a web browser to explore the graph")

        except ImportError as e:
            print(f"Note: {e}")
        except Exception as e:
            print(f"Warning: Could not generate visualization: {e}")

        output_path = "data/lexical_graph.gexf"
        print(f"\nSaving complete graph to {output_path}")
        graph_manager.save_to_gexf(output_path)
        print(f"Graph saved successfully to {output_path}")

        try:
            print(f"\nExtracting subgraph for '{example_term}'...")
            subgraph_path = graph_manager.export_subgraph(example_term, depth=2)
            print(f"Subgraph exported to {subgraph_path}")
        except NodeNotFoundError:
            print(
                f"Warning: Could not extract subgraph for '{example_term}' (term not found)"
            )

        elapsed_time = time.time() - start_time
        print(f"\nDemonstration completed in {elapsed_time:.2f} seconds")

    except GraphError as e:
        print(f"Graph error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
    finally:
        db_manager.close()


if __name__ == "__main__":
    graph_demo()
