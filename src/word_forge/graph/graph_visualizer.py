"""
Manages the generation of interactive graph visualizations.

Encapsulates logic for creating 2D and 3D visualizations using libraries
like Pyvis and Plotly. Handles configuration of visual elements (nodes, edges,
layout) based on GraphConfig settings. Adheres to Eidosian principles of
modularity, clarity, and effective communication.

Architecture:
    ┌──────────────────┐      ┌────────────────────┐
    │  GraphManager    │◄────►│  GraphVisualizer   │
    │ (Orchestrator)   │      │ (Plotting & Config)│
    └────────┬─────────┘      └─────────┬──────────┘
             │                          │
             ▼                          ▼
    ┌──────────────────┐      ┌────────────────────┐
    │    GraphLayout   │      │ Visualization Libs │
    │  (Positions)     │      │ (Pyvis, Plotly)    │
    └──────────────────┘      └────────────────────┘
"""

from __future__ import annotations

import json
import logging
import math
import traceback
import webbrowser
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Final, List, Optional, Union, cast

import networkx as nx

# Optional dependencies for visualization
try:
    from pyvis.network import Network as PyvisNetwork  # type: ignore[import-untyped]

    _pyvis_available = True
except ImportError:
    _pyvis_available = False
    PyvisNetwork = None  # Define for type checking

try:
    import plotly.graph_objects as go  # type: ignore[import-untyped]

    _plotly_available = True
except ImportError:
    _plotly_available = False
    go = None  # Define for type checking

_VISUALIZATION_INSTALL_HINT = 'pip install "word_forge[visualization]"'

# Emotional dimension thresholds for color classification
# These can be adjusted to tune sensitivity of emotional coloring
VALENCE_HIGH_THRESHOLD: Final[float] = 0.3  # Above this = positive
VALENCE_LOW_THRESHOLD: Final[float] = -0.3  # Below this = negative
AROUSAL_HIGH_THRESHOLD: Final[float] = 0.5  # Above this = high arousal


def _pipe_values(value: object) -> List[str]:
    """Return stable, non-empty values from a pipe-delimited attribute."""

    if value is None:
        return []
    return sorted(
        {part.strip() for part in str(value).split("|") if part.strip()},
        key=str.casefold,
    )


def _finite_float(value: object, default: float) -> float:
    """Coerce a finite numeric value, returning ``default`` on bad metadata."""

    try:
        normalized = float(str(value))
    except (TypeError, ValueError, OverflowError):
        return default
    return normalized if math.isfinite(normalized) else default


# Import necessary components
from word_forge.exceptions import GraphVisualizationError, NodeNotFoundError
from word_forge.graph.graph_assertions import decode_edge_assertions
from word_forge.graph.graph_config import (
    PositionDict,  # Ensure PositionDict is imported
)
from word_forge.graph.graph_config import (
    ColorHex,
    RelationshipDimension,
    WordId,
)
from word_forge.graph.graph_html import (
    GraphViewerMetadata,
    atomic_write_graph_viewer,
    render_graph_viewer,
)

# Type hint for the main GraphManager to avoid circular imports
if TYPE_CHECKING:
    from .graph_manager import GraphManager


class GraphVisualizer:
    """
    Generates interactive 2D and 3D visualizations of the knowledge graph.

    Uses Pyvis for 2D visualizations and Plotly for 3D visualizations.
    Configures node size, color, edge properties, and layout based on the
    GraphManager's state and configuration.

    Attributes:
        manager (GraphManager): Reference to the main GraphManager for state access.
        logger (logging.Logger): Logger instance for this module.
        _config (GraphConfig): Reference to the graph configuration object.
    """

    def __init__(self, manager: GraphManager) -> None:
        """
        Initialize the GraphVisualizer with a reference to the GraphManager.

        Args:
            manager (GraphManager): The orchestrating GraphManager instance.
        """
        self.manager: GraphManager = manager
        self.logger: logging.Logger = logging.getLogger(__name__)
        # Use config from manager for consistency
        self._config = self.manager.config

    def visualize(
        self,
        output_path: Optional[str] = None,
        height: Optional[
            str
        ] = None,  # Pyvis expects string height/width (e.g., "800px")
        width: Optional[str] = None,
        use_3d: Optional[bool] = None,  # Explicitly choose 2D/3D
        dimensions_filter: Optional[List[RelationshipDimension]] = None,
        open_in_browser: bool = False,  # Option to automatically open
        *,
        focus_term: Optional[str] = None,
        focus_language: Optional[str] = None,
        depth: int = 1,
        max_nodes: Optional[int] = None,
        max_edges: Optional[int] = None,
    ) -> None:
        """
        Generate an interactive graph visualization (2D default, optionally 3D).

        Creates an HTML file containing the visualization. Uses Pyvis for 2D
        and Plotly for 3D. Filters graph elements based on provided dimensions.

        Args:
            output_path (Optional[str]): Path to save the HTML file. Defaults to config path.
            height (Optional[str]): Height of the visualization canvas (e.g., "800px"). Defaults to config.
            width (Optional[str]): Width of the visualization canvas (e.g., "1200px"). Defaults to config.
            use_3d (Optional[bool]): If True, generate a 3D plot using Plotly. If False or None,
                                     generate a 2D plot using Pyvis (respecting manager.dimensions if 3D).
            dimensions_filter (Optional[List[RelationshipDimension]]): List of relationship dimensions to include.
                                                                       If None, includes dimensions specified in config.active_dimensions.
            open_in_browser (bool): If True, automatically opens the generated HTML file.
            focus_term: Optional term whose bounded neighborhood should be shown.
            focus_language: Optional BCP 47 tag used to resolve the focus term.
            depth: Maximum graph distance from the focus term.
            max_nodes: Optional render-node limit overriding configuration.
            max_edges: Optional render-edge limit overriding configuration.

        Raises:
            GraphVisualizationError: If visualization libraries are missing or
                                     if generation fails.
            GraphError: If the graph is empty or positions are missing.
        """
        is_3d = use_3d if use_3d is not None else (self.manager.dimensions == 3)

        if is_3d:
            self.visualize_3d(
                output_path,
                dimensions_filter,
                open_in_browser,
                focus_term=focus_term,
                focus_language=focus_language,
                depth=depth,
                max_nodes=max_nodes,
                max_edges=max_edges,
            )
        else:
            self.visualize_2d(
                output_path,
                height,
                width,
                dimensions_filter,
                open_in_browser,
                focus_term=focus_term,
                focus_language=focus_language,
                depth=depth,
                max_nodes=max_nodes,
                max_edges=max_edges,
            )

    def visualize_2d(
        self,
        output_path: Optional[str] = None,
        height: Optional[str] = None,
        width: Optional[str] = None,
        dimensions_filter: Optional[List[RelationshipDimension]] = None,
        open_in_browser: bool = False,
        *,
        focus_term: Optional[str] = None,
        focus_language: Optional[str] = None,
        depth: int = 1,
        max_nodes: Optional[int] = None,
        max_edges: Optional[int] = None,
    ) -> None:
        """
        Generate an interactive 2D graph visualization using Pyvis.

        Args:
            output_path (Optional[str]): Path to save the HTML file. Defaults to config path.
            height (Optional[str]): Height of the visualization canvas (e.g., "800px"). Defaults to config.
            width (Optional[str]): Width of the visualization canvas (e.g., "1200px"). Defaults to config.
            dimensions_filter (Optional[List[RelationshipDimension]]): List of relationship dimensions to include.
                                                                       If None, includes dimensions specified in config.active_dimensions.
            open_in_browser (bool): If True, automatically opens the generated HTML file.

        Raises:
            GraphVisualizationError: If Pyvis is not installed or generation fails.
            GraphError: If the graph is empty or 2D positions are missing.
        """
        if not _pyvis_available:
            self.logger.error("Pyvis library is required for 2D visualization.")
            self.logger.error(
                f"Install with the visualization extras: {_VISUALIZATION_INSTALL_HINT}"
            )
            raise GraphVisualizationError(
                "Missing 'pyvis' library. Install the visualization extras to enable 2D graph rendering."
            )

        if self.manager.g.number_of_nodes() == 0:
            raise GraphVisualizationError("Cannot visualize an empty graph.")

        graph_to_visualize = self._select_graph(
            dimensions_filter,
            focus_term=focus_term,
            focus_language=focus_language,
            depth=depth,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )
        if graph_to_visualize.number_of_nodes() == 0:
            raise GraphVisualizationError(
                "No graph nodes match the selected focus and dimensions."
            )
        node_positions = self._positions_for_graph(graph_to_visualize, dimensions=2)

        vis_height = height or f"{self._config.vis_height}px"
        vis_width = width or "100%"
        save_path = self._resolve_output_path(
            output_path or self._config.visualization_path,
            "graph_2d.html",
        )

        self.logger.info(f"Generating 2D visualization (Pyvis) to: {save_path}")

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise GraphVisualizationError(
                f"Could not create directory for visualization: {save_path.parent}", e
            ) from e

        net = PyvisNetwork(
            height=vis_height,
            width=vis_width,
            directed=graph_to_visualize.is_directed(),
            notebook=False,
            neighborhood_highlight=False,
            bgcolor="#222222",
            font_color="white",
            cdn_resources="in_line",
        )

        # Configure Pyvis network appearance and add nodes/edges
        self._configure_pyvis_network(net, graph_to_visualize, node_positions)

        try:
            generated_html = str(net.generate_html(notebook=False))
            document = render_graph_viewer(
                generated_html,
                GraphViewerMetadata(
                    title="Lexical connection graph",
                    total_nodes=int(
                        graph_to_visualize.graph.get(
                            "word_forge_total_nodes",
                            graph_to_visualize.number_of_nodes(),
                        )
                    ),
                    total_edges=int(
                        graph_to_visualize.graph.get(
                            "word_forge_total_edges",
                            graph_to_visualize.number_of_edges(),
                        )
                    ),
                    rendered_nodes=graph_to_visualize.number_of_nodes(),
                    rendered_edges=graph_to_visualize.number_of_edges(),
                    height=vis_height,
                ),
            )
            atomic_write_graph_viewer(save_path, document)
            self.logger.info(f"2D visualization saved successfully to {save_path}")

            if open_in_browser:
                try:
                    webbrowser.open(f"file://{str(save_path.resolve())}")
                except Exception as wb_err:
                    self.logger.warning(
                        f"Could not automatically open visualization in browser: {wb_err}"
                    )

        except Exception as e:
            self.logger.error(f"Failed to generate or save Pyvis visualization: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}", exc_info=True)
            raise GraphVisualizationError(
                f"Error generating 2D visualization: {e}", e
            ) from e

    def visualize_3d(
        self,
        output_path: Optional[str] = None,
        dimensions_filter: Optional[List[RelationshipDimension]] = None,
        open_in_browser: bool = False,
        *,
        focus_term: Optional[str] = None,
        focus_language: Optional[str] = None,
        depth: int = 1,
        max_nodes: Optional[int] = None,
        max_edges: Optional[int] = None,
    ) -> None:
        """
        Generate an interactive 3D graph visualization using Plotly.

        Args:
            output_path (Optional[str]): Path to save the HTML file. Defaults to config path.
            dimensions_filter (Optional[List[RelationshipDimension]]): List of relationship dimensions to include.
                                                                       If None, includes dimensions specified in config.active_dimensions.
            open_in_browser (bool): If True, automatically opens the generated HTML file.

        Raises:
            GraphVisualizationError: If Plotly is not installed or generation fails.
            GraphError: If the graph is empty or 3D positions are missing.
        """
        if not _plotly_available:
            self.logger.error("Plotly library is required for 3D visualization.")
            self.logger.error(
                f"Install with the visualization extras: {_VISUALIZATION_INSTALL_HINT}"
            )
            raise GraphVisualizationError(
                "Missing 'plotly' library. Install the visualization extras to enable 3D graph rendering."
            )

        if self.manager.g.number_of_nodes() == 0:
            raise GraphVisualizationError("Cannot visualize an empty graph.")

        graph_to_visualize = self._select_graph(
            dimensions_filter,
            focus_term=focus_term,
            focus_language=focus_language,
            depth=depth,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )
        if graph_to_visualize.number_of_nodes() == 0:
            raise GraphVisualizationError(
                "No graph nodes match the selected focus and dimensions."
            )
        node_positions = self._positions_for_graph(graph_to_visualize, dimensions=3)
        save_path = self._resolve_output_path(
            output_path or self._config.visualization_path,
            "graph_3d.html",
        )

        self.logger.info(f"Generating 3D visualization (Plotly) to: {save_path}")

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise GraphVisualizationError(
                f"Could not create directory for visualization: {save_path.parent}", e
            ) from e

        # Create Plotly figure using the potentially updated positions
        fig = self._configure_plotly_figure(graph_to_visualize, node_positions)

        try:
            fig.write_html(
                str(save_path),
                include_plotlyjs=True,
                full_html=True,
                auto_open=False,
            )
            self.logger.info(f"3D visualization saved successfully to {save_path}")

            if open_in_browser:
                try:
                    webbrowser.open(f"file://{str(save_path.resolve())}")
                except Exception as wb_err:
                    self.logger.warning(
                        f"Could not automatically open visualization in browser: {wb_err}"
                    )

        except Exception as e:
            self.logger.error(f"Failed to generate or save Plotly visualization: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}", exc_info=True)
            raise GraphVisualizationError(
                f"Error generating 3D visualization: {e}", e
            ) from e

    @staticmethod
    def _resolve_output_path(raw_path: str, default_filename: str) -> Path:
        """Resolve a user path to a concrete HTML artifact path.

        Args:
            raw_path: File or directory requested by the caller.
            default_filename: Filename used when ``raw_path`` is a directory.

        Returns:
            Expanded HTML output path. Suffix-free paths are treated as
            directories so configured visualization directories work before
            they have been created.
        """

        path = Path(raw_path).expanduser()
        if path.suffix.lower() == ".html":
            return path
        if path.is_dir() or not path.suffix:
            return path / default_filename
        return path.with_suffix(".html")

    def _positions_for_graph(
        self,
        graph: nx.Graph,
        *,
        dimensions: int,
    ) -> PositionDict:
        """Return existing compatible positions or lay out only this graph view."""

        if graph.number_of_nodes() > 500:
            self.logger.info(
                "Graph too large (%d nodes) for fast Python-side layout; delegating entirely to browser-side vis.js physics.",
                graph.number_of_nodes(),
            )
            return {}

        existing_positions = self.manager.get_positions()
        if all(
            node_id in existing_positions
            and len(existing_positions[node_id]) >= dimensions
            for node_id in graph.nodes()
        ):
            return cast(
                PositionDict,
                {
                    cast(WordId, node_id): tuple(
                        float(value)
                        for value in existing_positions[cast(WordId, node_id)][
                            :dimensions
                        ]
                    )
                    for node_id in graph.nodes()
                },
            )

        try:
            return self.manager.layout.compute_positions(
                graph,
                dimensions=dimensions,
            )
        except Exception as exc:
            raise GraphVisualizationError(
                "Failed to compute layout for the selected graph view.", exc
            ) from exc

    def _select_graph(
        self,
        dimensions_filter: Optional[List[RelationshipDimension]],
        *,
        focus_term: Optional[str],
        focus_language: Optional[str],
        depth: int,
        max_nodes: Optional[int],
        max_edges: Optional[int],
    ) -> nx.Graph:
        """Build a deterministic, bounded graph view for interactive rendering.

        Args:
            dimensions_filter: Relationship dimensions to retain.
            focus_term: Optional center term for a hop-bounded neighborhood.
            focus_language: Optional language used to disambiguate ``focus_term``.
            depth: Maximum undirected hop distance from the focus node.
            max_nodes: Optional node limit overriding graph configuration.
            max_edges: Optional edge limit overriding graph configuration.

        Returns:
            A detached graph with deterministic node and edge selection.

        Raises:
            NodeNotFoundError: If the requested focus term is absent.
            ValueError: If view bounds are invalid.
        """

        if depth < 0:
            raise ValueError("Visualization depth cannot be negative")
        if focus_language is not None and focus_term is None:
            raise ValueError("focus_language requires focus_term")

        node_limit = self._config.limit_node_count if max_nodes is None else max_nodes
        edge_limit = self._config.limit_edge_count if max_edges is None else max_edges
        if node_limit is not None and node_limit <= 0:
            raise ValueError("max_nodes must be positive")
        if edge_limit is not None and edge_limit <= 0:
            raise ValueError("max_edges must be positive")

        selected = self._filter_graph_by_dimensions(dimensions_filter)
        focus_id: Optional[WordId] = None
        distances: Dict[WordId, int] = {}
        if focus_term is not None:
            focus_id = self.manager.query.get_node_id(
                focus_term,
                language=focus_language,
            )
            if focus_id is None:
                language_context = (
                    f" ({focus_language})" if focus_language is not None else ""
                )
                raise NodeNotFoundError(
                    f"Focus term {focus_term!r}{language_context} was not found"
                )
            if focus_id not in selected:
                selected.add_node(focus_id, **dict(self.manager.g.nodes[focus_id]))
            traversal_graph = (
                selected.to_undirected(as_view=True)
                if selected.is_directed()
                else selected
            )
            distances = {
                cast(WordId, node_id): distance
                for node_id, distance in nx.single_source_shortest_path_length(
                    traversal_graph,
                    focus_id,
                    cutoff=depth,
                ).items()
            }
            selected = selected.subgraph(distances).copy()
        elif selected.number_of_edges() > 0:
            connected_nodes = {
                cast(WordId, node_id)
                for edge in selected.edges()
                for node_id in edge[:2]
            }
            selected = selected.subgraph(connected_nodes).copy()
        elif self.manager.g.number_of_edges() > 0:
            selected = type(selected)()

        original_nodes = selected.number_of_nodes()
        original_edges = selected.number_of_edges()
        selected = self._limit_graph(
            selected,
            focus_id=focus_id,
            distances=distances,
            max_nodes=node_limit,
            max_edges=edge_limit,
        )
        selected.graph["word_forge_total_nodes"] = original_nodes
        selected.graph["word_forge_total_edges"] = original_edges
        if (
            selected.number_of_nodes() != original_nodes
            or selected.number_of_edges() != original_edges
        ):
            self.logger.info(
                "Bounded graph view from %d nodes/%d edges to %d nodes/%d edges.",
                original_nodes,
                original_edges,
                selected.number_of_nodes(),
                selected.number_of_edges(),
            )
        return selected

    @staticmethod
    def _limit_graph(
        graph: nx.Graph,
        *,
        focus_id: Optional[WordId],
        distances: Dict[WordId, int],
        max_nodes: Optional[int],
        max_edges: Optional[int],
    ) -> nx.Graph:
        """Apply deterministic degree, distance, provenance, and weight limits."""

        selected = graph
        if max_nodes is not None and selected.number_of_nodes() > max_nodes:
            far_distance = selected.number_of_nodes() + 1

            def node_rank(node_id: WordId) -> tuple[object, ...]:
                attributes = selected.nodes[node_id]
                return (
                    distances.get(node_id, far_distance if focus_id is not None else 0),
                    -int(selected.degree[node_id]),
                    str(attributes.get("normalized_term", attributes.get("term", ""))),
                    str(attributes.get("language", "und")),
                    str(node_id),
                )

            retained_node_ids = sorted(selected.nodes(), key=node_rank)[:max_nodes]
            selected = selected.subgraph(retained_node_ids).copy()

        if max_edges is not None and selected.number_of_edges() > max_edges:
            far_distance = selected.number_of_nodes() + 1

            def edge_rank(
                edge: tuple[WordId, WordId, Dict[str, Any]],
            ) -> tuple[object, ...]:
                source_id, target_id, attributes = edge
                weight = _finite_float(attributes.get("weight", 1.0), 1.0)
                assertions = max(
                    1,
                    int(_finite_float(attributes.get("assertion_count", 1), 1.0)),
                )
                source_distance = distances.get(source_id, far_distance)
                target_distance = distances.get(target_id, far_distance)
                return (
                    min(source_distance, target_distance),
                    max(source_distance, target_distance),
                    -assertions,
                    -weight,
                    str(source_id),
                    str(target_id),
                )

            retained_edges = sorted(selected.edges(data=True), key=edge_rank)[
                :max_edges
            ]
            limited = type(selected)()
            limited.graph.update(selected.graph)
            edge_node_ids = {
                node_id
                for source_id, target_id, _ in retained_edges
                for node_id in (source_id, target_id)
            }
            if focus_id is not None and focus_id in selected:
                edge_node_ids.add(focus_id)
            limited.add_nodes_from(
                (node_id, dict(selected.nodes[node_id])) for node_id in edge_node_ids
            )
            limited.add_edges_from(
                (source_id, target_id, dict(attributes))
                for source_id, target_id, attributes in retained_edges
            )
            selected = limited

        return selected

    def _filter_graph_by_dimensions(
        self, dimensions_filter: Optional[List[RelationshipDimension]]
    ) -> nx.Graph:
        """
        Create a subgraph containing only edges matching the dimension filter.

        Args:
            dimensions_filter (Optional[List[RelationshipDimension]]): List of dimensions to include.
                                                                       If None, uses config.active_dimensions.

        Returns:
            nx.Graph: A NetworkX graph object containing the filtered edges and involved nodes.
                      Returns a copy to prevent modification of the original graph view.
        """
        active_dimensions = set(
            self._config.active_dimensions
            if dimensions_filter is None
            else dimensions_filter
        )
        self.logger.debug(f"Filtering graph for dimensions: {active_dimensions}")

        if not active_dimensions:
            self.logger.warning(
                "Dimension filter is empty. Visualization might be empty."
            )
            return type(self.manager.g)()

        def edge_filter(u: WordId, v: WordId) -> bool:
            """Check if edge between u and v matches active dimensions."""
            if self.manager.g.is_multigraph():
                multi_graph = cast(
                    Union[nx.MultiGraph, nx.MultiDiGraph], self.manager.g
                )
                if not multi_graph.has_edge(u, v):
                    return False
                for key in multi_graph[u][v]:
                    edge_data = multi_graph.get_edge_data(u, v, key=key)
                    dimensions = (
                        str(
                            edge_data.get("dimensions", edge_data.get("dimension", ""))
                        ).split("|")
                        if edge_data
                        else []
                    )
                    if active_dimensions.intersection(dimensions):
                        return True
                return False
            else:
                edge_data = self.manager.g.get_edge_data(u, v)
                dimensions = (
                    str(
                        edge_data.get("dimensions", edge_data.get("dimension", ""))
                    ).split("|")
                    if edge_data is not None
                    else []
                )
                return bool(active_dimensions.intersection(dimensions))

        filtered_view = nx.subgraph_view(self.manager.g, filter_edge=edge_filter)
        filtered_graph = type(self.manager.g)()
        filtered_graph.add_nodes_from(filtered_view.nodes(data=True))
        filtered_graph.add_edges_from(filtered_view.edges(data=True))
        return filtered_graph

    def _configure_pyvis_network(
        self, net: PyvisNetwork, graph: nx.Graph, node_positions: PositionDict
    ) -> None:
        """
        Configure Pyvis network object with nodes, edges, and styling.

        Args:
            net (PyvisNetwork): The PyvisNetwork instance.
            graph (nx.Graph): The NetworkX graph to visualize (potentially filtered).
            node_positions (PositionDict): Dictionary mapping node IDs to positions.
        """
        if not _pyvis_available:
            self.logger.error("Pyvis not available, cannot configure network.")
            return

        self.logger.debug("Configuring Pyvis network...")

        # Sort all records so equal graph snapshots produce byte-stable exports.
        for raw_node_id, attrs in sorted(
            graph.nodes(data=True), key=lambda item: str(item[0])
        ):
            node_id = cast(WordId, raw_node_id)
            term = str(attrs.get("term", f"ID:{node_id}"))
            label = str(attrs.get("label", term))
            language = str(attrs.get("language", "und"))
            script = str(attrs.get("script", "Zzzz"))
            source = str(attrs.get("source", "unknown"))
            is_stub = bool(attrs.get("is_stub", False))
            node_size = self._calculate_node_size(node_id, graph)
            node_color = self._get_node_color(attrs)
            pos = node_positions.get(node_id)
            if pos is not None and len(pos) >= 2:
                pos_x: Optional[float] = float(pos[0] * 100)
                pos_y: Optional[float] = float(pos[1] * 100)
                node_physics = False
            else:
                pos_x = None
                pos_y = None
                node_physics = True

            title_parts = [
                f"Term: {escape(str(term))}",
                f"ID: {escape(str(node_id))}",
                f"Language: {escape(language)}",
                f"Script: {escape(script)}",
                f"Source: {escape(source)}",
            ]
            if is_stub:
                title_parts.append("Status: awaiting lexical enrichment")
            if "valence" in attrs and attrs["valence"] is not None:
                title_parts.append(f"Valence: {attrs['valence']:.2f}")
            if "arousal" in attrs and attrs["arousal"] is not None:
                title_parts.append(f"Arousal: {attrs['arousal']:.2f}")
            title = "\n".join(title_parts) if self._config.enable_tooltips else None

            node_kwargs: Dict[str, Any] = {
                "label": label if self._config.enable_labels else "",
                "title": title,
                "size": node_size,
                "color": node_color,
                "physics": node_physics,
                "wfTerm": term,
                "wfDisplayLabel": label,
                "wfLanguage": language,
                "wfScript": script,
                "wfSource": source,
                "wfStub": is_stub,
            }
            if pos_x is not None:
                node_kwargs["x"] = pos_x
            if pos_y is not None:
                node_kwargs["y"] = pos_y

            net.add_node(str(node_id), **node_kwargs)

        edge_records = sorted(
            graph.edges(data=True),
            key=lambda item: (
                min(str(item[0]), str(item[1])),
                max(str(item[0]), str(item[1])),
                str(item[2].get("assertions_json", "")),
                str(item[2].get("relationship_types", "")),
            ),
        )
        for edge_index, (raw_source_id, raw_target_id, attrs) in enumerate(
            edge_records
        ):
            source_id = cast(WordId, raw_source_id)
            target_id = cast(WordId, raw_target_id)
            assertions = decode_edge_assertions(
                attrs,
                default_source_id=source_id,
                default_target_id=target_id,
            )
            relationship_types = sorted(
                {
                    *(str(item["relationship"]) for item in assertions),
                    *_pipe_values(
                        attrs.get(
                            "relationship_types",
                            attrs.get("relationship", ""),
                        )
                    ),
                },
                key=str.casefold,
            )
            dimensions = sorted(
                {
                    *(str(item["dimension"]) for item in assertions),
                    *_pipe_values(
                        attrs.get("dimensions", attrs.get("dimension", "lexical"))
                    ),
                },
                key=str.casefold,
            )
            sources = sorted(
                {
                    *(str(item["source"]) for item in assertions),
                    *_pipe_values(attrs.get("sources", attrs.get("source", "unknown"))),
                },
                key=str.casefold,
            )
            rel_type = "|".join(relationship_types)
            dimension = "|".join(dimensions) or "lexical"

            # Get color based on relationship type, falling back to dimension-based color
            edge_color = self._get_edge_color(
                relationship_types[0] if relationship_types else "",
                dimensions[0] if dimensions else "lexical",
                attrs,
            )
            edge_width = self._calculate_edge_width(attrs.get("weight", 1.0))

            # Build informative tooltip
            title_parts = []
            if rel_type:
                title_parts.append(f"Type: {escape(rel_type.replace('|', ', '))}")
            title_parts.append(f"Dimension: {escape(dimension.replace('|', ', '))}")
            if sources:
                title_parts.append(f"Source: {escape(', '.join(sources))}")
            if attrs.get("assertion_count") is not None:
                title_parts.append(f"Assertions: {attrs['assertion_count']}")
            if attrs.get("valence") is not None:
                title_parts.append(f"Valence: {attrs['valence']:.2f}")
            if attrs.get("arousal") is not None:
                title_parts.append(f"Arousal: {attrs['arousal']:.2f}")
            if attrs.get("weight") is not None:
                title_parts.append(f"Weight: {attrs['weight']:.2f}")
            title = "\n".join(title_parts) if self._config.enable_tooltips else None

            # Use dashed style for cross-dimensional edges or emotional relationships
            style = attrs.get("style", "solid")
            if {"emotional", "affective"}.intersection(dimensions) and style == "solid":
                style = self._config.cross_dimension_edge_style

            display_source = source_id
            display_target = target_id
            directions = {(item["source_id"], item["target_id"]) for item in assertions}
            arrows = ""
            if bool(attrs.get("bidirectional")) or {
                (source_id, target_id),
                (target_id, source_id),
            }.issubset(directions):
                arrows = "to, from"
            elif len(directions) == 1:
                assertion_source, assertion_target = next(iter(directions))
                if {assertion_source, assertion_target} == {source_id, target_id}:
                    display_source = assertion_source
                    display_target = assertion_target
                    arrows = "to"

            edge_options: Dict[str, Any] = {
                "id": f"wf-edge-{edge_index}",
                "title": title,
                "color": edge_color,
                "width": edge_width,
                "label": (
                    ", ".join(relationship_types)
                    if self._config.enable_edge_labels
                    else ""
                ),
                "dashes": style == "dashed",
                "wfTypes": relationship_types,
                "wfDimensions": dimensions,
                "wfSources": sources,
                "wfAssertions": [dict(item) for item in assertions],
                "wfAssertionCount": len(assertions)
                or int(_finite_float(attrs.get("assertion_count", 1), 1.0)),
            }
            if arrows:
                edge_options["arrows"] = arrows
            net.add_edge(
                str(display_source),
                str(display_target),
                **edge_options,
            )

        # Apply general Pyvis options using set_options
        options_dict = {
            "physics": {"enabled": False},  # Disable physics for precomputed layout
            "layout": {
                "hierarchical": {
                    "enabled": self._config.default_layout
                    == "hierarchical"
                    # Add other hierarchical options if needed
                }
            },
            "nodes": {
                "font": {"color": "white"},
                "shape": "dot",  # Default shape
                # Add other global node options
            },
            "edges": {
                "smooth": {"enabled": True, "type": "continuous"},
                "font": {"color": "white", "size": 10, "align": "top"},
                # Add other global edge options
            },
            "interaction": {
                "tooltipDelay": 200,
                "navigationButtons": True,
                "keyboard": True,
            },
        }
        # Convert dict to JSON string for set_options
        options_json = json.dumps(options_dict)
        net.set_options(options_json)

        self.logger.debug("Pyvis network configuration complete.")

    def _configure_plotly_figure(
        self, graph: nx.Graph, node_positions: PositionDict
    ) -> go.Figure:
        """
        Configure Plotly figure object for 3D visualization.

        Args:
            graph (nx.Graph): The NetworkX graph to visualize (potentially filtered).
            node_positions (PositionDict): Dictionary mapping node IDs to positions.

        Returns:
            go.Figure: A Plotly Figure object.

        Raises:
            GraphVisualizationError: If Plotly is not available or no valid nodes found.
        """
        if not _plotly_available:
            raise GraphVisualizationError(
                "Plotly library missing. Install the visualization extras to configure 3D figures."
            )

        self.logger.debug("Configuring Plotly 3D figure...")

        edge_x: List[Optional[float]] = []
        edge_y: List[Optional[float]] = []
        edge_z: List[Optional[float]] = []

        for edge in graph.edges(data=True):
            u, v, data = edge
            pos_u = node_positions.get(u)
            pos_v = node_positions.get(v)

            # Explicitly check if positions are not None and have length 3
            if (
                pos_u is not None
                and pos_v is not None
                and len(pos_u) == 3
                and len(pos_v) == 3
            ):
                edge_x.extend([pos_u[0], pos_v[0], None])
                edge_y.extend([pos_u[1], pos_v[1], None])
                edge_z.extend([pos_u[2], pos_v[2], None])
            else:
                self.logger.warning(
                    f"Skipping edge ({u},{v}) due to missing or invalid 3D positions (pos_u: {type(pos_u)}, pos_v: {type(pos_v)})."
                )

        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            line=dict(width=self._config.min_edge_width, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_x: List[float] = []
        node_y: List[float] = []
        node_z: List[float] = []
        node_text: List[str] = []
        node_sizes: List[float] = []
        node_colors: List[ColorHex] = []
        valid_node_ids = []

        for node_id, attrs in graph.nodes(data=True):
            pos = node_positions.get(node_id)
            # Explicitly check if position is not None and has length 3
            if pos is not None and len(pos) == 3:
                valid_node_ids.append(node_id)
                node_x.append(pos[0])
                node_y.append(pos[1])
                node_z.append(pos[2])

                term = attrs.get("term", f"ID:{node_id}")
                hover_parts = [
                    f"Term: {escape(str(term))}",
                    f"Language: {escape(str(attrs.get('language', 'und')))}",
                    f"Script: {escape(str(attrs.get('script', 'Zzzz')))}",
                    f"ID: {escape(str(node_id))}",
                ]
                if "valence" in attrs and attrs["valence"] is not None:
                    hover_parts.append(f"Valence: {attrs['valence']:.2f}")
                if "arousal" in attrs and attrs["arousal"] is not None:
                    hover_parts.append(f"Arousal: {attrs['arousal']:.2f}")
                node_text.append("<br>".join(hover_parts))

                node_sizes.append(self._calculate_node_size(node_id, graph))
                node_colors.append(self._get_node_color(attrs))
            else:
                self.logger.warning(
                    f"Skipping node {node_id} due to missing or invalid 3D position (pos: {type(pos)})."
                )

        if not valid_node_ids:
            self.logger.error(
                "No nodes with valid 3D positions found. Cannot generate 3D plot."
            )
            raise GraphVisualizationError(
                "No nodes with valid 3D positions found for Plotly."
            )

        # Calculate sizeref based on valid node sizes
        valid_node_sizes = [s for s in node_sizes if s is not None]
        sizeref_value = (
            (max(valid_node_sizes) / (self._config.max_node_size * 1.5))
            if valid_node_sizes
            else 1
        )

        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers" + ("+text" if self._config.enable_labels else ""),
            hoverinfo="text" if self._config.enable_tooltips else "none",
            text=node_text if self._config.enable_tooltips else None,
            marker=dict(
                showscale=False,
                color=node_colors,
                size=node_sizes,
                sizeref=sizeref_value,  # Use calculated sizeref
                sizemin=self._config.min_node_size / 1.5,
                line_width=0.5,
            ),
            textfont=(
                dict(size=10, color="#CCCCCC") if self._config.enable_labels else None
            ),
            textposition="top center",
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                # Corrected: Use 'title' dict with 'text' and 'font' sub-properties
                title=dict(
                    text="<br>3D Knowledge Graph Visualization",
                    font=dict(size=16, color="white"),
                    x=0.5,  # Center title
                    xanchor="center",
                ),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                scene=dict(
                    xaxis=dict(
                        showgrid=False, zeroline=False, showticklabels=False, title=""
                    ),
                    yaxis=dict(
                        showgrid=False, zeroline=False, showticklabels=False, title=""
                    ),
                    zaxis=dict(
                        showgrid=False, zeroline=False, showticklabels=False, title=""
                    ),
                    bgcolor="#111111",
                ),
                paper_bgcolor="#1e1e1e",
                plot_bgcolor="#1e1e1e",
                font=dict(color="white"),  # Global font color
            ),
        )

        self.logger.debug("Plotly 3D figure configuration complete.")
        return fig

    def _calculate_node_size(self, node_id: WordId, graph: nx.Graph) -> float:
        """
        Calculate node size based on configured strategy.

        Args:
            node_id (WordId): The ID of the node.
            graph (nx.Graph): The graph containing the node (used for degree calculation).

        Returns:
            float: The calculated node size, clamped within configured min/max bounds.
        """
        strategy = self._config.node_size_strategy
        min_size = self._config.min_node_size
        max_size = self._config.max_node_size
        default_size = (min_size + max_size) / 2.0

        try:
            if strategy == "degree":
                if node_id not in graph:
                    return default_size
                degree = float(graph.degree(node_id))
                all_degrees = [float(d) for _, d in graph.degree()]
                max_degree = max(all_degrees) if all_degrees else 1
                size = min_size + (max_size - min_size) * (degree / max(1, max_degree))
                return max(min_size, min(size, max_size))
            elif strategy == "centrality":
                self.logger.warning(
                    "Node size strategy 'centrality' not fully implemented, using 'degree'."
                )
                if node_id not in graph:
                    return default_size
                degree = float(graph.degree(node_id))
                all_degrees = [float(d) for _, d in graph.degree()]
                max_degree = max(all_degrees) if all_degrees else 1
                size = min_size + (max_size - min_size) * (degree / max(1, max_degree))
                return max(min_size, min(size, max_size))
            else:
                return default_size
        except Exception as e:
            self.logger.warning(
                f"Error calculating node size for {node_id} using strategy '{strategy}': {e}. Using default size {default_size}."
            )
            return default_size

    def _calculate_edge_width(self, weight: Optional[float]) -> float:
        """
        Calculate edge width based on weight, clamped within configured bounds.

        Args:
            weight (Optional[float]): The edge weight (typically 0.0 to 1.0). Defaults to 0.5 if None.

        Returns:
            float: The calculated edge width.
        """
        min_width = self._config.min_edge_width
        max_width = self._config.max_edge_width
        effective_weight = _finite_float(weight, 0.5)
        width = min_width + (max_width - min_width) * effective_weight
        return max(min_width, min(width, max_width))

    def _get_node_color(self, node_attributes: Dict[str, Any]) -> ColorHex:
        """
        Determine node color based on attributes and configuration.

        Implements coloring based on valence if available and configured,
        otherwise uses a default color.

        Args:
            node_attributes (Dict[str, Any]): Dictionary of attributes for the node.

        Returns:
            ColorHex: A hex color string (e.g., "#RRGGBB").
        """
        valence = node_attributes.get("valence")
        if isinstance(valence, (int, float)):
            if valence > 0.5:
                return self._config.affective_relationship_colors.get(
                    "positive_valence", "#00cc66"
                )
            if valence < -0.5:
                return self._config.affective_relationship_colors.get(
                    "negative_valence", "#cc3300"
                )
            if valence > 0.1:
                return "#90EE90"
            if valence < -0.1:
                return "#FFA07A"
            return self._config.affective_relationship_colors.get(
                "valence_neutral", "#cccccc"
            )

        return "#6666ff"  # Default blueish color

    def _get_edge_color(
        self,
        rel_type: str,
        dimension: str,
        attrs: Dict[str, Any],
    ) -> ColorHex:
        """
        Determine edge color based on relationship type, dimension, and attributes.

        Prioritizes relationship type colors, then dimension-based colors,
        and finally considers emotional attributes like valence for coloring.

        Args:
            rel_type: The relationship type (e.g., 'synonym', 'emotional_synonym')
            dimension: The relationship dimension (e.g., 'lexical', 'emotional')
            attrs: Edge attributes dictionary

        Returns:
            ColorHex: A hex color string for the edge.
        """
        # First, try to get color by relationship type if it's a known type
        if rel_type and rel_type.lower() in self._config.relationship_colors:
            return self._config.relationship_colors[rel_type.lower()]

        # Check emotional relationship colors
        if rel_type and rel_type.lower() in self._config.emotional_relationship_colors:
            return self._config.emotional_relationship_colors[rel_type.lower()]

        # Check affective relationship colors
        if rel_type and rel_type.lower() in self._config.affective_relationship_colors:
            return self._config.affective_relationship_colors[rel_type.lower()]

        # For emotional dimensions, color by valence if available
        if dimension in ("emotional", "affective"):
            valence = attrs.get("valence")
            if isinstance(valence, (int, float)):
                if valence > VALENCE_HIGH_THRESHOLD:
                    return self._config.affective_relationship_colors.get(
                        "positive_valence", "#00cc66"
                    )
                elif valence < VALENCE_LOW_THRESHOLD:
                    return self._config.affective_relationship_colors.get(
                        "negative_valence", "#cc3300"
                    )
                else:
                    return self._config.affective_relationship_colors.get(
                        "valence_neutral", "#cccccc"
                    )

            # Color by arousal if valence not available
            arousal = attrs.get("arousal")
            if isinstance(arousal, (int, float)):
                if arousal > AROUSAL_HIGH_THRESHOLD:
                    return self._config.affective_relationship_colors.get(
                        "high_arousal", "#ff9900"
                    )
                else:
                    return self._config.affective_relationship_colors.get(
                        "low_arousal", "#3366cc"
                    )

        # Dimension-based default colors
        dimension_colors = {
            "lexical": "#4287f5",  # Blue
            "emotional": "#ff69b4",  # Hot pink
            "affective": "#ff9900",  # Orange
            "contextual": "#20b2aa",  # Light sea green
            "connotative": "#daa520",  # Goldenrod
        }

        return dimension_colors.get(dimension, "#aaaaaa")
