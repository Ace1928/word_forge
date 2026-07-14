"""
Manages graph layout computation and node positioning.

Encapsulates logic for applying various layout algorithms (2D and 3D)
to the knowledge graph, handling incremental updates, and storing
position data. Adheres to Eidosian principles of modularity, precision,
and adaptability.

Architecture:
    ┌──────────────────┐      ┌──────────────────┐
    │  GraphManager    │◄────►│   GraphLayout    │
    │ (Orchestrator)   │      │ (Position Calc & │
    └────────┬─────────┘      │  Algorithm Mgmt) │
             │                └──────────────────┘
             ▼
    ┌──────────────────┐
    │    NetworkX      │
    │ (Layout Algos)   │
    └──────────────────┘
"""

from __future__ import annotations

import functools  # Import functools
import logging
import math
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Dict, List, Optional, Union, cast

import networkx as nx

# Import necessary components
from word_forge.exceptions import GraphLayoutError
from word_forge.graph.graph_config import WordId  # Alias to avoid naming conflict
from word_forge.graph.graph_config import (
    LayoutAlgorithm,
    Position,
    PositionDict,
)

# Type hint for the main GraphManager to avoid circular imports
if TYPE_CHECKING:
    from .graph_manager import GraphManager

LayoutFunction = Callable[..., object]


class GraphLayout:
    """
    Computes and manages node positions for graph visualization.

    Supports various NetworkX layout algorithms in both 2D and 3D,
    adapting based on the GraphManager's dimension setting. Provides methods
    for full layout computation and incremental updates for newly added nodes.

    Attributes:
        manager: Reference to the main GraphManager for state access.
        logger: Logger instance for this module.
    """

    def __init__(self, manager: GraphManager) -> None:
        """
        Initialize the GraphLayout with a reference to the GraphManager.

        Args:
            manager: The orchestrating GraphManager instance.
        """
        self.manager: GraphManager = manager
        self.logger: logging.Logger = logging.getLogger(__name__)
        # Use config from manager for consistency
        self._config = self.manager.config

    def compute_layout(self, algorithm: Optional[LayoutAlgorithm] = None) -> None:
        """
        Compute node positions for the entire graph using a specified algorithm.

        Applies the chosen layout algorithm (or the default from config) to
        all nodes in the manager's graph. Handles both 2D and 3D layouts based
        on the manager's dimension setting. Stores the computed positions in
        the manager's `_positions` dictionary.

        Args:
            algorithm: The layout algorithm to use (e.g., 'force_directed').
                       If None, uses the default from GraphConfig.

        Raises:
            GraphLayoutError: If the layout computation fails or the graph is empty.
            AttributeError: If an invalid layout algorithm is specified.
        """
        if self.manager.g.number_of_nodes() == 0:
            self.logger.warning("Cannot compute layout for an empty graph.")
            self.manager._positions.clear()  # Ensure positions are cleared
            return

        self.manager._positions = self.compute_positions(
            self.manager.g,
            algorithm=algorithm,
            dimensions=self.manager.dimensions,
        )
        self.logger.info(
            "Layout computation complete. Stored %d node positions.",
            len(self.manager._positions),
        )

    def compute_positions(
        self,
        graph: nx.Graph,
        *,
        algorithm: Optional[LayoutAlgorithm] = None,
        dimensions: Optional[int] = None,
    ) -> PositionDict:
        """Compute deterministic positions without mutating manager state.

        Args:
            graph: Graph or bounded graph view to position.
            algorithm: Optional layout algorithm override.
            dimensions: Coordinate dimensions. Defaults to the manager setting.

        Returns:
            Floating-point coordinate tuples keyed by node ID.

        Raises:
            GraphLayoutError: If the selected layout cannot be computed.
            ValueError: If dimensions is not two or three.
        """

        if graph.number_of_nodes() == 0:
            return {}

        selected_dimensions = (
            self.manager.dimensions if dimensions is None else dimensions
        )
        if selected_dimensions not in (2, 3):
            raise ValueError(
                "Invalid number of dimensions specified: "
                f"{selected_dimensions}. Must be 2 or 3."
            )
        layout_name = algorithm or self._config.default_layout
        algorithm_name = str(getattr(layout_name, "value", layout_name))
        self.logger.info(
            "Computing %dD layout for %d nodes using '%s'.",
            selected_dimensions,
            graph.number_of_nodes(),
            algorithm_name,
        )

        try:
            layout_function = self._get_layout_function(
                algorithm_name,
                selected_dimensions,
            )
            raw_positions = layout_function(self._canonical_graph(graph))
            if not isinstance(raw_positions, Mapping):
                raise TypeError("layout algorithm returned a non-mapping result")
            positions: PositionDict = {}
            for node_id, raw_coordinates in raw_positions.items():
                if not isinstance(raw_coordinates, Iterable) or isinstance(
                    raw_coordinates, (str, bytes)
                ):
                    raise TypeError(
                        f"layout coordinates for node {node_id!r} are not iterable"
                    )
                coordinates = tuple(float(value) for value in raw_coordinates)
                if len(coordinates) != selected_dimensions or not all(
                    math.isfinite(value) for value in coordinates
                ):
                    raise ValueError(
                        f"layout coordinates for node {node_id!r} must contain "
                        f"{selected_dimensions} finite values"
                    )
                positions[cast(WordId, node_id)] = cast(Position, coordinates)
            return positions
        except AttributeError as exc:
            self.logger.error("Invalid layout algorithm '%s': %s", algorithm_name, exc)
            raise GraphLayoutError(
                f"Layout algorithm '{algorithm_name}' not found or failed.", exc
            ) from exc
        except Exception as exc:
            self.logger.error(
                "Layout computation using '%s' failed: %s",
                algorithm_name,
                exc,
                exc_info=True,
            )
            raise GraphLayoutError(f"Layout computation failed: {exc}", exc) from exc

    def update_layout_incrementally(self, new_node_ids: List[WordId]) -> None:
        """
        Update layout incrementally, focusing on positioning new nodes.

        Currently, this often involves recomputing the layout for stability,
        especially with force-directed algorithms. Future enhancements could
        implement true incremental updates for specific algorithms if feasible
        and beneficial. For now, it primarily recalculates the full layout.

        Args:
            new_node_ids: A list of IDs for the newly added nodes.
                          (Currently unused, but kept for future incremental logic).

        Raises:
            GraphLayoutError: If the layout computation fails.
        """
        if not new_node_ids:
            self.logger.debug("No new nodes provided for incremental layout update.")
            return

        self.logger.info(
            f"Received {len(new_node_ids)} new nodes. Updating layout incrementally."
        )

        default_algo = self._config.default_layout
        algo_str = str(getattr(default_algo, "value", default_algo))

        if algo_str != "force_directed" or not self.manager._positions:
            # Fallback to full recompute for unsupported algorithms or missing positions
            self.logger.debug(
                "Incremental update not supported for this layout. Recomputing full layout."
            )
            self.compute_layout(algorithm=cast(LayoutAlgorithm, algo_str))
            return

        try:
            fixed_nodes = [n for n in self.manager.g.nodes() if n not in new_node_ids]
            pos_init = {
                n: self.manager._positions[n]
                for n in fixed_nodes
                if n in self.manager._positions
            }
            k_value = getattr(self._config, "layout_k", None)
            iterations = self._config.layout_iterations
            updated_pos = nx.spring_layout(
                self.manager.g,
                pos=pos_init,
                fixed=fixed_nodes,
                dim=self.manager.dimensions,
                k=k_value,
                iterations=iterations,
                seed=self._config.layout_seed,
            )
            self.manager._positions.update(cast(PositionDict, updated_pos))
            self.logger.info(
                f"Incremental layout update complete. Total nodes positioned: {len(self.manager._positions)}."
            )
        except Exception as e:
            self.logger.error(
                f"Incremental layout update failed: {e}",
                exc_info=True,
            )
            raise GraphLayoutError(f"Incremental layout update failed: {e}") from e

    @staticmethod
    def _canonical_graph(graph: nx.Graph) -> nx.Graph:
        """Copy a graph with deterministic node and edge insertion order."""

        canonical = type(graph)()
        canonical.graph.update(graph.graph)
        canonical.add_nodes_from(
            (node_id, dict(attributes))
            for node_id, attributes in sorted(
                graph.nodes(data=True), key=lambda item: str(item[0])
            )
        )
        if graph.is_multigraph():
            multigraph = cast(Union[nx.MultiGraph, nx.MultiDiGraph], graph)
            canonical_multigraph = cast(
                Union[nx.MultiGraph, nx.MultiDiGraph], canonical
            )
            for source_id, target_id, key, attributes in sorted(
                multigraph.edges(keys=True, data=True),
                key=lambda item: (
                    str(item[0]),
                    str(item[1]),
                    str(item[2]),
                ),
            ):
                canonical_multigraph.add_edge(
                    source_id,
                    target_id,
                    key=key,
                    **dict(attributes),
                )
        else:
            canonical.add_edges_from(
                (source_id, target_id, dict(attributes))
                for source_id, target_id, attributes in sorted(
                    graph.edges(data=True),
                    key=lambda item: (str(item[0]), str(item[1])),
                )
            )
        return canonical

    def _get_layout_function(
        self, algorithm_name: str, dimensions: int
    ) -> LayoutFunction:
        """
        Retrieve the appropriate NetworkX layout function based on name and dimension.
        Ensures correct parameters are passed to the underlying layout function.

        Args:
            algorithm_name: The name of the layout algorithm.
            dimensions: The desired number of dimensions (2 or 3).

        Returns:
            callable: A function that takes a graph `G` and returns positions.

        Raises:
            AttributeError: If the algorithm name is invalid.
            ValueError: If the dimensions are not 2 or 3.
        """
        if dimensions not in [2, 3]:
            raise ValueError(
                f"Invalid number of dimensions specified: {dimensions}. Must be 2 or 3."
            )

        # Common parameters for spring_layout
        k_value = getattr(self._config, "layout_k", None)
        iterations = self._config.layout_iterations
        seed = self._config.layout_seed

        # Define base layout functions
        layout_map_base: Dict[str, LayoutFunction] = {
            "force_directed": nx.spring_layout,
            "spectral": nx.spectral_layout,
            "circular": nx.circular_layout,
            "hierarchical": lambda G: nx.nx_agraph.graphviz_layout(G, prog="dot"),
            "radial": lambda G: nx.nx_agraph.graphviz_layout(G, prog="twopi"),
            "grid": nx.spring_layout,  # Fallback for grid
        }

        try:
            base_func = layout_map_base[algorithm_name]

            # Handle dimension-specific logic and parameters
            if algorithm_name == "force_directed":
                # Use functools.partial to pre-set arguments for spring_layout
                return functools.partial(
                    base_func,
                    dim=dimensions,
                    k=k_value,
                    iterations=iterations,
                    seed=seed,
                )
            elif algorithm_name == "spectral":
                if dimensions == 3:
                    try:
                        # Attempt 3D spectral layout
                        return functools.partial(base_func, dim=3)
                    except TypeError:
                        self.logger.warning(
                            "Current NetworkX spectral_layout doesn't support dim=3. Falling back to 3D spring layout."
                        )
                        # Fallback to 3D spring layout
                        return functools.partial(
                            nx.spring_layout,
                            dim=3,
                            k=k_value,
                            iterations=iterations,
                            seed=seed,
                        )
                else:  # dimensions == 2
                    return base_func  # spectral_layout defaults to 2D
            elif algorithm_name == "circular":
                # Circular layout is inherently 2D. For 3D, use spring as fallback.
                if dimensions == 3:
                    self.logger.debug(
                        "Circular layout requested for 3D, using 3D spring layout as fallback."
                    )
                    return functools.partial(
                        nx.spring_layout,
                        dim=3,
                        k=k_value,
                        iterations=iterations,
                        seed=seed,
                    )
                else:
                    return base_func
            elif algorithm_name in ["hierarchical", "radial"]:
                if dimensions == 3:
                    self.logger.warning(
                        f"Layout '{algorithm_name}' is 2D only. Falling back to 3D spring layout."
                    )
                    return functools.partial(
                        nx.spring_layout,
                        dim=3,
                        k=k_value,
                        iterations=iterations,
                        seed=seed,
                    )
                else:
                    # Check for pygraphviz dependency
                    try:
                        import pygraphviz  # type: ignore[import-not-found] # noqa: F401

                        return base_func  # Return the lambda defined in layout_map_base
                    except ImportError:
                        self.logger.warning(
                            f"Layout '{algorithm_name}' requires pygraphviz. Falling back to 'force_directed'."
                        )
                        self.logger.warning("Install with: pip install pygraphviz")
                        # Fallback to 2D spring layout
                        return functools.partial(
                            nx.spring_layout,
                            dim=2,
                            k=k_value,
                            iterations=iterations,
                            seed=seed,
                        )
            elif algorithm_name == "grid":
                self.logger.debug(
                    "Grid layout requested, using spring layout as fallback."
                )
                return functools.partial(
                    nx.spring_layout,
                    dim=dimensions,
                    k=k_value,
                    iterations=iterations,
                    seed=seed,
                )
            else:
                # Should not be reached if algorithm_name is in layout_map_base
                raise AttributeError(f"Unhandled layout algorithm: {algorithm_name}")

        except KeyError:
            self.logger.error(f"Layout algorithm '{algorithm_name}' is not supported.")
            raise AttributeError(f"Unsupported layout algorithm: {algorithm_name}")
        except ImportError as ie:
            # Catch potential import errors from nx_agraph
            self.logger.error(
                f"Layout algorithm '{algorithm_name}' failed due to missing dependency: {ie}. Falling back to force_directed."
            )
            # Fallback to appropriate dimension spring layout
            return functools.partial(
                nx.spring_layout,
                dim=dimensions,
                k=k_value,
                iterations=iterations,
                seed=seed,
            )

    def _apply_layout(self) -> None:
        """
        Applies the computed positions to the graph nodes.

        Deprecated/Internal: Positions are now stored directly in
        `self.manager._positions`. This method might be repurposed if
        node attributes need direct updating in the future.
        """
        # This method is less relevant now as positions are stored centrally
        # in self.manager._positions. Keeping as a placeholder or for future use
        # if direct node attribute updates become necessary.
        self.logger.debug(
            "Layout positions are stored centrally; direct application to node attributes skipped."
        )
        # Example of direct application if needed later:
        # if self.manager._positions:
        #     for node_id, pos in self.manager._positions.items():
        #         if node_id in self.manager.g:
        #             self.manager.g.nodes[node_id['pos'] = pos
