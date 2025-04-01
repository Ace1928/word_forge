"""
Graph visualization configuration system for Word Forge.

This module defines the configuration schema for knowledge graph visualization,
including layout algorithms, color schemes, export options, and rendering
parameters used throughout the Word Forge system.

Architecture:
    ┌───────────────────┐
    │   GraphConfig     │
    └─────────┬─────────┘
              │
    ┌─────────┴─────────┐
    │    Components     │
    └───────────────────┘
    ┌─────┬─────┬───────┬─────┬─────┐
    │Vis  │Layout│Export│Size │Color│
    └─────┴─────┴───────┴─────┴─────┘
"""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from word_forge.configs.config_essentials import (
    DATA_ROOT,
    GraphColorScheme,
    GraphConfigError,
    GraphEdgeWeightStrategy,
    GraphExportFormat,
    GraphLayoutAlgorithm,
    GraphNodeSizeStrategy,
)
from word_forge.configs.config_types import EnvMapping, SQLTemplates


@dataclass
class GraphConfig:
    """
    Configuration for knowledge graph visualization.

    Controls layout algorithms, node/edge styling, export formats,
    and performance parameters for visualizing semantic relationships.

    Attributes:
        default_layout: Default layout algorithm for graph visualization
        default_color_scheme: Default color scheme for nodes and edges
        visualization_path: Path for saving visualization outputs
        default_export_format: Default format for exporting graphs
        default_export_path: Default path for saving exported graphs
        node_size_strategy: Strategy for calculating node sizes
        edge_weight_strategy: Strategy for calculating edge weights
        min_node_size: Minimum node size in pixels
        max_node_size: Maximum node size in pixels
        min_edge_width: Minimum edge width in pixels
        max_edge_width: Maximum edge width in pixels
        enable_labels: Whether to display labels on nodes
        enable_edge_labels: Whether to display labels on edges
        enable_tooltips: Whether to enable interactive tooltips
        high_quality_rendering: Whether to use high quality rendering
        animation_duration_ms: Duration of animations in milliseconds
        limit_node_count: Maximum number of nodes to render
        limit_edge_count: Maximum number of edges to render
        vis_width: Width of visualization in pixels
        vis_height: Height of visualization in pixels

    Usage:
        from word_forge.config import config

        # Access settings
        layout = config.graph.default_layout

        # Get visualization path
        viz_path = config.graph.get_visualization_path()

        # Create optimized config for interactive use
        interactive_config = config.graph.optimize_for_interactivity()
    """

    # Layout and visualization settings
    default_layout: GraphLayoutAlgorithm = GraphLayoutAlgorithm.FORCE_DIRECTED
    default_color_scheme: GraphColorScheme = GraphColorScheme.SEMANTIC

    # Paths for visualization and export
    visualization_path: str = str(DATA_ROOT / "visualizations")
    default_export_path: str = str(DATA_ROOT / "exports")
    default_export_format: GraphExportFormat = "svg"

    # Node and edge styling strategies
    node_size_strategy: GraphNodeSizeStrategy = "degree"
    edge_weight_strategy: GraphEdgeWeightStrategy = "similarity"

    # SQL templates for graph database operations
    sql_templates: SQLTemplates = field(
        default_factory=lambda: {
            "check_words_table": """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='words'
            """,
            "check_relationships_table": """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='relationships'
            """,
            "fetch_all_words": """
                SELECT id, term, definition FROM words
            """,
            "fetch_all_relationships": """
                SELECT word_id, related_term, relationship_type
                FROM relationships
            """,
            "get_all_words": """
                SELECT id, term, definition FROM words
            """,
            "get_all_relationships": """
                SELECT word_id, related_term, relationship_type
                FROM relationships
            """,
            "insert_sample_word": """
                INSERT OR IGNORE INTO words (term, definition, part_of_speech)
                VALUES (?, ?, ?)
            """,
            "insert_sample_relationship": """
                INSERT OR IGNORE INTO relationships
                (word_id, related_term, relationship_type)
                VALUES (?, ?, ?)
            """,
        }
    )

    # Sample relationships for initial graph population
    sample_relationships: List[tuple] = field(
        default_factory=lambda: [
            ("algorithm", "computation", "domain"),
            ("algorithm", "procedure", "synonym"),
            ("database", "storage", "function"),
            ("graph", "network", "synonym"),
            ("function", "procedure", "related"),
        ]
    )

    # Size constraints
    min_node_size: int = 5
    max_node_size: int = 30
    min_edge_width: float = 0.5
    max_edge_width: float = 5.0

    # Visualization dimensions
    vis_width: int = 1200
    vis_height: int = 800

    # Display options
    enable_labels: bool = True
    enable_edge_labels: bool = False
    enable_tooltips: bool = True
    high_quality_rendering: bool = True

    # Performance settings
    animation_duration_ms: int = 800
    limit_node_count: Optional[int] = 1000
    limit_edge_count: Optional[int] = 2000

    # Color mappings for semantic relationships
    relationship_colors: Dict[str, str] = field(
        default_factory=lambda: {
            "synonym": "#4287f5",  # Blue
            "antonym": "#f54242",  # Red
            "hypernym": "#42f584",  # Green
            "hyponym": "#a142f5",  # Purple
            "holonym": "#f5a142",  # Orange
            "meronym": "#42f5f5",  # Cyan
            "domain": "#7a42f5",  # Indigo
            "function": "#f542a7",  # Pink
            "related": "#42f5a1",  # Mint
            "default": "#aaaaaa",  # Gray
        }
    )

    # Export format options
    supported_export_formats: List[GraphExportFormat] = field(
        default_factory=lambda: ["graphml", "gexf", "json", "png", "svg", "pdf"]
    )

    # Environment variable mapping for configuration overrides
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_GRAPH_LAYOUT": ("default_layout", GraphLayoutAlgorithm),
        "WORD_FORGE_GRAPH_COLOR_SCHEME": ("default_color_scheme", GraphColorScheme),
        "WORD_FORGE_GRAPH_EXPORT_FORMAT": ("default_export_format", str),
        "WORD_FORGE_GRAPH_VIZ_PATH": ("visualization_path", str),
        "WORD_FORGE_GRAPH_EXPORT_PATH": ("default_export_path", str),
        "WORD_FORGE_GRAPH_HIGH_QUALITY": ("high_quality_rendering", bool),
        "WORD_FORGE_GRAPH_NODE_LIMIT": ("limit_node_count", int),
        "WORD_FORGE_GRAPH_VIS_WIDTH": ("vis_width", int),
        "WORD_FORGE_GRAPH_VIS_HEIGHT": ("vis_height", int),
    }

    # ==========================================
    # Cached Properties
    # ==========================================

    @cached_property
    def get_visualization_path(self) -> Path:
        """
        Get visualization path as a Path object.

        Returns:
            Path: Visualization directory path
        """
        return Path(self.visualization_path)

    @cached_property
    def get_export_path(self) -> Path:
        """
        Get export path as a Path object.

        Returns:
            Path: Export directory path
        """
        return Path(self.default_export_path)

    @cached_property
    def is_export_format_valid(self) -> bool:
        """
        Check if the configured export format is valid.

        Returns:
            bool: True if format is supported, False otherwise
        """
        return self.default_export_format in self.supported_export_formats

    # ==========================================
    # Public Methods
    # ==========================================

    def get_relationship_color(self, relationship_type: str) -> str:
        """
        Get color for a specific relationship type.

        Args:
            relationship_type: The type of relationship

        Returns:
            str: Hex color code for the relationship
        """
        return self.relationship_colors.get(
            relationship_type.lower(), self.relationship_colors["default"]
        )

    def with_layout(self, layout: GraphLayoutAlgorithm) -> "GraphConfig":
        """
        Create a new config with a different layout algorithm.

        Args:
            layout: New layout algorithm to use

        Returns:
            GraphConfig: New configuration instance
        """
        return GraphConfig(
            default_layout=layout,
            default_color_scheme=self.default_color_scheme,
            visualization_path=self.visualization_path,
            default_export_path=self.default_export_path,
            default_export_format=self.default_export_format,
            node_size_strategy=self.node_size_strategy,
            edge_weight_strategy=self.edge_weight_strategy,
            min_node_size=self.min_node_size,
            max_node_size=self.max_node_size,
            min_edge_width=self.min_edge_width,
            max_edge_width=self.max_edge_width,
            vis_width=self.vis_width,
            vis_height=self.vis_height,
            enable_labels=self.enable_labels,
            enable_edge_labels=self.enable_edge_labels,
            enable_tooltips=self.enable_tooltips,
            high_quality_rendering=self.high_quality_rendering,
            animation_duration_ms=self.animation_duration_ms,
            limit_node_count=self.limit_node_count,
            limit_edge_count=self.limit_edge_count,
            relationship_colors=self.relationship_colors,
            supported_export_formats=self.supported_export_formats,
            sql_templates=self.sql_templates,
            sample_relationships=self.sample_relationships,
        )

    def optimize_for_interactivity(self) -> "GraphConfig":
        """
        Create a new config optimized for interactive visualization.

        Returns:
            GraphConfig: New configuration with interactive-friendly settings
        """
        config = GraphConfig(
            default_layout=self.default_layout,
            default_color_scheme=self.default_color_scheme,
            visualization_path=self.visualization_path,
            default_export_path=self.default_export_path,
            default_export_format=self.default_export_format,
            node_size_strategy=self.node_size_strategy,
            edge_weight_strategy=self.edge_weight_strategy,
            min_node_size=self.min_node_size,
            max_node_size=self.max_node_size,
            min_edge_width=self.min_edge_width,
            max_edge_width=self.max_edge_width,
            vis_width=self.vis_width,
            vis_height=self.vis_height,
            enable_labels=True,
            enable_edge_labels=False,
            enable_tooltips=True,
            high_quality_rendering=False,  # Lower quality for better performance
            animation_duration_ms=300,  # Faster animations
            limit_node_count=200,  # Lower node count for responsiveness
            limit_edge_count=400,  # Lower edge count for responsiveness
            relationship_colors=self.relationship_colors,
            supported_export_formats=self.supported_export_formats,
            sql_templates=self.sql_templates,
            sample_relationships=self.sample_relationships,
        )

        return config

    def optimize_for_publication(self) -> "GraphConfig":
        """
        Create a new config optimized for high-quality export/publication.

        Returns:
            GraphConfig: New configuration with publication-quality settings
        """
        config = GraphConfig(
            default_layout=self.default_layout,
            default_color_scheme=self.default_color_scheme,
            visualization_path=self.visualization_path,
            default_export_path=self.default_export_path,
            default_export_format="svg",  # Vector format for publication
            node_size_strategy=self.node_size_strategy,
            edge_weight_strategy=self.edge_weight_strategy,
            min_node_size=8,  # Larger min size for visibility
            max_node_size=35,  # Larger max size for visibility
            min_edge_width=1.0,  # Thicker edges for clarity
            max_edge_width=6.0,  # Thicker edges for clarity
            vis_width=1600,  # Higher resolution for publication
            vis_height=1200,  # Higher resolution for publication
            enable_labels=self.enable_labels,
            enable_edge_labels=self.enable_edge_labels,
            enable_tooltips=False,  # No tooltips needed for static export
            high_quality_rendering=True,  # Maximum quality
            animation_duration_ms=0,  # No animation for static export
            limit_node_count=self.limit_node_count,
            limit_edge_count=self.limit_edge_count,
            relationship_colors=self.relationship_colors,
            supported_export_formats=self.supported_export_formats,
            sql_templates=self.sql_templates,
            sample_relationships=self.sample_relationships,
        )

        return config

    def get_export_filepath(self, graph_name: str) -> Path:
        """
        Generate full export filepath with proper extension.

        Args:
            graph_name: Base name for the graph file

        Returns:
            Path: Complete export path with filename and extension

        Raises:
            GraphConfigError: If the export format is not supported
        """
        if not self.is_export_format_valid:
            raise GraphConfigError(
                f"Export format '{self.default_export_format}' not supported. "
                f"Valid formats: {', '.join(self.supported_export_formats)}"
            )

        filename = f"{graph_name}.{self.default_export_format}"
        return self.get_export_path / filename

    def get_visualization_dimensions(self) -> Dict[str, int]:
        """
        Get width and height for visualization.

        Returns:
            Dict[str, int]: Width and height dimensions
        """
        return {
            "width": self.vis_width,
            "height": self.vis_height,
        }

    def get_display_settings(self) -> Dict[str, Any]:
        """
        Get display-related settings as a dictionary.

        Returns:
            Dict[str, Any]: Display configuration dictionary
        """
        return {
            "enable_labels": self.enable_labels,
            "enable_edge_labels": self.enable_edge_labels,
            "enable_tooltips": self.enable_tooltips,
            "high_quality_rendering": self.high_quality_rendering,
            "animation_duration_ms": self.animation_duration_ms,
            "min_node_size": self.min_node_size,
            "max_node_size": self.max_node_size,
            "min_edge_width": self.min_edge_width,
            "max_edge_width": self.max_edge_width,
            "vis_width": self.vis_width,
            "vis_height": self.vis_height,
        }

    def validate(self) -> None:
        """
        Validate the entire configuration for consistency and correctness.

        Raises:
            GraphConfigError: If any validation fails
        """
        errors = []

        # Validate export format
        if self.default_export_format not in self.supported_export_formats:
            errors.append(
                f"Export format '{self.default_export_format}' not supported. "
                f"Valid formats: {', '.join(self.supported_export_formats)}"
            )

        # Validate node size range
        if self.min_node_size >= self.max_node_size:
            errors.append(
                f"Minimum node size ({self.min_node_size}) must be less than "
                f"maximum node size ({self.max_node_size})"
            )

        # Validate edge width range
        if self.min_edge_width >= self.max_edge_width:
            errors.append(
                f"Minimum edge width ({self.min_edge_width}) must be less than "
                f"maximum edge width ({self.max_edge_width})"
            )

        # Validate visualization dimensions
        if self.vis_width <= 0:
            errors.append(f"Visualization width must be positive, got {self.vis_width}")
        if self.vis_height <= 0:
            errors.append(
                f"Visualization height must be positive, got {self.vis_height}"
            )

        # Validate node and edge limits
        if self.limit_node_count is not None and self.limit_node_count <= 0:
            errors.append(f"Node limit must be positive, got {self.limit_node_count}")

        if self.limit_edge_count is not None and self.limit_edge_count <= 0:
            errors.append(f"Edge limit must be positive, got {self.limit_edge_count}")

        if errors:
            raise GraphConfigError(
                f"Configuration validation failed: {'; '.join(errors)}"
            )


# ==========================================
# Module Exports
# ==========================================

__all__ = [
    # Configuration class
    "GraphConfig",
    # Type definitions
    "GraphLayoutAlgorithm",
    "GraphColorScheme",
    "GraphExportFormat",
    "GraphNodeSizeStrategy",
    "GraphEdgeWeightStrategy",
    # Error type
    "GraphConfigError",
]
