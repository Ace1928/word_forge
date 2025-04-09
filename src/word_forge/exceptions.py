"""
Common exceptions shared across Word Forge modules.

This module provides base exceptions used throughout the Word Forge system,
ensuring consistent error handling and avoiding circular imports.
"""

# Generic Exception Classes


class WordForgeError(Exception):
    """Base exception for all Word Forge errors."""

    def __init__(self, message: str, cause: Exception) -> None:
        """
        Initialize with detailed error message and optional cause.

        Args:
            message: Error description with context
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.__cause__ = cause
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        """Provide detailed error message including cause if available."""
        error_msg = self.message
        if self.cause:
            error_msg += f" | Cause: {str(self.cause)}"
        return error_msg


class DatabaseError(WordForgeError):
    """Base exception for database operations."""

    pass


class GraphError(WordForgeError):
    """Base exception for graph operations."""

    pass


class GraphAnalysisError(GraphError):
    """Base exception for graph analysis operations."""

    pass


class GraphExportError(GraphError):
    """Base exception for graph export operations."""

    pass


class GraphImportError(GraphError):
    """Base exception for graph import operations."""

    pass


class GraphUpdateError(GraphError):
    """Base exception for graph update operations."""

    pass


class GraphQueryError(GraphError):
    """Base exception for graph query operations."""

    pass


class GraphConnectionError(GraphError):
    """Base exception for graph connection operations."""

    pass


class GraphTraversalError(GraphError):
    """Base exception for graph traversal operations."""

    pass


class GraphStorageError(GraphError):
    """Base exception for graph storage operations."""

    pass


class GraphSerializationError(GraphError):
    """Base exception for graph serialization operations."""

    pass


class GraphIOError(GraphError):
    """Base exception for graph input/output operations."""

    pass


class GraphLayoutError(GraphError):
    """Base exception for graph layout operations."""

    pass


class QueueError(WordForgeError):
    """Base exception for queue operations."""

    pass


class ParserError(WordForgeError):
    """Base exception for parser operations."""

    pass


# Word Forge Specific Exceptions


# Database Specific Exceptions


# Graph Specific Exceptions


class NodeNotFoundError(GraphError):
    """
    Raised when a term lookup fails within the graph.

    This occurs when attempting to access a node that doesn't exist,
    typically during relationship or subgraph operations.
    """

    pass


class GraphDataError(GraphError):
    """
    Raised when graph data structure contains inconsistencies.

    This indicates a structural problem with the graph data itself,
    such as missing required node attributes or invalid edge structures.
    """

    pass


class GraphVisualizationError(GraphError):
    """
    Raised when graph visualization generation fails.

    This typically occurs during rendering operations, HTML generation,
    or when visualization libraries encounter errors.
    """

    pass


class GraphDimensionError(GraphError):
    """
    Raised when graph dimensional operations fail.

    This occurs when attempting to set invalid dimensions or
    when dimensional operations (like projection) fail.
    """

    pass


# Queue Specific Exceptions


# Parser Specific Exceptions
