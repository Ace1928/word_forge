# ...existing code...


class StorageFactory:
    """Factory for creating storage implementations."""

    @staticmethod
    def create_vector_storage(config):
        """Create and return a vector storage instance.

        Args:
            config: Configuration for the vector storage

        Returns:
            VectorStorage: An initialized vector storage instance
        """
        # Ensure we have default values for common configuration
        if "index_path" not in config:
            config["index_path"] = "vector_index"

        # Add any other defaults needed

        from word_forge.storage.vector_storage import VectorStorage

        return VectorStorage(config)

    # ...existing code...
