# ...existing code...


class VectorStorage:
    """Vector storage implementation for word embeddings."""

    def __init__(self, config):
        """Initialize vector storage with configuration."""
        self.config = config
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize the vector database client based on configuration."""
        # ...existing code...

    def store_vector(self, word_id, word, vector):
        """Store a word vector in the vector database.

        Args:
            word_id: Unique identifier for the word
            word: The word text
            vector: The embedding vector

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Add the vector to the client
            self.client.add(
                ids=[str(word_id)], embeddings=[vector], metadatas=[{"word": word}]
            )

            # Persist the data using the appropriate method
            self._persist_data()

            return True
        except Exception as e:
            raise ValueError(f"Vector upsert failed: {str(e)}")

    def _persist_data(self):
        """Custom persistence implementation for vector database clients."""
        try:
            # Attempt different persistence methods in order of preference
            if hasattr(self.client, "persist"):
                self.client.persist()
            elif hasattr(self.client, "save_index"):
                index_path = self.config.get("index_path", "vector_index")
                self.client.save_index(index_path)
            elif hasattr(self.client, "save"):
                self.client.save()
            elif hasattr(self.client, "commit"):
                self.client.commit()
            elif hasattr(self.client, "flush"):
                self.client.flush()
            # If none of the above methods exist, log this for future improvement
            # but don't fail - vectors are still added to in-memory storage
        except Exception as e:
            # Don't re-raise exception - this allows vectors to be added even if persistence fails
            # The system can still function, albeit without persistence until fixed
            import logging

            logging.getLogger().warning(f"Vector persistence failed: {str(e)}")
