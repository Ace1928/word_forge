# ...existing code...


class EmbeddingService:
    """Service for generating and managing word embeddings."""

    def __init__(self, vector_storage, embedding_model):
        """Initialize the embedding service.

        Args:
            vector_storage: Storage for vectors
            embedding_model: Model to generate embeddings
        """
        self.vector_storage = vector_storage
        self.embedding_model = embedding_model

    async def process_word(self, word_id, word):
        """Process a word by generating its embedding and storing it.

        Args:
            word_id: ID of the word
            word: Word text to process

        Returns:
            bool: True if successful
        """
        # Generate embedding
        vector = await self.embedding_model.generate_embedding(word)

        # Store in vector database
        success = self.vector_storage.store_vector(word_id, word, vector)

        return success

    # ...existing code...
