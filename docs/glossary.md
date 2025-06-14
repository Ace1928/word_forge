# Word Forge Glossary

This living glossary defines key terms used throughout the project. New entries should remain in **alphabetical order** to make lookup simple.

- **ChromaDB** — vector database used for embedding storage and similarity search.
- **Conversation Manager** — orchestrates multi-step conversation sessions using several language models and persists messages to the database.
- **Eidosian** — design philosophy emphasizing type safety, clear layering and continuous self‑improvement.
- **Emotion Manager** — analyzes text to derive sentiment and emotional valence, integrating results with the conversation and graph modules.
- **Graph Manager** — central orchestrator that builds the semantic network from lexical and emotional data using NetworkX.
- **Graph Worker** — background thread that keeps the lexical graph up to date and saves periodic snapshots.
- **Lexical Data** — structured information about words including definitions, usage examples and relationships.
- **NetworkX** — library used for graph operations and network analysis.
- **NumPy** — array library providing efficient numeric computation used by vector features.
- **Queue Manager** — thread-safe task queue coordinating asynchronous worker threads.
- **Vector Store** — component providing persistent storage for vector embeddings.
- **Vector Worker** — background thread that generates embeddings for new or updated content and inserts them into the vector store.
- **Worker Manager** — utility orchestrating multiple background workers such as vector and graph processors.
- **Word Forge** — the toolkit for building and exploring a semantic network of terms with vector search and emotion analysis.
- **WordNet** — lexical database from Princeton that supplies synonyms and sense relationships.

