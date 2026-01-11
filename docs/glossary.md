# Word Forge Glossary

This living glossary defines key terms used throughout the project. New entries should remain in **alphabetical order** to make lookup simple.

## A

- **Arousal** — Emotional intensity dimension ranging from calm (0.0) to excited (1.0). Used alongside valence for dimensional emotion analysis.

## C

- **ChromaDB** — Vector database used for embedding storage and similarity search. Provides persistent storage with efficient nearest-neighbor queries.
- **Circuit Breaker** — Pattern for preventing cascading failures by temporarily stopping requests to failing services.
- **Configuration Component** — Dataclass implementing the `ConfigComponent` protocol, providing type-safe settings for a specific subsystem.
- **Conversation Manager** — Orchestrates multi-step conversation sessions using several language models and persists messages to the database.

## D

- **DBManager** — Core class handling all database operations including connection pooling, transactions, and schema management.
- **Dimensional Analysis** — Emotion analysis using continuous dimensions (valence/arousal) rather than discrete categories.

## E

- **Eidosian** — Design philosophy emphasizing type safety, clear layering, self-documenting code, and continuous self‑improvement.
- **Emotion Manager** — Unified interface for emotional processing, integrating VADER, TextBlob, and optional LLM analysis.
- **Emotion Vector** — Mathematical representation of emotional state in multidimensional space (valence, arousal, dominance).

## F

- **FAISS** — Facebook AI Similarity Search library used as an alternative vector index backend.
- **Force-Directed Layout** — Graph visualization algorithm using physics simulation to arrange nodes naturally.

## G

- **Graph Builder** — Component that constructs the NetworkX graph from database relationships.
- **Graph Manager** — Central orchestrator that builds the semantic network from lexical and emotional data using NetworkX.
- **Graph Worker** — Background thread that keeps the lexical graph up to date and saves periodic snapshots.

## L

- **Lexical Data** — Structured information about words including definitions, usage examples, and relationships.
- **LRU Cache** — Least Recently Used cache pattern for memoizing expensive function calls.

## N

- **NetworkX** — Python library used for graph operations and network analysis.
- **NumPy** — Array library providing efficient numeric computation used by vector features.

## P

- **Parser Refiner** — Main parsing pipeline that extracts lexical entries from text and enriches them with relationships.
- **Priority Queue** — Queue data structure where items are processed based on priority level rather than arrival order.

## Q

- **Queue Manager** — Thread-safe task queue coordinating asynchronous worker threads with priority support.
- **Queue Processor** — Protocol defining the interface for components that process items from the queue.

## R

- **Relationship** — Connection between words (synonyms, antonyms, hypernyms, etc.) stored in the database and graph.
- **Result Pattern** — Monadic error handling pattern that avoids exceptions for cross-component error propagation.

## S

- **Sentence Transformer** — Neural network model for generating dense vector embeddings from text.
- **SQLite** — Embedded relational database used for persistent storage.
- **Synset** — WordNet concept grouping related word senses.

## T

- **Term Extractor** — NLP component that discovers and extracts significant terms from text content.
- **Transaction** — Database operation unit that ensures atomicity and consistency.

## V

- **VADER** — Valence Aware Dictionary and sEntiment Reasoner, rule-based sentiment analysis tool.
- **Valence** — Emotional pleasantness dimension ranging from negative (-1.0) to positive (1.0).
- **Vector Store** — Component providing persistent storage and similarity search for vector embeddings.
- **Vector Worker** — Background thread that generates embeddings for new or updated content and inserts them into the vector store.

## W

- **Word Forge** — The toolkit for building and exploring a semantic network of terms with vector search and emotion analysis.
- **WordNet** — Lexical database from Princeton that supplies synonyms, definitions, and sense relationships.
- **Worker Manager** — Utility orchestrating multiple background workers such as vector and graph processors.
- **Worker Pool** — Collection of worker threads processing items in parallel.
