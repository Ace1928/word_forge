# Performance and scale

## Fast default path

Manual CRUD, counts, exact lookup, and bounded text search execute directly in
SQLite. Count operations do not load tables into Python memory. Search limits
are validated and capped at 1,000 results per call.

## Concurrency

SQLite uses foreign keys, write-ahead logging, and normal synchronous mode. This
supports concurrent readers and serialized writes well for a local application.
Worker pools are intended for enrichment jobs, not for bypassing SQLite's write
coordination.

## Neural indexes

Embeddings are derived data. Build them in batches, keep the original model name
with index metadata, and rebuild after changing embedding dimensions. FAISS and
HNSW favor in-process speed; Chroma favors a richer persistence/query surface.

## Graphs

Full force-directed layouts become expensive as edge counts grow. Use bounded
subgraphs for interactive exploration, perform full analysis offline, and avoid
regenerating HTML for every small write.

## Benchmarking

Run the deterministic core benchmark from a development install:

```bash
python scripts/benchmark_core.py
```

Treat local numbers as hardware-specific. Production changes should compare the
same dataset, Python version, optional extras, and warm-up policy before and
after the change.
