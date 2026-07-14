# Multilingual embedding models

Word Forge uses `intfloat/multilingual-e5-small` as its portable semantic-search
default. It is MIT-licensed, produces 384-dimensional vectors, and supports the
100-language XLM-R training set, with lower quality possible for low-resource
languages. Its model files are roughly 471 MB. See the
[official model card](https://huggingface.co/intfloat/multilingual-e5-small).

The model was trained with asymmetric retrieval prefixes. Word Forge therefore
embeds searches as `query: …` and indexed terms, definitions, and examples as
`passage: …`, including for non-English content. These prefixes are part of the
embedding space and changing them requires rebuilding the index.

## Reproducible model isolation

One `VectorStore` owns one Sentence Transformers instance. The worker reuses
that instance instead of loading a second copy. A deterministic collection name
derived from the complete model identifier keeps vectors from different models
separate. Explicit dimensions must exactly match the loaded model; Word Forge
never pads or truncates embeddings because doing so corrupts semantic geometry.

Choose another public Sentence Transformers model with either interface:

```bash
word_forge start language --vector-model MODEL_ID
word_forge vector index --embedder MODEL_ID
```

Known E5 variants receive their documented formatting automatically:

- Standard multilingual E5 uses `query:` for searches and `passage:` for
  indexed documents.
- `intfloat/multilingual-e5-large-instruct` uses
  `Instruct: <task>\nQuery: <query>` for searches and raw documents, as specified
  by its [official model card](https://huggingface.co/intfloat/multilingual-e5-large-instruct).
- Unknown models receive no guessed prompt. Their own Sentence Transformers
  configuration remains authoritative.

Set `WORD_FORGE_VECTOR_MODEL` or use a typed JSON/YAML configuration file to
change the default. Re-run `word_forge vector index` after any model or prompt
contract change.
