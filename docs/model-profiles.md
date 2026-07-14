# Local model profiles

Word Forge treats generative enrichment as optional. Definitions, relationships,
and usage examples from lexical sources work offline and are preferred over
generated text. A model is loaded lazily only when a source-backed example is
missing and the user explicitly enables a model or profile.

## Choose a profile

```bash
python -m pip install -e ".[llm]"
word_forge models list
word_forge models recommend
word_forge start language --llm-profile portable
```

`word_forge models list --json` returns the detected accelerator, dependency
versions, available memory, readiness issues, and warnings for automation.

| Profile | Model | Parameters | Context | Operational RAM minimum / recommended |
|---|---|---:|---:|---:|
| `off` | None | 0 | 0 | 0 / 0 GiB |
| `portable` | `Qwen/Qwen2.5-0.5B-Instruct` | 0.49B | 32,768 | 2.5 / 4 GiB |
| `gemma3-tiny` | `google/gemma-3-270m-it` | 0.27B | 32,768 | 2 / 3 GiB |
| `gemma4-edge` | `google/gemma-4-E2B-it` | 5.1B total (2.3B effective) | 131,072 | 14 / 20 GiB |

The RAM figures are conservative Word Forge operational estimates for standard
Transformers loading, not vendor guarantees. Quantized runtimes can use less;
context growth and multimodal inputs can use substantially more.

## Why portable is the CPU default

The portable Qwen model is Apache-2.0, ungated, 0.49B parameters, supports more
than 29 languages, and uses the normal Transformers causal-language-model API.
That combination makes first-run automation predictable. See the
[Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

Gemma 3 270M is smaller and is a good constrained-device option, but Hugging Face
requires users to accept the Gemma terms before downloading it. Its model card
documents the 270M/1B 32K context and Gemma 3's broad multilingual training:
[Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3) and
[270M repository](https://huggingface.co/google/gemma-3-270m-it).

Gemma 4 E2B is the modern high-capability option requested for Word Forge. Google
documents 2.3B effective parameters, 5.1B total parameters with embeddings, a
128K context, Apache-2.0 licensing, and multilingual training over 140 languages.
It requires Transformers 5.5 or newer and is not the CPU default because standard
unquantized weights are much larger. See the
[Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) and
[E2B repository](https://huggingface.co/google/gemma-4-E2B-it).

## Selection behavior

- `--llm-profile auto` selects `off` when the LLM extra is unavailable.
- On a CPU with enough memory, `auto` selects the ungated `portable` profile.
- On CUDA with at least 12 GiB accelerator memory and sufficient system RAM,
  `auto` selects `gemma4-edge`.
- Explicit profiles are rejected before workers or databases start when their
  hard dependency or available-memory requirements are not met.
- `--llm-model MODEL_ID` bypasses the catalog for advanced users.
- `--llm-profile off` always preserves the deterministic offline path.

Model output is enrichment, not authority. Applications should retain provenance,
review generated lexical claims, and apply model-provider terms independently of
Word Forge's MIT-licensed source code.
