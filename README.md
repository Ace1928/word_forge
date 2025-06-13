# Word Forge

Word Forge is a lexical data processing and enrichment system.

## Packaging

The project relies on **setuptools** to build distribution artifacts. Running

```bash
python -m build
```

will create the standard `src/word_forge.egg-info` directory along with wheel
and source distributions. Because this directory is generated automatically, it
is excluded from version control via `.gitignore`.

