# Test suite

The test suite covers both the stable local-first API and the original advanced
subsystems retained from Word Forge's experimental history.

```bash
python -m pytest
```

Tests use temporary databases and must not write into the repository. Optional
vector or model tests skip when their feature extra is not installed; the skip
reason states the exact missing capability. Network model loading is never a
requirement for the default suite.

Focused release checks:

```bash
python -m pytest tests/test_lexicon.py tests/test_database_migrations.py
python -m pytest tests/test_web_api.py
npm test
```

Fixtures contain synthetic data only. Add a regression test whenever changing
a public command, data migration, validation rule, or failure mode.
