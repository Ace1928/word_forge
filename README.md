# Word Forge

Word Forge is a modular system for building and enriching a lexical database. It integrates multiple resources, including WordNet, OpenThesaurus and transformer-based models, to collect definitions, examples and semantic relations.

## NLTK Data

Several components rely on datasets distributed with NLTK. These files are downloaded automatically the first time Word Forge accesses WordNet or related features via `ensure_nltk_data()`.

Ensure the running environment has internet access on the initial run so these resources can be retrieved.
