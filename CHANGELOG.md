# Changelog

All notable changes are documented here. This project follows Semantic
Versioning and the structure described by Keep a Changelog.

## [Unreleased]

### Added

- Standalone `Lexicon` API for local multilingual CRUD, search, relationships,
  translations, and statistics.
- Friendly `init`, `add`, `show`, `find`, `relate`, `translate`, `stats`, and
  `doctor` commands with JSON output.
- Cross-platform user-data paths and explicit portable-directory overrides.
- Dependency-free observability and optional service discovery inside the
  package.
- Production packaging, documentation, and clean-install quality gates.
- Authenticated loopback-only visual workspace with a coordinated accessible
  list and high-performance Sigma network view.
- Reproducible TypeScript browser build with locked dependencies and tests.
- Transactional schema migrations, stable lexical UUIDs, optimistic revisions,
  change history, derived-index invalidation records, and recoverable jobs.
- Independently versioned lexical senses with provenance, confidence, review
  state, and soft deletion.

### Changed

- Removed runtime coupling to Eidosian Forge's private `eidosian_core` and
  shared CLI modules.
- Made neural vector and LLM processing explicitly opt-in.
- Moved generated databases, graphs, indexes, logs, and models out of the source
  and installation directories.

### Removed

- Generated databases, vector files, visualizations, legacy source snapshots,
  and local editor state from the distributable repository.

## [0.1.0] - 2025-04-01

### Added

- Initial experimental lexical database, graph, parser, worker, emotion, and
  vector components.

[Unreleased]: https://github.com/Ace1928/word_forge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Ace1928/word_forge/releases/tag/v0.1.0
