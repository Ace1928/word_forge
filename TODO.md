# Word Forge Advanced MVP TODO

This list distills the current expansion plan into concrete deliverables. Items are ordered approximately in the sequence they should be tackled. Each completed item should be committed and documented in the changelog.

1. **Finalize Test Coverage**
   - [ ] Ensure each core module has unit tests: database, queue, parser, worker, CLI, vectorizer, graph, conversation, and emotion.
     - [x] Queue manager
   - [ ] Stub or mock heavy dependencies (torch, chromadb) so tests run without external resources.
   - [ ] Integrate tests into CI pipeline.

2. **Vectorization Enhancements**
   - [ ] Confirm `VectorStore` persists embeddings to disk and reloads correctly.
   - [ ] Flesh out `VectorWorker` polling logic for detecting new or updated entries.
   - [ ] Add CLI commands for building and querying the vector index.

3. **Graph Module**
   - [x] Verify `GraphManager` can rebuild the network from the database.
   - [ ] Implement incremental updates in `GraphWorker` to avoid full rebuilds.
   - [ ] Provide export utilities (GEXF/GraphML) for external visualization tools.

4. **Conversation & Emotion**
   - [ ] Store conversations and messages using `ConversationManager`.
   - [ ] Integrate `EmotionManager` to annotate messages with valence/arousal.
   - [ ] Expose conversation commands through the CLI.

5. **Worker Orchestration**
   - [ ] Create a `WorkerManager` to start, stop, and monitor all background threads.
   - [ ] Allow modules to be enabled/disabled via configuration flags.

6. **Documentation Updates**
   - [ ] Expand `docs/overview.md` with diagrams showing module interactions.
   - [ ] Keep `docs/glossary.md` updated as new terminology arises.
   - [ ] Maintain `upgrade_plan.md` with notes on design decisions and future ideas.

