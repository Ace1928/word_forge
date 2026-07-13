# Visual workspace

The Word Forge workspace is the simplest way to use the living lexicon. It is a
local web application: the database and server stay on your computer, while a
normal browser provides a responsive interface.

## Start it

```bash
python -m pip install ".[gui]"
word-forge gui
```

Word Forge prints and opens a URL containing a one-time session secret. The
secret is moved from the URL into temporary browser storage immediately. A new
secret is created every time the workspace starts.

Use a specific portable lexicon or another local port when needed:

```bash
word-forge gui --db-path ./research.sqlite --port 9876
```

## Main areas

- **Overview** shows exact counts for words, language-aware lexemes, languages,
  and connections.
- **Network** renders a live WebGL graph. Search and language filters keep the
  view meaningful instead of drawing an unreadable whole-database hairball.
  Zoom controls and a structured accessible list provide equivalent access.
- **Lexicon** shows normal searchable entry cards. This remains the primary
  information view for people who do not use or cannot see the graph.
- **Operations** checks database integrity, creates managed online snapshots,
  lists their sizes and dates, and restores a selected snapshot only after the
  user types `RESTORE`.

## Edit an entry

1. Search for a term and select its card or graph node.
2. Use **Edit** in the inspector.
3. Change the definition, part of speech, examples, base term, or source.
4. Save. The hidden revision guard prevents an older editor from overwriting a
   newer change; a conflict is shown instead of losing data.

Use **+ Sense** for a separate meaning of the same spelling. Each sense has its
own definition, domain, register, source, confidence, review state, UUID, and
revision. Use **Remove** to tombstone a sense without erasing its history.

Use **+ Relation** to connect English base-layer terms with a named semantic
edge. Use **+ Translation** for language-aware one-way or bidirectional links.
Every successful mutation updates the inspector, counts, accessible list, and
graph through the sequenced live event channel.

## Back up and restore

The Operations page does not copy a live database file naively. **Create
snapshot** uses SQLite's online backup mechanism, validates the result, flushes
it, and atomically publishes it in the database's `backups` directory.

Restoring is intentionally explicit:

1. Select a managed snapshot.
2. Type `RESTORE` exactly.
3. Choose **Restore selected snapshot**.

Normal operations drain before replacement, every SQLite connection is closed,
the snapshot is validated again, and the workspace reopens the restored file.

## Live updates and recovery

Every event has a monotonic sequence number. A briefly disconnected browser
asks for missed events when it reconnects. If its gap is older than the bounded
replay history, the server sends `resync.required` and the workspace refetches
authoritative database state. Slow tabs never block database writers.

## Security boundary

The workspace binds only to `127.0.0.1`, `localhost`, or `::1`. It checks the
Host and Origin headers, requires the session secret for HTTP and WebSocket
access, rejects unknown request fields, and serves a restrictive content
security policy with no remote scripts, fonts, analytics, or telemetry.

Loopback is a single-user convenience boundary, not multi-user authentication.
Do not expose the port through a public tunnel or reverse proxy without adding
reviewed TLS, authentication, authorization, and deployment-specific limits.
