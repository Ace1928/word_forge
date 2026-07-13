# Security and privacy

## Local data

Word Forge stores lexicon content locally by default. It does not require an
account, telemetry endpoint, or hosted database. File permissions are inherited
from the current user's operating-system settings.

## External model providers

When an Ollama-compatible or other remote provider is configured, text sent for
generation or embedding leaves the local process. Review that provider's data
handling before sending confidential lexicon content. Core commands do not call
model providers.

## Input handling

- SQLite operations use parameters rather than interpolating user values.
- Search escapes SQLite wildcard characters.
- Terms reject NUL characters and excessive lengths.
- Language tags and result limits are validated before database work.
- JSON output preserves Unicode and never evaluates stored text.

## Generated HTML

Graph visualization can contain stored terms and definitions. Treat generated
HTML as potentially sensitive and do not publish it without reviewing the
underlying data. Open it only in an up-to-date browser.

## Local workspace

`word-forge gui` accepts loopback hosts only. It validates HTTP host headers,
rejects cross-origin mutations and WebSocket connections, applies a restrictive
content security policy, and ships all browser assets inside the wheel. It is
not an authenticated multi-user server. Do not expose it through a tunnel or
reverse proxy without adding reviewed authentication, TLS, and authorization.

## Secrets

Do not put API keys, browser cookies, or credentials in a lexicon, configuration
file, sample, test fixture, command history, or Git commit. Use the secret
mechanism provided by the external service or deployment environment.

See the repository-level [security policy](../SECURITY.md) for reporting a
vulnerability.
