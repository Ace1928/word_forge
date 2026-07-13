# Security policy

## Supported versions

Security fixes are provided for the latest minor release on the default branch.
Older experimental snapshots are not supported.

## Report a vulnerability

Do not open a public issue for a vulnerability or suspected credential leak.
Use GitHub's private vulnerability reporting feature for
`Ace1928/word_forge`. Include affected versions, impact, reproduction steps,
and any safe proof of concept.

You can expect an acknowledgement within seven days. A fix timeline depends on
severity and whether upstream libraries are involved. Please allow coordinated
disclosure before publishing details.

## Scope

Useful reports include SQL or command injection, unsafe file writes, path
traversal, secret exposure, malicious package contents, untrusted HTML handling,
and unexpected transmission of lexicon content to external providers.

Dependency vulnerabilities with no reachable Word Forge impact may be closed
with an explanation and tracked through normal maintenance.
