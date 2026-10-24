# Source-only manuscript audit

**Status:** `PASS`

This audit scans only canonical manuscript/code/config sources. Tests, temporary trees, build outputs, pytest sandboxes, and release archives are excluded.

## Evidence firewall

- Generated `TBD`, `pending`, and `NOT YET AUTHORIZED` entries remain execution-time interfaces, not scientific results.
- Operational progress counters cannot authorize manuscript claims.
- Missing empirical values are never estimated or copied by hand.

## Counts

| Check | Count |
|---|---:|
| source files | 165 |
| placeholder markers | 140 |
| intentional placeholders | 140 |
| blocking placeholders | 0 |
| provisional terms | 290 |
| labels | 25 |
| references | 2 |
| citations | 33 |
| bibliography entries | 33 |

## Submission blockers

- None in the source-only structural audit.

## Intentional unfinished evidence

The current paper remains pre-final wherever RQ2--RQ5 or cross-regime synthesis gates are incomplete. These placeholders are allowed only because the prose labels them as pending/provisional and the final submission gate rejects them.

## Machine-readable companion

See `SOURCE_ONLY_MANUSCRIPT_AUDIT.json` for path-level findings and SHA-256 bindings.
