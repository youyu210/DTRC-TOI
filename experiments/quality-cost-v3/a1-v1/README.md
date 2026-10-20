# A1 table frame-23 dual-tolerance diagnosis

This directory records the preregistered 2x2 Newton/PCG tolerance diagnosis from
one schema-v3 S22 checkpoint. Each configuration was repeated three times. The
strict-Newton configurations reached the 2,000-iteration diagnostic cap; this
is a censored diagnostic result rather than a converged reference.

`A1_REPORT.md` contains the main finding and `analysis.json` contains every
derived metric. Large checkpoint arrays, exported states, and Newton traces are
kept in the local workspace. `EVIDENCE_MANIFEST.json` records their sizes and
SHA-256 hashes so the committed report remains auditable without putting about
150 MiB of generated data in Git history.
