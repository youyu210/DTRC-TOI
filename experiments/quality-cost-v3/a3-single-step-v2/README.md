# A3 paired single-step matrix

This directory contains the completed quality/cost matrix for eight frozen
states, fourteen configurations, and three repeats (336 runs). All runs use one
budget-corrected executable and exact schema-v3 checkpoints.

`A3_REPORT.md` is the review summary and `analysis.json` contains paired errors,
reference resolution, speedups, and rejection counts. Large per-run arrays stay
in the local workspace; `EVIDENCE_MANIFEST.json` records every raw file hash.

The earlier `a3-single-step-v1-prebudgetfix` attempt is retained separately and
is excluded from every result here.
