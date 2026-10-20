# A2 frozen-state checkpoints

Eight schema-v3 checkpoints were generated after the representative frame list
was frozen. Each scene was replayed once, and all four checkpoints for that
scene were written by the same process.

| Scene | Checkpoints | Contact counts |
|---|---|---|
| table | S2, S21, S22, S39 | 2, 2, 51,750, 40,375 |
| bunny | S2, S47, S48, S49 | 0, 4,621, 5,851, 5,942 |

Every manifest reports `stiffgipc-step-boundary-v3`, `complete=true`, and the
expected physical frame. All later candidate and reference runs for a state
must load the same directory listed here.

The checkpoint-generation replay and the selection replay are separate GPU
runs. Table remained close through S39 (maximum free-cloth position RMS
`1.810535812e-05 m`). Bunny diverged nonlinearly by S49 (maximum position RMS
`8.570457274e-03 m`), although the contact counts remain representative of the
selected late-contact region. Consequently, the selection run is not treated
as a pointwise numerical reference. Candidate quality is evaluated only by
paired single steps from the exact same saved checkpoint.

`analysis.json` contains the replay differences and checkpoint manifest hashes.
All local binary arrays and traces are indexed by `EVIDENCE_MANIFEST.json`.
