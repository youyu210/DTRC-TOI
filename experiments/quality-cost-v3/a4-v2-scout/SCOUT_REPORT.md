# A4-v2 same-run post-activation state selection

The correction scout ran Graph N1 for 50 frames in `table` and `bunny` while
saving checkpoints S1 through S49 in the same executions. The first frozen
selection minimized alpha across all Newton updates; pre-candidate verification
showed that its minima occurred at `k=0`, before the six-update termination gate.
That file is retained as `A4_V2_SELECTED_STATES.json`.

The candidate selection used for A4-v2 is
`A4_V2_POSTACTIVATION_SELECTED_STATES.json`. It excludes the eight A2 states and
minimizes naturally accepted alpha among Newton updates with `k>=5`:

| State | checkpoint | source post-activation alpha | restored alpha | source/restored Newton |
|---|---|---:|---:|---:|
| table-f28 | table/S27 | 0.507492565 | 0.543367218 | 14 / 14 |
| bunny-f46 | bunny/S45 | 0.5 | 0.5 | 105 / 105 |

Both restore checks passed before candidate runs. No CCD alpha was modified.
The executable SHA-256 is
`84c429f7e30dcd15f420cb22b50b90972045905d3981481eb97f8201d0178ae9`.
