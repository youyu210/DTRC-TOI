# A4 exact checkpoints

The frozen natural small-alpha states were materialized as schema-v3 table/S38
and bunny/S45 checkpoints before any A4 candidate run. Both manifests are
complete and report the expected physical frame.

The independent prefix replay differs from the A2 selection trajectory by a
maximum free-cloth position RMS of `2.319054630e-05 m` for table and
`6.069220436e-03 m` for bunny. This is recorded repeat divergence; A4 compares
all methods from the exact same saved checkpoint and does not use the selection
trajectory as a numerical reference.

The binary SHA-256 is `84c429f7e30dcd15f420cb22b50b90972045905d3981481eb97f8201d0178ae9`.
