# P0 checkpoint validation

- Schema: `stiffgipc-step-boundary-v3`
- Byte-identical load/save roundtrip: `True`
- Restored initial state equals checkpoint state: `True`
- Truncated checkpoint rejected: `True`
- Position continuation max RMS: `1.6613252206998826e-09` m
- Position repeat-noise max RMS: `1.6699410111790925e-09` m
- Velocity continuation max RMS: `1.6613252206998825e-07` m/s
- Velocity repeat-noise max RMS: `1.6699410111790925e-07` m/s
- Default path compatible with frozen V2 on table/bunny: `True`
- Native RHS audits passed: `True`
- P0 numerical/repeatability checks passed: `True`

See `analysis.json` for all pairwise errors and per-run control-flow counts.
