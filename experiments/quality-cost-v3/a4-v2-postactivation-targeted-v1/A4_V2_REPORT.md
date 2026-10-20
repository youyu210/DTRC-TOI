# A4-v2 post-activation-alpha targeted matrix

Two states selected before candidate runs from naturally accepted small line-search steps were run with six configurations and three repeats (36 paired single steps). No CCD alpha was changed. Every method starts from the same schema-v3 checkpoint within each state.

## Reference resolution

| State | strict2 complete | strict4 complete | chosen | strict2↔strict4 max position RMS (m) | repeat-noise max RMS (m) | resolved band |
|---|---:|---:|---|---:|---:|---|
| table-f28 | 3/3 | 3/3 | strict4 | 1.67228e-05 | 2.03433e-07 | unresolved |
| bunny-f46 | 3/3 | 3/3 | strict4 | 1.7185e-06 | 1.75979e-07 | 0.0001 |

## Results

### table-f28 (natural minimum accepted alpha 0.507493)

| Config | Newton | seconds | speedup | position RMS max (m) | raw / validated band | history rejects | shadow history-veto samples | active samples | observed min alpha |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| graph_n1 | 14 | 0.7125 | 1.000× | 1.75461e-05 | 0.0001 / unresolved | 0 | 0 | 30 | 0.505457 |
| toi | 9 | 0.4920 | 1.448× | 7.55034e-05 | >0.0001 / unresolved | 0 | 0 | 15 | 0.506058 |
| comp03 | 10 | 0.5317 | 1.338× | 3.79777e-05 | >0.0001 / unresolved | 0 | 0 | 17 | 0.501861 |
| gate03 | 10 | 0.5340 | 1.339× | 3.58685e-05 | >0.0001 / unresolved | 0 | 0 | 18 | 0.505939 |

### bunny-f46 (natural minimum accepted alpha 0.5)

| Config | Newton | seconds | speedup | position RMS max (m) | raw / validated band | history rejects | shadow history-veto samples | active samples | observed min alpha |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| graph_n1 | 105 | 3.2565 | 1.000× | 0.00012176 | >0.0001 / >0.0001 | 0 | 0 | 306 | 0.5 |
| toi | 6 | 0.3787 | 8.568× | 0.00220247 | >0.0001 / >0.0001 | 0 | 0 | 6 | 1 |
| comp03 | 12 | 0.5689 | 5.808× | 0.000912196 | >0.0001 / >0.0001 | 0 | 0 | 23 | 0.548392 |
| gate03 | 11 | 0.5371 | 6.003× | 0.000916895 | >0.0001 / >0.0001 | 0 | 0 | 21 | 0.549564 |

## Findings

- The compensated method recorded zero history-only rejections and zero shadow history-veto samples at εr=0.03 on the restored trajectories.
- State selection minimized accepted alpha only among Newton k>=5. A separate restore preflight before candidate runs reproduced post-activation minima of about 0.543/0.5, so this matrix tests the intended active history window.
- Timing and output differences between compensated and residual-gated runs cannot be attributed to history when the history-only counter is zero; independent GPU trajectories remain a source of run-to-run divergence.
- The strict reference is a numerical reference. Its disagreement and repeat noise determine the finest defensible error band above.

Derived values are in `analysis.json`; raw file hashes are in `EVIDENCE_MANIFEST.json`.
