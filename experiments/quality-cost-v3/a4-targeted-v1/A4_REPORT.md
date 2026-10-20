# A4 natural-small-alpha targeted matrix

Two states selected before candidate runs from naturally accepted small line-search steps were run with six configurations and three repeats (36 paired single steps). No CCD alpha was changed. Every method starts from the same schema-v3 checkpoint within each state.

## Reference resolution

| State | strict2 complete | strict4 complete | chosen | strict2↔strict4 max position RMS (m) | repeat-noise max RMS (m) | resolved band |
|---|---:|---:|---|---:|---:|---|
| table-f39 | 3/3 | 3/3 | strict4 | 2.28242e-06 | 1.77402e-08 | 0.0001 |
| bunny-f46 | 3/3 | 3/3 | strict4 | 4.83305e-06 | 3.38588e-08 | 0.0001 |

## Results

### table-f39 (natural minimum accepted alpha 0.0510726)

| Config | Newton | seconds | speedup | position RMS max (m) | raw / validated band | history rejects | shadow history-veto samples | active samples | observed min alpha |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| graph_n1 | 96 | 3.6610 | 1.000× | 3.14426e-05 | >0.0001 / >0.0001 | 0 | 0 | 276 | 0.653719 |
| toi | 6 | 0.3792 | 9.666× | 0.000779284 | >0.0001 / >0.0001 | 0 | 0 | 6 | 1 |
| comp03 | 15 | 0.7549 | 4.847× | 0.000262204 | >0.0001 / >0.0001 | 0 | 0 | 33 | 0.653512 |
| gate03 | 15 | 0.7555 | 4.843× | 0.000262357 | >0.0001 / >0.0001 | 0 | 0 | 33 | 0.65346 |

### bunny-f46 (natural minimum accepted alpha 0.0907136)

| Config | Newton | seconds | speedup | position RMS max (m) | raw / validated band | history rejects | shadow history-veto samples | active samples | observed min alpha |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| graph_n1 | 57 | 1.9012 | 1.000× | 6.43481e-06 | 0.0001 / 0.0001 | 0 | 0 | 159 | 0.5 |
| toi | 7 | 0.4170 | 4.562× | 0.00191239 | >0.0001 / >0.0001 | 0 | 0 | 9 | 0.5 |
| comp03 | 11 | 0.5516 | 3.447× | 0.000939764 | >0.0001 / >0.0001 | 0 | 0 | 21 | 0.5 |
| gate03 | 11 | 0.5522 | 3.456× | 0.000942581 | >0.0001 / >0.0001 | 0 | 0 | 21 | 0.5 |

## Findings

- The compensated method recorded zero history-only rejections and zero shadow history-veto samples at εr=0.03 on the restored trajectories.
- The selection minima occurred in the first Newton update, before residual/history gating became active at six accepted updates. Within the active termination window the minimum positive alphas were about 0.653/0.5. This A4-v1 result is therefore inconclusive for history behavior under smaller post-activation steps.
- Timing and output differences between compensated and residual-gated runs cannot be attributed to history when the history-only counter is zero; independent GPU trajectories remain a source of run-to-run divergence.
- The strict reference is a numerical reference. Its disagreement and repeat noise determine the finest defensible error band above.

Derived values are in `analysis.json`; raw file hashes are in `EVIDENCE_MANIFEST.json`.
