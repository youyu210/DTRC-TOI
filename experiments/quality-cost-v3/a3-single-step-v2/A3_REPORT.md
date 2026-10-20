# A3 paired single-step quality/cost matrix

Eight frozen states, 14 configurations, and three repeats produced 336 paired single steps. Every comparison starts from the exact same schema-v3 checkpoint. Heavy Newton tracing was off; the 120 s / 2000-Newton diagnostic limits remained active.

## Reference resolution

| State | strict2 complete | strict4 complete | Chosen reference | strict2↔strict4 max position RMS (m) | repeat-noise max RMS (m) | finest resolved band |
|---|---:|---:|---|---:|---:|---|
| table-f22 | 3/3 | 3/3 | strict4 | 5.4158e-08 | 2.29561e-10 | 1e-06 |
| table-f23 | 3/3 | 0/3 | strict2 | nan | 1.27597e-06 | unresolved |
| table-f40 | 3/3 | 3/3 | strict4 | 4.7739e-06 | 1.87821e-08 | 0.0001 |
| table-f03 | 3/3 | 3/3 | strict4 | 1.81503e-08 | 5.79149e-10 | 1e-06 |
| bunny-f48 | 3/3 | 3/3 | strict4 | 3.25609e-06 | 5.54886e-07 | 0.0001 |
| bunny-f49 | 3/3 | 3/3 | strict4 | 3.48608e-06 | 1.14232e-07 | 0.0001 |
| bunny-f50 | 3/3 | 3/3 | strict4 | 0.000105333 | 1.22709e-06 | unresolved |
| bunny-f03 | 3/3 | 3/3 | strict4 | 2.56156e-08 | 3.40239e-09 | 1e-06 |

## Aggregate candidate results

Raw quality is the worst paired error band across all eight states and three repeats. A band requires position/velocity RMS and 10× maximum-error limits. Speedup is paired to Graph N1 within the same state and repeat, then summarized across states. Validated counts additionally require the strict-reference disagreement and repeat noise to be no more than 10% of that band.

| Config | Complete | median state speedup [range] | worst position RMS (m) | worst velocity RMS (m/s) | worst raw band | resolved states | validated ≤1/10/100 μm | history rejects |
|---|---:|---:|---:|---:|---|---:|---:|
| graph_n1 | 24/24 | 1.000× [1.000, 1.000] | 0.000146868 | 0.0146868 | >0.0001 | 6/8 | 3/3/5 | 0 |
| graph_n2 | 24/24 | 1.136× [0.987, 1.700] | 0.000148145 | 0.0148145 | >0.0001 | 6/8 | 2/3/3 | 0 |
| graph_n4 | 24/24 | 1.182× [0.993, 4.907] | 0.000207418 | 0.0207418 | >0.0001 | 6/8 | 2/3/3 | 0 |
| graph_n8 | 24/24 | 1.523× [0.993, 5.847] | 0.000591953 | 0.0591953 | >0.0001 | 6/8 | 2/2/3 | 0 |
| toi | 24/24 | 6.762× [0.992, 12.645] | 0.00338577 | 0.338577 | >0.0001 | 6/8 | 2/3/3 | 0 |
| comp001 | 24/24 | 1.116× [0.992, 2.312] | 0.000578542 | 0.0578542 | >0.0001 | 6/8 | 3/3/3 | 0 |
| comp01 | 24/24 | 1.646× [0.993, 4.779] | 0.000834614 | 0.0834614 | >0.0001 | 6/8 | 3/3/3 | 0 |
| comp03 | 24/24 | 3.327× [0.994, 6.893] | 0.00153042 | 0.153042 | >0.0001 | 6/8 | 3/3/3 | 0 |
| comp10 | 24/24 | 5.390× [0.992, 8.437] | 0.0024577 | 0.24577 | >0.0001 | 6/8 | 2/3/3 | 0 |
| gate01 | 24/24 | 1.651× [0.988, 4.743] | 0.000880839 | 0.0880839 | >0.0001 | 6/8 | 3/3/3 | 0 |
| gate03 | 24/24 | 3.301× [0.992, 6.892] | 0.00144977 | 0.144977 | >0.0001 | 6/8 | 3/3/3 | 0 |
| gate10 | 24/24 | 5.441× [0.986, 8.413] | 0.00284928 | 0.284928 | >0.0001 | 6/8 | 2/3/3 | 0 |

## Per-state medians

### table-f22 (contact_onset)

| Config | Newton | seconds | speedup | position RMS max (m) | velocity RMS max (m/s) | raw / validated band | exits |
|---|---:|---:|---:|---:|---:|---|---|
| graph_n1 | 8 | 0.3577 | 1.000× | 1.93409e-07 | 1.93409e-05 | 1e-06 / 1e-06 | original_direction |
| graph_n2 | 7 | 0.3255 | 1.102× | 1.12622e-06 | 0.000112622 | 1e-05 / 1e-05 | original_direction |
| graph_n4 | 6 | 0.2929 | 1.221× | 6.1746e-06 | 0.00061746 | 1e-05 / 1e-05 | original_direction |
| graph_n8 | 5 | 0.2629 | 1.360× | 1.84595e-05 | 0.00184595 | 0.0001 / 0.0001 | original_direction |
| toi | 6 | 0.2910 | 1.229× | 6.17462e-06 | 0.000617462 | 1e-05 / 1e-05 | toi |
| comp001 | 8 | 0.3578 | 1.000× | 1.934e-07 | 1.934e-05 | 1e-06 / 1e-06 | original_direction |
| comp01 | 8 | 0.3588 | 1.000× | 1.93402e-07 | 1.93402e-05 | 1e-06 / 1e-06 | original_direction |
| comp03 | 8 | 0.3580 | 0.999× | 1.93409e-07 | 1.93409e-05 | 1e-06 / 1e-06 | original_direction |
| comp10 | 7 | 0.3264 | 1.099× | 1.12622e-06 | 0.000112622 | 1e-05 / 1e-05 | compensated |
| gate01 | 8 | 0.3581 | 1.000× | 1.93402e-07 | 1.93402e-05 | 1e-06 / 1e-06 | original_direction |
| gate03 | 8 | 0.3591 | 0.998× | 1.93406e-07 | 1.93406e-05 | 1e-06 / 1e-06 | original_direction |
| gate10 | 7 | 0.3228 | 1.108× | 1.12622e-06 | 0.000112622 | 1e-05 / 1e-05 | gated |

### table-f23 (dense_contact)

| Config | Newton | seconds | speedup | position RMS max (m) | velocity RMS max (m/s) | raw / validated band | exits |
|---|---:|---:|---:|---:|---:|---|---|
| graph_n1 | 125 | 5.1451 | 1.000× | 3.4843e-05 | 0.0034843 | >0.0001 / unresolved | original_direction |
| graph_n2 | 89 | 3.7889 | 1.361× | 7.26386e-05 | 0.00726386 | >0.0001 / unresolved | original_direction |
| graph_n4 | 21 | 1.0478 | 4.907× | 0.000207418 | 0.0207418 | >0.0001 / unresolved | original_direction |
| graph_n8 | 17 | 0.8799 | 5.847× | 0.000221921 | 0.0221921 | >0.0001 / unresolved | original_direction |
| toi | 8 | 0.4739 | 10.869× | 0.000293727 | 0.0293727 | >0.0001 / unresolved | toi |
| comp001 | 120 | 4.9732 | 1.037× | 3.61136e-05 | 0.00361136 | >0.0001 / unresolved | original_direction |
| comp01 | 122 | 5.0240 | 1.022× | 3.61015e-05 | 0.00361015 | >0.0001 / unresolved | original_direction |
| comp03 | 14 | 0.7460 | 6.893× | 0.000234532 | 0.0234532 | >0.0001 / unresolved | compensated |
| comp10 | 11 | 0.6079 | 8.437× | 0.000247668 | 0.0247668 | >0.0001 / unresolved | compensated |
| gate01 | 122 | 5.0246 | 1.025× | 3.52019e-05 | 0.00352019 | >0.0001 / unresolved | original_direction |
| gate03 | 14 | 0.7480 | 6.892× | 0.000234438 | 0.0234438 | >0.0001 / unresolved | gated |
| gate10 | 11 | 0.6128 | 8.413× | 0.000248001 | 0.0248001 | >0.0001 / unresolved | gated |

### table-f40 (contact_rearrangement)

| Config | Newton | seconds | speedup | position RMS max (m) | velocity RMS max (m/s) | raw / validated band | exits |
|---|---:|---:|---:|---:|---:|---|---|
| graph_n1 | 143 | 5.6697 | 1.000× | 1.2506e-05 | 0.0012506 | 0.0001 / 0.0001 | original_direction |
| graph_n2 | 81 | 3.3344 | 1.700× | 0.000126057 | 0.0126057 | >0.0001 / >0.0001 | original_direction |
| graph_n4 | 61 | 2.5671 | 2.197× | 0.000161255 | 0.0161255 | >0.0001 / >0.0001 | original_direction |
| graph_n8 | 34 | 1.5646 | 3.624× | 0.000279804 | 0.0279804 | >0.0001 / >0.0001 | original_direction |
| toi | 7 | 0.4475 | 12.645× | 0.00100845 | 0.100845 | >0.0001 / >0.0001 | toi |
| comp001 | 81 | 3.3246 | 1.719× | 0.000126544 | 0.0126544 | >0.0001 / >0.0001 | compensated |
| comp01 | 59 | 2.4845 | 2.270× | 0.000167993 | 0.0167993 | >0.0001 / >0.0001 | compensated |
| comp03 | 34 | 1.5569 | 3.628× | 0.000280004 | 0.0280004 | >0.0001 / >0.0001 | compensated |
| comp10 | 15 | 0.8100 | 7.022× | 0.000675289 | 0.0675289 | >0.0001 / >0.0001 | compensated |
| gate01 | 59 | 2.4853 | 2.277× | 0.000167909 | 0.0167909 | >0.0001 / >0.0001 | gated |
| gate03 | 34 | 1.5541 | 3.632× | 0.000279613 | 0.0279613 | >0.0001 / >0.0001 | gated |
| gate10 | 15 | 0.8106 | 6.996× | 0.000674762 | 0.0674762 | >0.0001 / >0.0001 | gated |

### table-f03 (quiet_motion)

| Config | Newton | seconds | speedup | position RMS max (m) | velocity RMS max (m/s) | raw / validated band | exits |
|---|---:|---:|---:|---:|---:|---|---|
| graph_n1 | 3 | 0.1832 | 1.000× | 8.06691e-08 | 8.06691e-06 | 1e-06 / 1e-06 | original_direction |
| graph_n2 | 3 | 0.1845 | 0.993× | 8.05905e-08 | 8.05905e-06 | 1e-06 / 1e-06 | original_direction |
| graph_n4 | 3 | 0.1849 | 0.997× | 8.05294e-08 | 8.05294e-06 | 1e-06 / 1e-06 | original_direction |
| graph_n8 | 3 | 0.1843 | 1.000× | 8.06064e-08 | 8.06064e-06 | 1e-06 / 1e-06 | original_direction |
| toi | 3 | 0.1830 | 1.001× | 8.05967e-08 | 8.05967e-06 | 1e-06 / 1e-06 | original_direction |
| comp001 | 3 | 0.1845 | 0.992× | 8.04601e-08 | 8.04601e-06 | 1e-06 / 1e-06 | original_direction |
| comp01 | 3 | 0.1828 | 1.002× | 8.06202e-08 | 8.06202e-06 | 1e-06 / 1e-06 | original_direction |
| comp03 | 3 | 0.1830 | 1.003× | 8.0884e-08 | 8.0884e-06 | 1e-06 / 1e-06 | original_direction |
| comp10 | 3 | 0.1831 | 1.003× | 8.05538e-08 | 8.05538e-06 | 1e-06 / 1e-06 | original_direction |
| gate01 | 3 | 0.1840 | 0.996× | 8.05378e-08 | 8.05378e-06 | 1e-06 / 1e-06 | original_direction |
| gate03 | 3 | 0.1834 | 0.998× | 8.05639e-08 | 8.05639e-06 | 1e-06 / 1e-06 | original_direction |
| gate10 | 3 | 0.1870 | 0.986× | 8.05003e-08 | 8.05003e-06 | 1e-06 / 1e-06 | original_direction |

### bunny-f48 (contact_onset)

| Config | Newton | seconds | speedup | position RMS max (m) | velocity RMS max (m/s) | raw / validated band | exits |
|---|---:|---:|---:|---:|---:|---|---|
| graph_n1 | 112 | 3.7565 | 1.000× | 0.000106184 | 0.0106184 | >0.0001 / >0.0001 | original_direction |
| graph_n2 | 109 | 3.6345 | 1.034× | 0.000107015 | 0.0107015 | >0.0001 / >0.0001 | original_direction |
| graph_n4 | 102 | 3.3972 | 1.127× | 0.000117377 | 0.0117377 | >0.0001 / >0.0001 | original_direction |
| graph_n8 | 45 | 1.6372 | 2.294× | 0.000591953 | 0.0591953 | >0.0001 / >0.0001 | original_direction |
| toi | 7 | 0.4136 | 9.063× | 0.00324286 | 0.324286 | >0.0001 / >0.0001 | toi |
| comp001 | 45 | 1.6250 | 2.312× | 0.000578542 | 0.0578542 | >0.0001 / >0.0001 | compensated |
| comp01 | 21 | 0.8954 | 4.275× | 0.000834614 | 0.0834614 | >0.0001 / >0.0001 | compensated |
| comp03 | 19 | 0.8207 | 4.620× | 0.00103088 | 0.103088 | >0.0001 / >0.0001 | compensated |
| comp10 | 10 | 0.5148 | 7.304× | 0.00199075 | 0.199075 | >0.0001 / >0.0001 | compensated |
| gate01 | 22 | 0.9261 | 4.134× | 0.000880839 | 0.0880839 | >0.0001 / >0.0001 | gated |
| gate03 | 17 | 0.7412 | 5.015× | 0.00144977 | 0.144977 | >0.0001 / >0.0001 | gated |
| gate10 | 11 | 0.5448 | 6.895× | 0.00284928 | 0.284928 | >0.0001 / >0.0001 | gated |

### bunny-f49 (dense_contact)

| Config | Newton | seconds | speedup | position RMS max (m) | velocity RMS max (m/s) | raw / validated band | exits |
|---|---:|---:|---:|---:|---:|---|---|
| graph_n1 | 147 | 4.6061 | 1.000× | 1.0018e-05 | 0.0010018 | 0.0001 / 0.0001 | original_direction |
| graph_n2 | 98 | 3.1938 | 1.448× | 6.51927e-05 | 0.00651927 | >0.0001 / >0.0001 | original_direction |
| graph_n4 | 90 | 2.9663 | 1.553× | 7.91571e-05 | 0.00791571 | >0.0001 / >0.0001 | original_direction |
| graph_n8 | 83 | 2.7621 | 1.668× | 0.000103317 | 0.0103317 | >0.0001 / >0.0001 | original_direction |
| toi | 8 | 0.4400 | 10.468× | 0.0031529 | 0.31529 | >0.0001 / >0.0001 | toi |
| comp001 | 84 | 2.7805 | 1.655× | 0.000104132 | 0.0104132 | >0.0001 / >0.0001 | compensated |
| comp01 | 24 | 0.9686 | 4.779× | 0.000709938 | 0.0709938 | >0.0001 / >0.0001 | compensated |
| comp03 | 17 | 0.7365 | 6.274× | 0.00153042 | 0.153042 | >0.0001 / >0.0001 | compensated |
| comp10 | 12 | 0.5629 | 8.182× | 0.00153921 | 0.153921 | >0.0001 / >0.0001 | compensated |
| gate01 | 24 | 0.9711 | 4.743× | 0.000711589 | 0.0711589 | >0.0001 / >0.0001 | gated |
| gate03 | 16 | 0.6862 | 6.690× | 0.00123799 | 0.123799 | >0.0001 / >0.0001 | gated |
| gate10 | 12 | 0.5623 | 8.200× | 0.00155298 | 0.155298 | >0.0001 / >0.0001 | gated |

### bunny-f50 (contact_rearrangement)

| Config | Newton | seconds | speedup | position RMS max (m) | velocity RMS max (m/s) | raw / validated band | exits |
|---|---:|---:|---:|---:|---:|---|---|
| graph_n1 | 73 | 2.5773 | 1.000× | 0.000146868 | 0.0146868 | >0.0001 / unresolved | original_direction |
| graph_n2 | 63 | 2.2389 | 1.169× | 0.000148145 | 0.0148145 | >0.0001 / unresolved | original_direction |
| graph_n4 | 63 | 2.2528 | 1.143× | 0.000148589 | 0.0148589 | >0.0001 / unresolved | original_direction |
| graph_n8 | 49 | 1.8410 | 1.379× | 0.000186849 | 0.0186849 | >0.0001 / unresolved | original_direction |
| toi | 10 | 0.5773 | 4.461× | 0.00338577 | 0.338577 | >0.0001 / unresolved | toi |
| comp001 | 61 | 2.1911 | 1.196× | 0.000148161 | 0.0148161 | >0.0001 / unresolved | compensated |
| comp01 | 24 | 1.0251 | 2.510× | 0.000740381 | 0.0740381 | >0.0001 / unresolved | compensated |
| comp03 | 20 | 0.9108 | 3.027× | 0.00127759 | 0.127759 | >0.0001 / unresolved | compensated |
| comp10 | 12 | 0.6425 | 3.758× | 0.0024577 | 0.24577 | >0.0001 / unresolved | compensated |
| gate01 | 23 | 1.0093 | 2.431× | 0.000748098 | 0.0748098 | >0.0001 / unresolved | gated |
| gate03 | 19 | 0.8682 | 2.969× | 0.00141446 | 0.141446 | >0.0001 / unresolved | gated |
| gate10 | 12 | 0.6464 | 3.987× | 0.00246856 | 0.246856 | >0.0001 / unresolved | gated |

### bunny-f03 (quiet_motion)

| Config | Newton | seconds | speedup | position RMS max (m) | velocity RMS max (m/s) | raw / validated band | exits |
|---|---:|---:|---:|---:|---:|---|---|
| graph_n1 | 3 | 0.1168 | 1.000× | 8.04865e-08 | 8.04865e-06 | 1e-06 / 1e-06 | original_direction |
| graph_n2 | 3 | 0.1181 | 0.987× | 8.05132e-08 | 8.05132e-06 | 1e-06 / 1e-06 | original_direction |
| graph_n4 | 3 | 0.1176 | 0.993× | 8.03617e-08 | 8.03617e-06 | 1e-06 / 1e-06 | original_direction |
| graph_n8 | 3 | 0.1179 | 0.993× | 8.05806e-08 | 8.05806e-06 | 1e-06 / 1e-06 | original_direction |
| toi | 3 | 0.1179 | 0.992× | 8.05787e-08 | 8.05787e-06 | 1e-06 / 1e-06 | original_direction |
| comp001 | 3 | 0.1175 | 0.992× | 8.04042e-08 | 8.04042e-06 | 1e-06 / 1e-06 | original_direction |
| comp01 | 3 | 0.1176 | 0.993× | 8.03708e-08 | 8.03708e-06 | 1e-06 / 1e-06 | original_direction |
| comp03 | 3 | 0.1174 | 0.994× | 8.06146e-08 | 8.06146e-06 | 1e-06 / 1e-06 | original_direction |
| comp10 | 3 | 0.1177 | 0.992× | 8.03727e-08 | 8.03727e-06 | 1e-06 / 1e-06 | original_direction |
| gate01 | 3 | 0.1185 | 0.988× | 8.03638e-08 | 8.03638e-06 | 1e-06 / 1e-06 | original_direction |
| gate03 | 3 | 0.1181 | 0.992× | 8.05097e-08 | 8.05097e-06 | 1e-06 / 1e-06 | original_direction |
| gate10 | 3 | 0.1178 | 0.989× | 8.04279e-08 | 8.04279e-06 | 1e-06 / 1e-06 | original_direction |

## Findings

- No candidate is validated within the 100 μm band on all six reference-resolved states; Graph N1 reaches five of six. The production tolerance remains unapproved.
- Pure TOI has the largest median state speedup, but its worst paired position RMS is in the millimeter range.
- Residual compensation records zero history-only rejections in every tested configuration. The selected states therefore show no independent protection from the historical budget.
- comp03 and gate03 have similar median speedups and error scale; neither establishes a global 100 μm guarantee.

## Interpretation limits

The chosen strict result is a numerical reference, not an exact physical solution. Reference disagreement and repeat noise above quantify its resolution. A diagnostic-budget result is censored and never selected as the reference. These one-step results rank local quality/cost behavior; long-horizon trajectory validation remains a later stage.

Derived values are in `analysis.json`; raw file hashes are in `EVIDENCE_MANIFEST.json`.
