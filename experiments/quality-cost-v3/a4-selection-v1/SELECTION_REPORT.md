# A4 natural fractional-step states

A4 was triggered because A3 recorded zero history-only rejections for every
compensated configuration. Two additional states were selected before running
the A4 candidates. The selection uses the smallest naturally accepted alpha in
each scene after excluding the eight A2 states.

| State | Checkpoint | Newton | PCG | contacts start→end | unique contact states | min α |
|---|---:|---:|---:|---:|---:|---:|
| table-f39 | S38 | 96 | 4,999 | 38,718→40,372 | 84 | 0.0510726 |
| bunny-f46 | S45 | 68 | 4,285 | 5,105→4,968 | 58 | 0.0907136 |

No CCD alpha is changed or injected. The frozen A4 matrix is Graph N1, TOI,
comp03, gate03, strict2, and strict4, each repeated three times (36 steps).
