# A2 representative-state selection

The selection was frozen before any A2/A3 candidate run. Both scenes used the
same Graph N1 solver for the first 50 physical frames with opt-in Newton
diagnostics. The deterministic ranking in `run_a2_state_selection.py` selected
one unique frame per preregistered category and scene.

| State | Category | Checkpoint | Newton | Contacts start→end | Unique contact states | min α |
|---|---|---:|---:|---:|---:|---:|
| table-f22 | contact onset | S21 | 8 | 2→51,750 | 8 | 0.847535 |
| table-f23 | dense contact | S22 | 120 | 51,750→66,692 | 82 | 0.146007 |
| table-f40 | contact rearrangement | S39 | 145 | 40,372→41,739 | 141 | 0.169134 |
| table-f03 | quiet motion | S02 | 3 | 2→2 | 1 | 1.0 |
| bunny-f48 | contact onset | S47 | 120 | 4,636→5,789 | 117 | 0.143601 |
| bunny-f49 | dense contact | S48 | 170 | 5,789→6,282 | 159 | 0.160758 |
| bunny-f50 | contact rearrangement | S49 | 182 | 6,282→6,136 | 166 | 0.120628 |
| bunny-f03 | quiet motion | S02 | 3 | 0→0 | 1 | 1.0 |

`SELECTED_STATES.json` contains all 100 frame metrics, exact input hashes, the
ranking output, and the frozen-state marker. Large states and traces remain in
the local workspace and are indexed by `EVIDENCE_MANIFEST.json`.
