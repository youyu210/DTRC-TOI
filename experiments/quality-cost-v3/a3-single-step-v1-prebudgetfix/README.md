# Preserved A3 pre-budget-fix attempt

This partial matrix is retained as failure evidence. The runner supplied
`--diagnostic-max-ms 120000` and `--diagnostic-max-newton 2000` without
`--trace-newton`, but the implementation accidentally guarded both limits with
`trace_newton`. Short runs completed, while `table-f23/strict4/r01` exceeded the
150-second process timeout and was terminated with incomplete output.

No result in this directory is used by the final A3 comparison. The corrected
matrix starts from an empty `a3-single-step-v2` directory and one rebuilt
binary.
