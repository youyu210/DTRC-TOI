# A3 trace-independent diagnostic budget smoke

The rebuilt executable loaded table S22 with strict4 tolerances and ran without
`--trace-newton`. With `--diagnostic-max-newton 5` and
`--diagnostic-max-ms 500`, it stopped after exactly five Newton updates,
reported `exit_reason=diagnostic_budget`, incremented
`diagnostic_budget_hits`, wrote finite output arrays, and returned exit code 2.

This verifies that experiment safety budgets no longer depend on the heavy
Newton-trace instrumentation. Normal runs with both diagnostic limits set to
zero retain the original stopping path.
