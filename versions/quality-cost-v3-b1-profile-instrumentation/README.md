# quality-cost-v3 B1 profiling binary

This local snapshot is the exact executable and runtime DLL set used for the
B1 `stack10` profiling matrix. Detailed profiling is opt-in through
`--profile-stages`; without that flag the instrumentation only adds guarded
host-side branches and does not insert profiling synchronizations.

The profile separates gradient/Hessian operations, global linear-system
operations, and converter operations. It also records profiling syncs, D2H
reads, resize calls, and capacity-triggered GPU allocation requests. Binary
files are retained locally and ignored by Git; `MANIFEST.json` records hashes.
