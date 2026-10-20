# A2 batch-checkpoint smoke

- Release build completed with CUDA 12.6 and Visual Studio 2022.
- A three-frame table run saved schema-v3 `S1` and `S2` checkpoints from one
  process.
- `metadata.json` records the requested frame list `[1, 2]` and output root.
- The benchmark completed with finite state and no Newton/diagnostic cap.
- A duplicate frame list (`1,1`) was rejected before simulation with exit code
  1 and `Invalid checkpoint frame list/output root`.

The new options only schedule calls to the existing validated schema-v3
`save_checkpoint`; they do not change solver mathematics or runtime state.
