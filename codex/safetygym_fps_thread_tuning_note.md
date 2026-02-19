# SafetyGym FPS Thread Tuning Note

- In local CPU intervention runs, PyTorch update time can dominate FPS when thread counts are high.
- Empirical check showed large speedup by limiting CPU threads:
  - `torch_num_threads=1`
  - `torch_num_interop_threads=1`
- Added these as train CLI args (default `1`) for SafetyGym FastSAC entrypoints.
