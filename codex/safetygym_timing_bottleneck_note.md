# SafetyGym Timing Bottleneck Note

- Added `--profile_timing` to SafetyGym FastSAC train entrypoints.
- Training now logs per-window timing breakdown metrics:
  - action selection
  - env step
  - replay/data handling
  - update step
  - episode-end handling
  - misc
- Use these percentages to diagnose low FPS before changing hyperparameters or env settings.
