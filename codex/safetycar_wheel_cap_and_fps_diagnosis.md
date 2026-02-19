# SafetyCar Wheel Cap and FPS Diagnosis

- Differential keyboard mixing now preserves additive wheel commands up to `2` per wheel:
  - `W+A -> (2, 0)`
  - `W+D -> (0, 2)`
  - `S+A -> (0, -2)`
  - `S+D -> (-2, 0)`
- Removed normalization that previously forced combo commands back to unit magnitude.
- Added car actuation scaling in env setup:
  - action/control range and force range can be expanded via `car_wheel_command_limit` and `car_force_scale`.
- Added controller timing metric in training logs:
  - `train/live_controller_ms_per_step`
  - Use with `train_timing/*` metrics to separate controller/render costs from update costs.
