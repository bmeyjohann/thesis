# SafetyGym Update Decimation for Interactive FPS

- Added train-time knobs:
  - `update_every`: run optimizer cycles every N environment steps.
  - `updates_per_cycle`: number of optimizer cycles when an update is triggered.
- Default behavior remains equivalent to prior setup (`1`, `1`).
- Intended use: improve interactive steering FPS by reducing update frequency during human intervention sessions.
