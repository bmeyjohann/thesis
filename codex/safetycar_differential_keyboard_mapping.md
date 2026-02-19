# SafetyCar Differential Keyboard Mapping

- Implemented dedicated 2D wheel-mixing control for SafetyCar intervention:
  - `W/S`: both wheels forward/backward
  - `A/D`: opposite wheel directions for turning
  - key combinations are summed and saturated per-wheel to `[-1, 1]`
- Removed L2 vector normalization for controller output to avoid unintuitive wheel command shrinkage.
- Added optional `controller_fps_limit` (default `0`, uncapped) for the pygame control overlay.
