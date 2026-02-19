# SafetyGym Grippy Motion Regression Fix

- Issue: initial reduced-slip tuning made the point agent nearly immobile.
- Cause: damping/frictionloss values were too high.
- Fix: switched to milder values in `SurfaceConfigWrapper`:
  - floor friction: `[1.0, 0.02, 0.001]`
  - x/y damping/frictionloss: `0.015` / `0.0005`
  - z damping/frictionloss: `0.0075` / `0.00025`
- Validation: constant-forward rollout now shows substantial movement again.
