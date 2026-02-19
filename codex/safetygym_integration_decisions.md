# Safety-Gymnasium integration decisions (THE-11)

## Scope and architecture
- Use `safety-gymnasium/` (modern API), not legacy `safety-gym/`.
- Keep Safety-Gymnasium train/eval scripts separate from OGBench scripts.
- Use FastSAC-style state-vector MLP agents for all Safety-Gymnasium variants (own/PVP/HILSERL).
- Default env is `SafetyPointGoal2-v0` with Goal variants selectable via CLI.

## Human intervention behavior
- Human intervention uses a dedicated focused pygame control window.
- Intervention is deterministic keyboard override (no intervention-probability decay scheduling).
- Wrapper emits `teacher_intervened`, `teacher_action`, `student_action`, and intervention diagnostics.

## Reward modes
- `sparse`: reward on goal reached only.
- `dense`: Euclidean progress-delta shaping (`d_prev - d_cur`) scaled by `--dense_reward_scale`.
- `none`: zero reward.
- `--step_penalty` applies in all modes.

## Outcome classification
- Primary classification is based on goal + episode step count:
  - success: `goal_met=True`
  - timeout: `episode_steps >= max_episode_steps`
  - kill: premature episode end without goal
- `terminated`/`truncated` are logged as diagnostics, not primary truth.

## Mandatory metrics schema
- Episode metrics are validated against a required schema (intervention/cost/outcome/final-distance fields).
- Non-finite mandatory values are treated as errors.
