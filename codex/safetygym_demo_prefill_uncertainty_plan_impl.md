## SafetyGym Demo Prefill + Uncertainty Implementation Notes (2026-02-19)

### Added training capabilities
- New optional online demo-prefill phase by episode count:
  - `--prefill_demo_episodes`
  - `--prefill_max_steps_per_episode`
  - `--prefill_policy {student,random,zero}`
- New optional demo-only pretrain phase:
  - `--demo_pretrain_updates`
  - `--demo_pretrain_batch_size`
- New optional critic reset after pretrain:
  - `--critic_reset_after_pretrain`

### Critic uncertainty instrumentation
- Added per-step critic disagreement metrics from twin Q critics (`abs(Q1-Q2)`).
- Added split uncertainty logs for:
  - all steps
  - intervention steps
  - no-intervention steps
- Added pre-intervention rise metrics:
  - delta, slope, z-score over a trailing window.
- Added oversight signal-only logging:
  - `--uncertainty_oversight_mode {off,signal_only}`
  - `--uncertainty_oversight_threshold`
  - `--uncertainty_oversight_ema_alpha`

### Performance profiling
- Timing profiler now tracks uncertainty compute time:
  - `train_timing/uncertainty_ms_mean`
  - `train_timing/pct_uncertainty`
- Added matrix profiling helper:
  - `scripts/profile_safetygym_perf_matrix.sh`

### Replay buffer behavior
- `main_rb` now receives all online transitions.
- `demo_rb` (when enabled) receives teacher-intervened transitions.
- Existing PVP split (`novice_rb` / `human_rb`) remains intact for batch construction.
