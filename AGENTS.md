# Repository Guidelines

## General Guidelines

- Whenever I need to tell you to do something in a different way or am dissatisfied with your work, append concise info on what you made wrong and how to do it correctly to `AGENTS.md`. Also save a short note in codex/[appropriate_naming_for_info].md.
- Save other relevant info that should be preserved in codex/[appropriate_naming_for_info].md
- Always inform me if you save info in a file.
- When providing runnable commands for copy/paste, default to multi-line format with trailing `\` line continuations.
- Whenever project architecture changes, immediately update all architecture-related sections in `AGENTS.md` in the same work so this file stays aligned with the actual implementation and is never outdated.
- Proactively suggest automation improvements that reduce manual setup/debugging and burden for the human and increase how much Codex can run autonomously (including when escalation is needed), and include concrete next steps to implement that automation. This is about Codex being able to run for longer without requiring human intervention or oversight, so closing the look such that Codex can larger tasks autonomously by checking its own implementation and iterating on that or doing more thing by receiving the appropriate tooling.
- Subagent permission policy: subagents must not execute commands that require escalated permissions. If escalation is needed (e.g., SSH/cluster operations), subagents should return the required command/context and the main agent executes it with explicit user approval.
- STT clarification: whenever the user says or writes `1B` or `1p`, interpret it as `WANDB` every time. This mixup is caused by the user's speech-to-text tool.
- Use the dedicated `apply_patch` tool for file edits, and in general follow tool/API contracts literally: required keyword arguments, return signatures, scheduler semantics, and call-site expectations should match the active implementation exactly.
- Prefer explicit, environment-family-specific wiring over generic routing when it improves debuggability: keep maze and manipulation entrypoints, wrappers, and control paths separate and directly inspectable.
- Keep train, eval, and launch commands in lockstep: checkpoints should carry the settings needed for eval, auto-sync should be preferred over redundant manual flags, and run commands should preserve established hyperparameters while explicitly including any flags required to realize the intended mode.
- Keep reward and intervention logic deterministic, task-local, and single-source: avoid hidden decay schedules or overlapping reward paths, ground gating in meaningful environment signals, and expose thresholds/diagnostics so behavior can be tuned from data.
- Debug and review through the live codepath and existing signals first: inspect the active implementation before answering, reuse already-computed tensors/metrics before adding new forwards, log the loss and buffer diagnostics needed for comparison, and back interface changes with a targeted smoke test.
- For non-learning/debug investigations, explicitly verify end-to-end signal integrity: (1) observation/goal content, (2) action transform/clipping consistency, (3) intervention action storage format in replay/pref/demo buffers, and (4) presence/non-zero progression of the intended W&B metrics (not just summary endpoints).
- Correction noted: if a task stalls or takes longer than expected, provide frequent concrete progress updates and partial results immediately instead of waiting for a full bundle; isolate likely external/runtime issues (e.g., WSL/GLFW display) before iterating code-side fixes.
- Correction noted: when adjusting teacher/oracle behavior, keep the fix in the live oracle path and remove stale wrapper-side special cases from prior experiments so target switching stays aligned with the vanilla control flow.
- Correction noted: for clearance guards around carried objects, explicitly exclude the currently grasped target object from obstacle-height checks; otherwise the guard can chase the end effector upward indefinitely.

# Knowledge Management
- The Linear team corresponding to this workspace is named Thesis, issues have the format THE-XXX
- Use this info when requested to access Linear

## Project Structure & Module Organization
- `train_*.py`: Entry points for training (RSL‑RL, PointMaze variants).
- `train_fast_sac_ogbench_maze.py`: Maze-focused FastSAC entrypoint (enforces maze env family).
- `train_fast_sac_ogbench_manip.py`: Manipulation-focused FastSAC entrypoint (enforces manip env family).
- `ogbench_utils/fastsac_ogbench_maze_cli.py`: Maze-specific FastSAC parser defaults and validation.
- `ogbench_utils/fastsac_ogbench_manip_cli.py`: Manip-specific FastSAC parser defaults and validation.
- `ogbench_utils/fastsac_ogbench_maze_env.py`: Maze-specific wrapper/env assembly (`clip_actions=1.0`).
- `ogbench_utils/fastsac_ogbench_manip_env.py`: Manip-specific wrapper/env assembly (no global L2 clip), including optional train/eval human render wiring (`train_render_mode`, `eval_render_mode` with single-eval-env enforcement).
- `ogbench_utils/fastsac_ogbench_maze_train.py`: Maze-specific FastSAC orchestration used by `train_fast_sac_ogbench_maze.py`.
- `ogbench_utils/fastsac_ogbench_manip_train.py`: Manip-specific FastSAC orchestration used by `train_fast_sac_ogbench_manip.py`.
- `ogbench_utils/fastsac_ogbench_types.py`: Dataclass containers for FastSAC model/buffer/AMP/logging components.
- `ogbench_utils/fastsac_ogbench_env.py`: Shared helper utilities (tensor shaping, reward-mode trackers, legacy family helpers) used by FastSAC OGBench loop/setup modules.
- `ogbench_utils/fastsac_ogbench_setup.py`: Device/run-dir/model/buffer/updater/logging initialization helpers for FastSAC OGBench.
- `ogbench_utils/fastsac_ogbench_loop.py`: Demo prefill, evaluation metrics rollout, and main FastSAC training loop.
- `eval_interactive.py`: Load a trained model and render episodes.
- `train_fast_sac_safetygym.py`, `train_fastsac_pvp_safetygym.py`, `train_fastsac_hilserl_safetygym.py`: Safety-Gymnasium state-vector FastSAC-family training entrypoints for own/PVP/HILSERL comparisons with human intervention.
- `eval_interactive_safetygym.py`: Safety-Gymnasium interactive eval with policy/random/human controllers and optional human intervention overlay.
- `safetygym_utils/`: Shared Safety-Gymnasium env/controller/wrapper/metrics/SAC helpers; reward modes are centralized here (`sparse`, `dense`, `none`) and must stay train/eval lockstep.
- `xmltodict.py`: Local minimal xmltodict-compatible shim (parse/unparse subset) used to keep Safety-Gymnasium runnable in offline environments where pip installs are unavailable.
- `config/`: YAML configs (e.g., `ogbench_config.yaml`).
- `rsl_rl/`, `fasttd3/`, `ogbench/`, `IsaacLab/`: Vendor/submodules used by training.
- `scripts/`: Cluster helpers (copy logs, SSH, allocation).
- `scripts/run_safetygym_*_local.sh`: Local launchers for Safety-Gymnasium own/PVP/HILSERL runs.
- `scripts/run_safetycar_own_human_prefill20_local.sh`: Local SafetyCar own-method launcher that collects 20 human-intervened prefill episodes into replay before online learning, uses dense reward, and sets `prefill_policy=zero` so the human effectively provides the demo actions during prefill.
- `scripts/run_safetycar_own_dense_only_no_prefill_local.sh`: Local SafetyCar own-method dense-only baseline launcher with no demo prefill; online human intervention remains available, but optimization is driven by dense goal-distance reward only (`pref_rank_weight=0.0`).
- `scripts/run_safetycar_pretrain_dense_plus_sparse_local.sh`: Long-running SafetyCar pretraining launcher using `reward_mode=dense_plus_sparse`, no intervention, no preference loss, regular eval/checkpointing, and manual stop-on-success workflow.
- `scripts/smoke_safetygym_all.sh`, `scripts/safetygym_matrix_local.sh`, `scripts/verify_safetygym_ckpt_eval_lockstep.sh`, `scripts/profile_safetygym_perf_matrix.sh`, `scripts/review_safetygym_checkpoints.sh`: Safety-Gymnasium automation helpers for smoke checks, matrix runs, visual checkpoint review, perf breakdown profiling, and checkpoint/eval compatibility checks.
- `scripts/smoke_fastsac_ogbench_modularization.sh`: Compile-smoke check for FastSAC OGBench maze/manip CLI+env+train modules and shared setup/loop modules.
- `scripts/run_cube_single_static_demo_dense_online.sh`: Single-cube deterministic reset test run (fixed reset seed + demo prefill + dense cube reward mode).
- `scripts/run_cube_triple_task2_relative_only_layernorm_utd_sweep.sh`: Task2 manipulation launcher that mirrors the baseline run with relative-only observations (compact proprio + relative cube features), LayerNorm enabled, and sequential UTD sweep (`num_updates=1` then `5`).
- `scripts/run_cube_triple_task2_relative_only_layernorm_manual_gate_prefill.sh`: Task2 manipulation launcher for single-env human-gated intervention training with relative-only observations, LayerNorm, W&B logging, and teacher demo prefill into the replay buffer before learning starts.
- `scripts/run_cube_triple_task5_relative_only_layernorm_utd_sweep.sh`: Task5 manipulation launcher mirroring the task2 relative-only setup (compact proprio + relative cube features), LayerNorm enabled, and sequential UTD sweep (`num_updates=1` then `5`).
- `scripts/run_cube_triple_task2_task5_relonly_layernorm_newteacher_utd1_compare.sh`: Two-run manipulation comparison launcher that runs task2 then task5 for `60k` steps each with relative-only observations, LayerNorm, fixed `num_updates=1`, and run names tagged `newteacher` for teacher-behavior comparisons.
- `sc_venv_template/`: Virtual gridworld environment setup for HPC (Isaac Sim and Isaac Lab use an apptainer container instead of a virtual environment).
- `logs/`, `models/`, `wandb/`: Outputs, artifacts, offline tracking.
- `ogbench_utils/env_wrappers_maze.py`: maze-only wrapper stack (`FlexibleObsWrapper` + reward/intervention wrappers) and maze goal-marker coloring helper.
- `ogbench_utils/env_wrappers_maze.py`: maze-only wrapper stack and local concrete wrappers (`MazeFlexibleObsWrapper`, `MazeDetailedRewardWrapper`) plus maze goal-marker coloring helper.
- `ogbench_utils/env_wrappers_manip.py`: manipulation-only wrapper stack and local concrete wrappers (`ManipDetailedRewardWrapper`, `ManipGoalConditionedObsWrapper`, `CubeTeacherInfoAdapter`, `ManipRelativeCubeFeaturesWrapper`, `ManipRelativeOnlyObsWrapper`) plus cube reward-mode tracker utilities; `relative_only_obs` keeps compact proprio + optional relative features and force-disables goal concat to avoid absolute-goal leakage.
- `ogbench_utils/env_wrappers_common.py`: shared wrapper helpers (`FixedResetSeedWrapper`, environment-family inference, intervention wrapper construction).
- `ogbench_utils/intervention_wrappers.py`: local copy of intervention wrapper implementation used by `env_wrappers_common.py` (no dependency on `ogbench.wrappers` for intervention logic), including manual-gate control-panel diagnostics (`Q_min`, `Q_dis`), `agent_mode='always'` support, component-wise intervention thresholds (`xyz`, `yaw`, `gripper`) with optional adaptive tightening near pick/place targets, and `manual_gripper` semantics where manual gate gives full teacher takeover while off-gate steps still mirror the teacher gripper sign (`open` and `close`) when the student disagrees.
- `ogbench/ogbench/manipspace/oracles/markov/cube_markov.py`: cube markov oracle keeps the vanilla phase sequence, but now gates XY travel in phases 1, 5, and 9 behind an all-cubes clearance check so the effector first lifts to `max(block_z) + 0.05` before lateral motion; `ogbench_utils/intervention_wrappers.py` uses the normal oracle reset-on-done-or-target-switch flow for `cube_markov`.
- `ogbench_utils/cli.py`, `ogbench_utils/fastsac_ogbench_loop.py`: demo prefill supports a dedicated env count via `--demo_prefill_num_envs`; prefill rendering is forced off so fast teacher collection can run in parallel before single-env human-visible training starts, multi-env prefill is row-wise copied into single-env replay buffers safely, prefill progress is logged to W&B under `DemoPrefill/*`, console heartbeats are emitted during prefill, and episode-target prefill now stages per-env trajectories so `--demo_prefill_episodes N` means exactly `N` committed completed demo episodes (no overshoot).
- `fasttd3/fast_sac/fast_sac_utils.py`, `ogbench_utils/fastsac_ogbench_loop.py`: in `pref_sampling_mode=linked`, W&B buffer logging now reports pref-pair count from replay rows marked `teacher_intervened` rather than the standalone `PreferencePairBuffer`, so `/Buffers/pref_pairs` reflects collectable linked preference pairs stored in replay.
- `safetygym_utils/sac.py`: Safety-Gymnasium SAC update now supports manipulation-style preference optimization controls (`margin`, `bradley_terry`, `lagrangian`, `pref_stopgrad_positive`, dual lambda/EMA/violation settings) and logs the associated preference metrics.
- `safetygym_utils/train.py`: Safety-Gymnasium console output now includes concise human-readable summaries for prefill, pretrain, train, and eval in addition to the existing JSON/W&B logging; eval summaries explicitly surface cost/collision metrics while reward remains controlled solely by `RewardModeWrapper`.
- `safetygym_utils/wrappers.py`, `train_fast_sac_safetygym.py`, `eval_interactive_safetygym.py`: Safety-Gymnasium reward modes now include `dense_plus_sparse`, which adds the sparse goal-completion bonus on top of dense goal-distance progress while still keeping collision cost separate from reward.
- `train_fast_sac_safetygym.py`, `train_fastsac_pvp_safetygym.py`, `train_fastsac_hilserl_safetygym.py`: Safety-Gymnasium entrypoints expose the manip-style preference/Lagrangian CLI flags so launches stay aligned across variants, even when the active training variant does not consume preference pairs.
- `ogbench/ogbench/wrappers/vec_env_wrapper.py`: forwards teacher/action-delta diagnostics into `extras['log']`, including aggregate deltas and component-wise intervention diagnostics (`/Teacher/diag_delta_xyz_l2`, `/Teacher/diag_delta_yaw_abs`, `/Teacher/diag_delta_gripper_abs`, threshold values/scales, divergence flags).
- `ogbench/ogbench/wrappers/vec_env_wrapper.py`: emits cube subgoal/progress episode metrics (`cubes_solved`, `cubes_total`, `cubes_solved_fraction`, `cube_max_target_error`) into `extras['log']` for W&B panels.
- `ogbench_utils/fastsac_ogbench_maze_env.py`, `ogbench_utils/fastsac_ogbench_manip_env.py`: apply explicit family-specific vector action clipping (`maze`: L2 clip `1.0`, `manip`: no global L2 clip) to avoid distorting manipulation action semantics.
- `ogbench_utils/fastsac_ogbench_maze_train.py`, `ogbench_utils/fastsac_ogbench_manip_train.py`: FastSAC OGBench is state-only (`obs_mode='state'` enforced); pixel observation handling was removed from this pipeline (use DRQ-v2 entrypoints for pixel training).
- `ogbench_utils/buffers.py`: counterfactual replay buffer is removed; FastSAC uses replay/demo buffers plus optional preference pair buffer.
- `ogbench_utils/buffers.py`: `PreferenceTDBuffer` is removed as deprecated/unused; only `PreferencePairBuffer` remains for preference supervision.
- `ogbench_utils/update.py`: FastSAC updater no longer applies pixel random-shift augmentation or counterfactual critic penalties.

## Build, Test, and Development Commands
- To run/test anything locally, ALWAYS activate the local virtual env first: `conda activate fasttd3`
- See `README.md`
- Starting jobs on the cluster with automatic account detection works like this but is reserved for the human Use this info to create appropriate sbatch files and submit them: `bash submit_job.sh [experiment_name].sbatch`
- Logs: tail `logs/<name>_<JOBID>.log` for artifacts.
- If possible, ALWAYS run local smoke tests before telling me something is working and done.
- Always at least attempt a local smoke test before cluster submission to avoid queue/time waste.
- Assume local testing is CPU-only: validate startup/initialization and short-step execution, but keep smoke tests lightweight (no computationally intensive long training loops).

## Coding Style & Naming Conventions
- Python: PEP 8, 4 spaces, max line length 120.
- Names: modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE_CASE`.
- Scripts: executable with shebang; keep flags explicit and self-documenting.

## Testing Guidelines
- Framework: no unit test suite at root; add minimal smoke tests when changing training/eval.
- Quick check: run a short job and verify logs and model saved.

## Commit & Pull Request Guidelines
- Commits: imperative, concise, scoped (e.g., “Adjust SLURM script for booster queue”).
- Include why + scope when touching `run_experiment.sbatch`/`submit_job.sh` and configs.
- PRs: clear description, reproducible command(s), affected configs/paths, and sample logs/metrics (screenshots or snippets).
- Link related issues; note breaking changes and migration steps.

## Security & Configuration Tips
- WANDB runs offline by default as compute node on cluster does not have internet access; set `WANDB_MODE=run` to sync online.
- SLURM account replacement uses `.slurm_account`; review before submitting.
- Keep large artifacts out of Git; rely on `logs/`, `models/`, and external storage.

## Cluster Access Notes
- The cluster project workspace path is `/p/project1/hai_1074/meyjohann1/thesis` and mirrors the local repo structure (`logs/`, `models/`, `wandb/`, `run_*.sbatch`, `train_*.py`).
- For repeated non-interactive cluster access after one MFA login, use SSH connection multiplexing (`ControlMaster`/`ControlPersist`) with a control socket under `~/.ssh/cm/`.
- Prefer `scripts/cluster_readonly.sh` for cluster read operations (logs/wandb inspection). It enforces a read-only command allowlist and should be the default path for non-mutating cluster triage.
- Typical flow:
  - Start master once: `ssh -MNf juwels-booster.fz-juelich.de` (requires key passphrase + TOTP once).
  - Verify reuse: `ssh -O check juwels-booster.fz-juelich.de`.
  - Reuse for commands/log triage without re-prompt while master is alive: `ssh juwels-booster.fz-juelich.de 'cd /p/project1/hai_1074/meyjohann1/thesis && ...'`.
  - Close when done: `ssh -O exit juwels-booster.fz-juelich.de`.
- Cluster submission workflow (non-interactive):
  - Keep local + cluster repos in sync first (use local commit/push, then `git pullall` on cluster repo).
  - Use `bash submit_job.sh -y <script>.sbatch` to accept defaults without interactive prompts.
  - `-y` requires SLURM account auto-detection to work (or `SLURM_ACCOUNT`/`.slurm_account` to be set), otherwise it exits with a clear error.
  - After submit, verify queued/running jobs with `squeue -u "$USER"` or `squeue -j <JOBID>`.
- For log access, prefer direct paths in the cluster project dir, e.g.:
  - `/p/project1/hai_1074/meyjohann1/thesis/logs/drqv2_*.out`
  - `/p/project1/hai_1074/meyjohann1/thesis/logs/<run_name>_<JOBID>.log`

## W&B Debugging Workflow
- Canonical term is `wandb` (voice-to-text may transcribe this as `1B`; interpret `1B` as `wandb`).
- Cluster pipeline expectation:
  - Training jobs write offline runs to local `wandb/` on cluster (compute nodes have no internet).
  - `submit_job.sh` launches background sync on login node (`WANDB_MODE=online wandb sync --sync-all`) to upload those offline runs.
- Standard triage order:
  1. Inspect Slurm and per-run logs first (`logs/*.out`, `logs/*.err`, `logs/<run>_<JOBID>.log`).
  2. Inspect offline `wandb/` folders for the same job IDs (`offline-run-*` presence/timestamps).
  3. Check online W&B project/runs for corresponding uploaded runs.
- Interpretation:
  - No offline `wandb` run folder usually means failure happened before/during wandb initialization.
  - Offline folder exists but no online run usually means post-run sync failure (upload stage), not training-stage logging failure.
