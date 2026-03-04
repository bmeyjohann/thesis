# Repository Guidelines

## General Guidelines

- Whenever I need to tell you to do something in a different way or am dissatisfied with your work, append concise info on what you made wrong and how to do it correctly to `AGENTS.md`. Also save a short note in codex/[appropriate_naming_for_info].md.
- Save other relevant info that should be preserved in codex/[appropriate_naming_for_info].md
- Always inform me if you save info in a file.
- When providing runnable commands for copy/paste, default to multi-line format with trailing `\` line continuations.
- Correction noted: never invoke `apply_patch` through `exec_command`; use the dedicated `apply_patch` tool for file edits.
- Whenever project architecture changes, immediately update all architecture-related sections in `AGENTS.md` in the same work so this file stays aligned with the actual implementation and is never outdated.
- Proactively suggest automation improvements that reduce manual setup/debugging and burden for the human and increase how much Codex can run autonomously (including when escalation is needed), and include concrete next steps to implement that automation. This is about Codex being able to run for longer without requiring human intervention or oversight, so closing the look such that Codex can larger tasks autonomously by checking its own implementation and iterating on that or doing more thing by receiving the appropriate tooling.
- Subagent permission policy: subagents must not execute commands that require escalated permissions. If escalation is needed (e.g., SSH/cluster operations), subagents should return the required command/context and the main agent executes it with explicit user approval.
- STT clarification: whenever the user says or writes `1B` or `1p`, interpret it as `WANDB` every time. This mixup is caused by the user's speech-to-text tool.
- Correction noted: for cube state-observation variants, use FastSAC manipulation entrypoint (`train_fast_sac_ogbench_manip.py`) rather than DrQ-v2 scripts.
- Correction noted: for intervention behavior, do not use intervention-probability decay scheduling; keep interventions deterministic by teacher-student deviation/tolerance rules.
- Correction noted: when modularizing maze/manip pipelines, prefer explicit per-environment modules and entrypoints over shared env-family flag dispatch so debugging paths stay readable.
- Correction noted: keep OGBench wrapper stacks split by environment family (maze vs manip) in separate modules to avoid cross-family reward/observation wiring mistakes.
- Correction noted: for `eval_interactive.py`, default to checkpoint-args auto-sync and avoid adding redundant manual flags when the checkpoint already contains the required train-time observation/wrapper settings.
- Correction noted: for manipulation checkpoint visual evals, use `eval_interactive_manip.py` (not `eval_interactive.py`) so manip-specific wrapper/config auto-sync is applied correctly.
- Correction noted: when changing eval control-panel APIs, keep loop call-sites and panel methods in sync (e.g., `set_intervention_active`), and run at least a tiny method-level smoke test to catch missing-method regressions.
- Correction noted: in `eval_interactive_manip.py`, keep control-panel/keyboard return signatures aligned with loop unpacking (including FPS delta channel) to avoid runtime tuple-unpack crashes.
- Correction noted: `eval_interactive_manip.py` policy inference must load/apply checkpoint `obs_normalizer_state` (and checkpoint LayerNorm architecture flags) to mirror train/eval behavior.
- Correction noted: do not keep compatibility/facade wrapper routers for OGBench when they obscure flow; select maze/manip wrapper builders directly in each train/eval entrypoint with explicit local `if` logic.
- Correction noted: define concrete wrapper classes locally in `ogbench_utils` (maze/manip/common) instead of relying on `ogbench.wrappers` indirection so wrapper behavior is directly inspectable during debugging.
- Correction noted: when disagreement/Q diagnostics are needed, derive them from critic outputs already computed during the update pass and log those, instead of adding another critic forward in the rollout path.
- Correction noted: in FastSAC OGBench, `batch_size` is the total per-update sample count (not divided by `num_envs`), then split by demo/pref ratios.
- Correction noted: FastSAC OGBench preference optimization supports `pref_loss_type` = `margin` / `bradley_terry` / `lagrangian`; lagrangian uses dual parameters (`pref_lambda_*`, `pref_violation_clip`) and logs `Train/pref_lambda`, `Train/pref_violation`, `Train/pref_violation_ema`.
- Correction noted: FastSAC OGBench supports optional LayerNorm in MLP backbone/policy head/critic heads via `--use_layer_norm` and `--layer_norm_eps` for stabilization experiments.
- Correction noted: in manipulation `agent_reward_progress` mode, use `diag/cube_max_target_error` first as progress signal; sparse-mode wrapper `dense_reward` can be constant zero and should not drive intervention gating.
- Correction noted: for cube state runs with `cube_reward_mode` enabled, keep a single dense reward source by forcing wrapper `reward_type='sparse'` and using `CubeRewardModeTracker` for dense shaping.
- Correction noted: log critic loss components separately (`replay`, `pref`, `pref_weighted`, `total`) to diagnose replay-vs-preference optimization behavior.
- Correction noted: always log buffer sizes/capacities every log step (replay/demo/pref and variant-specific buffers) with explicit zero values when disabled, so WANDB panels stay consistent across train scripts/environments.
- Correction noted: when asked to locate a specific loss path, inspect and reference the active implementation lines directly (including reducer choice like min-vs-per-critic) instead of answering with a generic or outdated snippet.
- Correction noted: when the user references a specific line range/block (e.g., critic-side preference ranking), keep responses scoped to that exact block and avoid conflating it with later actor/BC sections.
- Keep train and eval scripts in lockstep: any change to training-time observation/action/reward/wrapper/model I/O must be mirrored in eval tooling so newly produced checkpoints remain directly runnable in eval without manual fixes.
- For non-learning/debug investigations, explicitly verify end-to-end signal integrity: (1) observation/goal content, (2) action transform/clipping consistency, (3) intervention action storage format in replay/pref/demo buffers, and (4) presence/non-zero progression of the intended W&B metrics (not just summary endpoints).

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
- `scripts/smoke_safetygym_all.sh`, `scripts/safetygym_matrix_local.sh`, `scripts/verify_safetygym_ckpt_eval_lockstep.sh`, `scripts/profile_safetygym_perf_matrix.sh`, `scripts/review_safetygym_checkpoints.sh`: Safety-Gymnasium automation helpers for smoke checks, matrix runs, visual checkpoint review, perf breakdown profiling, and checkpoint/eval compatibility checks.
- `scripts/smoke_fastsac_ogbench_modularization.sh`: Compile-smoke check for FastSAC OGBench maze/manip CLI+env+train modules and shared setup/loop modules.
- `scripts/run_cube_single_static_demo_dense_online.sh`: Single-cube deterministic reset test run (fixed reset seed + demo prefill + dense cube reward mode).
- `sc_venv_template/`: Virtual gridworld environment setup for HPC (Isaac Sim and Isaac Lab use an apptainer container instead of a virtual environment).
- `logs/`, `models/`, `wandb/`: Outputs, artifacts, offline tracking.
- `ogbench_utils/env_wrappers_maze.py`: maze-only wrapper stack (`FlexibleObsWrapper` + reward/intervention wrappers) and maze goal-marker coloring helper.
- `ogbench_utils/env_wrappers_maze.py`: maze-only wrapper stack and local concrete wrappers (`MazeFlexibleObsWrapper`, `MazeDetailedRewardWrapper`) plus maze goal-marker coloring helper.
- `ogbench_utils/env_wrappers_manip.py`: manipulation-only wrapper stack and local concrete wrappers (`ManipDetailedRewardWrapper`, `ManipGoalConditionedObsWrapper`, `CubeTeacherInfoAdapter`, `ManipRelativeCubeFeaturesWrapper`) plus cube reward-mode tracker utilities.
- `ogbench_utils/env_wrappers_common.py`: shared wrapper helpers (`FixedResetSeedWrapper`, environment-family inference, intervention wrapper construction).
- `ogbench_utils/intervention_wrappers.py`: local copy of intervention wrapper implementation used by `env_wrappers_common.py` (no dependency on `ogbench.wrappers` for intervention logic), including manual-gate control-panel diagnostics (`Q_min`, `Q_dis`) and `agent_mode='always'` support.
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
- Correction noted: manipulation gripper-lock intervention must use target-local grasp evidence (contact + near target + non-fully-closed opening) rather than raw contact alone to avoid finger self-contact false positives.
- Correction noted: FastSAC manip training `train_render_mode=human` must launch/sync the manip passive viewer (`launch_passive_viewer`/`sync_passive_viewer`) via the vector env wrapper and use `MUJOCO_GL=glfw`; passing `render_mode='human'` alone is insufficient.
- Correction noted: instantiate manipulation Markov oracles with keyword `env=...` (not positional), because `CubeMarkovOracle` has `max_step` as first positional argument and positional env causes constructor failure.
- Correction noted: never trigger eval logging with `total_env_steps % eval_interval == 0` when using vectorized envs; use a `next_eval_step` threshold (`>=`) so eval/WANDB logs are not skipped by step-size jumps.
- Correction noted: when the user asks for UI/debug changes in training-time human intervention mode (`agent_manual_gripper`), implement them in the training intervention wrapper path first (not eval tooling) and keep console output episode-level unless step-level is explicitly requested.
- Correction noted: when the user asks for an updated run command, preserve their previously established hyperparameters/mode choices and only change the explicitly requested knobs (e.g., teacher type, batch size, reward semantics), and explain reward-flag semantics clearly.
- Correction noted: when providing run commands for preference/intervention experiments, never omit required boolean enablers (`--use_intervention`, `--pref_buffer_enable`) or loss selector flags (`--pref_loss_type`) when related hyperparameters are present; otherwise the run silently uses defaults and mismatches intent.
- Correction noted: FastSAC OGBench supports linked preference sampling via `--pref_sampling_mode linked`; replay stores `student_actions`, and preference loss is computed on sampled replay rows where executed-vs-student action delta exceeds `--pref_linked_action_epsilon`.
