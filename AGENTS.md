# Repository Guidelines

## Core Working Rules

- When the user corrects a workflow or is dissatisfied, add one concise durable rule here and a short local note under `codex/`.
- Save other durable project context under `codex/` and always tell the user when doing so.
- Never commit `codex/`, `.codex/config.toml`, runtime credentials, local datasets, model artifacts, or machine-specific state.
- Use `apply_patch` for file edits and follow tool/API contracts literally.
- Default copy/paste commands to multiline shell syntax with trailing `\` continuations.
- Keep train, eval, plotting, and launch commands synchronized. Checkpoints should serialize settings required to reconstruct evaluation.
- Update this file in the same change whenever project architecture or durable operating constraints change.
- Prefer explicit environment-family entrypoints and control paths over opaque generic routing.
- Inspect the live implementation and current artifacts before answering; do not infer active flags, paths, or results from old runs.
- Use targeted smoke tests and `py_compile`/syntax checks for changed codepaths before reporting completion.
- Give frequent concrete progress updates and partial findings during long investigations.
- Proactively suggest automation that reduces manual setup, validation, or supervision.
- Subagents must not run escalated commands. They return the command/context and the main agent handles approval.
- Interpret speech-to-text `1B` or `1p` as `WANDB`.
- The Linear team is `Thesis`; issues use `THE-XXX`.

## Git And Publishing

- The worktree may be dirty. Never revert unrelated user changes or use destructive Git commands without explicit approval.
- Treat submodules as independent repositories: commit and publish the submodule before committing its parent pointer.
- Never push a submodule or embedded repository unless its remote is confirmed to be user-owned.
- For supervisor/upstream-owned remotes, keep commits local or provide patches; do not push even to a new branch without explicit ownership confirmation.
- Push only coherent, granular commits. Keep generated artifacts and local experiment outputs out of commits unless they are intentional reusable fixtures.
- Do not amend commits unless explicitly requested.

## Experiment Operations

- GPU experiments must use the `autonomous-research` skill and experiment queue; do not probe `/dev/dxg` or assume sandbox GPU access.
- Queue wrappers under `scripts/generated/` must be self-contained, export the full config, and invoke repo launchers by absolute path.
- Smoke-check generated launchers against the live parser before queue submission.
- `stop_now` is queue-wide. Use `cancel_job` only for queued jobs; do not use `stop_now` to target one running job.
- Monitor GPU utilization and increase concurrency only when memory, utilization, and environment construction permit it.
- Log resolved controller, environment/layout, checkpoint, reward, seed, device, and intervention settings before expensive imports.
- Training comparisons require identical benchmark manifests, seeds/layout cohorts, horizons, reward/cost semantics, and episode quotas.
- A shared seed is not a paired layout cohort when actor construction or asynchronous partial resets consume shared RNG; replay a precomputed per-episode terrain/start/goal manifest for strict method comparisons.
- Use fixed per-environment episode quotas in vectorized eval; first-N-completions biases results toward fast failures.
- Always include the relevant no-safety/goal-only baseline in method-comparison reports.
- Reports should include learning curves, final metrics, trajectory plots with cost markers, checkpoint paths, and interactive eval commands.
- Human/intervention-method reports must plot cumulative intervention frequency and sampled-batch teacher fraction over training; final policy metrics alone are incomplete.
- Unitree loss ablations use literal names: `pref_only` has anchored preference learning but no environmental TD/RL or BC; `pref_rl` is the original thesis objective; `bc_rl` and `bc_only` disable preference learning.

## Debugging Invariants

- Verify end-to-end signal integrity: observation/goal content, action transform/clipping, executed action storage, intervention flags, replay routing, and nonzero metric progression.
- Keep reward and intervention logic deterministic, task-local, single-source, and independently configurable.
- Do not pass sentinel values such as `0` or `-1` into library constructors when they mean “use default”; omit the argument.
- Distinguish `configured`, `connected`, and `receiving data` in runtime status messages.
- Debug/status UIs must fit their default window and must not open competing pygame displays.
- Resolve timestamped run/checkpoint paths from disk; never infer them from queue submission time.
- Before giving a runnable launcher command, verify that the referenced script exists; if no launcher exists, invoke the live Python entrypoint with its verified parser flags instead of inventing a launcher name.
- When eval results differ, first verify checkpoint, environment generator, seed/reset logic, horizon, cost semantics, and train/eval wrapper parity.
- Do not claim a visualization or coordinate-frame fix until the generated image or live tensors verify it.
- Multimodality diagnostics must test both temporal mode switching and low-magnitude mean-action collapse; sign flips alone are insufficient.

## Safety-Gym Architecture

- Main custom training uses `train_fast_sac_safetygym_minimal.py` with `safetygym_utils/`; interactive evaluation uses `eval_interactive_safetygym.py`.
- Keep car and Point action abstractions explicit. Point uses `point_action_mode`; do not overload car action modes.
- `obs_frame_stack` and optional `temporal_encoder=attention` must be reconstructed from checkpoint args in eval and plotting.
- Diagnostic privileged observations use `obs_mask_mode=privileged_geometry` or `privileged_geometry_rich`; label them as diagnostics, not deployable policies.
- Euclidean-distance-only learner reward means `reward_mode=dense`, `success_reward_scale=0.0`, and no clearance shaping.
- Intervention score configuration is independent of learner reward. Obstacle-aware gates may use potential/clearance fields while learner reward remains obstacle-blind.
- `teacher_override_mode=reward_progress` and `pcpo_value_progress` are online gate paths; keep launcher, controller, and WANDB diagnostics aligned.
- Use `fixed_layout_preset=car_single_block` and `tools/plot_safetygym_intervention_tuner.py` before tuning obstacle gates on random layouts.
- Validate teacher-gated rollouts for zero cost before training students; teacher-only quality does not prove the gate intervenes early enough.
- Native Safety-Gym cost, visual footprint cost, and keepout cost are different benchmarks. Use `footprint_cost_mode=visual` only for rendered-contact semantics.
- Pair `teacher_clearance_source=visual_footprint` with visual footprint costs; reserve `keepout` for conservative diagnostics.
- `pref_sampling_mode=linked` must compute TD loss on the full sampled batch and preference loss on intervened rows from that same raw batch.
- Apply observation preprocessing exactly once in linked preference learning.
- Preference action augmentation (`pref_action_noise_copies/std`) affects preference rows only; do not jitter observations without geometry-aware transforms.
- Action-delta weighting uses `pref_action_delta_weight_scale/max`; report it separately from behavior cloning.
- HIL-SERL, BC/HG-DAgger, PVP, EIL, and the thesis preference method are distinct baselines. Do not combine or relabel them.
- A faithful PVP comparison uses the dedicated TD3-style two-buffer implementation, not the approximate shared FastSAC variant.
- Replay insertion must distinguish full-vector batches from single-env annotated rows and use blocking CUDA-to-CPU copies for immediate CPU storage.
- Human input attached to Windows while training in WSL requires a host sender and WSL receiver; do not rely on SDL passthrough.
- A gamepad intervention run needs a real environment step cap (`env_fps_limit`), not only a controller-window FPS cap.
- Safety-Gym pygame telemetry must share/suppress the environment viewer rather than opening a competing display.
- Windows MuJoCo viewer compatibility fixes belong in the vendored Safety-Gym viewer shim, not in training logic.

## Unitree / Isaac Navigation Architecture

- The active integration is the submodule `external/unitree_rl_mjlab`.
- Native rough-terrain locomotion inspection uses `eval_unitree_velocity_policy.py` and `scripts/run_unitree_mjlab_velocity_inspect_local.sh`; keep it separate from the high-level navigation task.
- Rough-terrain inspection must override the velocity task's small contact capacity when needed; the local launcher defaults `NCONMAX=256` to prevent seed-dependent MuJoCo Warp initialization overflow.
- G1 and Go2 velocity checkpoints are not interchangeable: validate the checkpoint actor output before environment construction (`29` G1 joints versus `12` Go2 joints).
- Supervisor velocity checkpoints may use split `actor_state_dict`/`critic_state_dict` format; adapt them in memory for the installed legacy RSL-RL runner and preserve the original checkpoint bytes.
- When inspecting a legacy scan-blind velocity checkpoint on rough terrain, preserve rough physics but remove the task height-scan terms to match the checkpoint observation space; do not infer its original training terrain from dimensions alone.
- Rough/Rough2/Parkour are low-level velocity-tracking tasks. Terrain avoidance requires a later high-level route/command policy rather than relabeling the low-level locomotion objective.
- The G1 asset has no articulated head or neck; head-mounted RGB-D inspection uses a rigid `robot/torso_link` camera with configurable pitch/FOV and must not alter the low-level checkpoint observation shape unless a new vision policy is trained.
- Core files are `train_unitree_nav_thesis.py`, `unitree_nav_layout.py`, `unitree_nav_eval_manifest.py`, `unitree_nav_geom_teacher.py`, `eval_unitree_nav_baselines.py`, `plot_unitree_nav_rollout.py`, and `eval_interactive_unitree_nav.py`.
- Local launchers are `scripts/run_unitree_mjlab_nav_{smoke,baseline_eval,plot,thesis,interactive}_local.sh`.
- Use the same low-level locomotion checkpoint across train, batch eval, plots, and interactive eval. The current default is the omni-finetuned G1 run under `external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/2026-07-12_10-35-19_omni_finetune_model1499_20260712`.
- Navigation diversity uses a persistent pre-generated terrain tile bank plus randomized tile assignment and bounded radial goals. Do not rebuild the full terrain on every reset or Next action.
- Do not describe tile/start/goal resampling as novel terrain generation. Report bank size, unique tile IDs or geometry hashes, and held-out generator seeds when assessing Unitree layout diversity.
- All-blocked Unitree benchmarks must force a goal behind a sampled obstacle and validate the blocked-corridor postcondition for every initial and partial vector reset; probability alone is not sufficient.
- Random multi-obstacle Unitree benchmarks must not silently choose only the nearest feasible blocked goal; record goal-path length and distinct blocking-component counts to verify layout diversity.
- Before new human ablations, audit tile, start-position, goal-distance, and blocking-component diversity; repeated finite-bank layouts or obstacle-relative goal templates must not be presented as independent environment randomization.
- The canonical persistent benchmark uses 6 obstacles, width 1.0-1.4 m, height 1.0 m, platform width 2.0 m, and a 5x10 tile bank.
- The revised human-ablation benchmark uses a persistent 10x20 tile bank; historical scripted-teacher comparisons remain on their saved 5x10 checkpoint manifests.
- Set both `env_cfg.seed` and `env_cfg.scene.terrain.terrain_generator.seed`; forced blocked-goal selection also requires NumPy and Torch seeding.
- Apply goal-obstacle clearance repair after forced goal-through-obstacle placement in train and eval.
- On partial vector resets, update goals and layout metadata only for `done_idx`.
- Height-scan observations flatten as row=lateral and column=forward. Verify mappings against `terrain_scan.data.hit_pos_w`.
- Scanner density and temporal context are first-class settings: keep `height_scan_resolution` and `scan_history` synchronized across train, batch eval, interactive eval, plotting, checkpoint metadata, and the scan-CNN encoder. A 0.25 m resolution preserves the 3x3 m footprint while producing a 13x13 scan; history 5 yields `9 + 5*169 = 854` policy inputs.
- Rectangular Unitree scans use `height_scan_forward_size` and `height_scan_lateral_size`; preserve both in checkpoint manifests. History stacking accepts flattened rectangular scans when `scan_history` is explicit. The current scan CNN and scripted scan teachers remain square-only, so rectangular scratch runs must use the MLP encoder and policy/human control unless rectangular shape metadata is implemented there.
- Heightfield raster plotting maps columns to world x and rows to world y; pass `(nrow, ncol)` directly to `imshow` without transposing.
- Interactive Unitree obstacle overlays must use the exact reset-time `obstacle_cells` geometry used by goal filtering; keep scan-detected points as a separately labeled observation trail rather than reconstructing a second privileged heightfield map.
- `geom_scan_teacher` uses privileged obstacle geometry clipped to the current scanner footprint and is a diagnostic bridge, not a final observation-only teacher.
- Observation-only scan teachers support `teacher_scan_planner=heuristic|astar`; account for the coarse 7x7 scan with physical clearance and cell padding.
- Teacher state includes bypass-side commitment and optional command smoothing. Keep `teacher_geom_side_frame=body` as the conservative default unless a matched audit supports another mode.
- `teacher_goal_stop_dist` must be strictly inside `success_dist`; with success 0.50, use stop distance 0.40.
- Goal arrival is a successful terminal event. Configure `goal_reached` as a timeout-class termination so the environment's generic failure-termination reward does not penalize successful episodes, while tracking success from the explicit goal term.
- `navigation_episode_mode=continuous_goals` must preserve simulator state on success, sample the next goal with a verified obstacle-clearance postcondition, and terminate only the learning segment so replay returns never span two goals; `episodic` retains full reset-on-goal behavior.
- Clearance-or-stall gating must suppress/release stall intervention inside success distance.
- Audit teachers on the exact student-training layout distribution before producing labels.
- `policy_encoder=scan_cnn` infers the square scan side and accepts stacked scan channels; checkpoint eval must reconstruct resolution and history from saved args.
- Unitree policy checkpoints are eval manifests. Before environment creation, restore task/layout, obstacle dimensions, goal radius, scan and action history, architecture, and training-time action smoothing; do not restore teacher/intervention settings.
- Unitree training runs used for decisions must schedule deterministic teacher-free checkpoint evaluation and log it separately under `eval/`; teacher-gated training metrics are not policy-quality evidence.
- Unitree intervention comparisons use the single `train_unitree_nav_thesis.py --method` entrypoint. Keep `thesis`, `hilserl`, `eil`, `pvp`, `hg_dagger`, and goal-only `sac` distinct while sharing the exact teacher, intervention gate, layout manifest, initialization, transition budget, and teacher-free eval cohort.
- Unitree PVP uses separate novice/human replay buffers and its dedicated TD3-style proxy-value update. HG-DAgger is actor-only ensemble behavior cloning on intervention states. HIL-SERL oversamples intervention transitions without BC or preference loss. EIL applies good/bad Q-threshold losses, including pre-intervention learner rows.
- Use `scripts/generated/run_unitree_competitor_method_20260718.sh` as the queue unit for matched comparisons and `scripts/generated/plot_unitree_competitor_artifacts_20260718.sh` for common learning curves and trajectory artifacts.
- For goal-only reward diagnostics, `dense_progress_exp` adds near-goal exponential potential progress to the linear distance-progress term; keep both scale and temperature in the checkpoint args.
- Unitree goal-signal diagnostics may mask proprioception and height scans independently while preserving checkpoint dimensions. N-step replay rows must stay within one vector environment and one episode, and SAC discounts them by `gamma ** effective_n_steps`.
- Unitree vector training must report updates per inserted transition; scale `updates_per_step` with `num_envs` when targeting UTD 1. Never bootstrap auto-reset transitions without a preserved terminal observation.
- `action_history` appends previous executed high-level commands as temporal context while scan history remains CNN channels. Keep this synchronized across train, eval, plotting, and actor/critic reconstruction.
- Future Unitree navigation training masks the final-heading command by default (`mask_goal_heading`), retaining only body-relative forward/lateral goal displacement while preserving the nine-value context shape for checkpoint compatibility. Eval and plotting must restore this mask from new checkpoints; old checkpoints remain unmasked.
- Diagnose goal-heading masking with a same-checkpoint, same-layout, same-seed A/B; do not attribute changes from runs that also alter terrain generation, gating, or initialization.
- Unitree navigation intervention clearance can be tied to the geometric teacher with `intervention_clearance_mode=teacher_ratio`; checkpoint/eval paths must preserve the trigger/release ratios and the teacher clearance. Use `strict_min_size_obstacles` for navigation terrain so nominal obstacle dimensions are not reduced by tile-boundary clipping or central-platform erasure.
- Correction noted: do not assume teacher-relative intervention ratios below `1.0` are safe merely because a short rollout probe is zero-cost. The Unitree `2/3` engage and `5/6` release ratios allowed late, unrecoverable takeovers in parallel training; audit teacher- and student-executed costs across multiple simultaneous layouts before accepting adaptive thresholds.
- Unitree intervention diagnostics must distinguish trigger conditions from actual gate engagements and release blockers. Intervention trajectory plots use crimson path segments for teacher-executed actions; do not infer engagement causes from condition fractions alone.
- `--init-checkpoint` restores networks, alpha, and preference state but not replay or optimizer state. Label it a warm-start continuation and preserve teacher warmup.
- Interactive eval with `--controller policy` is policy-only unless manual keyboard control is used; no teacher gate is active.
- Unitree human collection uses `unitree_nav_gamepad.py` over the Windows raw-state publisher on port `8794`; hold the configured gate button to replace only the high-level navigation action, never the low-level locomotion policy.
- Do not describe Unitree gamepad evaluation/NPZ collection as human-intervention training. A complete human pipeline must either update `train_unitree_nav_thesis.py` online from gamepad takeovers or provide and verify an explicit offline trainer that consumes the recorded Unitree dataset.
- Unitree online-human runs and raw datasets are immutable, but relaunching with a reused `ONLINE_RUN_NAME` must allocate a unique suffixed run directory rather than crash or overwrite existing results.
- The Windows `C:\\Data\\thesis` checkout is a separate Git worktree. Its Unitree human-control launcher must invoke the authoritative `/home/benjamin/thesis` WSL collector via `wsl.exe` and reuse the already-present Windows Safety-Gym raw gamepad sender, rather than assuming uncommitted code is synchronized.
- PowerShell launchers that emit Bash commands must avoid nested quote literals. Build shell-escape sequences with explicit character codes or pass values through a robust transport, then run a native PowerShell parse check before reporting them runnable.
- Human Unitree datasets are immutable chunked NPZ directories with a manifest. Preserve policy and human transitions, intervention edges, and transport status; derive cleaned subsets separately after auditing the raw collection.
- Goal-only checkpoints may restore `disable_obstacles=True`; human collection must restore their policy shape while explicitly forcing the obstacle benchmark back on.
- Human-control status must report transport connection, fresh receipt, state age, axes, and gate state; "configured" is not evidence that Windows input is arriving in WSL.
- Windows controller publishers must use the existing `C:\Users\benja\miniconda3\envs\thesis\python.exe` environment, not a transient `uv` environment with only Pygame.
- Windows-to-WSL human collection must probe the Windows publisher port before launching WSL; fail locally with the sender traceback rather than beginning a run that can only report connection refused.
- WSL loopback does not reach the Windows publisher on this machine. Discover the Windows `vEthernet (WSL*)` IPv4 address dynamically and pass it as `GAMEPAD_HOST`; do not hardcode `127.0.0.1` or a stale `172.28.x.x` address.
- Windows gamepad launcher cleanup must terminate the nested PowerShell process tree so the Python publisher cannot remain orphaned on port `8794`; reject an already-owned port before starting.
- When PowerShell itself originated through WSL interop, nested `wsl.exe` returns no output or exit code. Use the standalone Windows sender plus shared endpoint file and start the WSL receiver/collector from a separate WSL terminal.
- Cross-Windows/WSL gamepad freshness must use the WSL-local packet receipt timestamp, not the sender wall clock; host clock skew can exceed the stale timeout even while packets arrive continuously.
- Unitree human takeover UI must distinguish raw controller receipt from actual stick-gated control: show a high-contrast live-takeover indicator and log policy, gamepad, and executed high-level actions at takeover edges.
- Unitree human steering uses deadzoned stick-command magnitude as the intervention gate, not a button: default to stick mode with a configurable post-deadzone command-norm threshold and make that threshold visible in the overlay.
- An obstacle-free Unitree goal-only checkpoint must not define the intervention benchmark through checkpoint restoration. After architecture restore, apply the strict blocked profile with goal/start clearance repair, strict minimum obstacle dimensions, and a verified obstacle between start and goal.
- Unitree Windows gamepad lateral control is inverted relative to the environment body-y convention; force the runtime lateral inversion so a persisted profile cannot silently reverse human steering.
- Interactive Unitree scan overlays must read the current unmasked environment observation, never policy-masked input, and must not retain historical hit trails; keep the live scan toggle for uncluttered visual inspection.
- A goal-only actor intended to initialize obstacle-intervention training must see the same obstacle observation/layout distribution during pretraining. Log costs separately but keep them out of the learner reward when measuring goal-only behavior.
- Obstacle-avoidance learning diagnostics must replay one blocked-layout manifest for every checkpoint and include a scan-blind `direct_goal` controller; obstacle-free checkpoint evaluations are not comparable evidence.
- Forced blocked Unitree goals must be sampled beyond a selected obstacle using obstacle-relative distance multipliers and accepted only after both goal-clearance and blocked-corridor postconditions pass; do not repair them with an unconstrained generic goal resample.
- The interactive Unitree student view must remain non-privileged: body-frame live height-scan samples plus body-relative goal direction only, with terrain geometry, trajectories, and cost history hidden.
- Interactive controls: Right changes tile/layout, `R` resets, Space pauses, `N` single-steps, `T` toggles autopilot, `V` toggles costly RGB rendering, WASD/QE manually control.
- Interactive visualization keeps policy inference every simulation step. Up/Down changes RGB render stride; `+/-` changes top-down UI stride; `--sim-fps 0` means unlimited simulation stepping.
- Actual Isaac camera videos require `RecordVideo`/camera rendering. Do not substitute trajectory animations when a camera video was requested.
- Headless `RecordVideo` uses `ViewerCfg` configured before `gym.make`; interactive camera calls do not configure captured output.
- Cluster RTX camera recording may segfault. Run camera recording locally when needed and state clearly when only posthoc trajectory video is available.

## Legacy OGBench / Manipulation

- OGBench and maze work is maintenance-only unless explicitly requested; do not expand these sections in this file.
- Keep maze and manipulation entrypoints separate. Manipulation uses `train_fast_sac_ogbench_manip.py`; dedicated comparison trainers exist for BC, HG-DAgger, and PVP-TD3.
- Local OGBench defaults WANDB online; cluster launchers use offline mode.
- Configure `MUJOCO_GL` before importing OGBench. Windows uses `glfw`, not `egl`.
- Human VR intervention must use the saved mapping profile and persistent transport; runtime profiles/datasets live in user config/data directories, not `codex/`.
- Human partial gripper overrides must merge only the gripper channel; inactive mirroring must not trigger full-policy takeover.
- Keep no-op handling centralized and opt-in; gripper-only commands must not trigger arm IK.
- Historical details and experiment-specific corrections belong in local `codex/` notes or Git history, not this operating guide.

## Validation Checklist

- Run syntax/import checks for changed Python files and shell syntax checks for launchers.
- Run the smallest live smoke test that exercises the changed path.
- Confirm train/eval manifests and checkpoint metadata agree.
- Inspect generated trajectory/visual artifacts rather than relying only on scalar summaries.
- Report anything not tested, unavailable GPU/runtime dependencies, and any publish blocker explicitly.
