# Unitree Multimodal Locomotion Suite

## Scope

The first suite trains no-memory locomotion students. It is followed by GRU and GRU-plus-privileged-height-reconstruction ablations.

Physical cells cross five geometries (`flat`, `random_rough`, `cobblestone`, `stairs`, `stepping_stones`) with three conditions (`rigid`, `slippery`, `sand_drag`). `sand_drag` is a labeled proxy using contact grip plus increased joint damping; it is not deformable granular simulation.

Sensor students are trained separately for `height_scan`, `depth`, `mono_rgb`, and `stereo_rgb`. Camera students use 80x60 frames during the initial feasibility suite.

The common expert actor observes 285 values: angular velocity (3), projected gravity (3), commanded planar/yaw velocity (3), gait phase (2), joint positions (29), joint velocities (29), previous action (29), and a privileged local height scan (187). The critic adds 15 privileged base/foot/contact values; students never receive those critic-only values.

## Stage Order

1. Screen the robust 9,999-iteration rough-terrain supervisor on every physical cell.
2. Fine-tune only cells exceeding one termination per 1,000 steps or falling below 0.05 mean step reward.
3. Select each checkpoint by lowest matched fall rate (reward breaks ties), then evaluate it on its own cell over five fixed seeds.
4. Distill four independent no-memory students from teacher-controlled rollouts across all cells.
5. Evaluate each final student over every physical cell and five fixed seeds.
6. Aggregate seed means/deviations and render the transfer matrix.
7. Train GRU students for all modalities with 25-step truncated backpropagation.
8. Train GRU-plus-reconstruction students for depth, mono RGB, and stereo RGB.
9. Evaluate and plot the combined memory/reconstruction ablation.

Every training cell has a runtime smoke gate. A failure stops that stage and preserves the command and log path in its state file.

## State And Outputs

- Expert state: `artifacts/unitree_multimodal/sequence/state.json`
- Expert logs: `artifacts/unitree_multimodal/sequence/logs/`
- Expert checkpoints: `logs/rsl_rl/g1_multimodal_cell_experts/`
- Expert evaluation: `artifacts/unitree_multimodal/sequence/evaluation/`
- Adaptive expert state: `artifacts/unitree_multimodal/adaptive_experts/state.json`
- Selected expert manifest: `artifacts/unitree_multimodal/sequence/evaluation/selected_experts.json`
- Student state/checkpoints: `artifacts/unitree_multimodal/students_nomemory/`
- Transfer results: `artifacts/unitree_multimodal/modality_matrix_results/`
- Recurrent students: `artifacts/unitree_multimodal/students_recurrent/`
- Recurrent matrix: `artifacts/unitree_multimodal/recurrent_matrix_results/`
- Fixed evaluation plan: `artifacts/unitree_multimodal/modality_transfer_matrix_plan.csv`

The initial scratch expert state and logs were preserved as `state_scratch_20260823_1813.json` and `logs_scratch_20260823_1813/` after matched evaluation showed substantially worse fall reliability than the existing supervisor.

For the feasibility matrix, only failed supervisor cells receive 2,000 warm-start fine-tuning iterations. Student distillation uses 5,000 steps per cell. Evaluation remains at five fixed seeds and 1,000 steps.

## Status

```bash
cd /home/benjamin/thesis
cat artifacts/unitree_multimodal/sequence/state.json
tail -40 artifacts/unitree_multimodal/sequence/logs/*.train.log
nvidia-smi
```

## Restart

The launchers are resumable and skip completed jobs. Start only missing processes after checking with `pgrep -af unitree_multimodal` and the state files.

```bash
./scripts/run_unitree_adaptive_expert_sequence.sh
./scripts/run_unitree_multimodal_posteval_watcher.sh
./scripts/run_unitree_modality_students_watcher.sh
./scripts/run_unitree_modality_matrix_watcher.sh
./scripts/run_unitree_modality_matrix_plot_watcher.sh
```

Do not start duplicate watcher processes. The modality stages wait on explicit completion artifacts and consume no GPU while blocked.
## Shared multimodal students

The primary experiment trains three checkpoints, not one checkpoint per sensor:

- `nomemory`: feed-forward shared student
- `gru`: recurrent shared student
- `gru_reconstruction`: recurrent student with privileged height reconstruction

Every checkpoint contains encoders for height scan, depth, monocular RGB, and
stereo RGB. Training balances one-hot modality assignments across parallel
environments and holds them for 500 steps before rotating, matching evaluation
with a persistent sensor rather than alternating sensors every frame. The
sequence cycles through all 15 geometry/material
cells in 500-step chunks with 32 environments for five rounds per architecture.
This yields 80,000 environment samples per cell while halving simulator
iterations relative to the original 16-environment, 5,000-step schedule.

After training, each checkpoint is evaluated with every input modality on every
physical cell and three fixed seeds (540 evaluations). The best architecture is
selected once using the declared aggregate locomotion score, then rendered for
all 60 modality/geometry/material combinations.

Artifacts are written below `artifacts/unitree_multimodal/students_shared` and
`artifacts/unitree_multimodal/shared_matrix_results`.

The privileged expert observation normalizer exactly follows RSL-RL and divides
by `stored_std + 0.01`. Using a small clamp instead is invalid because several
constant expert features have a stored standard deviation of zero.
The student also reuses these fixed statistics for proprioception and height
scans; the statistics are embedded in every checkpoint and replayed by all
evaluation and rendering paths.

An initial corrected-teacher run that still used raw student inputs is retained
only as a diagnostic. A controlled comparison showed that fixed expert-derived
student normalization sharply reduced falls and action jitter, so production
training starts from scratch with normalized proprioception and height scans.
