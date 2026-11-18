# Human intervention curriculum

Start the first round from ordinary Windows PowerShell:

```powershell
& 'C:\home\benjamin\thesis\scripts\run_unitree_human_curriculum_windows.ps1'
```

Open http://localhost:8080. Each fresh policy starts paused; press Pause / Run
when ready and intervene with the gamepad sticks. At 4,500 navigation decisions
the policy saves and the next condition opens paused. Refresh the browser if it
still shows the preceding server. Each run is approximately 15 minutes of active
simulation, so the first round is about one hour, plus pauses and startup.

Startup reports WSL/receipt checks, gamepad JSON readiness, and simulator
initialization separately. Connection and packet reads are bounded; if no JSON
arrives within approximately 30 seconds, startup fails with sender logs and
does not start training. The page becomes available after simulator initialization.

### Display and simulation speed

New curricula use `stabilized_visual_goal_v3`: stabilized camera images contain
a bright green, non-colliding cylinder at the current goal. It is rendered in
the policy camera, not just the browser, and excluded from terrain scans.
RGB gives its visible silhouette a fixed green highlight before encoding to
avoid near-black lighting. The mask uses the same camera's measured ray ranges
and the synthetic cylinder geometry; it does not expose an obstacle map.
Scene occlusion and field of view still apply. PHP marks the visible cylinder
silhouette with black/white stripes before its encoder so the goal stays
identifiable beyond PHP's 3 m depth clipping range. These stripes are synthetic
goal identity, not metric depth. There is no appended compass/distance strip in
v3 images. Both modes display the same processed images that their encoders receive.
Existing v2/v1/legacy trials retain their original inputs; start a fresh curriculum
to use v3. The v1 profile had only the appended goal strip, not a scene cylinder.

RGB+height and PHP human launchers now default to the **Policy view** page:
the current encoder-input image, its 0.2 s and 5 s history frames, and a
body-relative goal compass decoded from the current policy tensor. RGB+height
also shows the current 11x17 policy scan; PHP does not expose navigation scans.
The expandable context panel shows all nine current context values. There is
no privileged world map, trajectory, material label or extra RGB view in PHP.

The display reuses cached encoder inputs; it never renders another simulator
camera or constructs the Viser 3D scene. RGB sensor content remains 80x60; the
v3 profile keeps the 80x60 encoder image. PHP displays the
87x58 cropped/resized normalized depth, as quantized grayscale,
not the uncropped sensor image. Only older v1/v2 profiles append a strip. These are encoder inputs, not the learned
feature vectors themselves, so equal visual access does not imply identical
human and machine perception. History frames retain their original body frame.

**Simulation speed** selects 0.5x, 1x, 1.5x or 2x. Camera observations remain
5 Hz in simulation time (about 5 displayed new images/s at 1x); the page does
not invent interpolated images or acquire additional frames to look smoother.
HTTP display requests do not overlap and only receive the latest cached state.
Policy sensor resolution and sampling are unchanged. The old diagnostic 3D
viewer remains available via `--no-observation-only` on the local launcher.

The status shows requested versus achieved simulation speed and measured tick
time. The run summary retains physics, perception and learning timings. A speed target cannot exceed
hardware throughput; skipping display frames cannot fix slow physics or learning.
The physics timestep, four-tick action hold, 5 Hz decision rate in simulation
time, history ages, learner updates and saved transitions remain unchanged.
At 2x, five simulated decisions per second require ten decisions per wall-clock
second, so the human has less reaction time. Speed changes are recorded in the
run summary; keep speed consistent across comparison trials.

The 15-minute estimate is simulation time: wall time is approximately 15 minutes
divided by achieved speed, plus startup/pauses/saves. These controls take effect
when a new viewer process starts; do not restart an active trial just to apply
them. If Windows localhost forwarding fails, use `http://<WSL-IP>:8080`, finding
the current IP with `wsl.exe -d Ubuntu --exec hostname -I`.

| First-round order | Sensors | Objective | BC weight |
| --- | --- | --- | ---: |
| 1 | RGB + height scan | Preference + RL | 0 |
| 2 | Simulator depth + frozen PHP | Preference + RL | 0 |
| 3 | RGB + height scan | Preference + RL + BC | 0.1 |
| 4 | Simulator depth + frozen PHP | Preference + RL + BC | 0.1 |

These are four independent fresh policies, not successive warm starts. Round one
uses seed 7; later rounds use 8 and 9 and rotate order. Human behavior and
policy-dependent reset timing mean these are not strictly paired trajectories.
The same seeded challenge arena configuration is retained within each round.

For three repetitions of all four conditions from the beginning:

```powershell
& 'C:\home\benjamin\thesis\scripts\run_unitree_human_curriculum_windows.ps1' -Repetitions 3
```

This creates 12 runs (about three active hours). To extend the completed first
round without repeating it, pass its printed session directory with
`-SessionPath 'C:\home\benjamin\thesis\local\human_curricula\SESSION' -Repetitions 3`.
The launcher preserves the original decision budget and first seed on resume.
`-Steps` selects the per-run decision budget for a new curriculum. `-PlanOnly`
creates/displays the plan without starting a sender or simulation.

Stop and save before the decision limit preserves the checkpoint and halts the
curriculum. A failure also halts it. Partial/failed trials are never automatically
repeated or labeled complete. A per-session file lock prevents two coordinators;
the existing GPU lock and port ownership checks protect the simulator and sender.
Completed trials have atomic receipts containing the actual run/checkpoint path,
sensor, objective, seed and verified decision count. Session state/receipts are in
`local/human_curricula/`; checkpoints are under
`/home/benjamin/thesis/models/unitree_mjlab_nav_human/`, and immutable raw data under
`/home/benjamin/thesis/local/unitree_human_interventions/`.

## Selected candidate settings

### Stabilized Visual Goal Profile

New curricula use `stabilized_visual_goal_v3`, with `navigation_image_profile=stabilized_goal_v3`.
The existing launcher command is unchanged. Resuming an old `desktop_history_candidate_v1`
curriculum explicitly retains legacy images; start a fresh curriculum for the new profile.
Learning hyperparameters and rewards are unchanged. Old checkpoints restore legacy input
processing; new checkpoints embed the visual profile and camera calibration.

One sensor capture is reused for the policy and UI. Rotation-only image warping compensates
roll/pitch to the calibrated mounting angle while retaining camera yaw. Simulated torso
orientation acts as an ideal attitude measurement; real deployment needs calibrated,
time-aligned IMU/kinematic orientation, and is not verified here. Stabilization cannot
recover unseen pixels or remove translation/parallax. RGB missing pixels are black;
depth missing pixels use the existing far-depth convention. A pooled 4x4 validity mask
is additionally supplied to the policy alongside each encoded frame.

Only older v1/v2 profiles append a 12-pixel goal strip BEFORE encoding, without covering scene pixels.
It contains a full-circle body-relative compass (up=forward, left=left) and a distance
bar clipped at the existing goal-distance scale. Both RGB and PHP encoders receive it,
and the UI displays those same inputs. This auxiliary strip is synthetic goal
annotation, separate from the v2 scene cylinder described above. Numerical goal context is retained.
The frozen encoders are unchanged; usefulness of the new annotation requires evaluation.

New RGB encoder images are 80x60 and PHP normalized depth+annotation images are 87x58.
Including validity features, new policy dimensions are RGB+height 810 and PHP 153.
The legacy dimensions below remain valid for old profiles. Raw sensor frames remain
unaltered in datasets; `policy_image_*` and `vision_valid_mask` save exact preprocessing
outputs alongside them, including corresponding `next_*` fields.

| Setting | New human profile |
| --- | --- |
| Perception frame ages | 0, 0.2, 5 seconds = decision lags 0, 1, 25 at 5 Hz |
| Context | Current 9D body/goal context; no past action inputs |
| RGB + height inputs | 762 = 9 + 3*(187 raw scan values + 64 frozen RGB features) |
| PHP depth inputs | 105 = 9 + 3*32 frozen PHP features |
| Heads | Existing LayerNorm MLP, hidden dimension 256 |
| Actor / critic / alpha LR | 0.0001 / 0.0003 / 0.0003 |
| Returns / TD batch / preference batch | n1 / 256 / 256 independent preference pairs |
| Update schedule | Start after 128 replay rows; one update per decision; actor every two updates |
| Alpha | Learned from student-controlled states; initial 0.01; bounds 1e-5 to 0.1 |
| Target entropy / preference lambda | -3 / fixed 1 |
| BC arms | 0.1, applied to intervention samples; preference and RL retained |

The completed desktop screen did not identify a uniform improvement from changing
entropy or batch size. The later campaign supports keeping preference learning;
it does not establish a universal architecture or horizon winner. The newest
history screen's mixed recent/old layout completed 15/16 reused validation cases
with zero collision cost/falls, versus 15/16 and cost 50 for short three-frame
history. It used 25.9% fewer scripted-teacher decisions, but had lower mean progress.
This is one exploratory training seed, not confirmation on unseen cases.

The desktop timing result used square scan CNNs. These human trials transfer the
timing to rectangular scan + RGB or PHP features using the existing MLP heads;
neither modality was validated by that desktop experiment. No automatic head
replacement or pretrained-policy conversion is implied. The PHP encoder remains
stateless, the stored frames retain their original robot-relative coordinates,
and the low-level locomotion controller still uses height scans. This does not
make the whole robot stereo-only. Raw sensor acquisition is unchanged; compact
feature storage does not demonstrate reduced camera power or acquisition cost.

## Desktop evidence and evaluation

The imported `artifacts/desktop_experiments_20261024/` contains the complete
scientific files from `desktop_screen_20261018`, `desktop_campaign_20261019`, and
`desktop_scan_history_20261023`: reports, all condition specs, attempt records,
raw metrics/evaluations, full galleries, checkpoints and versioned source.
The transfer manifest records SHA256 for every imported file and explicitly
lists exclusions (installed runtime, caches, Git internals and W&B spool files).
No experiments were rerun. The campaign's missing CNN3 seed78 final validation
remains missing (16 trajectories); it is not imputed as zero.

Start with each imported README and `analysis/index.html`. The human evaluation
entrypoint is `eval_unitree_browser_policy.py`; it restores sensor/history from
the saved checkpoint and uses BrowserTrainingSession for identical decision
timing. Inspect saved starts/goals/orientations before claiming paired evaluation.
The curriculum collects the requested human runs; it does not silently add a
teacher-controlled condition or a new automatic evaluation campaign.

Verification: 44 focused CPU tests; native PowerShell plan/sequence tests covering
four conditions, expansion to 12, completion skipping, early stop and failure;
four real simulator/Viser smokes, each 32 decisions / 128 environment ticks and
31 learner updates, synthetic takeovers, expected BC activation, fixed lambda,
student alpha, independent preference sampling and donor-free checkpoint reload.
These checks validate the pipeline, not policy quality or physical gamepad use.
