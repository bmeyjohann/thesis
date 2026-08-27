#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${REPO_ROOT:-$PWD}"
if [[ ! -f "$ROOT_DIR/train_unitree_nav_thesis.py" ]]; then
  ROOT_DIR="$SCRIPT_ROOT"
fi
if [[ ! -f "$ROOT_DIR/train_unitree_nav_thesis.py" ]]; then
  echo "Could not resolve thesis repo root; set REPO_ROOT=/home/benjamin/thesis" >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"

for arg in "$@"; do
  if [[ "$arg" == *=* ]]; then
    export "$arg"
  else
    echo "Unsupported positional argument: $arg" >&2
    exit 2
  fi
done

UNITREE_CACHE_ROOT="${UNITREE_CACHE_ROOT:-$HOME/.cache/unitree-nav}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$UNITREE_CACHE_ROOT/matplotlib}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$UNITREE_CACHE_ROOT/warp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$UNITREE_CACHE_ROOT/xdg}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "$MPLCONFIGDIR" "$WARP_CACHE_PATH" "$XDG_CACHE_HOME"

TASK="${TASK:-Unitree-G1-Nav-Obstacles-Safe-Collision}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-16}"
TOTAL_STEPS="${TOTAL_STEPS:-20000}"
RUN_NAME="${RUN_NAME:-unitree_nav_thesis_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/models/unitree_mjlab_nav_thesis}"
LOW_LEVEL_POLICY_PATH="${LOW_LEVEL_POLICY_PATH:-$ROOT_DIR/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/2026-07-12_10-35-19_omni_finetune_model1499_20260712}"

ESCAPE_ALL_DIRECTIONS_FLAG=()
if [[ "${TEACHER_ESCAPE_ALL_DIRECTIONS:-0}" == "1" ]]; then
  ESCAPE_ALL_DIRECTIONS_FLAG=(--teacher-escape-all-directions)
fi

REQUIRE_BLOCKED_CORRIDOR_FLAG=()
if [[ "${REQUIRE_BLOCKED_CORRIDOR:-0}" == "1" ]]; then
  REQUIRE_BLOCKED_CORRIDOR_FLAG=(--require-blocked-corridor)
fi

DEBUG_GOAL_THROUGH_OBSTACLE_FLAG=()
if [[ "${DEBUG_GOAL_THROUGH_OBSTACLE:-0}" == "1" ]]; then
  DEBUG_GOAL_THROUGH_OBSTACLE_FLAG=(--debug-goal-through-obstacle)
fi

RESAMPLE_TERRAIN_TILES_FLAG=()
if [[ "${RESAMPLE_TERRAIN_TILES:-1}" == "1" ]]; then
  RESAMPLE_TERRAIN_TILES_FLAG=(--resample-terrain-tiles)
fi

PREF_STOPGRAD_POSITIVE_FLAG=()
if [[ "${PREF_STOPGRAD_POSITIVE:-1}" == "1" ]]; then
  PREF_STOPGRAD_POSITIVE_FLAG=(--pref-stopgrad-positive)
fi

MASK_GOAL_HEADING_FLAG=()
if [[ "${MASK_GOAL_HEADING:-1}" == "1" ]]; then
  MASK_GOAL_HEADING_FLAG=(--mask-goal-heading)
fi

MASK_HEIGHT_SCAN_FLAG=()
if [[ "${MASK_HEIGHT_SCAN:-0}" == "1" ]]; then
  MASK_HEIGHT_SCAN_FLAG=(--mask-height-scan)
fi

MASK_PROPRIOCEPTION_FLAG=()
if [[ "${MASK_PROPRIOCEPTION:-0}" == "1" ]]; then
  MASK_PROPRIOCEPTION_FLAG=(--mask-proprioception)
fi

STRICT_MIN_SIZE_OBSTACLES_FLAG=()
if [[ "${STRICT_MIN_SIZE_OBSTACLES:-1}" == "1" ]]; then
  STRICT_MIN_SIZE_OBSTACLES_FLAG=(--strict-min-size-obstacles)
fi

DISABLE_OBSTACLES_FLAG=()
if [[ "${DISABLE_OBSTACLES:-0}" == "1" ]]; then
  DISABLE_OBSTACLES_FLAG=(--disable-obstacles)
fi

EVAL_AT_END_FLAG=(--eval-at-end)
if [[ "${EVAL_AT_END:-1}" != "1" ]]; then
  EVAL_AT_END_FLAG=(--no-eval-at-end)
fi

EVAL_FAIL_FAST_FLAG=()
if [[ "${EVAL_FAIL_FAST:-0}" == "1" ]]; then
  EVAL_FAIL_FAST_FLAG=(--eval-fail-fast)
fi

ACTOR_BC_ONLY_FLAG=()
if [[ "${ACTOR_BC_ONLY:-0}" == "1" ]]; then
  ACTOR_BC_ONLY_FLAG=(--actor-bc-only)
fi

DETERMINISTIC_STUDENT_FLAG=()
if [[ "${DETERMINISTIC_STUDENT:-0}" == "1" ]]; then
  DETERMINISTIC_STUDENT_FLAG=(--deterministic-student)
fi

PVP_ENV_REWARD_FLAG=()
if [[ "${PVP_INCLUDE_ENV_REWARD_IN_TD:-0}" == "1" ]]; then
  PVP_ENV_REWARD_FLAG=(--pvp-include-env-reward-in-td)
fi

PVP_TD_START_FLAG=(--pvp-stop-td-on-intervention-start)
if [[ "${PVP_STOP_TD_ON_INTERVENTION_START:-1}" != "1" ]]; then
  PVP_TD_START_FLAG=(--no-pvp-stop-td-on-intervention-start)
fi

# Every standard training run should expose teacher-free policy quality. Set
# EVAL_INTERVAL=0 explicitly only for short infrastructure smoke tests.
EVAL_INTERVAL_RESOLVED="${EVAL_INTERVAL:-${CHECKPOINT_INTERVAL:-5000}}"

CONTINUOUS_BLOCKED_ARGS=(--no-continuous-goal-require-blocked-corridor)
if [[ "${CONTINUOUS_GOAL_REQUIRE_BLOCKED_CORRIDOR:-0}" == "1" ]]; then
  CONTINUOUS_BLOCKED_ARGS=(--continuous-goal-require-blocked-corridor)
fi

exec "$PYTHON_BIN" "$ROOT_DIR/train_unitree_nav_thesis.py" \
  --task "$TASK" \
  --method "${METHOD:-thesis}" \
  --device "$DEVICE" \
  --seed "${SEED:-0}" \
  --num-envs "$NUM_ENVS" \
  --episode-length-s "${EPISODE_LENGTH_S:-16.0}" \
  --navigation-episode-mode "${NAVIGATION_EPISODE_MODE:-episodic}" \
  --continuous-environment-horizon-s "${CONTINUOUS_ENVIRONMENT_HORIZON_S:-3600}" \
  --continuous-goal-distance-min "${CONTINUOUS_GOAL_DISTANCE_MIN:-0.0}" \
  --continuous-goal-distance-max "${CONTINUOUS_GOAL_DISTANCE_MAX:-0.0}" \
  --continuous-goal-region-mode "${CONTINUOUS_GOAL_REGION_MODE:-assigned_tile}" \
  --continuous-goal-blocked-probability "${CONTINUOUS_GOAL_BLOCKED_PROBABILITY:-1.0}" \
  "${CONTINUOUS_BLOCKED_ARGS[@]}" \
  --continuous-goal-resample-attempts "${CONTINUOUS_GOAL_RESAMPLE_ATTEMPTS:-256}" \
  --continuous-goal-boundary-margin "${CONTINUOUS_GOAL_BOUNDARY_MARGIN:-0.5}" \
  --low-level-policy-path "$LOW_LEVEL_POLICY_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --run-name "$RUN_NAME" \
  --wandb-project "${WANDB_PROJECT:-thesis-unitree-nav}" \
  --wandb-group "${WANDB_GROUP:-}" \
  --wandb-mode "${WANDB_MODE:-online}" \
  --total-steps "$TOTAL_STEPS" \
  --replay-capacity "${REPLAY_CAPACITY:-500000}" \
  --learning-starts "${LEARNING_STARTS:-1000}" \
  --random-steps "${RANDOM_STEPS:-500}" \
  --teacher-warmup-steps "${TEACHER_WARMUP_STEPS:-1000}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --updates-per-step "${UPDATES_PER_STEP:-1}" \
  --n-step "${N_STEP:-1}" \
  --policy-frequency "${POLICY_FREQUENCY:-2}" \
  --hidden-dim "${HIDDEN_DIM:-256}" \
  --policy-encoder "${POLICY_ENCODER:-mlp}" \
  --height-scan-resolution "${HEIGHT_SCAN_RESOLUTION:-0.5}" \
  --height-scan-pattern "${HEIGHT_SCAN_PATTERN:-grid}" \
  --height-scan-frustum-near "${HEIGHT_SCAN_FRUSTUM_NEAR:-0.25}" \
  --height-scan-frustum-far "${HEIGHT_SCAN_FRUSTUM_FAR:-4.0}" \
  --height-scan-frustum-fov-deg "${HEIGHT_SCAN_FRUSTUM_FOV_DEG:-70.0}" \
  --height-scan-frustum-side "${HEIGHT_SCAN_FRUSTUM_SIDE:-17}" \
  --height-scan-forward-size "${HEIGHT_SCAN_FORWARD_SIZE:-0.0}" \
  --height-scan-lateral-size "${HEIGHT_SCAN_LATERAL_SIZE:-0.0}" \
  --scan-history "${SCAN_HISTORY:-1}" \
  --scan-history-stride "${SCAN_HISTORY_STRIDE:-1}" \
  --action-history "${ACTION_HISTORY:-0}" \
  --goal-encoding "${GOAL_ENCODING:-cartesian}" \
  --goal-distance-scale "${GOAL_DISTANCE_SCALE:-14.0}" \
  --velocity-scale "${VELOCITY_SCALE:-1.0}" \
  "${MASK_HEIGHT_SCAN_FLAG[@]}" \
  "${MASK_PROPRIOCEPTION_FLAG[@]}" \
  "${MASK_GOAL_HEADING_FLAG[@]}" \
  --student-action-smoothing "${STUDENT_ACTION_SMOOTHING:-0.0}" \
  --pad-obs-to-dim "${PAD_OBS_TO_DIM:-0}" \
  --use-layer-norm \
  --lr-actor "${LR_ACTOR:-0.0003}" \
  --lr-critic "${LR_CRITIC:-0.0003}" \
  --gamma "${GAMMA:-0.99}" \
  --tau "${TAU:-0.005}" \
  --max-grad-norm "${MAX_GRAD_NORM:-10.0}" \
  --alpha-init "${ALPHA_INIT:-0.001}" \
  --alpha-min "${ALPHA_MIN:-0.0}" \
  --alpha-max "${ALPHA_MAX:-0.05}" \
  --actor-bc-weight "${ACTOR_BC_WEIGHT:-0.2}" \
  "${ACTOR_BC_ONLY_FLAG[@]}" \
  --hilserl-demo-ratio "${HILSERL_DEMO_RATIO:-0.5}" \
  --eil-threshold "${EIL_THRESHOLD:-0.0}" \
  --eil-good-margin "${EIL_GOOD_MARGIN:-0.01}" \
  --eil-bad-margin "${EIL_BAD_MARGIN:-0.01}" \
  --eil-pair-margin "${EIL_PAIR_MARGIN:-0.05}" \
  --eil-bad-pre-steps "${EIL_BAD_PRE_STEPS:-8}" \
  --pvp-proxy-value-bound "${PVP_PROXY_VALUE_BOUND:-1.0}" \
  --pvp-cql-coefficient "${PVP_CQL_COEFFICIENT:-1.0}" \
  --pvp-policy-delay "${PVP_POLICY_DELAY:-2}" \
  --pvp-target-policy-noise "${PVP_TARGET_POLICY_NOISE:-0.2}" \
  --pvp-target-noise-clip "${PVP_TARGET_NOISE_CLIP:-0.5}" \
  --hg-ensemble-size "${HG_ENSEMBLE_SIZE:-5}" \
  --init-actor-checkpoint "${INIT_ACTOR_CHECKPOINT:-}" \
  --init-checkpoint "${INIT_CHECKPOINT:-}" \
  --student-controller "${STUDENT_CONTROLLER:-actor}" \
  "${DETERMINISTIC_STUDENT_FLAG[@]}" \
  --learner-reward-mode "${LEARNER_REWARD_MODE:-dense_progress}" \
  --dense-progress-scale "${DENSE_PROGRESS_SCALE:-1.0}" \
  --dense-progress-exp-scale "${DENSE_PROGRESS_EXP_SCALE:-1.0}" \
  --dense-progress-exp-temperature "${DENSE_PROGRESS_EXP_TEMPERATURE:-1.0}" \
  --goal-turn-alignment-scale "${GOAL_TURN_ALIGNMENT_SCALE:-0.0}" \
  --reverse-action-penalty "${REVERSE_ACTION_PENALTY:-0.0}" \
  --lateral-action-penalty "${LATERAL_ACTION_PENALTY:-0.0}" \
  --success-bonus "${SUCCESS_BONUS:-1.0}" \
  --failure-penalty "${FAILURE_PENALTY:-0.0}" \
  --teacher-type "${TEACHER_TYPE:-geom_scan}" \
  --intervention-gate-mode "${INTERVENTION_GATE_MODE:-clearance_or_stall}" \
  --intervention-clearance-threshold "${INTERVENTION_CLEARANCE_THRESHOLD:-0.65}" \
  --intervention-release-clearance "${INTERVENTION_RELEASE_CLEARANCE:-0.8}" \
  --intervention-clearance-mode "${INTERVENTION_CLEARANCE_MODE:-teacher_ratio}" \
  --intervention-clearance-trigger-ratio "${INTERVENTION_CLEARANCE_TRIGGER_RATIO:-1.25}" \
  --intervention-clearance-release-ratio "${INTERVENTION_CLEARANCE_RELEASE_RATIO:-1.4166666667}" \
  --intervention-stall-steps "${INTERVENTION_STALL_STEPS:-30}" \
  --intervention-progress-epsilon "${INTERVENTION_PROGRESS_EPSILON:-0.04}" \
  --intervention-release-steps "${INTERVENTION_RELEASE_STEPS:-8}" \
  --intervention-release-progress-tolerance "${INTERVENTION_RELEASE_PROGRESS_TOLERANCE:-0.005}" \
  --intervention-release-action-delta-max "${INTERVENTION_RELEASE_ACTION_DELTA_MAX:-0.35}" \
  --pref-rank-weight "${PREF_RANK_WEIGHT:-1.0}" \
  --pref-rank-margin "${PREF_RANK_MARGIN:-0.05}" \
  --pref-loss-type "${PREF_LOSS_TYPE:-lagrangian}" \
  --pref-lambda-lr "${PREF_LAMBDA_LR:-0.01}" \
  --pref-lambda-max "${PREF_LAMBDA_MAX:-10.0}" \
  --pref-action-delta-min "${PREF_ACTION_DELTA_MIN:-0.05}" \
  --teacher-scan-block-threshold "${TEACHER_SCAN_BLOCK_THRESHOLD:-0.12}" \
  --teacher-scan-block-delta "${TEACHER_SCAN_BLOCK_DELTA:-0.0}" \
  --teacher-scan-planner "${TEACHER_SCAN_PLANNER:-heuristic}" \
  --teacher-scan-astar-clearance "${TEACHER_SCAN_ASTAR_CLEARANCE:-0.6}" \
  --teacher-scan-astar-cell-padding "${TEACHER_SCAN_ASTAR_CELL_PADDING:-0.35}" \
  --teacher-scan-astar-resolution "${TEACHER_SCAN_ASTAR_RESOLUTION:-0.25}" \
  --teacher-scan-astar-waypoint-index "${TEACHER_SCAN_ASTAR_WAYPOINT_INDEX:-3}" \
  --teacher-scan-astar-side-penalty "${TEACHER_SCAN_ASTAR_SIDE_PENALTY:-8.0}" \
  --teacher-scan-astar-commit-steps "${TEACHER_SCAN_ASTAR_COMMIT_STEPS:-30}" \
  --teacher-sector-half-width "${TEACHER_SECTOR_HALF_WIDTH:-0.35}" \
  --teacher-align-angle "${TEACHER_ALIGN_ANGLE:-0.55}" \
  --teacher-max-vx "${TEACHER_MAX_VX:-0.95}" \
  --teacher-max-vy "${TEACHER_MAX_VY:-0.45}" \
  --teacher-yaw-gain "${TEACHER_YAW_GAIN:-1.2}" \
  --teacher-clearance-weight "${TEACHER_CLEARANCE_WEIGHT:-0.0}" \
  --teacher-clearance-power "${TEACHER_CLEARANCE_POWER:-2.0}" \
  --teacher-speed-clearance-scale "${TEACHER_SPEED_CLEARANCE_SCALE:-0.0}" \
  --teacher-num-sectors "${TEACHER_NUM_SECTORS:-13}" \
  --teacher-min-forward-scale "${TEACHER_MIN_FORWARD_SCALE:-0.12}" \
  --teacher-escape-risk-threshold "${TEACHER_ESCAPE_RISK_THRESHOLD:-0.0}" \
  --teacher-escape-forward-scale "${TEACHER_ESCAPE_FORWARD_SCALE:-0.0}" \
  --teacher-escape-lateral-scale "${TEACHER_ESCAPE_LATERAL_SCALE:-1.0}" \
  --teacher-escape-radius "${TEACHER_ESCAPE_RADIUS:-1.0}" \
  --teacher-bypass-angle "${TEACHER_BYPASS_ANGLE:-0.0}" \
  --teacher-goal-stop-dist "${TEACHER_GOAL_STOP_DIST:-0.40}" \
  --teacher-wall-follow-steps "${TEACHER_WALL_FOLLOW_STEPS:-0}" \
  --teacher-wall-follow-angle "${TEACHER_WALL_FOLLOW_ANGLE:-0.9}" \
  --teacher-wall-follow-clear-risk "${TEACHER_WALL_FOLLOW_CLEAR_RISK:-0.15}" \
  --teacher-rollout-horizon "${TEACHER_ROLLOUT_HORIZON:-0.0}" \
  --teacher-rollout-clearance "${TEACHER_ROLLOUT_CLEARANCE:-0.65}" \
  --teacher-rollout-samples "${TEACHER_ROLLOUT_SAMPLES:-8}" \
  --teacher-rollout-clearance-weight "${TEACHER_ROLLOUT_CLEARANCE_WEIGHT:-20.0}" \
  --teacher-rollout-forward-bias "${TEACHER_ROLLOUT_FORWARD_BIAS:-0.05}" \
  --teacher-emergency-radius "${TEACHER_EMERGENCY_RADIUS:-0.0}" \
  --teacher-emergency-hard-radius "${TEACHER_EMERGENCY_HARD_RADIUS:-0.0}" \
  --teacher-emergency-speed-scale "${TEACHER_EMERGENCY_SPEED_SCALE:-0.75}" \
  --teacher-emergency-repulsion-weight "${TEACHER_EMERGENCY_REPULSION_WEIGHT:-0.8}" \
  --teacher-emergency-tangent-weight "${TEACHER_EMERGENCY_TANGENT_WEIGHT:-1.2}" \
  --teacher-emergency-goal-weight "${TEACHER_EMERGENCY_GOAL_WEIGHT:-0.4}" \
  --teacher-geom-planner "${TEACHER_GEOM_PLANNER:-astar}" \
  --teacher-geom-scan-margin "${TEACHER_GEOM_SCAN_MARGIN:-0.15}" \
  --teacher-geom-back-margin "${TEACHER_GEOM_BACK_MARGIN:-0.35}" \
  --teacher-geom-max-forward "${TEACHER_GEOM_MAX_FORWARD:-1.75}" \
  --teacher-geom-max-lateral "${TEACHER_GEOM_MAX_LATERAL:-1.75}" \
  --teacher-geom-lookahead "${TEACHER_GEOM_LOOKAHEAD:-1.6}" \
  --teacher-geom-clearance "${TEACHER_GEOM_CLEARANCE:-0.5}" \
  --teacher-geom-grid-resolution "${TEACHER_GEOM_GRID_RESOLUTION:-0.15}" \
  --teacher-geom-waypoint-index "${TEACHER_GEOM_WAYPOINT_INDEX:-3}" \
  --teacher-geom-side-penalty "${TEACHER_GEOM_SIDE_PENALTY:-8.0}" \
  --teacher-geom-side-frame "${TEACHER_GEOM_SIDE_FRAME:-body}" \
  --teacher-geom-disengage-clear-steps "${TEACHER_GEOM_DISENGAGE_CLEAR_STEPS:-20}" \
  --teacher-geom-command-smoothing "${TEACHER_GEOM_COMMAND_SMOOTHING:-0.0}" \
  --teacher-geom-waypoint-commit-distance "${TEACHER_GEOM_WAYPOINT_COMMIT_DISTANCE:-0.0}" \
  --teacher-geom-waypoint-reach-dist "${TEACHER_GEOM_WAYPOINT_REACH_DIST:-0.25}" \
  --teacher-geom-emergency-radius "${TEACHER_GEOM_EMERGENCY_RADIUS:-0.8}" \
  --intervention-delta "${INTERVENTION_DELTA:-0.35}" \
  --intervene-on-blocked-goal \
  --success-dist "${SUCCESS_DIST:-0.5}" \
  --goal-distance-min "${GOAL_DISTANCE_MIN:-2.8}" \
  --goal-distance-max "${GOAL_DISTANCE_MAX:-4.0}" \
  --resample-terrain-tiles \
  --min-goal-obstacle-clearance "${MIN_GOAL_OBSTACLE_CLEARANCE:-0.9}" \
  --goal-clearance-resample-attempts "${GOAL_CLEARANCE_RESAMPLE_ATTEMPTS:-50}" \
  --min-start-obstacle-clearance "${MIN_START_OBSTACLE_CLEARANCE:-0.0}" \
  --start-clearance-resample-attempts "${START_CLEARANCE_RESAMPLE_ATTEMPTS:-20}" \
  --blocked-corridor-radius "${BLOCKED_CORRIDOR_RADIUS:-0.45}" \
  --blocked-corridor-ignore-end-radius "${BLOCKED_CORRIDOR_IGNORE_END_RADIUS:-0.75}" \
  --blocked-corridor-min-cells "${BLOCKED_CORRIDOR_MIN_CELLS:-1}" \
  --blocked-corridor-resample-attempts "${BLOCKED_CORRIDOR_RESAMPLE_ATTEMPTS:-100}" \
  --blocked-goal-max-distance "${BLOCKED_GOAL_MAX_DISTANCE:-0.0}" \
  --blocked-goal-distance-sampling "${BLOCKED_GOAL_DISTANCE_SAMPLING:-nearest}" \
  --blocked-goal-placement-mode "${BLOCKED_GOAL_PLACEMENT_MODE:-obstacle_multiplier}" \
  --blocked-goal-distance-multiplier-min "${BLOCKED_GOAL_DISTANCE_MULTIPLIER_MIN:-1.0}" \
  --blocked-goal-distance-multiplier-max "${BLOCKED_GOAL_DISTANCE_MULTIPLIER_MAX:-2.0}" \
  --blocked-goal-candidate-attempts "${BLOCKED_GOAL_CANDIDATE_ATTEMPTS:-64}" \
  --debug-obstacle-width-min "${DEBUG_OBSTACLE_WIDTH_MIN:-1.0}" \
  --debug-obstacle-width-max "${DEBUG_OBSTACLE_WIDTH_MAX:-1.4}" \
  "${STRICT_MIN_SIZE_OBSTACLES_FLAG[@]}" \
  --debug-obstacle-height-min "${DEBUG_OBSTACLE_HEIGHT_MIN:-1.0}" \
  --debug-obstacle-height-max "${DEBUG_OBSTACLE_HEIGHT_MAX:-1.0}" \
  --debug-num-obstacles "${DEBUG_NUM_OBSTACLES:-6}" \
  --debug-platform-width "${DEBUG_PLATFORM_WIDTH:-2.0}" \
  --debug-obstacle-border-width "${DEBUG_OBSTACLE_BORDER_WIDTH:-0.0}" \
  --debug-terrain-rows "${DEBUG_TERRAIN_ROWS:-5}" \
  --debug-terrain-cols "${DEBUG_TERRAIN_COLS:-10}" \
  --debug-goal-distance "${DEBUG_GOAL_DISTANCE:-3.2}" \
  --debug-goal-obstacle-min-dist "${DEBUG_GOAL_OBSTACLE_MIN_DIST:-0.8}" \
  --debug-goal-obstacle-max-dist "${DEBUG_GOAL_OBSTACLE_MAX_DIST:-2.2}" \
  --goal-through-obstacle-prob "${GOAL_THROUGH_OBSTACLE_PROB:-0.7}" \
  --log-interval "${LOG_INTERVAL:-200}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-5000}" \
  --eval-interval "$EVAL_INTERVAL_RESOLVED" \
  --eval-num-envs "${EVAL_NUM_ENVS:-8}" \
  --eval-num-episodes "${EVAL_NUM_EPISODES:-16}" \
  --eval-seed "${EVAL_SEED:-941}" \
  --eval-layout-manifest "${EVAL_LAYOUT_MANIFEST:-}" \
  --eval-timeout-s "${EVAL_TIMEOUT_S:-3600}" \
  "${EVAL_AT_END_FLAG[@]}" \
  "${EVAL_FAIL_FAST_FLAG[@]}" \
  "${ESCAPE_ALL_DIRECTIONS_FLAG[@]}" \
  "${REQUIRE_BLOCKED_CORRIDOR_FLAG[@]}" \
  "${DEBUG_GOAL_THROUGH_OBSTACLE_FLAG[@]}" \
  "${RESAMPLE_TERRAIN_TILES_FLAG[@]}" \
  "${PREF_STOPGRAD_POSITIVE_FLAG[@]}" \
  "${PVP_ENV_REWARD_FLAG[@]}" \
  "${PVP_TD_START_FLAG[@]}" \
  "${DISABLE_OBSTACLES_FLAG[@]}" \
  ${EXTRA_ARGS:-}
