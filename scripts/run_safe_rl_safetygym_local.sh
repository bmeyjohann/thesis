#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
SAFE_RL_DIR="$ROOT/safe_rl"
PYTHON="${PYTHON:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"

cd "$SAFE_RL_DIR"

for arg in "$@"; do
  case "$arg" in
    *=*)
      export "$arg"
      ;;
    *)
      echo "Unsupported argument: $arg" >&2
      exit 2
      ;;
  esac
done

ALG="${ALG:-P3O}"
ENV_ID="${ENV_ID:-SafetyCarGoal1-v0}"
NUM_ENVS="${NUM_ENVS:-8}"
MAX_ITER="${MAX_ITER:-200}"
SEED="${SEED:-1}"
COST_LIMIT="${COST_LIMIT:-0.0}"
DEVICE="${DEVICE:-cpu}"
WANDB_PROJECT="${WANDB_PROJECT:-thesis-safetygym}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/safe_rl_thesis}"

case "$ALG" in
  P3O)
    CONFIG="${CONFIG:-config/safety_gymnasium_p3o_thesis.yaml}"
    ;;
  PPOL_PID)
    CONFIG="${CONFIG:-config/safety_gymnasium_ppol_pid_thesis.yaml}"
    ;;
  CPO)
    CONFIG="${CONFIG:-config/safety_gymnasium_cpo_thesis.yaml}"
    ;;
  PCPO)
    CONFIG="${CONFIG:-config/safety_gymnasium_pcpo_thesis.yaml}"
    ;;
  *)
    echo "Unsupported ALG=$ALG. Use P3O, PPOL_PID, CPO, or PCPO." >&2
    exit 2
    ;;
esac

echo "Configured Safe-RL Safety-Gym training"
echo "repo:       $SAFE_RL_DIR"
echo "env:        $ENV_ID"
echo "alg:        $ALG"
echo "config:     $CONFIG"
echo "num envs:   $NUM_ENVS"
echo "max iter:   $MAX_ITER"
echo "cost limit: $COST_LIMIT"
echo "seed:       $SEED"
echo "device:     $DEVICE"
echo "wandb:      $WANDB_PROJECT"
echo "run tag:    $RUN_TAG"
echo "init ckpt:  ${INIT_CHECKPOINT:-<none>}"
echo "term goal:  ${TERMINATE_ON_GOAL:-0}"
echo "term cost:  ${TERMINATE_ON_COST:-0}"

export PYTHONPATH="$ROOT:$SAFE_RL_DIR:$ROOT/safety-gymnasium:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_DIR="${WANDB_DIR:-$ROOT/logs/wandb}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/logs/cache}"
export GIT_PYTHON_REFRESH="${GIT_PYTHON_REFRESH:-quiet}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
mkdir -p "$WANDB_DIR" "$XDG_CACHE_HOME"

INIT_FLAG=()
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  INIT_FLAG=(--init_checkpoint "$INIT_CHECKPOINT")
fi
if [[ -n "${INIT_ACTOR_STD_OVERRIDE:-}" ]]; then
  INIT_FLAG+=(--init_actor_std_override "$INIT_ACTOR_STD_OVERRIDE")
fi
TERMINATION_FLAGS=()
if [[ "${TERMINATE_ON_GOAL:-0}" == "1" ]]; then
  TERMINATION_FLAGS+=(--terminate_on_goal)
fi
if [[ "${TERMINATE_ON_COST:-0}" == "1" ]]; then
  TERMINATION_FLAGS+=(--terminate_on_cost)
fi
P3O_FLAGS=()
if [[ -n "${P3O_KAPPA:-}" ]]; then
  P3O_FLAGS+=(--p3o_kappa "$P3O_KAPPA")
fi
if [[ -n "${P3O_KAPPA_MAX:-}" ]]; then
  P3O_FLAGS+=(--p3o_kappa_max "$P3O_KAPPA_MAX")
fi
if [[ -n "${P3O_RHO:-}" ]]; then
  P3O_FLAGS+=(--p3o_rho "$P3O_RHO")
fi

"$PYTHON" -u scripts/train/train_safety_gymnasium.py \
  --env_id "$ENV_ID" \
  --num_envs "$NUM_ENVS" \
  --config "$CONFIG" \
  --cost_limits "$COST_LIMIT" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --log_dir "$LOG_ROOT/$RUN_TAG" \
  --wandb_project "$WANDB_PROJECT" \
  "${INIT_FLAG[@]}" \
  "${TERMINATION_FLAGS[@]}" \
  "${P3O_FLAGS[@]}" \
  --max_iterations "$MAX_ITER" \
  --num_steps_per_env "${NUM_STEPS_PER_ENV:-1024}" \
  --learning_rate "${LEARNING_RATE:-0.0003}" \
  --num_learning_epochs "${NUM_LEARNING_EPOCHS:-10}" \
  --num_mini_batches "${NUM_MINI_BATCHES:-4}" \
  --gamma "${GAMMA:-0.99}" \
  --lam "${LAM:-0.95}" \
  --entropy_coef "${ENTROPY_COEF:-0.0}"
