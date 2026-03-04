#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV=${CONDA_ENV:-fasttd3}
MODEL_DIR=${MODEL_DIR:-}
RUN_NAME=${RUN_NAME:-}
CHECKPOINT_MODE=${CHECKPOINT_MODE:-all}   # all|final|latest
NUM_EPISODES=${NUM_EPISODES:-1}
FPS=${FPS:-30}
SEED=${SEED:-1}
PROMPT_BETWEEN=${PROMPT_BETWEEN:-1}

EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/review_safetygym_checkpoints.sh [options] [-- extra eval args]

Options:
  --model_dir <path>         Directory containing checkpoints (step_*.pt/final.pt)
  --run_name <name>          Convenience shortcut for models/safetygym/<name>
  --checkpoint_mode <mode>   all|final|latest (default: all)
  --num_episodes <n>         Episodes per checkpoint (default: 1)
  --fps <n>                  Eval fps (default: 30)
  --seed <n>                 Base seed (default: 1)
  --prompt_between <0|1>     Wait for Enter between checkpoints (default: 1)
  -h, --help                 Show this help

Examples:
  scripts/review_safetygym_checkpoints.sh \
    --run_name safetygym_dense_reward_test \
    --checkpoint_mode all \
    --num_episodes 1 \
    --fps 30

  scripts/review_safetygym_checkpoints.sh \
    --model_dir models/safetygym/my_run \
    --checkpoint_mode latest \
    --prompt_between 0 \
    -- --render_mode human --env_name SafetyCarGoal2-v0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --run_name)
      RUN_NAME="$2"
      shift 2
      ;;
    --checkpoint_mode)
      CHECKPOINT_MODE="$2"
      shift 2
      ;;
    --num_episodes)
      NUM_EPISODES="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --prompt_between)
      PROMPT_BETWEEN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${MODEL_DIR}" && -n "${RUN_NAME}" ]]; then
  MODEL_DIR="models/safetygym/${RUN_NAME}"
fi
if [[ -z "${MODEL_DIR}" ]]; then
  echo "Please set --model_dir or --run_name." >&2
  exit 2
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Model directory not found: ${MODEL_DIR}" >&2
  exit 2
fi

final_ckpt="${MODEL_DIR}/final.pt"
mapfile -t step_ckpts < <(find "${MODEL_DIR}" -maxdepth 1 -type f -name 'step_*.pt' | sort -V)
ckpts=()

case "${CHECKPOINT_MODE}" in
  final)
    if [[ -f "${final_ckpt}" ]]; then
      ckpts+=("${final_ckpt}")
    fi
    ;;
  latest)
    if [[ ${#step_ckpts[@]} -gt 0 ]]; then
      ckpts+=("${step_ckpts[-1]}")
    elif [[ -f "${final_ckpt}" ]]; then
      ckpts+=("${final_ckpt}")
    fi
    ;;
  all)
    if [[ ${#step_ckpts[@]} -gt 0 ]]; then
      ckpts+=("${step_ckpts[@]}")
    fi
    if [[ -f "${final_ckpt}" ]]; then
      ckpts+=("${final_ckpt}")
    fi
    ;;
  *)
    echo "Invalid --checkpoint_mode: ${CHECKPOINT_MODE} (expected all|final|latest)" >&2
    exit 2
    ;;
esac

if [[ ${#ckpts[@]} -eq 0 ]]; then
  echo "No checkpoints found in ${MODEL_DIR}" >&2
  exit 1
fi

echo "Reviewing ${#ckpts[@]} checkpoint(s) from ${MODEL_DIR}"
echo "Checkpoint mode: ${CHECKPOINT_MODE}"
echo

for i in "${!ckpts[@]}"; do
  ckpt="${ckpts[$i]}"
  idx=$((i + 1))
  run_seed=$((SEED + i))
  echo "=== [${idx}/${#ckpts[@]}] ${ckpt} ==="
  conda run -n "${CONDA_ENV}" python eval_interactive_safetygym.py \
    --model_path "${ckpt}" \
    --controller policy \
    --num_episodes "${NUM_EPISODES}" \
    --fps "${FPS}" \
    --seed "${run_seed}" \
    "${EXTRA_ARGS[@]}"
  echo
  if [[ "${PROMPT_BETWEEN}" == "1" && "${idx}" -lt "${#ckpts[@]}" ]]; then
    read -r -p "Press Enter for next checkpoint..."
  fi
done
