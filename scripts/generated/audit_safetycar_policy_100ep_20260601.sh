#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
PY="${PY:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
MODEL_PATH="${MODEL_PATH:-${1:-}}"
if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is required, either as env var or first positional arg" >&2
  exit 2
fi
AUDIT_NAME="${AUDIT_NAME:-${2:-$(basename "$(dirname "$MODEL_PATH")")_100ep_$(date +%Y%m%d_%H%M%S)}}"
ENV_NAME="${ENV_NAME:-SafetyCarGoal1-v0}"
EPISODES="${EPISODES:-100}"
PLOT_MAX_EPISODES="${PLOT_MAX_EPISODES:-25}"
OUT_DIR="$ROOT/logs/safetygym_eval_audits/$AUDIT_NAME"
ARGS_PATH="$(dirname "$MODEL_PATH")/args.json"
if [[ -z "${SEED:-}" && -f "$ARGS_PATH" ]]; then
  SEED="$("$PY" - "$ARGS_PATH" <<'PY'
import json
import sys
from pathlib import Path

args_path = Path(sys.argv[1])
try:
    cfg = json.loads(args_path.read_text())
    print(int(cfg.get("seed", 1)) + 10000)
except Exception:
    print(1)
PY
)"
fi
SEED="${SEED:-1}"

cd "$ROOT"
mkdir -p "$OUT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

echo "[audit] model=$MODEL_PATH"
echo "[audit] env=$ENV_NAME seed=$SEED episodes=$EPISODES out=$OUT_DIR"

"$PY" "$ROOT/eval_interactive_safetygym.py" \
  --model_path "$MODEL_PATH" \
  --controller policy \
  --policy_format fastsac \
  --render_mode none \
  --env_name "$ENV_NAME" \
  --seed "$SEED" \
  --layout_curriculum "${LAYOUT_CURRICULUM:-car_random_blocked_filter}" \
  --terminate_on_goal \
  --reward_mode "${REWARD_MODE:-dense}" \
  --dense_reward_scale "${DENSE_REWARD_SCALE:-1.0}" \
  --success_reward_scale "${SUCCESS_REWARD_SCALE:-0.0}" \
  --step_penalty "${STEP_PENALTY:-0.0}" \
  --clearance_penalty_scale "${CLEARANCE_PENALTY_SCALE:-0.0}" \
  --car_action_mode raw_wheels \
  --scale_actor_to_env_bounds \
  --use_layer_norm \
  --actor_hidden_dim "${ACTOR_HIDDEN_DIM:-256}" \
  --load_checkpoint_args \
  --num_episodes "$EPISODES" \
  --fps 0 \
  --save_episode_plots \
  --episode_plot_dir "$OUT_DIR" \
  --episode_plot_max_episodes "$PLOT_MAX_EPISODES" \
  --episode_plot_reward_surface
