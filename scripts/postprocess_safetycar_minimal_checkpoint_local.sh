#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
cd "$ROOT"

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

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is required" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig-codex}"
mkdir -p "$MPLCONFIGDIR"
export MPLCONFIGDIR

MODEL_PATH="$(python3 - <<'PY' "$MODEL_PATH"
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"

MODEL_DIR="$(dirname "$MODEL_PATH")"
RUN_DIR="${RUN_DIR:-$(python3 - <<'PY' "$MODEL_DIR"
from pathlib import Path
import sys
model_dir = Path(sys.argv[1]).resolve()
run_name = model_dir.name
print((Path('/home/benjamin/thesis/logs/safetygym_minimal') / run_name).resolve())
PY
)}"

TAG="${TAG:-$(basename "${MODEL_PATH%.pt}")}"
POLICY_DIR="${POLICY_DIR:-${RUN_DIR}/manual_policy_maps_${TAG}}"
EPISODE_PLOT_DIR="${EPISODE_PLOT_DIR:-${RUN_DIR}/manual_eval_plots_${TAG}}"

"$PYTHON_BIN" /home/benjamin/thesis/tools/visualize_safetygym_policy_map.py \
  --model_path "$MODEL_PATH" \
  --output_dir "$POLICY_DIR" \
  --tag "$TAG" \
  --device "${VIZ_DEVICE:-cpu}" \
  --grid_resolution "${VIZ_GRID_RESOLUTION:-48}" \
  --quiver_stride "${VIZ_QUIVER_STRIDE:-4}" \
  --seed "${VIZ_SEED:-0}" \
  --num_rollouts "${VIZ_NUM_ROLLOUTS:-4}" \
  --headings_deg "${VIZ_HEADINGS_DEG:-0,90,180,270}"

"$PYTHON_BIN" /home/benjamin/thesis/eval_interactive_safetygym.py \
  --model_path "$MODEL_PATH" \
  --controller policy \
  --render_mode none \
  --num_episodes "${NUM_EPISODES:-9}" \
  --fps 0 \
  --no_show_episode_controls \
  --save_episode_plots \
  --episode_plot_dir "$EPISODE_PLOT_DIR"
