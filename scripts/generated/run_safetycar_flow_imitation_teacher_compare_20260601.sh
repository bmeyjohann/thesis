#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/benjamin/thesis}"
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-thesis-flow-imitation}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

BASE_DATASET="${BASE_DATASET:-/home/benjamin/.config/thesis/datasets/safetygym/SafetyCarGoal1-v0__scriptedgeo_clearance_or_progress_blocked_seed7_20260523__20260523_211034.npz}"
RICH_RAW_DATASET="${RICH_RAW_DATASET:-$ROOT/data/safetygym_datasets/safetycar_goal1_scriptedgeo_randomblocked_rich_raw_70k_20260515.npz}"
STUDENT_CKPT="${STUDENT_CKPT:-$ROOT/models/safetygym_minimal/safetycar_scriptedgeo_bc70k_shapedrl_finetune_30k_20260515/step_10000.pt}"
RICH_STUDENT_CKPT="${RICH_STUDENT_CKPT:-$ROOT/models/safetygym_ppo/safetycar_ppo_bc_scriptedgeo_rich_raw_finetune_probe_30k_20260515/ppo_step_10000_steps.zip}"
OUT_ROOT="${OUT_ROOT:-$ROOT/logs/safetygym_flow_imitation_teacher_20260601}"
MODEL_ROOT="${MODEL_ROOT:-$OUT_ROOT/models}"
EVAL_ROOT="${EVAL_ROOT:-$OUT_ROOT/eval}"
SEED="${SEED:-7}"
TRAIN_STEPS="${TRAIN_STEPS:-15000}"
EVAL_EPISODES="${EVAL_EPISODES:-64}"
PLOT_EPISODES="${PLOT_EPISODES:-9}"
mkdir -p "$OUT_ROOT" "$MODEL_ROOT" "$EVAL_ROOT"

train_variant() {
  local name="$1"
  local dataset="$2"
  local ctx="$3"
  local selector="$4"
  local max_rows="$5"
  local dir="$MODEL_ROOT/$name"
  mkdir -p "$dir"
  echo "[train] $name ctx=$ctx selector=$selector dataset=$dataset"
  "$PYTHON_BIN" "$ROOT/train_safetygym_flow_imitation.py" \
    --dataset_path "$dataset" \
    --env_name SafetyCarGoal1-v0 \
    --output_dir "$MODEL_ROOT" \
    --exp_name "$name" \
    --seed "$SEED" \
    --device auto \
    --train_steps "$TRAIN_STEPS" \
    --batch_size 512 \
    --learning_rate 3e-4 \
    --weight_decay 1e-4 \
    --hidden_dim 256 \
    --num_layers 3 \
    --dropout 0.0 \
    --context_len "$ctx" \
    --flow_loss_weight 1.0 \
    --decision_loss_weight 1.0 \
    --train_noise_scale 1.0 \
    --sample_noise_scale 1.0 \
    --sample_steps 12 \
    --eval_num_samples 8 \
    --eval_sample_selector "$selector" \
    --intervention_threshold 0.5 \
    --val_fraction 0.1 \
    --log_interval 1000 \
    --save_interval 10000 \
    --max_rows "$max_rows" \
    --skip_env_action_space \
    --car_wheel_command_limit 1.0 \
    --car_force_scale 1.0 \
    --car_action_mode raw_wheels \
    > "$dir/train_stdout.log" 2> "$dir/train_stderr.log"
}

eval_variant() {
  local name="$1"
  local obs_mode="$2"
  local student_ckpt="${3:-$STUDENT_CKPT}"
  local policy_format="${4:-auto}"
  local out="$EVAL_ROOT/$name"
  mkdir -p "$out"
  echo "[eval] $name obs=$obs_mode"
  "$PYTHON_BIN" "$ROOT/eval_interactive_safetygym.py" \
    --model_path "$student_ckpt" \
    --policy_format "$policy_format" \
    --controller policy \
    --intervention_mode human \
    --human_input_device flow_imitation \
    --imitation_checkpoint_path "$MODEL_ROOT/$name/best.pt" \
    --intervention_threshold 0.5 \
    --teacher_override_mode clearance \
    --teacher_override_clearance_threshold -1.0 \
    --env_name SafetyCarGoal1-v0 \
    --seed 424242 \
    --render_mode none \
    --num_episodes "$EVAL_EPISODES" \
    --fps 0 \
    --no_show_episode_controls \
    --layout_curriculum car_random_blocked_filter \
    --layout_curriculum_level 0 \
    --obs_mask_mode "$obs_mode" \
    --car_action_mode raw_wheels \
    --car_wheel_command_limit 1.0 \
    --car_force_scale 1.0 \
    --reward_mode dense \
    --dense_reward_scale 1.0 \
    --success_reward_scale 0.0 \
    --step_penalty 0.0 \
    --cost_penalty 0.0 \
    --clearance_penalty_scale 0.0 \
    --terminate_on_goal \
    --save_episode_plots \
    --episode_plot_dir "$out" \
    --episode_plot_max_episodes "$PLOT_EPISODES" \
    > "$out/stdout.log" 2> "$out/stderr.log"
}

train_variant flow_ctx8_first "$BASE_DATASET" 8 first 0
train_variant flow_ctx8_mean "$BASE_DATASET" 8 mean 0
train_variant flow_ctx8_max_turn "$BASE_DATASET" 8 max_turn 0
train_variant flow_ctx16_max_turn "$BASE_DATASET" 16 max_turn 0
train_variant flow_rich_raw_ctx8_max_turn "$RICH_RAW_DATASET" 8 max_turn 50000

eval_variant flow_ctx8_first privileged_geometry
eval_variant flow_ctx8_mean privileged_geometry
eval_variant flow_ctx8_max_turn privileged_geometry
eval_variant flow_ctx16_max_turn privileged_geometry
eval_variant flow_rich_raw_ctx8_max_turn privileged_geometry_rich "$RICH_STUDENT_CKPT" ppo

"$PYTHON_BIN" - "$OUT_ROOT" "$MODEL_ROOT" "$EVAL_ROOT" <<'PY'
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
model_root = Path(sys.argv[2])
eval_root = Path(sys.argv[3])
variants = [
    "flow_ctx8_first",
    "flow_ctx8_mean",
    "flow_ctx8_max_turn",
    "flow_ctx16_max_turn",
    "flow_rich_raw_ctx8_max_turn",
]
episode_re = re.compile(r"episode=(?P<episode>\d+) .*?cost=(?P<cost>[-+0-9.eE]+) .*?len=(?P<length>\d+) .*?interventions=(?P<interventions>\d+)")

def json_lines(path: Path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out

rows = []
offline = []
for name in variants:
    train_logs = json_lines(model_root / name / "train_stdout.log")
    train_last = next((x for x in reversed(train_logs) if "val" in x), {})
    if train_last:
        val = train_last.get("val") or {}
        offline.append({
            "variant": name,
            "val_gate_accuracy": val.get("decision_accuracy", float("nan")),
            "val_gate_recall": val.get("decision_recall", float("nan")),
            "val_action_mse_selected": val.get("action_mse_selected", float("nan")),
            "val_action_mse_best_of_samples": val.get("action_mse_best_of_samples", float("nan")),
            "val_hard_turn_mode_match": val.get("hard_turn_mode_match", float("nan")),
            "val_pred_far_from_teacher_modes": val.get("pred_far_from_teacher_modes", float("nan")),
        })
    out = eval_root / name
    text = (out / "stdout.log").read_text()
    summary = None
    costs = []
    for line in text.splitlines():
        m = episode_re.search(line)
        if m:
            costs.append(float(m.group("cost")))
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                summary = json.loads(line)
            except json.JSONDecodeError:
                pass
    if summary is None:
        raise RuntimeError(f"missing eval summary for {name}")
    rows.append({
        "variant": name,
        "episodes": int(summary.get("eval/episodes", len(costs))),
        "violations": float(sum(costs)),
        "episodes_with_cost": int(sum(1 for c in costs if c > 0.0)),
        "success_rate": float(summary.get("eval/goal_success_rate", 0.0)),
        "collision_cost_rate": float(summary.get("eval/collision_cost_rate", 0.0)),
        "mean_episode_length": float(summary.get("eval/mean_episode_length", summary.get("eval/episode_length_mean", 0.0))),
        "mean_intervention_fraction": float(summary.get("eval/teacher_fraction_steps", summary.get("eval/intervention_fraction_mean", 0.0))),
        "contact_sheet": str(out / "episode_contact_sheet.png"),
    })

with (out_root / "eval_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
with (out_root / "offline_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(offline[0].keys()))
    writer.writeheader()
    writer.writerows(offline)

md = ["| variant | violations | cost eps | success | int frac | val selected MSE | val best-of-samples MSE | hard-turn match |",
      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
offline_by_name = {r["variant"]: r for r in offline}
for row in rows:
    off = offline_by_name.get(row["variant"], {})
    md.append(
        f"| {row['variant']} | {row['violations']:.0f} | {row['episodes_with_cost']} | "
        f"{row['success_rate']:.3f} | {row['mean_intervention_fraction']:.3f} | "
        f"{float(off.get('val_action_mse_selected', float('nan'))):.4f} | "
        f"{float(off.get('val_action_mse_best_of_samples', float('nan'))):.4f} | "
        f"{float(off.get('val_hard_turn_mode_match', float('nan'))):.3f} |"
    )
(out_root / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(json.dumps({"eval_summary": str(out_root / "eval_summary.csv"), "offline_summary": str(out_root / "offline_summary.csv"), "summary_md": str(out_root / "summary.md")}, sort_keys=True), flush=True)
PY

echo "[done] $OUT_ROOT"
