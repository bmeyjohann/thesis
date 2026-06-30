#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/benjamin/thesis}"
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-thesis-gate-dagger}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

BASE_DATASET="${BASE_DATASET:-/home/benjamin/.config/thesis/datasets/safetygym/SafetyCarGoal1-v0__scriptedgeo_clearance_or_progress_blocked_seed7_20260523__20260523_211034.npz}"
STUDENT_CKPT="${STUDENT_CKPT:-$ROOT/models/safetygym_minimal/safetycar_scriptedgeo_bc70k_shapedrl_finetune_30k_20260515/step_10000.pt}"
OUT_ROOT="${OUT_ROOT:-$ROOT/logs/safetygym_gate_sample_dagger_20260602}"
DATA_ROOT="${DATA_ROOT:-$OUT_ROOT/datasets}"
MODEL_ROOT="${MODEL_ROOT:-$OUT_ROOT/models}"
EVAL_ROOT="${EVAL_ROOT:-$OUT_ROOT/eval}"
SEED="${SEED:-7}"
BASE_ROWS="${BASE_ROWS:-5000}"
SAMPLE_SIZES="${SAMPLE_SIZES:-500,1000,2000,5000,10000,20000,30000}"
SAMPLE_SEEDS="${SAMPLE_SEEDS:-1,2,3}"
GATE_TRAIN_STEPS="${GATE_TRAIN_STEPS:-2000}"
BC_EPOCHS="${BC_EPOCHS:-25}"
DAGGER_STEPS="${DAGGER_STEPS:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-32}"
PLOT_EPISODES="${PLOT_EPISODES:-6}"
mkdir -p "$OUT_ROOT" "$DATA_ROOT" "$MODEL_ROOT" "$EVAL_ROOT"

BASE_SLICE="$DATA_ROOT/base_${BASE_ROWS}.npz"
if [[ ! -s "$BASE_SLICE" ]]; then
  "$PYTHON_BIN" "$ROOT/tools/slice_safetygym_transition_dataset.py" \
    --input "$BASE_DATASET" \
    --output "$BASE_SLICE" \
    --max_rows "$BASE_ROWS"
fi

echo "[gate-sample] dataset=$BASE_DATASET sizes=$SAMPLE_SIZES seeds=$SAMPLE_SEEDS"
mkdir -p "$OUT_ROOT/gate_sample_efficiency"
"$PYTHON_BIN" "$ROOT/tools/probe_safetygym_gate_sample_efficiency.py" \
  --dataset_path "$BASE_DATASET" \
  --output_dir "$OUT_ROOT/gate_sample_efficiency" \
  --sizes "$SAMPLE_SIZES" \
  --seeds "$SAMPLE_SEEDS" \
  --context_len 8 \
  --train_steps "$GATE_TRAIN_STEPS" \
  --batch_size 512 \
  --device auto \
  > "$OUT_ROOT/gate_sample_efficiency/stdout.log" \
  2> "$OUT_ROOT/gate_sample_efficiency/stderr.log"

train_bc() {
  local name="$1"
  shift
  local out="$MODEL_ROOT/$name"
  mkdir -p "$out"
  echo "[train-bc] $name datasets=$*"
  local dataset_args=()
  for dataset in "$@"; do
    dataset_args+=(--dataset_path "$dataset")
  done
  "$PYTHON_BIN" "$ROOT/train_safetygym_human_imitation.py" \
    "${dataset_args[@]}" \
    --env_name SafetyCarGoal1-v0 \
    --output_dir "$MODEL_ROOT" \
    --exp_name "$name" \
    --seed "$SEED" \
    --device auto \
    --epochs "$BC_EPOCHS" \
    --batch_size 512 \
    --learning_rate 3e-4 \
    --weight_decay 1e-4 \
    --hidden_dim 256 \
    --num_layers 3 \
    --dropout 0.0 \
    --architecture mlp \
    --context_len 8 \
    --action_loss_weight 1.0 \
    --decision_loss_weight 1.0 \
    --non_intervention_action_weight 0.0 \
    --val_fraction 0.1 \
    --intervention_threshold 0.5 \
    --car_wheel_command_limit 1.0 \
    --car_force_scale 1.0 \
    --car_action_mode raw_wheels \
    > "$out/train_stdout.log" \
    2> "$out/train_stderr.log"
}

collect_dagger() {
  local round="$1"
  local ckpt="$2"
  local out_dataset="$3"
  local out="$OUT_ROOT/dagger_round${round}_collect"
  mkdir -p "$out"
  echo "[collect-dagger] round=$round ckpt=$ckpt steps=$DAGGER_STEPS"
  "$PYTHON_BIN" "$ROOT/scripts/collect_safetygym_dagger_gate_labels.py" \
    --base_model_path "$STUDENT_CKPT" \
    --base_policy fastsac \
    --learned_teacher_path "$ckpt" \
    --dataset_path "$out_dataset" \
    --env_name SafetyCarGoal1-v0 \
    --seed "$((SEED + 100 * round))" \
    --device auto \
    --render_mode none \
    --max_steps "$DAGGER_STEPS" \
    --num_episodes 1000 \
    --layout_curriculum car_random_blocked_filter \
    --layout_curriculum_level 0 \
    --obs_mask_mode privileged_geometry \
    --car_action_mode raw_wheels \
    --car_wheel_command_limit 1.0 \
    --car_force_scale 1.0 \
    --reward_mode dense \
    --dense_reward_scale 1.0 \
    --success_reward_scale 0.0 \
    --step_penalty 0.0 \
    --clearance_penalty_scale 0.0 \
    --terminate_on_goal \
    --learned_intervention_threshold 0.5 \
    --teacher_override_mode clearance_or_progress \
    --teacher_override_clearance_threshold 0.08 \
    --teacher_override_clearance_exit_threshold 0.12 \
    --teacher_progress_score_mode potential_field \
    --teacher_progress_clearance_scale 4.0 \
    --teacher_progress_clearance_margin 0.12 \
    --teacher_progress_bad_steps 3 \
    --teacher_progress_good_steps 5 \
    --teacher_progress_epsilon 1e-4 \
    --action_disagreement_margin 0.25 \
    > "$out/stdout.log" \
    2> "$out/stderr.log"
}

eval_bc() {
  local name="$1"
  local ckpt="$2"
  local device="$3"
  local out="$EVAL_ROOT/${name}_${device}"
  mkdir -p "$out"
  echo "[eval] name=$name device=$device"
  "$PYTHON_BIN" "$ROOT/eval_interactive_safetygym.py" \
    --model_path "$STUDENT_CKPT" \
    --policy_format auto \
    --controller policy \
    --intervention_mode human \
    --human_input_device "$device" \
    --imitation_checkpoint_path "$ckpt" \
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
    --obs_mask_mode privileged_geometry \
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
    > "$out/stdout.log" \
    2> "$out/stderr.log"
}

train_bc round0_base5k "$BASE_SLICE"
eval_bc round0_base5k "$MODEL_ROOT/round0_base5k/best.pt" imitation
eval_bc round0_base5k "$MODEL_ROOT/round0_base5k/best.pt" bc_gate_scripted_geo

DAGGER1="$DATA_ROOT/dagger_round1_${DAGGER_STEPS}.npz"
collect_dagger 1 "$MODEL_ROOT/round0_base5k/best.pt" "$DAGGER1"
train_bc round1_base5k_plus_dagger "$BASE_SLICE" "$DAGGER1"
eval_bc round1_base5k_plus_dagger "$MODEL_ROOT/round1_base5k_plus_dagger/best.pt" imitation
eval_bc round1_base5k_plus_dagger "$MODEL_ROOT/round1_base5k_plus_dagger/best.pt" bc_gate_scripted_geo

DAGGER2="$DATA_ROOT/dagger_round2_${DAGGER_STEPS}.npz"
collect_dagger 2 "$MODEL_ROOT/round1_base5k_plus_dagger/best.pt" "$DAGGER2"
train_bc round2_base5k_plus_dagger "$BASE_SLICE" "$DAGGER1" "$DAGGER2"
eval_bc round2_base5k_plus_dagger "$MODEL_ROOT/round2_base5k_plus_dagger/best.pt" imitation
eval_bc round2_base5k_plus_dagger "$MODEL_ROOT/round2_base5k_plus_dagger/best.pt" bc_gate_scripted_geo

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
    "round0_base5k",
    "round1_base5k_plus_dagger",
    "round2_base5k_plus_dagger",
]
devices = ["imitation", "bc_gate_scripted_geo"]
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

train_rows = []
for name in variants:
    logs = json_lines(model_root / name / "train_stdout.log")
    final = next((x for x in reversed(logs) if "val" in x), {})
    val = final.get("val", {}) if final else {}
    train_rows.append({
        "variant": name,
        "rows": final.get("rows", float("nan")),
        "intervened_rows": final.get("intervened_rows", float("nan")),
        "non_intervened_rows": final.get("non_intervened_rows", float("nan")),
        "val_decision_accuracy": val.get("decision_accuracy", float("nan")),
        "val_decision_precision": val.get("decision_precision", float("nan")),
        "val_decision_recall": val.get("decision_recall", float("nan")),
        "val_decision_bce": val.get("decision_bce", float("nan")),
        "val_action_mse_intervened": val.get("action_mse_intervened", float("nan")),
    })

collect_rows = []
for round_id in [1, 2]:
    logs = json_lines(out_root / f"dagger_round{round_id}_collect" / "stdout.log")
    final = logs[-1] if logs else {}
    collect_rows.append({
        "round": round_id,
        "rows": final.get("rows", float("nan")),
        "oracle_intervention_rate": final.get("oracle_intervention_rate", float("nan")),
        "learned_intervention_rate": final.get("learned_intervention_rate", float("nan")),
        "gate_disagreement_rate": final.get("gate_disagreement_rate", float("nan")),
        "action_disagreement_rate_when_oracle_intervenes": final.get("action_disagreement_rate_when_oracle_intervenes", float("nan")),
    })

eval_rows = []
for name in variants:
    for device in devices:
        out = eval_root / f"{name}_{device}"
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
            raise RuntimeError(f"missing eval summary for {name} {device}")
        eval_rows.append({
            "variant": name,
            "controller": device,
            "episodes": int(summary.get("eval/episodes", len(costs))),
            "violations": float(sum(costs)),
            "episodes_with_cost": int(sum(1 for c in costs if c > 0.0)),
            "success_rate": float(summary.get("eval/goal_success_rate", 0.0)),
            "collision_cost_rate": float(summary.get("eval/collision_cost_rate", 0.0)),
            "mean_episode_length": float(summary.get("eval/mean_episode_length", summary.get("eval/episode_length_mean", 0.0))),
            "mean_intervention_fraction": float(summary.get("eval/teacher_fraction_steps", summary.get("eval/intervention_fraction_mean", 0.0))),
            "contact_sheet": str(out / "episode_contact_sheet.png"),
        })

def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

write_csv(out_root / "dagger_train_summary.csv", train_rows)
write_csv(out_root / "dagger_collect_summary.csv", collect_rows)
write_csv(out_root / "dagger_eval_summary.csv", eval_rows)

md = ["# Safety-Gym Gate Sample + DAgger Probe", ""]
md += ["## DAgger Train", "", "| variant | rows | int rows | non-int rows | val acc | val precision | val recall | val action MSE |",
       "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for r in train_rows:
    md.append(
        f"| {r['variant']} | {int(float(r['rows']))} | {int(float(r['intervened_rows']))} | {int(float(r['non_intervened_rows']))} | "
        f"{float(r['val_decision_accuracy']):.3f} | {float(r['val_decision_precision']):.3f} | "
        f"{float(r['val_decision_recall']):.3f} | {float(r['val_action_mse_intervened']):.4f} |"
    )
md += ["", "## DAgger Collection", "", "| round | rows | oracle int rate | learned int rate | gate disagree | action disagree |",
       "| ---: | ---: | ---: | ---: | ---: | ---: |"]
for r in collect_rows:
    md.append(
        f"| {r['round']} | {int(float(r['rows']))} | {float(r['oracle_intervention_rate']):.3f} | "
        f"{float(r['learned_intervention_rate']):.3f} | {float(r['gate_disagreement_rate']):.3f} | "
        f"{float(r['action_disagreement_rate_when_oracle_intervenes']):.3f} |"
    )
md += ["", "## Closed-Loop Eval", "", "| variant | controller | cost eps | success | int frac |",
       "| --- | --- | ---: | ---: | ---: |"]
for r in eval_rows:
    md.append(
        f"| {r['variant']} | {r['controller']} | {r['episodes_with_cost']} | "
        f"{float(r['success_rate']):.3f} | {float(r['mean_intervention_fraction']):.3f} |"
    )
(out_root / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(json.dumps({
    "gate_sample_summary": str(out_root / "gate_sample_efficiency" / "gate_sample_efficiency_summary.json"),
    "dagger_train_summary": str(out_root / "dagger_train_summary.csv"),
    "dagger_collect_summary": str(out_root / "dagger_collect_summary.csv"),
    "dagger_eval_summary": str(out_root / "dagger_eval_summary.csv"),
    "summary_md": str(out_root / "summary.md"),
}, sort_keys=True), flush=True)
PY
