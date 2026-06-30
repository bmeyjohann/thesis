#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/benjamin/thesis}"
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-thesis}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

DATASET_PATH="${DATASET_PATH:-/home/benjamin/.config/thesis/datasets/safetygym/SafetyCarGoal1-v0__scriptedgeo_clearance_or_progress_blocked_seed7_20260523__20260523_211034.npz}"
STUDENT_CKPT="${STUDENT_CKPT:-$ROOT/models/safetygym_minimal/safetycar_scriptedgeo_bc70k_shapedrl_finetune_30k_20260515/step_10000.pt}"
OUT_ROOT="${OUT_ROOT:-$ROOT/logs/safetygym_temporal_bc_teacher_20260601}"
MODEL_ROOT="${MODEL_ROOT:-$OUT_ROOT/models}"
EVAL_ROOT="${EVAL_ROOT:-$OUT_ROOT/eval}"
SEED="${SEED:-7}"
EPOCHS="${EPOCHS:-35}"
EVAL_EPISODES="${EVAL_EPISODES:-64}"
PLOT_EPISODES="${PLOT_EPISODES:-9}"
mkdir -p "$OUT_ROOT" "$MODEL_ROOT" "$EVAL_ROOT"

train_variant() {
  local name="$1"
  local arch="$2"
  local ctx="$3"
  local layers="$4"
  local heads="$5"
  local dir="$MODEL_ROOT/$name"
  mkdir -p "$dir"
  echo "[train] $name arch=$arch ctx=$ctx"
  "$PYTHON_BIN" "$ROOT/train_safetygym_human_imitation.py" \
    --dataset_path "$DATASET_PATH" \
    --env_name SafetyCarGoal1-v0 \
    --output_dir "$MODEL_ROOT" \
    --exp_name "$name" \
    --seed "$SEED" \
    --device auto \
    --epochs "$EPOCHS" \
    --batch_size 256 \
    --learning_rate 3e-4 \
    --weight_decay 1e-4 \
    --hidden_dim 256 \
    --num_layers "$layers" \
    --dropout 0.0 \
    --architecture "$arch" \
    --num_heads "$heads" \
    --context_len "$ctx" \
    --action_loss_weight 1.0 \
    --decision_loss_weight 1.0 \
    --non_intervention_action_weight 0.0 \
    --val_fraction 0.1 \
    --intervention_threshold 0.5 \
    --surface_mode default \
    --car_wheel_command_limit 1.0 \
    --car_force_scale 1.0 \
    --car_action_mode raw_wheels \
    > "$dir/train_stdout.log" 2> "$dir/train_stderr.log"
}

eval_variant() {
  local name="$1"
  local controller="$2"
  local ckpt="$MODEL_ROOT/$name/best.pt"
  local out="$EVAL_ROOT/${name}_${controller}"
  mkdir -p "$out"
  echo "[eval] $name controller=$controller"
  local device_args=()
  if [[ "$controller" == "pure_bc" ]]; then
    device_args=(--human_input_device imitation)
  else
    device_args=(--human_input_device bc_gate_scripted_geo)
  fi
  "$PYTHON_BIN" "$ROOT/eval_interactive_safetygym.py" \
    --model_path "$STUDENT_CKPT" \
    --controller policy \
    --intervention_mode human \
    "${device_args[@]}" \
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
    > "$out/stdout.log" 2> "$out/stderr.log"
}

train_variant mlp_ctx8 mlp 8 3 4
train_variant mlp_ctx16 mlp 16 3 4
train_variant temporal_cnn_ctx16 temporal_cnn 16 3 4
train_variant attention_ctx16 attention 16 2 4

for name in mlp_ctx8 mlp_ctx16 temporal_cnn_ctx16 attention_ctx16; do
  eval_variant "$name" pure_bc
  eval_variant "$name" hybrid_scripted_move
done

"$PYTHON_BIN" - "$OUT_ROOT" "$DATASET_PATH" "$MODEL_ROOT" "$EVAL_ROOT" <<'PY'
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from safetygym_utils.imitation_teacher import (
    build_window_features,
    load_imitation_checkpoint,
    normalize_observations,
    scale_unit_action,
)

out_root = Path(sys.argv[1])
dataset_path = Path(sys.argv[2])
model_root = Path(sys.argv[3])
eval_root = Path(sys.argv[4])
variants = ["mlp_ctx8", "mlp_ctx16", "temporal_cnn_ctx16", "attention_ctx16"]
controllers = ["pure_bc", "hybrid_scripted_move"]
episode_re = re.compile(r"episode=(?P<episode>\d+) .*?cost=(?P<cost>[-+0-9.eE]+) .*?len=(?P<length>\d+) .*?interventions=(?P<interventions>\d+)")

def parse_jsonl_last(path: Path):
    last = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                pass
    return last

eval_rows = []
for name in variants:
    train_last = parse_jsonl_last(model_root / name / "train_stdout.log") or {}
    for controller in controllers:
        out = eval_root / f"{name}_{controller}"
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
            raise RuntimeError(f"missing eval summary for {out}")
        cost_arr = np.asarray(costs, dtype=np.float64)
        eval_rows.append({
            "variant": name,
            "controller": controller,
            "episodes": int(summary.get("eval/episodes", len(costs))),
            "violations": float(cost_arr.sum()) if cost_arr.size else float(summary.get("eval/collision_cost_sum", 0.0)),
            "episodes_with_cost": int((cost_arr > 0.0).sum()) if cost_arr.size else 0,
            "success_rate": float(summary.get("eval/goal_success_rate", 0.0)),
            "first_goal_success_rate": float(summary.get("eval/first_goal_success_rate", 0.0)),
            "collision_cost_rate": float(summary.get("eval/collision_cost_rate", 0.0)),
            "mean_episode_length": float(summary.get("eval/mean_episode_length", summary.get("eval/episode_length_mean", 0.0))),
            "mean_intervention_fraction": float(summary.get("eval/teacher_fraction_steps", summary.get("eval/intervention_fraction_mean", 0.0))),
            "train_final_val_accuracy": float((train_last.get("val") or {}).get("decision_accuracy", float("nan"))),
            "train_final_val_recall": float((train_last.get("val") or {}).get("decision_recall", float("nan"))),
            "train_final_val_precision": float((train_last.get("val") or {}).get("decision_precision", float("nan"))),
            "train_final_val_action_mse": float((train_last.get("val") or {}).get("action_mse_intervened", float("nan"))),
            "contact_sheet": str(out / "episode_contact_sheet.png"),
        })

data = np.load(dataset_path, allow_pickle=True)
obs = np.asarray(data["observations"], dtype=np.float32)
scripted_action = np.asarray(data["actions"], dtype=np.float32)
student = np.asarray(data["student_actions"], dtype=np.float32)
scripted_gate = np.asarray(data["teacher_intervened"]).astype(bool).reshape(-1)
episode_ids = np.asarray(data["episode_ids"], dtype=np.int64).reshape(-1)

def corr(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

corr_rows = []
heat_cols = {"scripted_gate": scripted_gate.astype(np.float32)}
for name in variants:
    ckpt = model_root / name / "best.pt"
    model, state = load_imitation_checkpoint(ckpt, device="cpu")
    meta = dict(state.get("metadata") or {})
    features = build_window_features(
        observations=obs,
        student_actions=student,
        teacher_intervened=scripted_gate,
        episode_ids=episode_ids,
        context_len=int(meta.get("context_len", 1)),
    )
    probs = []
    actions = []
    with torch.inference_mode():
        for start in range(0, features.shape[0], 4096):
            xb = torch.as_tensor(features[start : start + 4096], dtype=torch.float32)
            xb = normalize_observations(xb, state["obs_mean"], state["obs_std"])
            action_unit, logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            actions.append(scale_unit_action(action_unit, state["action_low"], state["action_high"]).cpu().numpy())
    probs = np.concatenate(probs, axis=0).reshape(-1)
    actions = np.concatenate(actions, axis=0)
    pred_gate = probs >= 0.5
    tp = int(np.logical_and(pred_gate, scripted_gate).sum())
    fp = int(np.logical_and(pred_gate, ~scripted_gate).sum())
    tn = int(np.logical_and(~pred_gate, ~scripted_gate).sum())
    fn = int(np.logical_and(~pred_gate, scripted_gate).sum())
    corr_rows.append({
        "variant": name,
        "gate_prob_corr": corr(scripted_gate.astype(float), probs),
        "gate_binary_corr": corr(scripted_gate.astype(float), pred_gate.astype(float)),
        "gate_accuracy": float((pred_gate == scripted_gate).mean()),
        "gate_precision": float(tp / max(1, tp + fp)),
        "gate_recall": float(tp / max(1, tp + fn)),
        "gate_pred_fraction": float(pred_gate.mean()),
        "left_action_corr_intervened": corr(scripted_action[scripted_gate, 0], actions[scripted_gate, 0]),
        "right_action_corr_intervened": corr(scripted_action[scripted_gate, 1], actions[scripted_gate, 1]),
        "delta_corr_intervened": corr(
            np.linalg.norm(scripted_action[scripted_gate] - student[scripted_gate], axis=1),
            np.linalg.norm(actions[scripted_gate] - student[scripted_gate], axis=1),
        ),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    })
    heat_cols[f"{name}_prob"] = probs.astype(np.float32)
    heat_cols[f"{name}_gate"] = pred_gate.astype(np.float32)
    heat_cols[f"{name}_delta"] = np.linalg.norm(actions - student, axis=1).astype(np.float32)

def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

write_csv(out_root / "eval_summary.csv", eval_rows)
write_csv(out_root / "offline_similarity.csv", corr_rows)

names = list(heat_cols)
arr = np.stack([np.asarray(heat_cols[n], dtype=np.float64).reshape(-1) for n in names], axis=1)
C = np.full((len(names), len(names)), np.nan)
for i in range(len(names)):
    for j in range(len(names)):
        C[i, j] = corr(arr[:, i], arr[:, j])
fig, ax = plt.subplots(figsize=(max(9, 0.75 * len(names)), max(8, 0.68 * len(names))), dpi=160)
im = ax.imshow(C, vmin=-1, vmax=1, cmap="coolwarm")
ax.set_title("Temporal BC teacher offline correlation")
ax.set_xticks(range(len(names)))
ax.set_yticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(names, fontsize=8)
for i in range(len(names)):
    for j in range(len(names)):
        if np.isfinite(C[i, j]):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=6)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(out_root / "offline_correlation_heatmap.png")
plt.close(fig)

with (out_root / "summary.md").open("w") as f:
    f.write("| variant | controller | violations | episodes_with_cost | success | cost_rate | intervention_frac | val_acc | val_recall |\\n")
    f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\\n")
    for row in eval_rows:
        f.write(
            f"| {row['variant']} | {row['controller']} | {row['violations']:.0f} | {row['episodes_with_cost']} | "
            f"{row['success_rate']:.3f} | {row['collision_cost_rate']:.4f} | {row['mean_intervention_fraction']:.3f} | "
            f"{row['train_final_val_accuracy']:.3f} | {row['train_final_val_recall']:.3f} |\\n"
        )
print(json.dumps({
    "eval_summary": str(out_root / "eval_summary.csv"),
    "offline_similarity": str(out_root / "offline_similarity.csv"),
    "heatmap": str(out_root / "offline_correlation_heatmap.png"),
    "summary_md": str(out_root / "summary.md"),
}, indent=2, sort_keys=True))
PY

echo "[done] $OUT_ROOT"
