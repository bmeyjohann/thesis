#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/benjamin/thesis}"
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-thesis}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

STUDENT_CKPT="${STUDENT_CKPT:-$ROOT/models/safetygym_minimal/safetycar_scriptedgeo_bc70k_shapedrl_finetune_30k_20260515/step_10000.pt}"
BC_TEACHER_CKPT="${BC_TEACHER_CKPT:-$ROOT/logs/safetygym_minimal/safetycar_scriptedgeo_blocked_bc_probe_20260523_blocked_repeat_seed7_v2/hotkey_bc_eval/step_30000/best.pt}"
DATASET_PATH="${DATASET_PATH:-/home/benjamin/.config/thesis/datasets/safetygym/SafetyCarGoal1-v0__scriptedgeo_clearance_or_progress_blocked_seed7_20260523__20260523_211034.npz}"
OUT_ROOT="${OUT_ROOT:-$ROOT/logs/safetygym_eval_plots/bc_gate_scripted_move_compare_20260528}"
EPISODES="${EPISODES:-64}"
PLOT_EPISODES="${PLOT_EPISODES:-12}"
SEED="${SEED:-424242}"

mkdir -p "$OUT_ROOT"

COMMON_ARGS=(
  "$ROOT/eval_interactive_safetygym.py"
  --model_path "$STUDENT_CKPT"
  --controller policy
  --intervention_mode human
  --env_name SafetyCarGoal1-v0
  --seed "$SEED"
  --render_mode none
  --num_episodes "$EPISODES"
  --fps 0
  --no_show_episode_controls
  --layout_curriculum car_random_blocked_filter
  --layout_curriculum_level 0
  --obs_mask_mode privileged_geometry
  --car_action_mode raw_wheels
  --car_wheel_command_limit 1.0
  --car_force_scale 1.0
  --reward_mode dense
  --dense_reward_scale 1.0
  --success_reward_scale 0.0
  --step_penalty 0.0
  --cost_penalty 0.0
  --clearance_penalty_scale 0.0
  --terminate_on_goal
  --save_episode_plots
  --episode_plot_max_episodes "$PLOT_EPISODES"
)

run_variant() {
  local name="$1"
  shift
  local dir="$OUT_ROOT/$name"
  mkdir -p "$dir"
  echo "[eval] $name -> $dir"
  "$PYTHON_BIN" "${COMMON_ARGS[@]}" \
    --episode_plot_dir "$dir" \
    "$@" \
    > "$dir/stdout.log" 2> "$dir/stderr.log"
}

run_variant scripted_gate_scripted_move \
  --human_input_device scripted_geo \
  --teacher_override_mode clearance_or_progress \
  --teacher_override_clearance_threshold 0.08 \
  --teacher_override_clearance_exit_threshold 0.14 \
  --teacher_progress_bad_steps 3 \
  --teacher_progress_good_steps 5 \
  --teacher_progress_epsilon 0.0005 \
  --teacher_progress_trigger_mode worse \
  --teacher_progress_release_mode improve

run_variant bc_gate_bc_move \
  --human_input_device imitation \
  --imitation_checkpoint_path "$BC_TEACHER_CKPT" \
  --intervention_threshold 0.5 \
  --teacher_override_mode clearance \
  --teacher_override_clearance_threshold -1.0

run_variant bc_gate_scripted_move \
  --human_input_device bc_gate_scripted_geo \
  --imitation_checkpoint_path "$BC_TEACHER_CKPT" \
  --intervention_threshold 0.5 \
  --teacher_override_mode clearance \
  --teacher_override_clearance_threshold -1.0

"$PYTHON_BIN" - "$OUT_ROOT" "$DATASET_PATH" "$BC_TEACHER_CKPT" <<'PY'
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
bc_ckpt = Path(sys.argv[3])
variants = [
    ("scripted_gate_scripted_move", "Scripted gate + scripted movement"),
    ("bc_gate_bc_move", "BC gate + BC movement"),
    ("bc_gate_scripted_move", "BC gate + scripted movement"),
]

episode_re = re.compile(r"episode=(?P<episode>\d+) .*?cost=(?P<cost>[-+0-9.eE]+) .*?len=(?P<length>\d+) .*?interventions=(?P<interventions>\d+)")

rows = []
for key, label in variants:
    log_path = out_root / key / "stdout.log"
    text = log_path.read_text()
    summary = None
    costs = []
    lengths = []
    interventions = []
    for line in text.splitlines():
        match = episode_re.search(line)
        if match:
            costs.append(float(match.group("cost")))
            lengths.append(int(match.group("length")))
            interventions.append(int(match.group("interventions")))
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                summary = json.loads(line)
            except json.JSONDecodeError:
                pass
    if summary is None:
        raise RuntimeError(f"No JSON summary found in {log_path}")
    cost_arr = np.asarray(costs, dtype=np.float64)
    rows.append({
        "variant": key,
        "label": label,
        "episodes": int(summary.get("eval/episodes", len(costs))),
        "total_violations": float(cost_arr.sum()) if cost_arr.size else float(summary.get("eval/collision_cost_sum", 0.0)),
        "episodes_with_cost": int((cost_arr > 0.0).sum()) if cost_arr.size else 0,
        "episode_cost_rate": float((cost_arr > 0.0).mean()) if cost_arr.size else 0.0,
        "mean_cost": float(summary.get("eval/mean_episode_cost", summary.get("eval/episode_cost_sum_mean", 0.0))),
        "collision_cost_rate": float(summary.get("eval/collision_cost_rate", 0.0)),
        "goal_success_rate": float(summary.get("eval/goal_success_rate", 0.0)),
        "first_goal_success_rate": float(summary.get("eval/first_goal_success_rate", 0.0)),
        "mean_episode_length": float(summary.get("eval/mean_episode_length", summary.get("eval/episode_length_mean", 0.0))),
        "mean_intervention_fraction": float(summary.get("eval/teacher_fraction_steps", summary.get("eval/intervention_fraction_mean", 0.0))),
        "mean_intervention_steps": float(summary.get("eval/teacher_intervention_steps", summary.get("eval/intervention_steps_mean", 0.0))),
        "contact_sheet": str(out_root / key / "episode_contact_sheet.png"),
    })

csv_path = out_root / "summary_table.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

md_path = out_root / "summary_table.md"
headers = [
    "variant",
    "total_violations",
    "episodes_with_cost",
    "goal_success_rate",
    "first_goal_success_rate",
    "mean_episode_length",
    "mean_intervention_fraction",
]
with md_path.open("w") as f:
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for row in rows:
        f.write(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    f"{row['total_violations']:.3f}",
                    str(row["episodes_with_cost"]),
                    f"{row['goal_success_rate']:.3f}",
                    f"{row['first_goal_success_rate']:.3f}",
                    f"{row['mean_episode_length']:.2f}",
                    f"{row['mean_intervention_fraction']:.3f}",
                ]
            )
            + " |\n"
        )

data = np.load(dataset_path, allow_pickle=True)
obs = np.asarray(data["observations"], dtype=np.float32)
scripted_applied = np.asarray(data["actions"], dtype=np.float32)
student = np.asarray(data["student_actions"], dtype=np.float32)
scripted_gate = np.asarray(data["teacher_intervened"]).astype(bool).reshape(-1)
episode_ids = np.asarray(data["episode_ids"], dtype=np.int64).reshape(-1)

model, state = load_imitation_checkpoint(bc_ckpt, device="cpu")
meta = dict(state.get("metadata") or {})
features = build_window_features(
    observations=obs,
    student_actions=student,
    teacher_intervened=scripted_gate,
    episode_ids=episode_ids,
    context_len=int(meta.get("context_len", 8)),
)
probs = []
bc_raw_actions = []
with torch.inference_mode():
    for start in range(0, features.shape[0], 4096):
        xb = torch.as_tensor(features[start : start + 4096], dtype=torch.float32)
        xb = normalize_observations(xb, state["obs_mean"], state["obs_std"])
        action_unit, logits = model(xb)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        bc_raw_actions.append(scale_unit_action(action_unit, state["action_low"], state["action_high"]).cpu().numpy())
bc_prob = np.concatenate(probs, axis=0).reshape(-1)
bc_gate = bc_prob >= 0.5
bc_raw_actions = np.concatenate(bc_raw_actions, axis=0).astype(np.float32)

bc_applied = np.where(bc_gate[:, None], bc_raw_actions, student)
# The dataset contains the scripted applied action. On the few BC-positive/scripted-negative
# rows, this approximates the hybrid scripted movement as the recorded applied action.
hybrid_applied = np.where(bc_gate[:, None], scripted_applied, student)
scripted_delta = np.linalg.norm(scripted_applied - student, axis=1)
bc_delta = np.linalg.norm(bc_applied - student, axis=1)
hybrid_delta = np.linalg.norm(hybrid_applied - student, axis=1)

cols = {
    "scripted_gate": scripted_gate.astype(np.float32),
    "bc_prob": bc_prob.astype(np.float32),
    "bc_gate": bc_gate.astype(np.float32),
    "hybrid_gate": bc_gate.astype(np.float32),
    "scripted_left": scripted_applied[:, 0],
    "scripted_right": scripted_applied[:, 1],
    "bc_left": bc_applied[:, 0],
    "bc_right": bc_applied[:, 1],
    "hybrid_left": hybrid_applied[:, 0],
    "hybrid_right": hybrid_applied[:, 1],
    "scripted_delta": scripted_delta,
    "bc_delta": bc_delta,
    "hybrid_delta": hybrid_delta,
}

def corr_matrix(columns: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    names = list(columns)
    arr = np.stack([np.asarray(columns[name], dtype=np.float64).reshape(-1) for name in names], axis=1)
    corr = np.full((len(names), len(names)), np.nan, dtype=np.float64)
    for i in range(len(names)):
        for j in range(len(names)):
            xi = arr[:, i]
            xj = arr[:, j]
            if xi.size >= 2 and np.std(xi) > 1e-12 and np.std(xj) > 1e-12:
                corr[i, j] = float(np.corrcoef(xi, xj)[0, 1])
    return names, corr

def plot_corr(names: list[str], corr: np.ndarray, path: Path, title: str) -> None:
    fig_w = max(9.0, 0.68 * len(names))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.85), dpi=160)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title(title)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            val = corr[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

names, corr = corr_matrix(cols)
all_corr_path = out_root / "three_variant_correlation_all_rows.png"
plot_corr(names, corr, all_corr_path, "Three teacher variants: all dataset rows")

intervened_cols = {name: values[scripted_gate] for name, values in cols.items() if name != "scripted_gate"}
names_int, corr_int = corr_matrix(intervened_cols)
int_corr_path = out_root / "three_variant_correlation_scripted_intervened_rows.png"
plot_corr(names_int, corr_int, int_corr_path, "Three teacher variants: scripted-intervened rows")

corr_summary = {
    "dataset_rows": int(scripted_gate.size),
    "scripted_intervention_fraction": float(scripted_gate.mean()),
    "bc_intervention_fraction": float(bc_gate.mean()),
    "bc_label_probability_corr": float(np.corrcoef(scripted_gate.astype(float), bc_prob)[0, 1]),
    "bc_label_binary_corr": float(np.corrcoef(scripted_gate.astype(float), bc_gate.astype(float))[0, 1]),
    "all_rows_heatmap": str(all_corr_path),
    "scripted_intervened_rows_heatmap": str(int_corr_path),
}
(out_root / "correlation_summary.json").write_text(json.dumps(corr_summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({"summary_table": str(csv_path), "correlation": corr_summary}, indent=2, sort_keys=True))
PY

echo "[done] artifacts written to $OUT_ROOT"
