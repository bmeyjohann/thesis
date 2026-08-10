#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HUMAN_LOG = ROOT / "models/unitree_mjlab_nav_human/unitree_human_scratch_20260803_092823_retry03/metrics.jsonl"
SAC_LOG = ROOT / "models/unitree_mjlab_nav_thesis/unitree_human_matched_sac_reward_only_seed0_28k_20260804/metrics.jsonl"
EVAL_ROOT = ROOT / "logs/unitree_mjlab/human_checkpoint_sweep_20260806"
OUTPUT = ROOT / "visualizations/unitree_human_vs_sac_20260806"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def series(data: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = [(float(row["step"]), float(row[key])) for row in data if key in row]
    return np.asarray([p[0] for p in pairs]) / 1000.0, np.asarray([p[1] for p in pairs])


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def eval_metrics(name: str) -> dict:
    data = json.loads((EVAL_ROOT / name / "policy_metrics.json").read_text())
    data["safe_success_rate"] = sum(
        bool(ep["success"]) and float(ep["cost_sum"]) == 0.0 for ep in data["episodes"]
    ) / len(data["episodes"])
    return data


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    human, sac = rows(HUMAN_LOG), rows(SAC_LOG)
    final = {"human intervention": eval_metrics("retry03_final"), "reward-only SAC": eval_metrics("sac_reward_only_final")}
    colors = {"human intervention": "#087e8b", "reward-only SAC": "#c44536"}
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)

    names = list(final)
    x = np.arange(len(names))
    axes[0, 0].bar(x - 0.18, [final[n]["success_rate"] for n in names], 0.36, label="success")
    axes[0, 0].bar(x + 0.18, [final[n]["safe_success_rate"] for n in names], 0.36, label="safe success")
    axes[0, 0].set(title="Teacher-free evaluation (100 paired episodes)", xticks=x, xticklabels=names, ylabel="rate")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].bar(x - 0.18, [final[n]["mean_cost_sum"] for n in names], 0.36, label="mean cost")
    axes[0, 1].bar(x + 0.18, [100 * final[n]["costful_episode_rate"] for n in names], 0.36, label="costful episodes (%)")
    axes[0, 1].set(title="Safety evaluation", xticks=x, xticklabels=names)
    axes[0, 1].legend(frameon=False)

    for key, label, style in (("intervention_fraction", "cumulative intervention", "-"), ("batch_teacher_fraction", "teacher rows in sampled batch", "--")):
        sx, sy = series(human, key)
        axes[0, 2].plot(sx, smooth(sy), style, label=label)
    axes[0, 2].axhline(0, color=colors["reward-only SAC"], linestyle=":", label="SAC intervention = 0")
    axes[0, 2].set(title="Human effort and replay exposure", xlabel="training steps (thousands)", ylabel="fraction")
    axes[0, 2].legend(frameon=False, fontsize=8)

    for data, name, linestyle in ((human, "human intervention", "-"), (sac, "reward-only SAC", "--")):
        for key, suffix in (("q_min_pi_mean", "Q(policy)"), ("q_min_data_mean", "Q(data)"), ("target_q_mean", "target Q")):
            sx, sy = series(data, key)
            if len(sx):
                axes[0, 3].plot(sx, smooth(sy), linestyle, label=f"{name}: {suffix}")
    axes[0, 3].set(title="Q-value scale", xlabel="training steps (thousands)", ylabel="Q")
    axes[0, 3].legend(frameon=False, fontsize=7)

    for data, name, linestyle in ((human, "human intervention", "-"), (sac, "reward-only SAC", "--")):
        for key, suffix in (("q_disagreement_data_mean", "data"), ("q_disagreement_pi_mean", "policy")):
            sx, sy = series(data, key)
            axes[1, 0].plot(sx, smooth(sy), linestyle, label=f"{name}: {suffix}")
    axes[1, 0].set(title="Twin-critic disagreement", xlabel="training steps (thousands)", ylabel="absolute disagreement")
    axes[1, 0].legend(frameon=False, fontsize=7)

    sx, lam = series(human, "pref_lambda")
    _, violation = series(human, "pref_violation")
    axes[1, 1].plot(sx, lam, color="#f28e2b", label="preference lambda")
    twin = axes[1, 1].twinx()
    twin.plot(sx, smooth(violation), color="#59a14f", label="preference violation")
    axes[1, 1].set(title="Preference constraint", xlabel="training steps (thousands)", ylabel="lambda")
    twin.set_ylabel("violation")
    lines = axes[1, 1].lines + twin.lines
    axes[1, 1].legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8)

    for data, name, linestyle in ((human, "human intervention", "-"), (sac, "reward-only SAC", "--")):
        sx, sy = series(data, "critic_loss_total")
        axes[1, 2].plot(sx, smooth(sy), linestyle, label=name)
    axes[1, 2].set(title="Critic loss", xlabel="training steps (thousands)", ylabel="loss", yscale="log")
    axes[1, 2].legend(frameon=False)

    width = 0.2
    metrics = (("front_obstacle_abs_steering", "|steering|"), ("front_obstacle_small_steering_fraction", "near-zero"), ("mean_front_obstacle_directional_consistency", "consistency"), ("front_obstacle_action_delta", "action change"))
    for idx, (key, label) in enumerate(metrics):
        axes[1, 3].bar(x + (idx - 1.5) * width, [final[n][key] for n in names], width, label=label)
    axes[1, 3].set(title="When an obstacle is visible ahead", xticks=x, xticklabels=names, ylabel="metric")
    axes[1, 3].legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.grid(alpha=0.2)
        if ax in (axes[0, 0], axes[0, 1], axes[1, 3]):
            ax.tick_params(axis="x", rotation=12)
    fig.suptitle("Unitree: human-intervention policy vs reward-only SAC", fontsize=17)
    fig.savefig(OUTPUT / "human_vs_sac_training_and_eval.png", dpi=180)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
