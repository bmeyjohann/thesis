#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HUMAN_EVAL = ROOT / "logs/unitree_human_hist1_action20_4run_paired100_20260813"
SAC_EVAL = ROOT / "logs/unitree_goalonly_sac_hist1_action20_nofail_paired100_20260813"
CURVES = ROOT / "logs/unitree_current_objective_curves_20260814"
OUTPUT = ROOT / "visualizations/unitree_current_objective_comparison_20260814"
HUMAN_HZ = 20.0

METHODS = {
    "Pref + BC + RL": {
        "color": "#007f7b",
        "evals": [HUMAN_EVAL / "pref_bc_rl_seed1", HUMAN_EVAL / "pref_bc_rl_seed2"],
        "curve": ["pref_bc_rl_seed1", "pref_bc_rl_seed2"],
        "train": [
            ROOT / "models/unitree_mjlab_nav_human/unitree_human_pref_bc_rl_seed1_20260813_142741/metrics.jsonl",
            ROOT / "models/unitree_mjlab_nav_human/unitree_human_pref_bc_rl_seed2_20260813_145006/metrics.jsonl",
        ],
    },
    "Pref + RL": {
        "color": "#e17c05",
        "evals": [HUMAN_EVAL / "pref_rl_seed1_step13000", HUMAN_EVAL / "pref_rl_seed2"],
        "curve": ["pref_rl_seed1", "pref_rl_seed2"],
        "train": [
            ROOT / "models/unitree_mjlab_nav_human/unitree_human_pref_rl_seed1_20260813_150806/metrics.jsonl",
            ROOT / "models/unitree_mjlab_nav_human/unitree_human_pref_rl_seed2_20260813_152522/metrics.jsonl",
        ],
    },
    "BC + RL": {
        "color": "#5f4690",
        "evals": [HUMAN_EVAL / "bc_rl_seed1", HUMAN_EVAL / "bc_rl_seed2"],
        "curve": ["bc_rl_seed1", "bc_rl_seed2"],
        "train": [
            ROOT / "models/unitree_mjlab_nav_human/unitree_human_bc_rl_seed1_20260813_181129/metrics.jsonl",
            ROOT / "models/unitree_mjlab_nav_human/unitree_human_bc_rl_seed2_20260813_175038/metrics.jsonl",
        ],
    },
    "Preference only": {
        "color": "#cc503e",
        "evals": [HUMAN_EVAL / "pref_only_seed1", HUMAN_EVAL / "pref_only_seed2"],
        "curve": ["pref_only_seed1", "pref_only_seed2"],
        "train": [
            ROOT / "models/unitree_mjlab_nav_human/unitree_human_pref_only_seed1_20260814_011751/metrics.jsonl",
            ROOT / "models/unitree_mjlab_nav_human/unitree_human_pref_only_seed2_20260814_021008/metrics.jsonl",
        ],
    },
    "SAC": {
        "color": "#4c78a8",
        "evals": [SAC_EVAL / f"sac_hist1_nofail_seed{seed}" for seed in (1, 2, 3)],
        "curve": [f"sac_seed{seed}" for seed in (1, 2, 3)],
        "train": [],
    },
}
STEPS = [1000, 4000, 7000, 10000, 13000]


def load_metrics(directory: Path) -> dict:
    data = json.loads((directory / "policy_metrics.json").read_text())
    episodes = data.get("episodes", [])
    data["safe_success_rate"] = float(np.mean([
        bool(ep["success"]) and float(ep["cost_sum"]) == 0.0 for ep in episodes
    ]))
    return data


def mean_sd(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def add_time_axis(ax: plt.Axes) -> None:
    sec = ax.secondary_xaxis(
        "top",
        functions=(
            lambda steps_k: steps_k * 1000 / HUMAN_HZ / 60,
            lambda minutes: minutes * HUMAN_HZ * 60 / 1000,
        ),
    )
    sec.set_xlabel("equivalent human supervision (minutes at 20 Hz)")


def interval_intervention(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows = [row for row in rows if "step" in row and "intervention_fraction" in row]
    steps = np.asarray([float(row["step"]) for row in rows])
    cumulative_count = steps * np.asarray([float(row["intervention_fraction"]) for row in rows])
    interval = np.diff(np.r_[0.0, cumulative_count]) / np.maximum(np.diff(np.r_[0.0, steps]), 1.0)
    # Smooth over approximately 1,000 environment steps.
    samples = max(1, int(round(1000 / np.median(np.diff(steps))))) if len(steps) > 1 else 1
    smooth = np.asarray([interval[max(0, i - samples + 1): i + 1].mean() for i in range(len(interval))])
    return steps, smooth


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    finals = {
        name: [load_metrics(path) for path in cfg["evals"]]
        for name, cfg in METHODS.items()
    }

    plt.rcParams.update({"font.size": 10.5, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    names = list(METHODS)
    x = np.arange(len(names))
    colors = [METHODS[name]["color"] for name in names]

    ax = axes[0, 0]
    for offset, key, label in [(-0.18, "success_rate", "success"), (0.18, "safe_success_rate", "safe success")]:
        vals, errs = zip(*(mean_sd([row[key] for row in finals[name]]) for name in names))
        ax.bar(x + offset, vals, 0.36, yerr=errs, capsize=3, label=label, alpha=0.9)
    ax.set(title="Final goal reaching", ylabel="episode rate", xticks=x, xticklabels=names, ylim=(0, 0.75))
    ax.legend(frameon=False)

    for ax, key, title, ylabel in [
        (axes[0, 1], "mean_cost_sum", "Final mean safety cost", "cost per episode"),
        (axes[0, 2], "costful_episode_rate", "Final episodes with safety cost", "episode rate"),
    ]:
        vals, errs = zip(*(mean_sd([row[key] for row in finals[name]]) for name in names))
        ax.bar(x, vals, yerr=errs, capsize=4, color=colors)
        ax.set(title=title, ylabel=ylabel, xticks=x, xticklabels=names)

    for ax, key, title in [
        (axes[1, 0], "success_rate", "Teacher-free success over training"),
        (axes[1, 1], "safe_success_rate", "Teacher-free safe success over training"),
    ]:
        for name, cfg in METHODS.items():
            means, sds, valid_steps = [], [], []
            for step in STEPS:
                rows = []
                for label in cfg["curve"]:
                    path = CURVES / f"{label}_step_{step}"
                    if (path / "policy_metrics.json").exists():
                        rows.append(load_metrics(path)[key])
                if rows:
                    mean, sd = mean_sd(rows)
                    valid_steps.append(step / 1000)
                    means.append(mean)
                    sds.append(sd)
            if valid_steps:
                xs = np.asarray(valid_steps)
                ys = np.asarray(means)
                es = np.asarray(sds)
                ax.plot(xs, ys, marker="o", linewidth=2, color=cfg["color"], label=name)
                ax.fill_between(xs, np.clip(ys - es, 0, 1), np.clip(ys + es, 0, 1), color=cfg["color"], alpha=0.13)
        ax.set(title=title, xlabel="online training steps (thousands)", ylabel="episode rate", ylim=(0, 0.75))
        add_time_axis(ax)
    axes[1, 0].legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1, 2]
    common_x = np.arange(100, 14001, 100, dtype=float)
    for name, cfg in METHODS.items():
        traces = []
        for path in cfg["train"]:
            if path.exists():
                steps, values = interval_intervention(path)
                traces.append(np.interp(common_x, steps, values, left=np.nan, right=np.nan))
        if traces:
            stack = np.asarray(traces)
            mean = np.nanmean(stack, axis=0)
            sd = np.nanstd(stack, axis=0, ddof=1) if len(stack) > 1 else np.zeros_like(mean)
            ax.plot(common_x / 1000, mean, linewidth=2, color=cfg["color"], label=name)
            ax.fill_between(common_x / 1000, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), color=cfg["color"], alpha=0.13)
    ax.axhline(0, color=METHODS["SAC"]["color"], linestyle="--", label="SAC")
    ax.set(title="Recent human intervention rate", xlabel="online training steps (thousands)", ylabel="fraction in recent 1k steps", ylim=(0, 1))
    add_time_axis(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[0]:
        ax.tick_params(axis="x", rotation=16)
    fig.suptitle("Unitree navigation objective ablation", fontsize=18, fontweight="bold")
    fig.text(
        0.5,
        -0.008,
        "Final: deterministic teacher-free evaluation on the same 100 blocked layouts per seed. Curves: same 20-layout checkpoint cohort; lines are seed means and shading is +/-1 SD.",
        ha="center",
        fontsize=10,
    )
    output = OUTPUT / "objective_ablation_with_teacher_free_learning_curves.png"
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
