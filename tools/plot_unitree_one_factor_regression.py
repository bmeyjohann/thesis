#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "baseline": "baseline",
    "heading_mask": "heading masked",
    "strict_obstacles": "strict obstacles",
    "adaptive_clearance": "adaptive clearance",
    "relaxed_timing": "relaxed gate timing",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main() -> int:
    output_dir = ROOT / "visualizations" / "unitree_one_factor_regression_20260714"
    output_dir.mkdir(parents=True, exist_ok=True)
    series = {}
    summary = {}
    for variant, label in VARIANTS.items():
        run_dir = ROOT / "models" / "unitree_mjlab_nav_thesis" / f"unitree_oat_{variant}_2500_20260714"
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        rows = read_jsonl(metrics_path)
        series[variant] = rows
        final = rows[-1]
        summary[variant] = {
            "label": label,
            "training_step": final.get("step"),
            "training_success_rate": final.get("success_rate"),
            "training_episode_cost_mean": final.get("episode_cost_mean"),
            "teacher_fraction_interval": final.get("teacher_fraction_interval"),
            "teacher_fraction_cumulative": final.get("teacher_fraction_cumulative"),
            "student_costful_step_rate": final.get("student_executed_costful_step_rate"),
            "teacher_costful_step_rate": final.get("teacher_executed_costful_step_rate"),
            "q_min_pi_mean": final.get("q_min_pi_mean"),
            "q_min_data_mean": final.get("q_min_data_mean"),
            "critic_loss_total": final.get("critic_loss_total"),
            "pref_lambda": final.get("pref_lambda"),
            "pref_violation_ema": final.get("pref_violation_ema"),
        }
        eval_root = (
            ROOT
            / "logs"
            / "unitree_mjlab"
            / "one_factor_regression_20260714"
            / variant
        )
        eval_paths = sorted(eval_root.glob("*/policy_metrics.json"))
        if eval_paths:
            eval_path = eval_paths[-1]
            summary[variant]["evaluation"] = json.loads(eval_path.read_text(encoding="utf-8"))

    metrics = [
        ("success_rate", "teacher-gated training success"),
        ("episode_cost_mean", "teacher-gated episode cost"),
        ("teacher_fraction_interval", "teacher fraction"),
        ("student_executed_costful_step_rate", "student costful-step rate"),
        ("q_min_pi_mean", "Q(policy)"),
        ("pref_lambda", "preference lambda"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=150)
    for ax, (key, title) in zip(axes.flat, metrics):
        for variant, rows in series.items():
            points = [(row["step"], row[key]) for row in rows if key in row]
            if points:
                x, y = zip(*points)
                ax.plot(x, y, label=VARIANTS[variant], linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output_dir / "training_curves.png")
    plt.close(fig)

    eval_variants = [variant for variant in VARIANTS if "evaluation" in summary.get(variant, {})]
    if eval_variants:
        eval_metrics = []
        for variant in eval_variants:
            evaluation = summary[variant]["evaluation"]
            episodes = evaluation["episodes"]
            safe_success = sum(
                bool(episode["success"]) and float(episode["cost_sum"]) == 0.0
                for episode in episodes
            ) / len(episodes)
            eval_metrics.append(
                {
                    "success": evaluation["success_rate"],
                    "safe_success": safe_success,
                    "costful_episode": evaluation["costful_episode_rate"],
                    "mean_cost": evaluation["mean_cost_sum"],
                    "collision_steps": evaluation["mean_collision_steps"],
                    "action_flips": evaluation["mean_action_sign_flips"],
                }
            )
            summary[variant]["evaluation"]["safe_success_rate"] = safe_success
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=150)
        eval_panels = [
            ("success", "success rate"),
            ("safe_success", "safe success rate"),
            ("costful_episode", "costful episode rate"),
            ("mean_cost", "mean episode cost"),
            ("collision_steps", "mean collision steps"),
            ("action_flips", "mean action sign flips"),
        ]
        labels = [VARIANTS[variant] for variant in eval_variants]
        for ax, (key, title) in zip(axes.flat, eval_panels):
            values = [metrics[key] for metrics in eval_metrics]
            ax.bar(range(len(labels)), values)
            ax.set_title(title)
            ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle("Policy-only 40-episode common benchmark")
        fig.tight_layout()
        fig.savefig(output_dir / "evaluation_comparison.png")
        plt.close(fig)

    trajectory_paths = {}
    for variant in VARIANTS:
        trajectory_dir = output_dir / "trajectories" / variant
        candidates = sorted(trajectory_dir.glob("*topdown*.png"))
        if candidates:
            trajectory_paths[variant] = candidates[-1]
    if trajectory_paths:
        fig, axes = plt.subplots(1, len(trajectory_paths), figsize=(5 * len(trajectory_paths), 5), dpi=150)
        if len(trajectory_paths) == 1:
            axes = [axes]
        for ax, (variant, image_path) in zip(axes, trajectory_paths.items()):
            ax.imshow(mpimg.imread(image_path))
            ax.set_title(VARIANTS[variant])
            ax.axis("off")
        fig.suptitle("Policy-only trajectories on the common seed-941 benchmark")
        fig.tight_layout()
        fig.savefig(output_dir / "trajectory_comparison.png", bbox_inches="tight")
        plt.close(fig)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output_dir / "training_curves.png")
    if eval_variants:
        print(output_dir / "evaluation_comparison.png")
    if trajectory_paths:
        print(output_dir / "trajectory_comparison.png")
    print(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
