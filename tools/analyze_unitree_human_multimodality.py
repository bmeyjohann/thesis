#!/usr/bin/env python3
"""Diagnose inconsistent human steering labels and temporal action jitter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_dataset(root: Path) -> dict[str, np.ndarray]:
    columns: dict[str, list[np.ndarray]] = {}
    for path in sorted((root / "parts").glob("*.npz")):
        with np.load(path) as part:
            for key in part.files:
                columns.setdefault(key, []).append(part[key])
    return {key: np.concatenate(values) for key, values in columns.items()}


def _steering(actions: np.ndarray) -> np.ndarray:
    # Lateral and yaw encode the same avoidance side and have different ranges.
    return 0.5 * (actions[:, 1] / 0.65 + actions[:, 2] / 0.85)


def _committed_sign(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(values > threshold, 1, np.where(values < -threshold, -1, 0))


def _burst_metrics(data: dict[str, np.ndarray], threshold: float) -> dict[str, float]:
    mask = data["intervened"].astype(bool)
    steering = _steering(data["human_actions"])
    starts = np.flatnonzero(data["intervention_start"])
    ends = np.flatnonzero(data["intervention_end"])
    switch_counts, minority_fractions = [], []
    for start in starts:
        end_candidates = ends[ends >= start]
        end = int(end_candidates[0]) if end_candidates.size else len(mask) - 1
        indices = np.arange(start, end + 1)
        indices = indices[mask[indices]]
        signs = _committed_sign(steering[indices], threshold)
        signs = signs[signs != 0]
        if not signs.size:
            continue
        switch_counts.append(int(np.sum(signs[1:] != signs[:-1])))
        positive = float(np.mean(signs > 0))
        minority_fractions.append(min(positive, 1.0 - positive))
    switches = np.asarray(switch_counts, dtype=float)
    minority = np.asarray(minority_fractions, dtype=float)
    return {
        "bursts_analyzed": int(len(switches)),
        "bursts_with_side_switch_fraction": float(np.mean(switches > 0)),
        "bursts_with_multiple_switches_fraction": float(np.mean(switches > 1)),
        "mean_side_switches_per_burst": float(np.mean(switches)),
        "median_side_switches_per_burst": float(np.median(switches)),
        "mean_minority_side_fraction": float(np.mean(minority)),
    }


def _neighbor_metrics(
    data: dict[str, np.ndarray], threshold: float, neighbors: int, model_path: Path | None
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    mask = data["intervened"].astype(bool)
    obs = data["observations"][mask].astype(np.float64)
    episodes = data["episode_index"][mask]
    human = _steering(data["human_actions"][mask])
    student = _steering(data["student_actions"][mask])
    final_actions = _checkpoint_actions(model_path, obs) if model_path is not None else None
    final_student = _steering(final_actions) if final_actions is not None else None

    scale = np.std(obs, axis=0)
    active = scale > 1e-5
    normalized = (obs[:, active] - np.mean(obs[:, active], axis=0)) / scale[active]
    # PCA keeps nearest-neighbor search tractable while retaining almost all
    # state variation represented by the goal context and rectangular scan.
    _, _, vt = np.linalg.svd(normalized, full_matrices=False)
    embedding = normalized @ vt[: min(32, vt.shape[0])].T
    tree = cKDTree(embedding)
    _, candidate_idx = tree.query(embedding, k=min(max(neighbors * 8, 32), len(obs)))

    ambiguity, local_consensus, local_abs_target, local_distance = [], [], [], []
    human_sign = _committed_sign(human, threshold)
    for row, candidates in enumerate(candidate_idx):
        candidates = candidates[(episodes[candidates] != episodes[row]) & (human_sign[candidates] != 0)]
        candidates = candidates[:neighbors]
        if len(candidates) < max(3, neighbors // 2):
            ambiguity.append(np.nan)
            local_consensus.append(np.nan)
            local_abs_target.append(np.nan)
            local_distance.append(np.nan)
            continue
        signs = human_sign[candidates]
        p_right = float(np.mean(signs > 0))
        ambiguity.append(2.0 * min(p_right, 1.0 - p_right))
        local_consensus.append(float(np.mean(human[candidates])))
        local_abs_target.append(float(np.mean(np.abs(human[candidates]))))
        local_distance.append(float(np.mean(np.linalg.norm(embedding[candidates] - embedding[row], axis=1))))

    ambiguity = np.asarray(ambiguity)
    local_consensus = np.asarray(local_consensus)
    local_abs_target = np.asarray(local_abs_target)
    valid = np.isfinite(ambiguity)
    high = valid & (ambiguity >= 0.6)
    low = valid & (ambiguity <= 0.2)
    metrics = {
        "intervention_rows": int(mask.sum()),
        "cross_episode_neighborhoods_valid": int(valid.sum()),
        "ambiguous_neighborhood_fraction": float(np.mean(ambiguity[valid] >= 0.6)),
        "strong_consensus_neighborhood_fraction": float(np.mean(ambiguity[valid] <= 0.2)),
        "mean_neighborhood_ambiguity": float(np.mean(ambiguity[valid])),
        "mean_neighborhood_cancellation_ratio": float(
            np.mean(1.0 - np.abs(local_consensus[valid]) / np.maximum(local_abs_target[valid], 1e-6))
        ),
        "student_abs_steering_ambiguous": float(np.mean(np.abs(student[high]))),
        "student_abs_steering_consensus": float(np.mean(np.abs(student[low]))),
        "human_abs_steering_ambiguous": float(np.mean(np.abs(human[high]))),
        "human_abs_steering_consensus": float(np.mean(np.abs(human[low]))),
        "student_near_zero_fraction_ambiguous": float(np.mean(np.abs(student[high]) < threshold)),
        "student_near_zero_fraction_consensus": float(np.mean(np.abs(student[low]) < threshold)),
    }
    if final_student is not None:
        metrics.update(
            {
                "final_student_abs_steering_ambiguous": float(np.mean(np.abs(final_student[high]))),
                "final_student_abs_steering_consensus": float(np.mean(np.abs(final_student[low]))),
                "final_student_near_zero_fraction_ambiguous": float(
                    np.mean(np.abs(final_student[high]) < threshold)
                ),
                "final_student_near_zero_fraction_consensus": float(
                    np.mean(np.abs(final_student[low]) < threshold)
                ),
                "final_student_local_consensus_correlation": float(
                    np.corrcoef(final_student[valid], local_consensus[valid])[0, 1]
                ),
            }
        )
    return metrics, ambiguity, human, student, final_student, final_actions


def _magnitude_metrics(
    data: dict[str, np.ndarray], final_actions: np.ndarray | None, threshold: float
) -> dict[str, object]:
    mask = data["intervened"].astype(bool)
    sources = {
        "human": data["human_actions"][mask],
        "online_student": data["student_actions"][mask],
    }
    if final_actions is not None:
        sources["final_student"] = final_actions
    result: dict[str, object] = {}
    for name, actions in sources.items():
        result[name] = {
            "mean_abs_per_action": np.mean(np.abs(actions), axis=0).tolist(),
            "median_abs_per_action": np.median(np.abs(actions), axis=0).tolist(),
            "near_zero_fraction_per_action": np.mean(np.abs(actions) < threshold, axis=0).tolist(),
            "mean_action_norm": float(np.mean(np.linalg.norm(actions, axis=1))),
            "small_command_fraction": float(np.mean(np.linalg.norm(actions, axis=1) < threshold)),
        }
    return result


def _checkpoint_actions(model_path: Path, obs: np.ndarray) -> np.ndarray:
    from safetygym_utils.sac import build_sac

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    saved = dict(checkpoint.get("args", {}))
    sac = build_sac(
        obs_dim=int(checkpoint.get("obs_dim", obs.shape[1])),
        act_dim=int(checkpoint.get("act_dim", 3)),
        hidden_actor=int(saved.get("hidden_dim", 256)),
        hidden_critic=int(saved.get("hidden_dim", 256)),
        num_critics=2,
        use_layer_norm=bool(saved.get("use_layer_norm", False)),
        layer_norm_eps=1e-5,
        init_scale=0.01,
        lr_actor=3e-4,
        lr_critic=3e-4,
        weight_decay=0.0,
        num_envs=1,
        device=torch.device("cpu"),
        alpha_init=0.01,
        temporal_encoder="unitree_scan_cnn" if saved.get("policy_encoder") == "scan_cnn" else "none",
        obs_frame_stack=int(saved.get("scan_history", 1)),
        unitree_action_history=int(saved.get("action_history", 0)),
    )
    sac.actor.load_state_dict(checkpoint["actor_state_dict"])
    means = []
    with torch.inference_mode():
        for start in range(0, len(obs), 1024):
            _, _, mean = sac.actor(torch.as_tensor(obs[start : start + 1024], dtype=torch.float32))
            means.append(mean.numpy())
    return np.concatenate(means)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steering-threshold", type=float, default=0.15)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = _load_dataset(args.dataset_dir)
    burst = _burst_metrics(data, args.steering_threshold)
    neighbor, ambiguity, human, student, final_student, final_actions = _neighbor_metrics(
        data, args.steering_threshold, args.neighbors, args.model_path
    )
    magnitude = _magnitude_metrics(data, final_actions, args.steering_threshold)
    summary = {
        "steering_threshold": args.steering_threshold,
        **burst,
        **neighbor,
        "action_magnitude": magnitude,
    }
    (args.output_dir / "multimodality_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    valid = np.isfinite(ambiguity)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes = axes.flat
    axes[0].hist(ambiguity[valid], bins=np.linspace(0, 1, 21), color="#087e8b")
    axes[0].set(title="Opposite-side labels in nearby states", xlabel="ambiguity (0 consensus, 1 balanced)", ylabel="rows")
    axes[1].hexbin(human[valid], student[valid], gridsize=35, mincnt=1, cmap="magma")
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].axvline(0, color="black", linewidth=0.7)
    axes[1].set(title="Human target vs online student", xlabel="human steering", ylabel="student steering")
    bins = np.linspace(0, 1, 6)
    centers = (bins[:-1] + bins[1:]) / 2
    student_magnitude = [np.mean(np.abs(student[valid][(ambiguity[valid] >= lo) & (ambiguity[valid] < hi)])) for lo, hi in zip(bins[:-1], bins[1:])]
    human_magnitude = [np.mean(np.abs(human[valid][(ambiguity[valid] >= lo) & (ambiguity[valid] < hi)])) for lo, hi in zip(bins[:-1], bins[1:])]
    axes[2].plot(centers, human_magnitude, marker="o", label="human")
    axes[2].plot(centers, student_magnitude, marker="o", label="student")
    if final_student is not None:
        final_magnitude = [np.mean(np.abs(final_student[valid][(ambiguity[valid] >= lo) & (ambiguity[valid] < hi)])) for lo, hi in zip(bins[:-1], bins[1:])]
        axes[2].plot(centers, final_magnitude, marker="o", label="final student")
    axes[2].set(title="Steering magnitude vs label ambiguity", xlabel="neighborhood ambiguity", ylabel="absolute steering")
    axes[2].legend(frameon=False)
    labels = ("forward", "lateral", "yaw")
    x = np.arange(3)
    width = 0.25
    for offset, (name, values) in enumerate(magnitude.items()):
        axes[3].bar(
            x + (offset - (len(magnitude) - 1) / 2) * width,
            values["mean_abs_per_action"],
            width,
            label=name.replace("_", " "),
        )
    axes[3].set(
        title="Mean command magnitude on intervention states",
        ylabel="mean absolute action",
        xticks=x,
        xticklabels=labels,
    )
    axes[3].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Unitree human-intervention multimodality diagnostics")
    fig.savefig(args.output_dir / "multimodality_diagnostics.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
