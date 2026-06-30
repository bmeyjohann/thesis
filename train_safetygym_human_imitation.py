#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from safetygym_utils.dataset_io import load_transition_dataset
from safetygym_utils.env import make_safety_env
from safetygym_utils.imitation_teacher import (
    action_to_unit_np,
    build_window_features,
    make_imitation_policy,
    normalize_observations,
    save_imitation_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a Safety-Gym human-intervention imitation teacher")
    p.add_argument(
        "--dataset_path",
        type=str,
        action="append",
        required=True,
        help="Dataset path. May be passed multiple times; comma-separated lists are also accepted.",
    )
    p.add_argument("--env_name", type=str, default="")
    p.add_argument("--output_dir", type=str, default="models/safetygym_imitation")
    p.add_argument("--exp_name", type=str, default="")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument(
        "--architecture",
        type=str,
        default="mlp",
        choices=["mlp", "temporal_cnn", "attention"],
        help="Imitation model architecture. mlp uses flattened frame stack; temporal_cnn/attention operate over context segments.",
    )
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--context_len", type=int, default=8)
    p.add_argument("--action_loss_weight", type=float, default=1.0)
    p.add_argument("--decision_loss_weight", type=float, default=1.0)
    p.add_argument("--non_intervention_action_weight", type=float, default=0.0)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--intervention_threshold", type=float, default=0.5)
    p.add_argument("--surface_mode", type=str, default="default", choices=["default", "grippy"])
    p.add_argument("--car_wheel_command_limit", type=float, default=2.0)
    p.add_argument("--car_force_scale", type=float, default=2.0)
    p.add_argument("--car_action_mode", type=str, default="raw_wheels", choices=["raw_wheels", "throttle_turn", "cardinal"])
    p.add_argument("--point_action_mode", type=str, default="native", choices=["native", "world_velocity"])
    p.add_argument(
        "--skip_env_action_space",
        action="store_true",
        default=False,
        help="Infer action bounds without constructing Safety-Gym; useful for offline-only imitation jobs.",
    )
    p.add_argument("--max_rows", type=int, default=0)
    return p


def _split_indices(n_rows: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n_rows, dtype=np.int64)
    rng.shuffle(idx)
    n_val = int(round(n_rows * min(max(float(val_fraction), 0.0), 0.9)))
    if n_val <= 0 and n_rows > 10:
        n_val = max(1, n_rows // 10)
    return idx[n_val:], idx[:n_val]


def _expand_dataset_paths(raw_paths: list[str]) -> list[str]:
    out: list[str] = []
    for raw in raw_paths:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                out.append(part)
    if not out:
        raise ValueError("At least one --dataset_path is required.")
    return out


def _load_merged_datasets(paths: list[str]) -> dict:
    merged = None
    episode_offset = 0
    metadatas = []
    for path in paths:
        data = load_transition_dataset(path)
        metadatas.append({"path": str(Path(path).expanduser()), "metadata": data.get("metadata", {})})
        eps = np.asarray(data["episode_ids"], dtype=np.int64).reshape(-1)
        if eps.size > 0:
            eps = eps - int(eps.min()) + int(episode_offset)
            episode_offset = int(eps.max()) + 1
        data = dict(data)
        data["episode_ids"] = eps
        if merged is None:
            merged = data
            continue
        for key in (
            "observations",
            "actions",
            "next_observations",
            "rewards",
            "dones",
            "truncations",
            "student_actions",
            "teacher_intervened",
            "episode_ids",
            "episode_steps",
        ):
            merged[key] = np.concatenate([merged[key], data[key]], axis=0)
        if merged.get("costs") is not None and data.get("costs") is not None:
            merged["costs"] = np.concatenate([merged["costs"], data["costs"]], axis=0)
    assert merged is not None
    merged["metadata"] = {
        "source_datasets": metadatas,
        "env_name": (metadatas[0]["metadata"] or {}).get("env_name", ""),
    }
    return merged


def _metrics(
    *,
    model,
    x: torch.Tensor,
    y_action_unit: torch.Tensor,
    y_intervene: torch.Tensor,
    batch_size: int,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    total = 0
    action_sum = 0.0
    bce_sum = 0.0
    correct = 0
    tp = fp = fn = 0
    with torch.inference_mode():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            stop = min(int(x.shape[0]), start + int(batch_size))
            xb = x[start:stop]
            ya = y_action_unit[start:stop]
            yi = y_intervene[start:stop]
            pred_a, logits = model(xb)
            mask = yi > 0.5
            if bool(mask.any()):
                action_loss = F.mse_loss(pred_a[mask], ya[mask], reduction="sum")
                action_sum += float(action_loss.detach().cpu())
            bce = F.binary_cross_entropy_with_logits(logits, yi, reduction="sum")
            bce_sum += float(bce.detach().cpu())
            pred_i = torch.sigmoid(logits) >= float(threshold)
            true_i = yi > 0.5
            correct += int((pred_i == true_i).sum().detach().cpu())
            tp += int((pred_i & true_i).sum().detach().cpu())
            fp += int((pred_i & ~true_i).sum().detach().cpu())
            fn += int((~pred_i & true_i).sum().detach().cpu())
            total += int(stop - start)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "action_mse_intervened": float(action_sum / max(1, int((y_intervene > 0.5).sum().detach().cpu()))),
        "decision_bce": float(bce_sum / max(1, total)),
        "decision_accuracy": float(correct / max(1, total)),
        "decision_precision": float(precision),
        "decision_recall": float(recall),
    }


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    dataset_paths = _expand_dataset_paths(args.dataset_path)
    data = _load_merged_datasets(dataset_paths)
    metadata = dict(data.get("metadata") or {})
    env_name = str(args.env_name or metadata.get("env_name") or "SafetyCarGoal2-v0")

    observations = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    student_actions = np.asarray(data["student_actions"], dtype=np.float32)
    teacher_intervened = np.asarray(data["teacher_intervened"], dtype=np.bool_).reshape(-1)
    episode_ids = np.asarray(data["episode_ids"], dtype=np.int64).reshape(-1)
    if int(args.max_rows) > 0:
        limit = int(args.max_rows)
        observations = observations[:limit]
        actions = actions[:limit]
        student_actions = student_actions[:limit]
        teacher_intervened = teacher_intervened[:limit]
        episode_ids = episode_ids[:limit]

    if bool(args.skip_env_action_space):
        if str(args.car_action_mode).strip().lower() in {"throttle_turn", "cardinal"}:
            action_low = np.full((int(actions.shape[1]),), -1.0, dtype=np.float32)
            action_high = np.full((int(actions.shape[1]),), 1.0, dtype=np.float32)
        elif str(args.point_action_mode).strip().lower() == "world_velocity":
            action_low = np.full((int(actions.shape[1]),), -1.0, dtype=np.float32)
            action_high = np.full((int(actions.shape[1]),), 1.0, dtype=np.float32)
        else:
            raise ValueError(
                "--skip_env_action_space only has built-in bounds for throttle_turn, cardinal, "
                "and point world_velocity action modes"
            )
    else:
        env = make_safety_env(
            env_name,
            render_mode="none",
            surface_mode=str(args.surface_mode),
            car_wheel_command_limit=float(args.car_wheel_command_limit),
            car_force_scale=float(args.car_force_scale),
            car_action_mode=str(args.car_action_mode),
            point_action_mode=str(args.point_action_mode),
            seed=int(args.seed),
        )
        action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
        action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
        env.close()

    features = build_window_features(
        observations=observations,
        student_actions=student_actions,
        teacher_intervened=teacher_intervened,
        episode_ids=episode_ids,
        context_len=int(args.context_len),
    )
    targets_action = action_to_unit_np(actions, action_low, action_high)
    targets_intervene = teacher_intervened.astype(np.float32)

    obs_mean = torch.as_tensor(features.mean(axis=0, keepdims=True), device=device, dtype=torch.float32)
    obs_std = torch.as_tensor(features.std(axis=0, keepdims=True) + 1e-6, device=device, dtype=torch.float32)
    x_all = torch.as_tensor(features, device=device, dtype=torch.float32)
    y_action = torch.as_tensor(targets_action, device=device, dtype=torch.float32)
    y_intervene = torch.as_tensor(targets_intervene, device=device, dtype=torch.float32)

    train_idx, val_idx = _split_indices(int(features.shape[0]), float(args.val_fraction), int(args.seed))
    train_idx_t = torch.as_tensor(train_idx, device=device, dtype=torch.long)
    val_idx_t = torch.as_tensor(val_idx, device=device, dtype=torch.long)

    segment_dim = int(observations.shape[1] + actions.shape[1] + 1)
    model = make_imitation_policy(
        architecture=str(args.architecture),
        obs_dim=int(features.shape[1]),
        act_dim=int(actions.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        context_len=int(args.context_len),
        segment_dim=segment_dim,
        num_heads=int(args.num_heads),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    pos = float(teacher_intervened.sum())
    neg = float(teacher_intervened.shape[0] - teacher_intervened.sum())
    pos_weight = torch.as_tensor([neg / max(1.0, pos)], device=device, dtype=torch.float32)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    exp_name = str(args.exp_name or f"{env_name.replace('-', '_')}_human_imitation_{stamp}")
    output_dir = Path(args.output_dir).expanduser() / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_path = output_dir / "best.pt"
    final_path = output_dir / "final.pt"
    n_train = int(train_idx.shape[0])
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        perm = train_idx_t[torch.randperm(n_train, device=device)]
        total_loss = 0.0
        total_rows = 0
        for start in range(0, n_train, int(args.batch_size)):
            idx = perm[start : start + int(args.batch_size)]
            xb = normalize_observations(x_all[idx], obs_mean, obs_std)
            ya = y_action[idx]
            yi = y_intervene[idx]
            pred_a, logits = model(xb)
            weights = torch.where(
                yi > 0.5,
                torch.ones_like(yi),
                torch.full_like(yi, float(args.non_intervention_action_weight)),
            )
            action_per_row = ((pred_a - ya) ** 2).mean(dim=1)
            action_loss = (action_per_row * weights).sum() / torch.clamp(weights.sum(), min=1.0)
            decision_loss = F.binary_cross_entropy_with_logits(logits, yi, pos_weight=pos_weight)
            loss = float(args.action_loss_weight) * action_loss + float(args.decision_loss_weight) * decision_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            total_loss += float(loss.detach().cpu()) * int(idx.numel())
            total_rows += int(idx.numel())

        train_metrics = _metrics(
            model=model,
            x=normalize_observations(x_all[train_idx_t], obs_mean, obs_std),
            y_action_unit=y_action[train_idx_t],
            y_intervene=y_intervene[train_idx_t],
            batch_size=int(args.batch_size),
            threshold=float(args.intervention_threshold),
        )
        val_metrics = (
            _metrics(
                model=model,
                x=normalize_observations(x_all[val_idx_t], obs_mean, obs_std),
                y_action_unit=y_action[val_idx_t],
                y_intervene=y_intervene[val_idx_t],
                batch_size=int(args.batch_size),
                threshold=float(args.intervention_threshold),
            )
            if int(val_idx.shape[0]) > 0
            else train_metrics
        )
        val_score = float(val_metrics["decision_bce"] + val_metrics["action_mse_intervened"])
        log = {
            "epoch": int(epoch),
            "loss": float(total_loss / max(1, total_rows)),
            "train": train_metrics,
            "val": val_metrics,
            "rows": int(features.shape[0]),
            "intervened_rows": int(teacher_intervened.sum()),
            "non_intervened_rows": int((~teacher_intervened).sum()),
        }
        print(json.dumps(log, sort_keys=True), flush=True)
        ckpt_meta = {
            "env_name": env_name,
            "dataset_path": ",".join(str(Path(p).expanduser()) for p in dataset_paths),
            "dataset_paths": [str(Path(p).expanduser()) for p in dataset_paths],
            "obs_dim": int(features.shape[1]),
            "env_obs_dim": int(observations.shape[1]),
            "act_dim": int(actions.shape[1]),
            "context_len": int(args.context_len),
            "segment_dim": int(segment_dim),
            "architecture": str(args.architecture),
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
            "dropout": float(args.dropout),
            "intervention_threshold": float(args.intervention_threshold),
            "rows": int(features.shape[0]),
            "intervened_rows": int(teacher_intervened.sum()),
            "non_intervened_rows": int((~teacher_intervened).sum()),
        }
        if val_score < best_val:
            best_val = val_score
            save_imitation_checkpoint(
                path=best_path,
                model=model,
                optimizer=opt,
                obs_mean=obs_mean,
                obs_std=obs_std,
                action_low=action_low,
                action_high=action_high,
                metadata=ckpt_meta,
                step=epoch,
            )

    save_imitation_checkpoint(
        path=final_path,
        model=model,
        optimizer=opt,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_low=action_low,
        action_high=action_high,
        metadata=ckpt_meta,
        step=int(args.epochs),
    )
    print(json.dumps({"best_checkpoint": str(best_path), "final_checkpoint": str(final_path)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
