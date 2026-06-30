#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from safetygym_utils.dataset_io import load_transition_dataset
from safetygym_utils.io import save_args_json
from safetygym_utils.maneuver_policy import (
    ManeuverBCConfig,
    ManeuverBCPolicy,
    maneuver_labels_from_actions,
    save_maneuver_policy,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a discrete maneuver BC policy for SafetyCar throttle-turn actions.")
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--exp_name", default="")
    p.add_argument("--output_dir", default="models/safetygym_maneuver_bc")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--hidden_dims", default="512,512,256")
    p.add_argument("--activation", default="tanh", choices=["tanh", "relu", "elu"])
    p.add_argument("--forward_turn_weight", type=float, default=0.25)
    p.add_argument("--forward_throttle", type=float, default=0.8)
    return p


def _split_indices(n_rows: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n_rows, dtype=np.int64)
    rng.shuffle(idx)
    n_val = int(round(n_rows * min(max(float(val_fraction), 0.0), 0.9)))
    return idx[n_val:], idx[:n_val]


def _metrics(policy: ManeuverBCPolicy, obs: torch.Tensor, labels: torch.Tensor, actions: torch.Tensor, batch_size: int) -> dict[str, float]:
    policy.eval()
    total = 0
    correct = 0
    ce_sum = 0.0
    turn_sum = 0.0
    turn_rows = 0
    with torch.inference_mode():
        for start in range(0, int(obs.shape[0]), int(batch_size)):
            stop = min(int(obs.shape[0]), start + int(batch_size))
            logits, pred_turn = policy(obs[start:stop])
            y = labels[start:stop]
            ce = F.cross_entropy(logits, y, reduction="sum")
            ce_sum += float(ce.detach().cpu())
            pred = torch.argmax(logits, dim=-1)
            correct += int((pred == y).sum().detach().cpu())
            total += int(stop - start)
            mask = y == 2
            if bool(mask.any()):
                target_turn = actions[start:stop, 1]
                loss = F.mse_loss(pred_turn[mask], target_turn[mask], reduction="sum")
                turn_sum += float(loss.detach().cpu())
                turn_rows += int(mask.sum().detach().cpu())
    return {
        "class_acc": float(correct / max(1, total)),
        "class_ce": float(ce_sum / max(1, total)),
        "forward_turn_mse": float(turn_sum / max(1, turn_rows)),
    }


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    data = load_transition_dataset(args.dataset_path)
    obs = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    if int(args.max_rows) > 0:
        obs = obs[: int(args.max_rows)]
        actions = actions[: int(args.max_rows)]
    labels = maneuver_labels_from_actions(actions)

    exp_name = str(args.exp_name or f"maneuver_bc_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir = Path(args.output_dir).expanduser() / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_args_json(out_dir / "args.json", vars(args))

    obs_mean = obs.mean(axis=0).astype(np.float32)
    obs_std = np.maximum(obs.std(axis=0).astype(np.float32), 1e-6)
    hidden_dims = tuple(int(x.strip()) for x in str(args.hidden_dims).split(",") if x.strip())
    policy = ManeuverBCPolicy(
        ManeuverBCConfig(
            obs_dim=int(obs.shape[1]),
            hidden_dims=hidden_dims,
            activation=str(args.activation),
            obs_mean=tuple(float(x) for x in obs_mean.tolist()),
            obs_std=tuple(float(x) for x in obs_std.tolist()),
            forward_throttle=float(args.forward_throttle),
        )
    ).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
    act_t = torch.as_tensor(actions, device=device, dtype=torch.float32)
    label_t = torch.as_tensor(labels, device=device, dtype=torch.long)
    train_idx, val_idx = _split_indices(int(obs.shape[0]), float(args.val_fraction), int(args.seed))
    train_idx_t = torch.as_tensor(train_idx, device=device, dtype=torch.long)
    val_idx_t = torch.as_tensor(val_idx, device=device, dtype=torch.long)

    class_counts = np.bincount(labels, minlength=3).astype(np.float64)
    class_weights = class_counts.sum() / np.maximum(1.0, 3.0 * class_counts)
    class_weights_t = torch.as_tensor(class_weights, device=device, dtype=torch.float32)

    best_score = -float("inf")
    best_path = out_dir / "best.pt"
    n_train = int(train_idx_t.numel())
    for epoch in range(1, int(args.epochs) + 1):
        perm = train_idx_t[torch.randperm(n_train, device=device)]
        policy.train()
        total_loss = 0.0
        rows = 0
        for start in range(0, n_train, int(args.batch_size)):
            idx = perm[start : start + int(args.batch_size)]
            logits, pred_turn = policy(obs_t[idx])
            y = label_t[idx]
            ce = F.cross_entropy(logits, y, weight=class_weights_t)
            mask = y == 2
            turn_loss = (
                F.mse_loss(pred_turn[mask], act_t[idx][mask, 1])
                if bool(mask.any())
                else torch.zeros((), device=device, dtype=torch.float32)
            )
            loss = ce + float(args.forward_turn_weight) * turn_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt.step()
            total_loss += float(loss.detach().cpu()) * int(idx.numel())
            rows += int(idx.numel())
        train = _metrics(policy, obs_t[train_idx_t], label_t[train_idx_t], act_t[train_idx_t], int(args.batch_size))
        val = _metrics(policy, obs_t[val_idx_t], label_t[val_idx_t], act_t[val_idx_t], int(args.batch_size))
        score = float(val["class_acc"] - 0.05 * val["forward_turn_mse"])
        record = {
            "epoch": epoch,
            "rows": int(obs.shape[0]),
            "train_loss": float(total_loss / max(1, rows)),
            "train_class_acc": train["class_acc"],
            "train_forward_turn_mse": train["forward_turn_mse"],
            "val_class_acc": val["class_acc"],
            "val_class_ce": val["class_ce"],
            "val_forward_turn_mse": val["forward_turn_mse"],
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        if score > best_score:
            best_score = score
            save_maneuver_policy(best_path, policy, metadata={"dataset_path": str(args.dataset_path), "epoch": epoch, "metrics": record})
    save_maneuver_policy(out_dir / "final.pt", policy, metadata={"dataset_path": str(args.dataset_path)})
    (out_dir / "summary.json").write_text(
        json.dumps({"best_score": best_score, "rows": int(obs.shape[0]), "class_counts": class_counts.astype(int).tolist()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
