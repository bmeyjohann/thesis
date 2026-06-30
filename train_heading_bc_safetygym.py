#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from safetygym_utils.heading_policy import HeadingBCConfig, HeadingBCPolicy, save_heading_policy


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a Safety-Gym heading-level BC policy.")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--hidden_dims", type=str, default="512,512,256")
    p.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu", "elu"])
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--heading_tolerance", type=float, default=0.20)
    p.add_argument("--forward_throttle", type=float, default=0.8)
    return p


def _parse_hidden_dims(text: str) -> tuple[int, ...]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("--hidden_dims must contain at least one width")
    return tuple(vals)


def _angle_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dot = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(dot)


def main() -> int:
    args = build_parser().parse_args()
    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    with np.load(Path(args.dataset_path).expanduser(), allow_pickle=True) as data:
        obs = np.asarray(data["observations"], dtype=np.float32)
        target = np.asarray(data["target_sincos"], dtype=np.float32)
        metadata = json.loads(str(data["metadata"].item())) if "metadata" in data.files else {}
    if obs.ndim != 2 or target.ndim != 2 or target.shape[1] != 2 or obs.shape[0] != target.shape[0]:
        raise ValueError(f"Bad dataset shapes: observations={obs.shape}, target_sincos={target.shape}")

    target = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-6)
    n = int(obs.shape[0])
    indices = np.arange(n)
    rng.shuffle(indices)
    val_n = int(np.clip(round(n * float(args.val_fraction)), 1, max(1, n - 1)))
    val_idx = indices[:val_n]
    train_idx = indices[val_n:]

    obs_mean = obs[train_idx].mean(axis=0)
    obs_std = obs[train_idx].std(axis=0) + 1e-6
    train_ds = TensorDataset(torch.as_tensor(obs[train_idx]), torch.as_tensor(target[train_idx]))
    val_obs = torch.as_tensor(obs[val_idx], dtype=torch.float32, device=device)
    val_target = torch.as_tensor(target[val_idx], dtype=torch.float32, device=device)
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=False)

    cfg = HeadingBCConfig(
        obs_dim=int(obs.shape[1]),
        hidden_dims=_parse_hidden_dims(args.hidden_dims),
        activation=str(args.activation),
        obs_mean=tuple(float(x) for x in obs_mean.tolist()),
        obs_std=tuple(float(x) for x in obs_std.tolist()),
        heading_tolerance=float(args.heading_tolerance),
        forward_throttle=float(args.forward_throttle),
    )
    policy = HeadingBCPolicy(cfg).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    best_metric = float("inf")
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    history: list[dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        policy.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_obs, batch_target in train_loader:
            batch_obs = batch_obs.to(device=device, dtype=torch.float32)
            batch_target = batch_target.to(device=device, dtype=torch.float32)
            pred = policy(batch_obs)
            loss = (1.0 - (pred * batch_target).sum(dim=-1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            opt.step()
            train_loss_sum += float(loss.detach().cpu()) * int(batch_obs.shape[0])
            train_count += int(batch_obs.shape[0])

        policy.eval()
        with torch.no_grad():
            val_pred = policy(val_obs)
            val_loss = float((1.0 - (val_pred * val_target).sum(dim=-1)).mean().detach().cpu())
            val_angle = float(_angle_error(val_pred, val_target).mean().detach().cpu())
            val_within_20 = float((_angle_error(val_pred, val_target) <= 0.20).float().mean().detach().cpu())
        train_loss = float(train_loss_sum / max(1, train_count))
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_angle_rad": val_angle,
            "val_within_0p20": val_within_20,
        }
        history.append(row)
        if val_angle < best_metric:
            best_metric = val_angle
            save_heading_policy(best_path, policy, metadata={**metadata, "best_epoch": int(epoch), "best_val_angle_rad": float(best_metric)})
        if epoch == 1 or epoch % 10 == 0 or epoch == int(args.epochs):
            print(row, flush=True)

    save_heading_policy(last_path, policy, metadata={**metadata, "last_epoch": int(args.epochs), "best_val_angle_rad": float(best_metric)})
    (out_dir / "train_history.json").write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
    print({"best_path": str(best_path), "last_path": str(last_path), "best_val_angle_rad": float(best_metric)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
