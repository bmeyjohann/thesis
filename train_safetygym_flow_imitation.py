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
from safetygym_utils.env import make_safety_env
from safetygym_utils.flow_imitation_teacher import (
    FlowInterventionImitationPolicy,
    save_flow_imitation_checkpoint,
    select_flow_sample,
)
from safetygym_utils.imitation_teacher import action_to_unit_np, build_window_features, normalize_observations


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Safety-Gym flow-matching intervention imitation teacher")
    p.add_argument("--dataset_path", action="append", required=True)
    p.add_argument("--env_name", type=str, default="")
    p.add_argument("--output_dir", type=str, default="models/safetygym_flow_imitation")
    p.add_argument("--exp_name", type=str, default="")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--train_steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--context_len", type=int, default=8)
    p.add_argument("--flow_loss_weight", type=float, default=1.0)
    p.add_argument("--decision_loss_weight", type=float, default=1.0)
    p.add_argument("--train_noise_scale", type=float, default=1.0)
    p.add_argument("--sample_noise_scale", type=float, default=1.0)
    p.add_argument("--sample_steps", type=int, default=12)
    p.add_argument("--eval_num_samples", type=int, default=8)
    p.add_argument("--eval_sample_selector", type=str, default="first", choices=["first", "mean", "max_norm", "max_abs", "saturated", "max_turn", "turn"])
    p.add_argument("--intervention_threshold", type=float, default=0.5)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--log_interval", type=int, default=500)
    p.add_argument("--save_interval", type=int, default=10000)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--surface_mode", type=str, default="default", choices=["default", "grippy"])
    p.add_argument("--car_wheel_command_limit", type=float, default=1.0)
    p.add_argument("--car_force_scale", type=float, default=1.0)
    p.add_argument("--car_action_mode", type=str, default="raw_wheels", choices=["raw_wheels", "throttle_turn", "cardinal"])
    p.add_argument("--point_action_mode", type=str, default="native", choices=["native", "world_velocity"])
    p.add_argument("--skip_env_action_space", action="store_true", default=False)
    return p


def _expand_dataset_paths(raw_paths: list[str]) -> list[str]:
    out: list[str] = []
    for raw in raw_paths:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                out.append(part)
    if not out:
        raise ValueError("At least one dataset path is required")
    return out


def _load_merged(paths: list[str]) -> dict:
    merged = None
    episode_offset = 0
    metadatas = []
    for path in paths:
        data = dict(load_transition_dataset(path))
        metadatas.append({"path": str(Path(path).expanduser()), "metadata": data.get("metadata", {})})
        eps = np.asarray(data["episode_ids"], dtype=np.int64).reshape(-1)
        if eps.size:
            eps = eps - int(eps.min()) + int(episode_offset)
            episode_offset = int(eps.max()) + 1
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
    assert merged is not None
    merged["metadata"] = {
        "source_datasets": metadatas,
        "env_name": (metadatas[0]["metadata"] or {}).get("env_name", ""),
    }
    return merged


def _split(n_rows: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(int(n_rows), dtype=np.int64)
    rng.shuffle(idx)
    n_val = int(round(int(n_rows) * min(max(float(val_fraction), 0.0), 0.9)))
    if n_val <= 0 and n_rows > 10:
        n_val = max(1, n_rows // 10)
    return idx[n_val:], idx[:n_val]


def _metrics(
    *,
    model: FlowInterventionImitationPolicy,
    x_norm: torch.Tensor,
    y_action: torch.Tensor,
    y_intervene: torch.Tensor,
    threshold: float,
    sample_steps: int,
    num_samples: int,
    sample_noise_scale: float,
    selector: str,
    max_eval_rows: int = 4096,
) -> dict[str, float]:
    model.eval()
    n = int(x_norm.shape[0])
    if n > int(max_eval_rows):
        idx = torch.linspace(0, n - 1, steps=int(max_eval_rows), device=x_norm.device).long()
        x_eval = x_norm[idx]
        ya = y_action[idx]
        yi = y_intervene[idx]
    else:
        x_eval = x_norm
        ya = y_action
        yi = y_intervene
    with torch.inference_mode():
        logits = model.gate_logit(x_eval)
        probs = torch.sigmoid(logits)
        pred_i = probs >= float(threshold)
        true_i = yi > 0.5
        tp = int((pred_i & true_i).sum().detach().cpu())
        fp = int((pred_i & ~true_i).sum().detach().cpu())
        fn = int((~pred_i & true_i).sum().detach().cpu())
        correct = int((pred_i == true_i).sum().detach().cpu())
        action_mse = float("nan")
        best_mse = float("nan")
        mode_match = float("nan")
        pred_far = float("nan")
        mask = true_i
        if bool(mask.any()):
            xs = x_eval[mask]
            ys = ya[mask]
            selected = []
            best_rows = []
            for row in range(int(xs.shape[0])):
                samples = model.sample_unit_actions(
                    xs[row],
                    sample_steps=int(sample_steps),
                    num_samples=int(num_samples),
                    noise_scale=float(sample_noise_scale),
                )
                selected.append(select_flow_sample(samples, selector))
                best_rows.append(torch.min(((samples - ys[row]) ** 2).mean(dim=-1)))
            pred = torch.stack(selected, dim=0)
            best = torch.stack(best_rows, dim=0)
            action_mse = float(((pred - ys) ** 2).mean().detach().cpu())
            best_mse = float(best.mean().detach().cpu())
            pred_np = pred.detach().cpu().numpy()
            y_np = ys.detach().cpu().numpy()
            hard_true = ((y_np[:, 0] > 0.8) & (y_np[:, 1] < -0.8)) | ((y_np[:, 0] < -0.8) & (y_np[:, 1] > 0.8))
            hard_pred = ((pred_np[:, 0] > 0.8) & (pred_np[:, 1] < -0.8)) | ((pred_np[:, 0] < -0.8) & (pred_np[:, 1] > 0.8))
            if hard_true.any():
                same_left = (y_np[:, 0] > 0.8) & (pred_np[:, 0] > 0.8) & (y_np[:, 1] < -0.8) & (pred_np[:, 1] < -0.8)
                same_right = (y_np[:, 0] < -0.8) & (pred_np[:, 0] < -0.8) & (y_np[:, 1] > 0.8) & (pred_np[:, 1] > 0.8)
                mode_match = float((same_left[hard_true] | same_right[hard_true]).mean())
            prototypes = np.asarray([[1, -1], [-1, 1], [1, 1], [-1, -1], [1, 0], [0, 1], [0, -1], [-1, 0]], dtype=np.float32)
            dist_pred = np.linalg.norm(pred_np[:, None, :] - prototypes[None, :, :], axis=2).min(axis=1)
            pred_far = float((dist_pred > 0.45).mean())
    return {
        "decision_accuracy": float(correct / max(1, int(x_eval.shape[0]))),
        "decision_precision": float(tp / max(1, tp + fp)),
        "decision_recall": float(tp / max(1, tp + fn)),
        "action_mse_selected": action_mse,
        "action_mse_best_of_samples": best_mse,
        "hard_turn_mode_match": mode_match,
        "pred_far_from_teacher_modes": pred_far,
    }


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    paths = _expand_dataset_paths(args.dataset_path)
    data = _load_merged(paths)
    metadata = dict(data.get("metadata") or {})
    env_name = str(args.env_name or metadata.get("env_name") or "SafetyCarGoal1-v0")

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
        action_low = np.full((int(actions.shape[1]),), -1.0, dtype=np.float32)
        action_high = np.full((int(actions.shape[1]),), 1.0, dtype=np.float32)
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
    x_norm_all = normalize_observations(x_all, obs_mean, obs_std)
    y_action = torch.as_tensor(targets_action, device=device, dtype=torch.float32)
    y_intervene = torch.as_tensor(targets_intervene, device=device, dtype=torch.float32)

    train_idx, val_idx = _split(int(features.shape[0]), float(args.val_fraction), int(args.seed))
    train_idx_t = torch.as_tensor(train_idx, device=device, dtype=torch.long)
    val_idx_t = torch.as_tensor(val_idx, device=device, dtype=torch.long)

    model = FlowInterventionImitationPolicy(
        obs_dim=int(features.shape[1]),
        act_dim=int(actions.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    pos = float(teacher_intervened.sum())
    neg = float(teacher_intervened.shape[0] - teacher_intervened.sum())
    pos_weight_value = 1.0 if neg <= 0.0 or pos <= 0.0 else neg / max(1.0, pos)
    pos_weight = torch.as_tensor([pos_weight_value], device=device, dtype=torch.float32)

    exp_name = str(args.exp_name or f"{env_name.replace('-', '_')}_flow_imitation_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir = Path(args.output_dir).expanduser() / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    final_path = output_dir / "final.pt"
    segment_dim = int(observations.shape[1] + actions.shape[1] + 1)
    ckpt_meta = {
        "env_name": env_name,
        "dataset_path": ",".join(str(Path(p).expanduser()) for p in paths),
        "dataset_paths": [str(Path(p).expanduser()) for p in paths],
        "obs_dim": int(features.shape[1]),
        "env_obs_dim": int(observations.shape[1]),
        "act_dim": int(actions.shape[1]),
        "context_len": int(args.context_len),
        "segment_dim": int(segment_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "intervention_threshold": float(args.intervention_threshold),
        "eval_sample_steps": int(args.sample_steps),
        "eval_num_samples": int(args.eval_num_samples),
        "eval_sample_noise_scale": float(args.sample_noise_scale),
        "eval_sample_selector": str(args.eval_sample_selector),
        "rows": int(features.shape[0]),
        "intervened_rows": int(teacher_intervened.sum()),
        "non_intervened_rows": int((~teacher_intervened).sum()),
    }

    best_score = float("inf")
    n_train = int(train_idx_t.numel())
    for step in range(1, int(args.train_steps) + 1):
        model.train()
        idx = train_idx_t[torch.randint(0, n_train, (int(args.batch_size),), device=device)]
        xb = x_norm_all[idx]
        yi = y_intervene[idx]
        logits = model.gate_logit(xb)
        decision_loss = F.binary_cross_entropy_with_logits(logits, yi, pos_weight=pos_weight)
        flow_loss = torch.zeros((), device=device)
        mask = yi > 0.5
        if bool(mask.any()):
            x1 = y_action[idx][mask]
            x0 = torch.randn_like(x1) * float(args.train_noise_scale)
            t = torch.rand((x1.shape[0], 1), device=device)
            xt = (1.0 - t) * x0 + t * x1
            target = x1 - x0
            pred = model.velocity(xb[mask], xt, t)
            flow_loss = F.mse_loss(pred, target)
        loss = float(args.decision_loss_weight) * decision_loss + float(args.flow_loss_weight) * flow_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.train_steps):
            train_metrics = _metrics(
                model=model,
                x_norm=x_norm_all[train_idx_t],
                y_action=y_action[train_idx_t],
                y_intervene=y_intervene[train_idx_t],
                threshold=float(args.intervention_threshold),
                sample_steps=int(args.sample_steps),
                num_samples=int(args.eval_num_samples),
                sample_noise_scale=float(args.sample_noise_scale),
                selector=str(args.eval_sample_selector),
            )
            val_metrics = _metrics(
                model=model,
                x_norm=x_norm_all[val_idx_t],
                y_action=y_action[val_idx_t],
                y_intervene=y_intervene[val_idx_t],
                threshold=float(args.intervention_threshold),
                sample_steps=int(args.sample_steps),
                num_samples=int(args.eval_num_samples),
                sample_noise_scale=float(args.sample_noise_scale),
                selector=str(args.eval_sample_selector),
            )
            log = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "decision_loss": float(decision_loss.detach().cpu()),
                "flow_loss": float(flow_loss.detach().cpu()),
                "train": train_metrics,
                "val": val_metrics,
                "rows": int(features.shape[0]),
                "intervened_rows": int(teacher_intervened.sum()),
                "non_intervened_rows": int((~teacher_intervened).sum()),
            }
            print(json.dumps(log, sort_keys=True), flush=True)
            score = float(val_metrics["action_mse_selected"] + (1.0 - val_metrics["decision_recall"]))
            if score < best_score:
                best_score = score
                save_flow_imitation_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=opt,
                    obs_mean=obs_mean,
                    obs_std=obs_std,
                    action_low=action_low,
                    action_high=action_high,
                    metadata=ckpt_meta,
                    step=int(step),
                )
        if int(args.save_interval) > 0 and step % int(args.save_interval) == 0:
            save_flow_imitation_checkpoint(
                path=output_dir / f"step_{step}.pt",
                model=model,
                optimizer=opt,
                obs_mean=obs_mean,
                obs_std=obs_std,
                action_low=action_low,
                action_high=action_high,
                metadata=ckpt_meta,
                step=int(step),
            )

    save_flow_imitation_checkpoint(
        path=final_path,
        model=model,
        optimizer=opt,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_low=action_low,
        action_high=action_high,
        metadata=ckpt_meta,
        step=int(args.train_steps),
    )
    print(json.dumps({"best_checkpoint": str(best_path), "final_checkpoint": str(final_path)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
