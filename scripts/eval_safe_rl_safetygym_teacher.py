#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_policy(obs_dim: int, act_dim: int, cfg: dict[str, Any], device: torch.device):
    from safe_rl.modules import ActorCritic

    policy_cfg = dict(cfg["policy"])
    policy_cfg.pop("class_name", None)
    alg_cfg = cfg["algorithm"]
    cost_limits = alg_cfg.get("cost_limits") or [1.0]
    policy_cfg["num_costs"] = len(cost_limits)
    return ActorCritic(obs_dim, obs_dim, act_dim, **policy_cfg).to(device)


def _extract_goal_metrics(info: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in ("goal_met", "goal_dist", "cost", "num_steps"):
        if key in info:
            try:
                metrics[key] = float(info[key])
            except Exception:
                pass
    return metrics


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import safety_gymnasium

    device = torch.device(args.device)
    cfg = _load_yaml(args.config)

    env = safety_gymnasium.make(args.env_id, render_mode=args.render_mode)
    obs, info = env.reset(seed=args.seed)
    obs_dim = int(obs.shape[0])
    act_dim = int(env.action_space.shape[0])

    policy = _make_policy(obs_dim, act_dim, cfg, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    rows: list[dict[str, Any]] = []
    total_steps = 0
    try:
        for ep in range(args.episodes):
            if ep > 0:
                obs, info = env.reset(seed=None if args.seed is None else args.seed + ep)
            ep_reward = 0.0
            ep_cost = 0.0
            ep_len = 0
            goal_seen = False
            first_goal_step: int | None = None
            final_info: dict[str, Any] = {}
            terminated = False
            truncated = False

            while not (terminated or truncated):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.inference_mode():
                    action = policy.act_inference(obs_t).squeeze(0).detach().cpu().numpy()
                obs, reward, cost, terminated, truncated, info = env.step(action)
                if args.render_mode == "human" and args.fps > 0:
                    time.sleep(max(0.0, 1.0 / args.fps))
                ep_reward += float(reward)
                ep_cost += float(cost)
                ep_len += 1
                total_steps += 1
                final_info = info or {}
                if bool(final_info.get("goal_met", False)):
                    goal_seen = True
                    if first_goal_step is None:
                        first_goal_step = ep_len
                if args.max_steps and ep_len >= args.max_steps:
                    truncated = True

            row: dict[str, Any] = {
                "episode": ep,
                "reward": ep_reward,
                "cost": ep_cost,
                "length": ep_len,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "goal_met_any": bool(goal_seen),
                "first_goal_step": first_goal_step if first_goal_step is not None else 0,
            }
            row.update({f"final_{k}": v for k, v in _extract_goal_metrics(final_info).items()})
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        env.close()

    def mean(key: str) -> float:
        vals = [float(r[key]) for r in rows if key in r]
        return sum(vals) / max(len(vals), 1)

    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "env_id": args.env_id,
        "episodes": len(rows),
        "mean_reward": mean("reward"),
        "mean_cost": mean("cost"),
        "mean_length": mean("length"),
        "goal_success_rate": sum(1 for r in rows if r.get("goal_met_any")) / max(len(rows), 1),
        "mean_first_goal_step": mean("first_goal_step"),
        "total_steps": total_steps,
        "checkpoint_iter": checkpoint.get("iter"),
    }
    print("SUMMARY " + json.dumps(summary), flush=True)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.checkpoint).stem
        with open(out_dir / f"{stem}_episodes.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
            writer.writeheader()
            writer.writerows(rows)
        with open(out_dir / f"{stem}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Safe-RL Safety-Gymnasium teacher checkpoint in a single env.")
    parser.add_argument("--env_id", default="SafetyCarGoal1-v0")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render_mode", default=None, choices=[None, "human", "rgb_array"], nargs="?")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--out_dir", default="logs/safetygym_safe_rl_teacher_eval")
    args = parser.parse_args()
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
