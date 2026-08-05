"""Finite evaluation and video capture for native Unitree MJLab velocity tasks."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

if "--video" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import torch


ROOT = Path(__file__).resolve().parent
UNITREE_REPO = ROOT / "external" / "unitree_rl_mjlab"
if str(UNITREE_REPO) not in sys.path:
    sys.path.insert(0, str(UNITREE_REPO))


def _actor_linear_shapes(checkpoint_path: Path) -> tuple[int, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a dictionary checkpoint, got {type(checkpoint).__name__}")
    state = checkpoint.get("actor_state_dict", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a model state dictionary")

    layers: list[tuple[int, torch.Tensor]] = []
    pattern = re.compile(
        r"(?:^|\.)(?:actor(?:\.network)?|mlp|network)\.(\d+)\.weight$"
    )
    for key, value in state.items():
        match = pattern.search(str(key))
        if match and isinstance(value, torch.Tensor) and value.ndim == 2:
            layers.append((int(match.group(1)), value))
    if not layers:
        raise ValueError("Could not identify actor linear layers in checkpoint")
    layers.sort(key=lambda item: item[0])
    return int(layers[0][1].shape[1]), int(layers[-1][1].shape[0])


def _split_to_legacy_checkpoint(checkpoint: dict) -> dict:
    actor = checkpoint.get("actor_state_dict")
    critic = checkpoint.get("critic_state_dict")
    if not isinstance(actor, dict) or not isinstance(critic, dict):
        raise TypeError("Split checkpoint requires actor_state_dict and critic_state_dict")

    state: dict[str, torch.Tensor] = {}
    for key, value in actor.items():
        if key.startswith("mlp."):
            state["actor." + key.removeprefix("mlp.")] = value
        elif key.startswith("network."):
            state["actor." + key.removeprefix("network.")] = value
        elif key.startswith("obs_normalizer."):
            state["actor_obs_normalizer." + key.removeprefix("obs_normalizer.")] = value
        elif key == "distribution.std_param":
            state["std"] = value
        elif key == "distribution.log_std_param":
            state["log_std"] = value
    for key, value in critic.items():
        if key.startswith("mlp."):
            state["critic." + key.removeprefix("mlp.")] = value
        elif key.startswith("network."):
            state["critic." + key.removeprefix("network.")] = value
        elif key.startswith("obs_normalizer."):
            state["critic_obs_normalizer." + key.removeprefix("obs_normalizer.")] = value
    return {
        "model_state_dict": state,
        "optimizer_state_dict": checkpoint.get("optimizer_state_dict", {}),
        "iter": int(checkpoint.get("iter", 0)),
        "infos": checkpoint.get("infos"),
    }


@contextlib.contextmanager
def _runner_checkpoint_path(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "actor_state_dict" not in checkpoint:
        yield checkpoint_path
        return

    converted = _split_to_legacy_checkpoint(checkpoint)
    with tempfile.NamedTemporaryFile(prefix="unitree-rsl-compat-", suffix=".pt", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(converted, temporary_path)
        print(
            "[velocity-eval] checkpoint_format=split "
            "runner_adapter=temporary_legacy_state_dict"
        )
        yield temporary_path
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_robot_compatibility(task_id: str, checkpoint_path: Path) -> tuple[int, int]:
    obs_dim, action_dim = _actor_linear_shapes(checkpoint_path)
    if task_id.startswith("Unitree-G1-"):
        expected_action_dim = 29
        robot = "G1"
    elif task_id.startswith("Unitree-Go2-"):
        expected_action_dim = 12
        robot = "Go2"
    else:
        raise ValueError(
            f"Unsupported velocity task {task_id!r}; expected Unitree-G1-* or Unitree-Go2-*"
        )
    if action_dim != expected_action_dim:
        raise ValueError(
            f"{robot} task {task_id!r} expects {expected_action_dim} joint actions, "
            f"but {checkpoint_path} produces {action_dim}"
        )
    return obs_dim, action_dim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Unitree-G1-Rough")
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-observation-mode",
        choices=("auto", "task"),
        default="auto",
        help=(
            "auto removes rough-task height scans when loading a scan-blind checkpoint; "
            "task requires the checkpoint to match the task observation space exactly"
        ),
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-dir", type=Path, default=ROOT / "videos" / "unitree_velocity")
    parser.add_argument("--video-length", type=int, default=300)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--viewer", choices=("none", "native", "viser"), default="none")
    parser.add_argument(
        "--nconmax",
        type=int,
        default=256,
        help="MuJoCo Warp contact capacity; rough layouts can exceed the task default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = args.checkpoint_file.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if args.num_envs < 1 or args.steps < 1:
        raise ValueError("num_envs and steps must be positive")
    if args.video and args.num_envs != 1:
        raise ValueError("Video evaluation requires --num-envs 1")

    checkpoint_obs_dim, checkpoint_action_dim = _validate_robot_compatibility(
        args.task, checkpoint_path
    )
    print(
        "[velocity-eval] "
        f"task={args.task} checkpoint={checkpoint_path} "
        f"checkpoint_obs_dim={checkpoint_obs_dim} "
        f"checkpoint_action_dim={checkpoint_action_dim} "
        f"device={args.device} num_envs={args.num_envs} seed={args.seed}"
    )

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.utils.wrappers import VideoRecorder

    configure_torch_backends()
    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    scan_blind_actor_dims = {"G1": 98, "Go2": 47}
    robot = "G1" if args.task.startswith("Unitree-G1-") else "Go2"
    observation_profile = "task"
    if (
        args.checkpoint_observation_mode == "auto"
        and checkpoint_obs_dim == scan_blind_actor_dims[robot]
        and "height_scan" in env_cfg.observations["actor"].terms
    ):
        del env_cfg.observations["actor"].terms["height_scan"]
        del env_cfg.observations["critic"].terms["height_scan"]
        observation_profile = "checkpoint_blind_height_scan"
    print(f"[velocity-eval] observation_profile={observation_profile}")
    env_cfg.seed = int(args.seed)
    env_cfg.scene.num_envs = int(args.num_envs)
    if args.nconmax < 1:
        raise ValueError("nconmax must be positive")
    env_cfg.sim.nconmax = max(int(args.nconmax), int(env_cfg.sim.nconmax or 0))
    print(f"[velocity-eval] sim_nconmax={env_cfg.sim.nconmax}")
    env_cfg.viewer.width = int(args.video_width)
    env_cfg.viewer.height = int(args.video_height)
    terrain_generator = getattr(
        getattr(env_cfg.scene, "terrain", None), "terrain_generator", None
    )
    if terrain_generator is not None and hasattr(terrain_generator, "seed"):
        terrain_generator.seed = int(args.seed)

    render_mode = "rgb_array" if args.video else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=render_mode)
    if args.video:
        args.video_dir.mkdir(parents=True, exist_ok=True)
        env = VideoRecorder(
            env,
            video_folder=str(args.video_dir),
            step_trigger=lambda step: step == 0,
            video_length=int(args.video_length),
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    with _runner_checkpoint_path(checkpoint_path) as runner_checkpoint:
        runner.load(
            str(runner_checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
    policy = runner.get_inference_policy(device=args.device)

    if args.viewer != "none":
        if args.video:
            raise ValueError("--viewer and --video are mutually exclusive")
        if args.viewer == "native":
            from mjlab.viewer import NativeMujocoViewer

            NativeMujocoViewer(env, policy).run()
        else:
            from mjlab.viewer import ViserPlayViewer

            ViserPlayViewer(env, policy).run()
        env.close()
        return 0

    obs, _ = env.reset()
    reward_sum = torch.zeros(args.num_envs, device=args.device)
    action_abs_sum = 0.0
    action_samples = 0
    termination_count = 0
    nonfinite_action_count = 0
    steps_to_run = max(int(args.steps), int(args.video_length) + 10 if args.video else 0)
    for _ in range(steps_to_run):
        with torch.inference_mode():
            actions = policy(obs)
        nonfinite_action_count += int((~torch.isfinite(actions)).sum().item())
        action_abs_sum += float(actions.abs().mean().item())
        action_samples += 1
        obs, rewards, dones, _ = env.step(actions)
        reward_sum += rewards
        termination_count += int(dones.sum().item())

    summary = {
        "task": args.task,
        "checkpoint": str(checkpoint_path),
        "checkpoint_obs_dim": checkpoint_obs_dim,
        "checkpoint_action_dim": checkpoint_action_dim,
        "observation_profile": observation_profile,
        "num_envs": int(args.num_envs),
        "steps": steps_to_run,
        "seed": int(args.seed),
        "mean_step_reward": float((reward_sum / steps_to_run).mean().item()),
        "mean_abs_action": action_abs_sum / max(action_samples, 1),
        "termination_count": termination_count,
        "nonfinite_action_count": nonfinite_action_count,
        "video_dir": str(args.video_dir.resolve()) if args.video else None,
    }
    print("[velocity-eval] " + json.dumps(summary, sort_keys=True))
    if args.video:
        summary_path = args.video_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"[velocity-eval] summary={summary_path.resolve()}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
