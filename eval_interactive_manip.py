#!/usr/bin/env python3
"""
Standalone interactive evaluator for OGBench manipulation tasks.

Focus:
- State observations only (fast debugging path).
- Random/keyboard controllers.
- Intervention diagnostics (teacher candidate availability, l2/angle deltas, reasons).

Example:
  python eval_interactive_manip.py \
    --env_name cube-double-v0 \
    --controller random \
    --intervention_mode agent \
    --teacher_type cube_plan \
    --tolerance_type l2 \
    --tolerance_value 0.02
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pygame
import torch

if "fasttd3/fast_sac" not in sys.path:
    sys.path.append("fasttd3/fast_sac")
from fast_sac_utils import EmpiricalNormalization

from ogbench_utils import GaussianPolicyHead, MLPBackbone, prepare_observation
from ogbench_utils.env_wrappers_manip import (
    CubeRewardModeTracker,
    build_ogbench_manip_wrapper,
    canonicalize_cube_reward_mode,
    cube_reward_mode_active,
)


def _yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    """Convert quaternion (w, x, y, z) to yaw (z-axis Euler angle)."""
    w, x, y, z = [float(v) for v in quat_wxyz]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _reward_dot_color(value: float, scale: float) -> tuple[int, int, int]:
    """Map signed reward value to red->green color."""
    s = max(1e-6, float(scale))
    x = math.tanh(float(value) / s)  # [-1, +1]
    t = 0.5 * (x + 1.0)  # [0, 1]
    red = int(max(0.0, min(255.0, 255.0 * (1.0 - t))))
    green = int(max(0.0, min(255.0, 255.0 * t)))
    return red, green, 40


def _reward_panel_values(info: Optional[dict], reward_components: Optional[dict], final_reward: float) -> dict[str, float]:
    """Canonical reward channels for compact UI display."""
    sparse_final = _component_scalar(
        reward_components,
        "success_sparse",
        _info_scalar(info, "sparse_reward", 0.0),
    )
    sparse_step = _component_scalar(
        reward_components,
        "solved_delta_sparse",
        0.0,
    ) + _component_scalar(reward_components, "drop_penalty", 0.0)
    dense = _component_scalar(
        reward_components,
        "dense_phase_reward",
        _info_scalar(info, "dense_reward", 0.0),
    )
    selected = _component_scalar(
        reward_components,
        "mode_total_pre_intervention",
        float(final_reward),
    )
    return {
        "sparse_final": float(sparse_final),
        "sparse_step": float(sparse_step),
        "dense": float(dense),
        "selected": float(selected),
    }


class CubeTeacherInfoAdapter(gym.Wrapper):
    """
    Inject target-block fields expected by cube teacher oracles.

    Some cube env variants expose block state in info but omit:
    - privileged/target_block
    - privileged/target_block_pos
    - privileged/target_block_yaw
    """

    def __init__(self, env: gym.Env, *, target_mode: str = "sequential", success_tolerance: float = 0.04):
        super().__init__(env)
        if target_mode not in {"fixed", "sequential"}:
            raise ValueError(f"Unknown target_mode={target_mode}")
        self.target_mode = target_mode
        self.success_tolerance = float(success_tolerance)

    def _cube_target_errors(self, out: dict, unwrapped) -> np.ndarray:
        num_cubes = int(getattr(unwrapped, "_num_cubes", 0))
        if num_cubes <= 0:
            return np.zeros(0, dtype=np.float32)
        errs = []
        for i in range(num_cubes):
            try:
                obj = np.asarray(out[f"privileged/block_{i}_pos"], dtype=np.float32)
            except Exception:
                obj = np.asarray(unwrapped._data.joint(f"object_joint_{i}").qpos[:3], dtype=np.float32)
            try:
                mocap_id = int(unwrapped._cube_target_mocap_ids[i])
                tar = np.asarray(unwrapped._data.mocap_pos[mocap_id], dtype=np.float32)
            except Exception:
                tar = obj
            errs.append(float(np.linalg.norm(obj - tar)))
        return np.asarray(errs, dtype=np.float32)

    def _select_target_block(self, out: dict, unwrapped, errs: np.ndarray) -> int:
        base_target = int(getattr(unwrapped, "_target_block", 0))
        if self.target_mode != "sequential" or errs.size == 0:
            return base_target
        unresolved = np.where(errs > self.success_tolerance)[0]
        if unresolved.size == 0:
            return base_target
        return int(unresolved[0])

    def _augment_info(self, info):
        if not isinstance(info, dict):
            return info
        out = dict(info)
        unwrapped = self.unwrapped
        errs = self._cube_target_errors(out, unwrapped)
        out["diag/cube_target_errors"] = errs
        out["diag/cubes_solved"] = int(np.sum(errs <= self.success_tolerance)) if errs.size else 0
        out["diag/cube_max_target_error"] = float(np.max(errs)) if errs.size else 0.0

        target_block = self._select_target_block(out, unwrapped, errs)
        out["privileged/target_block"] = int(target_block)
        out["diag/target_block_dynamic"] = int(target_block)

        try:
            target_idx = int(out["privileged/target_block"])
        except Exception:
            return out

        if target_idx < 0:
            return out

        try:
            mocap_pos = np.asarray(unwrapped._data.mocap_pos, dtype=np.float32)
            if target_idx < mocap_pos.shape[0] and "privileged/target_block_pos" not in out:
                out["privileged/target_block_pos"] = mocap_pos[target_idx].copy()
        except Exception:
            pass

        try:
            mocap_quat = np.asarray(unwrapped._data.mocap_quat, dtype=np.float32)
            if target_idx < mocap_quat.shape[0] and "privileged/target_block_yaw" not in out:
                yaw = _yaw_from_quat_wxyz(mocap_quat[target_idx])
                out["privileged/target_block_yaw"] = np.asarray([yaw], dtype=np.float32)
        except Exception:
            pass

        return out

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._augment_info(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._augment_info(info)


@dataclass
class StepDiagnostics:
    candidate: bool
    intervened: bool
    reason: Optional[str]
    delta_l2: float
    delta_l2_raw: float
    delta_angle_deg: float
    tolerance: float
    target_block: int
    cubes_solved: int
    cube_max_error: float
    target_error: float
    within_success_tol: bool
    goal_reached: bool


class RandomController:
    def __init__(self, action_space: gym.spaces.Box):
        self.action_space = action_space

    def action(self) -> np.ndarray:
        return np.asarray(self.action_space.sample(), dtype=np.float32)

    def close(self) -> None:
        return None


class IdleController:
    """Always output zero action."""

    def __init__(self, action_dim: int):
        self.action_dim = int(action_dim)

    def action(self) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)

    def close(self) -> None:
        return None


class KeyboardController:
    """
    Keyboard mapping for 5D manip action spaces:
    - a/d: x
    - w/s: y
    - q/e: z
    - z/c: wrist yaw
    - r/t: gripper close/open
    """

    def __init__(self, action_dim: int, magnitude: float = 1.0):
        self.action_dim = int(action_dim)
        self.magnitude = float(magnitude)
        pygame.init()
        self._screen = pygame.display.set_mode((760, 220))
        pygame.display.set_caption("Manip Keyboard Controls")
        self._font = pygame.font.SysFont("Arial", 18)
        self._intervention_active = False
        self._fps = 20.0
        self._reward_values = {
            "sparse_final": 0.0,
            "sparse_step": 0.0,
            "dense": 0.0,
            "selected": 0.0,
        }
        self._reward_scales = {
            "sparse_final": 1.0,
            "sparse_step": 1.0,
            "dense": 0.1,
            "selected": 0.5,
        }

    def set_status(
        self,
        *,
        intervention_active: bool,
        fps: float,
        rewards: Optional[dict[str, float]] = None,
    ) -> None:
        self._intervention_active = bool(intervention_active)
        self._fps = max(1.0, float(fps))
        if isinstance(rewards, dict):
            for key in self._reward_values.keys():
                if key in rewards:
                    val = float(rewards[key])
                    self._reward_values[key] = val
                    self._reward_scales[key] = max(self._reward_scales[key], abs(val), 1e-3)

    def _draw_help(self) -> None:
        bg = (120, 20, 20) if self._intervention_active else (20, 20, 20)
        self._screen.fill(bg)
        lines = [
            "Controls: WASD+Q/E=move, Z/C=yaw, R/T=gripper, ENTER/.=step, LEFT=prev episode, RIGHT=next episode, ESC=quit",
            "Speed: UP=faster, DOWN=slower",
            "Focus this control window for keyboard input. Env view is in native MuJoCo window.",
        ]
        y = 24
        for line in lines:
            surf = self._font.render(line, True, (230, 230, 230))
            self._screen.blit(surf, (10, y))
            y += 30
        status = self._font.render(
            f"FPS: {self._fps:.1f} | Teacher: {'ACTIVE' if self._intervention_active else 'idle'}",
            True,
            (230, 230, 230),
        )
        self._screen.blit(status, (10, 118))

        dots = [
            ("Sparse(final)", "sparse_final"),
            ("Sparse(step)", "sparse_step"),
            ("Dense", "dense"),
            ("Selected", "selected"),
        ]
        x = 16
        y_dot = 170
        for label, key in dots:
            value = self._reward_values[key]
            color = _reward_dot_color(value, self._reward_scales[key])
            pygame.draw.circle(self._screen, color, (x, y_dot), 10)
            text = self._font.render(f"{label}: {value:+.3f}", True, (230, 230, 230))
            self._screen.blit(text, (x + 16, y_dot - 10))
            x += 180
        pygame.display.flip()

    def action(self) -> tuple[np.ndarray, bool, bool, bool, bool, float]:
        quit_requested = False
        prev_requested = False
        next_requested = False
        advance_requested = False
        fps_delta = 0.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    quit_requested = True
                if event.key == pygame.K_RIGHT:
                    next_requested = True
                if event.key == pygame.K_LEFT:
                    prev_requested = True
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_PERIOD):
                    advance_requested = True
                if event.key == pygame.K_DOWN:
                    fps_delta -= 2.0
                if event.key == pygame.K_UP:
                    fps_delta += 2.0

        keys = pygame.key.get_pressed()
        a = np.zeros(self.action_dim, dtype=np.float32)
        mag = self.magnitude

        if self.action_dim >= 1:
            if keys[pygame.K_a]:
                a[0] -= mag
            if keys[pygame.K_d]:
                a[0] += mag
        if self.action_dim >= 2:
            if keys[pygame.K_w]:
                a[1] += mag
            if keys[pygame.K_s]:
                a[1] -= mag
        if self.action_dim >= 3:
            if keys[pygame.K_q]:
                a[2] += mag
            if keys[pygame.K_e]:
                a[2] -= mag
        if self.action_dim >= 4:
            if keys[pygame.K_z]:
                a[3] -= mag
            if keys[pygame.K_c]:
                a[3] += mag
        if self.action_dim >= 5:
            if keys[pygame.K_r]:
                a[4] += mag
            if keys[pygame.K_t]:
                a[4] -= mag

        self._draw_help()
        return a, prev_requested, next_requested, quit_requested, advance_requested, fps_delta

    def close(self) -> None:
        pygame.quit()


class EpisodeControlPanel:
    """Small control window for non-keyboard controllers (policy/random/idle)."""

    def __init__(self):
        pygame.init()
        self._screen = pygame.display.set_mode((760, 190))
        pygame.display.set_caption("Manip Eval Controls")
        self._font = pygame.font.SysFont("Arial", 18)
        self._intervention_active = False
        self._fps = 20.0
        self._reward_values = {
            "sparse_final": 0.0,
            "sparse_step": 0.0,
            "dense": 0.0,
            "selected": 0.0,
        }
        self._reward_scales = {
            "sparse_final": 1.0,
            "sparse_step": 1.0,
            "dense": 0.1,
            "selected": 0.5,
        }
        self._draw()

    def _draw(self) -> None:
        bg = (120, 20, 20) if self._intervention_active else (20, 20, 20)
        self._screen.fill(bg)
        lines = [
            "Controls: ENTER/.=step, LEFT=previous episode, RIGHT=next episode, ESC/Q=quit",
            "Speed: UP=faster, DOWN=slower",
            (
                "Teacher ACTIVE (red) - intervention engaged."
                if self._intervention_active
                else "Teacher idle (dark) - no intervention."
            ),
        ]
        y = 24
        for line in lines:
            surf = self._font.render(line, True, (230, 230, 230))
            self._screen.blit(surf, (10, y))
            y += 30

        status = self._font.render(f"FPS: {self._fps:.1f}", True, (230, 230, 230))
        self._screen.blit(status, (10, 108))

        dots = [
            ("Sparse(final)", "sparse_final"),
            ("Sparse(step)", "sparse_step"),
            ("Dense", "dense"),
            ("Selected", "selected"),
        ]
        x = 16
        y_dot = 150
        for label, key in dots:
            value = self._reward_values[key]
            color = _reward_dot_color(value, self._reward_scales[key])
            pygame.draw.circle(self._screen, color, (x, y_dot), 10)
            text = self._font.render(f"{label}: {value:+.3f}", True, (230, 230, 230))
            self._screen.blit(text, (x + 16, y_dot - 10))
            x += 180
        pygame.display.flip()

    def set_status(self, *, active: bool, fps: float, rewards: Optional[dict[str, float]] = None) -> None:
        self._intervention_active = bool(active)
        self._fps = max(1.0, float(fps))
        if isinstance(rewards, dict):
            for key in self._reward_values.keys():
                if key in rewards:
                    val = float(rewards[key])
                    self._reward_values[key] = val
                    self._reward_scales[key] = max(self._reward_scales[key], abs(val), 1e-3)
        self._draw()

    def set_intervention_active(self, active: bool) -> None:
        """Compatibility helper used by the eval loop."""
        self._intervention_active = bool(active)
        self._draw()

    def poll(self) -> tuple[bool, bool, bool, bool, float]:
        prev_requested = False
        next_requested = False
        quit_requested = False
        advance_requested = False
        fps_delta = 0.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    quit_requested = True
                elif event.key == pygame.K_RIGHT:
                    next_requested = True
                elif event.key == pygame.K_LEFT:
                    prev_requested = True
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_PERIOD):
                    advance_requested = True
                elif event.key == pygame.K_DOWN:
                    fps_delta -= 2.0
                elif event.key == pygame.K_UP:
                    fps_delta += 2.0
        return prev_requested, next_requested, quit_requested, advance_requested, fps_delta

    def close(self) -> None:
        pygame.quit()


class NativeViewer:
    """MuJoCo passive viewer wrapper for manip environments."""

    def __init__(self):
        self.enabled = False

    def maybe_launch(self, env: gym.Env) -> None:
        base = env.unwrapped
        launch_fn = getattr(base, "launch_passive_viewer", None)
        if not callable(launch_fn):
            return
        try:
            launch_fn(show_left_ui=False, show_right_ui=False)
            self.enabled = True
            print("native viewer: launched")
        except Exception as exc:
            self.enabled = False
            print(f"native viewer warning: launch failed: {exc}")

    def sync(self, env: gym.Env) -> None:
        if not self.enabled:
            return
        base = env.unwrapped
        sync_fn = getattr(base, "sync_passive_viewer", None)
        if not callable(sync_fn):
            return
        try:
            sync_fn()
        except Exception as exc:
            self.enabled = False
            print(f"native viewer warning: sync failed, disabling viewer: {exc}")

    def close(self, env: gym.Env) -> None:
        base = env.unwrapped
        close_fn = getattr(base, "close_passive_viewer", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass


def _pump_pygame_events() -> None:
    """Keep SDL windows responsive even while waiting between env steps."""
    try:
        if pygame.get_init():
            pygame.event.pump()
    except Exception:
        pass


def clip_action(
    action: np.ndarray,
    *,
    clip_l2: bool = True,
    binary_gripper_actions: bool = False,
    binary_gripper_threshold: float = 0.0,
) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    out = np.clip(out, -1.0, 1.0)
    if clip_l2:
        norm = float(np.linalg.norm(out))
        if norm > 1.0 and norm > 1e-8:
            out = out / norm
    if binary_gripper_actions and out.shape[-1] >= 5:
        out[4] = 1.0 if out[4] >= float(binary_gripper_threshold) else -1.0
    return out


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _build_cube_reward_tracker(args: argparse.Namespace) -> Optional[CubeRewardModeTracker]:
    mode = canonicalize_cube_reward_mode(str(getattr(args, "cube_reward_mode", "dense")))
    args.cube_reward_mode = mode
    if not cube_reward_mode_active(env_name=args.env_name, obs_mode=args.obs_mode, reward_mode=mode):
        print(
            f"warning: cube_reward_mode={mode} only applies to cube state environments; "
            "disabling cube reward tracker."
        )
        return None
    return CubeRewardModeTracker(
        mode=mode,
        num_envs=1,
        device=torch.device("cpu"),
        success_reward=float(args.cube_success_reward),
        grasp_reward=float(args.cube_subgoal_grasp_reward),
        place_reward=float(args.cube_subgoal_place_reward),
        drop_penalty=float(args.cube_subgoal_drop_penalty),
        grasp_error_threshold=float(args.cube_subgoal_grasp_error_threshold),
        progress_scale=float(args.cube_dense_progress_scale),
        progress_clip=float(args.cube_dense_progress_clip),
    )


def _coerce_args_dict(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return {k: v for k, v in vars(raw).items() if not k.startswith("_")}
    except TypeError:
        return {}


def _load_model_config_dict(model_path: Optional[str]) -> tuple[Optional[Path], dict[str, Any]]:
    if not model_path:
        return None, {}
    resolved = Path(model_path).expanduser().resolve()
    candidates = [resolved.parent / "args.json"]
    for cand in candidates:
        if cand.is_file():
            try:
                with cand.open("r", encoding="utf-8") as fp:
                    return cand, json.load(fp)
            except Exception as exc:
                print(f"warning: failed to read config {cand}: {exc}")
    return None, {}


def _load_checkpoint_args_dict(model_path: Optional[str]) -> dict[str, Any]:
    if not model_path:
        return {}
    try:
        ckpt = torch.load(str(Path(model_path).expanduser()), map_location="cpu", weights_only=False)
    except Exception:
        return {}
    return _coerce_args_dict(ckpt.get("args"))


_CLI_FLAG_ALIASES: dict[str, tuple[str, ...]] = {
    "env_name": ("--env_name",),
    "obs_mode": ("--obs_mode",),
    "include_goal": ("--include_goal",),
    "include_distance": ("--include_distance",),
    "include_direction": ("--include_direction",),
    "include_velocity": ("--include_velocity",),
    "include_relative_cube_features": ("--include_relative_cube_features",),
    "reward_type": ("--reward_type",),
    "dense_reward_scale": ("--dense_reward_scale",),
    "step_penalty": ("--step_penalty",),
    "cube_reward_mode": ("--cube_reward_mode",),
    "cube_success_reward": ("--cube_success_reward",),
    "cube_subgoal_grasp_reward": ("--cube_subgoal_grasp_reward",),
    "cube_subgoal_place_reward": ("--cube_subgoal_place_reward",),
    "cube_subgoal_drop_penalty": ("--cube_subgoal_drop_penalty",),
    "cube_subgoal_grasp_error_threshold": ("--cube_subgoal_grasp_error_threshold",),
    "cube_dense_progress_scale": ("--cube_dense_progress_scale",),
    "cube_dense_progress_clip": ("--cube_dense_progress_clip",),
    "teacher_type": ("--teacher_type",),
    "tolerance_type": ("--tolerance_type",),
    "tolerance_value": ("--tolerance_value",),
    "tolerance_channel_weights": ("--tolerance_channel_weights",),
    "binary_gripper_actions": ("--binary_gripper_actions",),
    "binary_gripper_threshold": ("--binary_gripper_threshold",),
    "hard_gripper_intervention": ("--hard_gripper_intervention",),
    "gripper_intervene_pick_radius": ("--gripper_intervene_pick_radius",),
    "gripper_intervene_place_radius": ("--gripper_intervene_place_radius",),
    "gripper_intervene_contact_threshold": ("--gripper_intervene_contact_threshold",),
    "intervention_mode": ("--intervention_mode",),
    "intervention_enable_after_steps": ("--intervention_enable_after_steps",),
    "hard_block_lethal": ("--hard_block_lethal",),
    "teacher_target_mode": ("--teacher_target_mode",),
    "cube_success_tolerance": ("--cube_success_tolerance",),
    "max_episode_steps": ("--max_episode_steps",),
    "static_reset_seed": ("--static_reset_seed",),
}


def _cli_flag_provided(flags: tuple[str, ...]) -> bool:
    argv = sys.argv[1:]
    for flag in flags:
        if flag in argv:
            return True
        prefix = flag + "="
        for arg in argv:
            if arg.startswith(prefix):
                return True
    return False


def apply_model_config_defaults(args: argparse.Namespace) -> None:
    config_path, config = _load_model_config_dict(getattr(args, "model_path", None))
    source = None
    if config:
        source = str(config_path)
    else:
        config = _load_checkpoint_args_dict(getattr(args, "model_path", None))
        if config:
            source = f"{args.model_path}::checkpoint.args"
    if not config:
        return
    keys = [
        "env_name",
        "obs_mode",
        "include_goal",
        "include_distance",
        "include_direction",
        "include_velocity",
        "include_relative_cube_features",
        "reward_type",
        "dense_reward_scale",
        "step_penalty",
        "cube_reward_mode",
        "cube_success_reward",
        "cube_subgoal_grasp_reward",
        "cube_subgoal_place_reward",
        "cube_subgoal_drop_penalty",
        "cube_subgoal_grasp_error_threshold",
        "cube_dense_progress_scale",
        "cube_dense_progress_clip",
        "teacher_type",
        "tolerance_type",
        "tolerance_value",
        "tolerance_channel_weights",
        "binary_gripper_actions",
        "binary_gripper_threshold",
        "hard_gripper_intervention",
        "gripper_intervene_pick_radius",
        "gripper_intervene_place_radius",
        "gripper_intervene_contact_threshold",
        "intervention_enable_after_steps",
        "hard_block_lethal",
        "teacher_target_mode",
        "cube_success_tolerance",
        "max_episode_steps",
        "static_reset_seed",
    ]
    applied = []
    for key in keys:
        if not hasattr(args, key) or key not in config:
            continue
        flags = _CLI_FLAG_ALIASES.get(key)
        if flags and _cli_flag_provided(flags):
            continue
        setattr(args, key, config[key])
        applied.append(key)
    if applied:
        applied_str = ", ".join(applied)
        print(f"loaded run config from {source} (applied: {applied_str})")


class FastSACPolicyController:
    def __init__(self, *, model_path: str, env: gym.Env, device: torch.device):
        self.model_path = str(model_path)
        self.device = device
        self.obs_normalizer: Optional[EmpiricalNormalization] = None
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        train_args = _coerce_args_dict(checkpoint.get("args"))
        obs_space = env.observation_space
        if not hasattr(obs_space, "shape") or obs_space.shape is None:
            raise TypeError(f"Expected Box observation space for state mode, got {type(obs_space)}")
        obs_dim = int(np.prod(obs_space.shape))
        act_dim = int(np.prod(env.action_space.shape))

        actor_hidden = int(train_args.get("actor_hidden_dim", 512))
        shared_hidden = int(train_args.get("shared_hidden_dim", actor_hidden))
        arch_shared = bool(train_args.get("arch_shared_trunk", False))
        use_layer_norm = bool(train_args.get("use_layer_norm", False))
        layer_norm_eps = float(train_args.get("layer_norm_eps", 1e-5))
        feature_dim = shared_hidden if arch_shared else actor_hidden
        init_scale = float(train_args.get("init_scale", 0.01))

        self.backbone = MLPBackbone(
            obs_dim,
            feature_dim,
            use_layer_norm=use_layer_norm,
            layer_norm_eps=layer_norm_eps,
        ).to(device)
        self.actor_head = GaussianPolicyHead(
            self.backbone.output_dim,
            act_dim,
            actor_hidden,
            init_scale,
            use_layer_norm=use_layer_norm,
            layer_norm_eps=layer_norm_eps,
        ).to(device)
        self.backbone.load_state_dict(checkpoint["actor_backbone"])
        self.actor_head.load_state_dict(checkpoint["actor_head"])
        self.backbone.eval()
        self.actor_head.eval()

        obs_norm_state = checkpoint.get("obs_normalizer_state")
        if isinstance(obs_norm_state, dict):
            obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
            obs_normalizer.load_state_dict(obs_norm_state)
            obs_normalizer.eval()
            self.obs_normalizer = obs_normalizer

    @torch.no_grad()
    def action(self, obs: Any) -> np.ndarray:
        obs_tensor = prepare_observation(
            obs,
            device=self.device,
            obs_mode="state",
            pixel_shape=None,
            flatten=True,
        )
        # Training uses float32 model weights; ensure eval inputs match dtype.
        if obs_tensor.dtype != torch.float32:
            obs_tensor = obs_tensor.float()
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        if self.obs_normalizer is not None:
            obs_tensor = self.obs_normalizer(obs_tensor)
        features = self.backbone(obs_tensor)
        _, _, mean = self.actor_head(features)
        return mean[0].detach().cpu().numpy().astype(np.float32)

    def close(self) -> None:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone manipulation intervention evaluator")
    cube_reward_choices = [
        "sparse_final",
        "sparse_intermediate",
        "dense",
    ]
    p.add_argument("--env_name", type=str, default="cube-double-v0")
    p.add_argument("--model_path", type=str, default=None, help="Path to FastSAC checkpoint (.pt) for policy control")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--static_reset_seed",
        type=int,
        default=None,
        help=(
            "Force deterministic reset seed on every episode. If omitted and --model_path is provided, "
            "this is auto-loaded from checkpoint/args when available."
        ),
    )
    p.add_argument("--num_episodes", type=int, default=1, help="0 => run forever")
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--controller", type=str, default="random", choices=["policy", "random", "keyboard", "human", "idle"])
    p.add_argument("--render_mode", type=str, default="human", choices=["human", "rgb_array"])
    p.add_argument("--mujoco_gl", type=str, default="auto", choices=["auto", "glfw", "egl"])
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--headless", action="store_true", default=False)

    p.add_argument("--obs_mode", type=str, default="state", choices=["state"])
    p.add_argument("--include_goal", action="store_true", default=True)
    p.add_argument("--include_distance", action="store_true", default=False)
    p.add_argument("--include_direction", action="store_true", default=False)
    p.add_argument("--include_velocity", action="store_true", default=False)
    p.add_argument(
        "--include_relative_cube_features",
        action="store_true",
        default=False,
        help="Append target-relative cube features to state observations (must match training config).",
    )

    p.add_argument(
        "--reward_type",
        type=str,
        default="none",
        choices=["sparse", "dense", "combined", "none"],
        help="Wrapper reward channel for diagnostics only; cube_reward_mode sets the optimized reward when active.",
    )
    p.add_argument("--dense_reward_scale", type=float, default=1.0)
    p.add_argument("--step_penalty", type=float, default=0.0)
    p.add_argument(
        "--cube_reward_mode",
        type=str,
        default="dense",
        choices=cube_reward_choices,
        help=(
            "Manip reward mode: sparse_final (episode success), "
            "sparse_intermediate (per cube solved), "
            "dense (phase-based progress: reach->grasp->carry->place)."
        ),
    )
    p.add_argument("--cube_success_reward", type=float, default=1.0)
    p.add_argument("--cube_subgoal_grasp_reward", type=float, default=0.25)
    p.add_argument("--cube_subgoal_place_reward", type=float, default=1.0)
    p.add_argument("--cube_subgoal_drop_penalty", type=float, default=0.0)
    p.add_argument("--cube_subgoal_grasp_error_threshold", type=float, default=0.08)
    p.add_argument("--cube_dense_progress_scale", type=float, default=1.0)
    p.add_argument(
        "--cube_dense_progress_clip",
        type=float,
        default=0.0,
        help="Optional max per-step phase progress contribution before scaling (0 disables clipping).",
    )

    p.add_argument(
        "--intervention_mode",
        type=str,
        default="none",
        choices=["none", "agent", "agent_always", "agent_reward_progress", "agent_manual_gripper"],
    )
    p.add_argument("--teacher_type", type=str, default="cube_plan", choices=["cube_plan", "cube_markov"])
    p.add_argument("--tolerance_type", type=str, default="l2", choices=["l2", "angle"])
    p.add_argument("--tolerance_value", type=float, default=0.02)
    p.add_argument(
        "--tolerance_channel_weights",
        type=str,
        default="",
        help="Optional per-action weights for l2 tolerance. One scalar or comma-separated list matching action dim.",
    )
    p.add_argument(
        "--binary_gripper_actions",
        action="store_true",
        default=False,
        help="If set, force action[4] to binary {-1,+1} using --binary_gripper_threshold.",
    )
    p.add_argument(
        "--binary_gripper_threshold",
        type=float,
        default=0.0,
        help="Threshold for binary gripper mapping: action[4] >= threshold -> +1 else -1.",
    )
    p.add_argument(
        "--hard_gripper_intervention",
        action="store_true",
        default=False,
        help="Force teacher takeover on gripper mismatch during critical manipulation phases.",
    )
    p.add_argument("--gripper_intervene_pick_radius", type=float, default=0.06)
    p.add_argument("--gripper_intervene_place_radius", type=float, default=0.06)
    p.add_argument("--gripper_intervene_contact_threshold", type=float, default=0.3)
    p.add_argument("--intervention_enable_after_steps", type=int, default=0)
    p.add_argument("--intervention_reward_patience_steps", type=int, default=5)
    p.add_argument("--intervention_reward_improvement_epsilon", type=float, default=1e-6)
    p.add_argument("--hard_block_lethal", action="store_true", default=False)

    p.add_argument("--action_scale", type=float, default=1.0)
    p.add_argument("--keyboard_scale", type=float, default=0.5)
    p.add_argument(
        "--clip_action_l2",
        action="store_true",
        default=False,
        help="Enable additional L2 action clipping (disabled by default to match training)",
    )
    p.add_argument(
        "--no_clip_action_l2",
        action="store_true",
        default=False,
        help="Deprecated alias; if set, disables L2 clipping.",
    )
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument(
        "--step_through",
        action="store_true",
        default=False,
        help="Advance exactly one environment step per ENTER/. key press.",
    )
    p.add_argument(
        "--freeze_when_idle",
        action="store_true",
        default=False,
        help="If set with keyboard/human controller, do not advance env while no key action is pressed.",
    )
    p.add_argument(
        "--print_action_obs_every",
        type=int,
        default=0,
        help="If >0, print action and observation snapshots every N steps",
    )
    p.add_argument(
        "--print_obs_dims",
        type=int,
        default=12,
        help="Number of observation dimensions to preview in debug prints",
    )
    p.add_argument(
        "--print_action_dims",
        type=int,
        default=8,
        help="Number of action dimensions to preview in debug prints",
    )
    p.add_argument(
        "--print_raw_obs_in_debug",
        action="store_true",
        default=False,
        help="Include base-environment raw observation preview in debug prints",
    )
    p.add_argument(
        "--print_obs_structure_once",
        action="store_true",
        default=False,
        help="Print a structured observation breakdown at first reset (includes cube/goal slices when inferable).",
    )
    p.add_argument(
        "--print_obs_delta_every",
        type=int,
        default=0,
        help="If >0, print obs delta norms every N steps (and cube-goal distance stats when goal vector is available).",
    )
    p.add_argument(
        "--focus_reward_metrics",
        action="store_true",
        default=False,
        help=(
            "Print focused reward-debug metrics in console. Core: target_grasp_detected, "
            "mode_total_pre_intervention, target_error, target_effector_dist, mode_dense. "
            "Plus extra diagnostics: gripper_contact, gripper_closure, target_cube_speed."
        ),
    )
    p.add_argument("--print_interventions", dest="print_interventions", action="store_true", default=True)
    p.add_argument(
        "--no_print_interventions",
        dest="print_interventions",
        action="store_false",
        help="Disable intervention-specific per-step prints (keeps periodic step prints).",
    )
    p.add_argument("--print_info_keys_once", action="store_true", default=False)
    p.add_argument("--teacher_target_mode", type=str, default="sequential", choices=["fixed", "sequential"])
    p.add_argument("--cube_success_tolerance", type=float, default=0.04)
    p.add_argument("--wandb_enable", action="store_true", default=False, help="Enable WANDB step/episode logging.")
    p.add_argument("--wandb_project", type=str, default="ogbench-manip-reward-debug")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_group", type=str, default="")
    p.add_argument("--wandb_tags", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="offline", choices=["offline", "online", "disabled"])
    p.add_argument(
        "--disable_control_panel",
        dest="disable_control_panel",
        action="store_true",
        default=False,
        help="Disable control panel window for policy/random/idle controllers.",
    )
    return p.parse_args()


def _extract_diag(info: dict, default_tol: float) -> StepDiagnostics:
    target_block = int(info.get("privileged/target_block", 0))
    cube_target_errors = info.get("diag/cube_target_errors", None)
    target_error = 0.0
    if cube_target_errors is not None:
        try:
            errs = np.asarray(cube_target_errors, dtype=np.float32).reshape(-1)
            if errs.size > 0:
                idx = int(np.clip(target_block, 0, errs.size - 1))
                target_error = float(errs[idx])
        except Exception:
            target_error = 0.0
    tol = float(info.get("teacher_tolerance_value", default_tol))
    cube_max_error = float(info.get("diag/cube_max_target_error", 0.0))
    cube_success_tol = float(info.get("diag/cube_success_tolerance", 0.04))
    return StepDiagnostics(
        candidate=bool(info.get("teacher_candidate_available", False)),
        intervened=bool(info.get("teacher_intervened", False)),
        reason=info.get("teacher_reason"),
        delta_l2=float(info.get("teacher_delta_l2", 0.0)),
        delta_l2_raw=float(info.get("teacher_delta_l2_raw", info.get("teacher_delta_l2", 0.0))),
        delta_angle_deg=float(info.get("teacher_delta_angle_deg", 0.0)),
        tolerance=tol,
        target_block=target_block,
        cubes_solved=int(info.get("diag/cubes_solved", 0)),
        cube_max_error=cube_max_error,
        target_error=target_error,
        within_success_tol=bool(cube_max_error <= cube_success_tol),
        goal_reached=bool(info.get("goal_reached", False)),
    )


def _info_scalar(info: Optional[dict], key: str, default: float = float("nan")) -> float:
    if not isinstance(info, dict) or key not in info:
        return float(default)
    try:
        arr = np.asarray(info[key], dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return float(default)
        return float(arr[0])
    except Exception:
        return float(default)


def _component_scalar(reward_components: Optional[dict], name: str, default: float = 0.0) -> float:
    if not isinstance(reward_components, dict):
        return float(default)
    value = reward_components.get(name, None)
    if value is None:
        return float(default)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float(default)
        return float(value.reshape(-1)[0].item())
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return float(default)
        return float(arr[0])
    except Exception:
        return float(default)


def _reward_channels_suffix(
    *,
    info: Optional[dict],
    base_reward: float,
    final_reward: float,
    reward_components: Optional[dict],
) -> str:
    wr_s_raw = _info_scalar(info, "sparse_reward_raw")
    wr_d_raw = _info_scalar(info, "dense_reward_raw")
    wr_t_raw = _info_scalar(info, "total_reward_raw")
    wr_s = _info_scalar(info, "sparse_reward")
    wr_d = _info_scalar(info, "dense_reward")
    wr_t = _info_scalar(info, "total_reward")
    gripper_contact = _component_scalar(
        reward_components,
        "gripper_contact_raw",
        _info_scalar(info, "proprio/gripper_contact", default=float("nan")),
    )
    text = (
        f" r_base={float(base_reward):+.3f}"
        f" wr_raw[s={wr_s_raw:+.3f},d={wr_d_raw:+.3f},t={wr_t_raw:+.3f}]"
        f" wr_sel[s={wr_s:+.3f},d={wr_d:+.3f},t={wr_t:+.3f}]"
        f" g_contact={gripper_contact:+.3f}"
        f" r_final={float(final_reward):+.3f}"
    )
    if reward_components is not None:
        dense_phase = _component_scalar(
            reward_components,
            "dense_phase_reward",
            _component_scalar(reward_components, "dense_target_distance_reward"),
        )
        dense_legacy = _component_scalar(reward_components, "dense_target_distance_reward")
        text += (
            " mode["
            f"success={_component_scalar(reward_components, 'success_sparse'):+.3f},"
            f"solved_delta={_component_scalar(reward_components, 'solved_delta_sparse'):+.3f},"
            f"dist={_component_scalar(reward_components, 'dense_target_error'):+.3f},"
            f"dense_phase={dense_phase:+.3f},"
            f"dense_legacy={dense_legacy:+.3f},"
            f"grasped={_component_scalar(reward_components, 'target_grasp_detected'):+.3f},"
            f"grasped_strict={_component_scalar(reward_components, 'target_grasp_detected_strict'):+.3f},"
            f"tcp_dist={_component_scalar(reward_components, 'target_effector_dist'):+.3f},"
            f"drop={_component_scalar(reward_components, 'drop_penalty'):+.3f},"
            f"pre={_component_scalar(reward_components, 'mode_total_pre_intervention'):+.3f}"
            "]"
        )
    return text


def _reward_focus_line(
    *,
    step: int,
    step_reward: float,
    episode_reward: float,
    reward_components: Optional[dict],
    reward_suffix: str,
) -> str:
    dense_mode = _component_scalar(
        reward_components,
        "dense_phase_reward",
        _component_scalar(reward_components, "dense_target_distance_reward", 0.0),
    )
    dense_mode_cum = _component_scalar(reward_components, "dense_phase_cumulative", 0.0)
    pre_total = _component_scalar(reward_components, "mode_total_pre_intervention", 0.0)
    target_error = _component_scalar(reward_components, "dense_target_error", 0.0)
    target_effector_dist = _component_scalar(reward_components, "target_effector_dist", 0.0)
    target_grasp = _component_scalar(reward_components, "target_grasp_detected", 0.0)
    gripper_contact = _component_scalar(reward_components, "gripper_contact_raw", 0.0)
    gripper_closure = _component_scalar(reward_components, "gripper_closure", 0.0)
    target_cube_speed = _component_scalar(reward_components, "target_cube_speed", 0.0)
    target_grasp_strict = _component_scalar(reward_components, "target_grasp_detected_strict", 0.0)

    def _bar(value: float, vmax: float, width: int = 8) -> str:
        v = max(0.0, min(float(value), float(max(vmax, 1e-8))))
        fill = int(round((v / float(max(vmax, 1e-8))) * width))
        fill = max(0, min(width, fill))
        return "[" + ("#" * fill) + ("." * (width - fill)) + "]"

    return (
        f"step={int(step):04d} "
        f"r_step={float(step_reward):+.4f} "
        f"r_ep={float(episode_reward):+.4f} "
        f"mode_dense={dense_mode:+.4f} "
        f"mode_dense_cum={dense_mode_cum:+.4f} "
        f"mode_pre={pre_total:+.4f} "
        f"E{_bar(target_error, 0.25)}{target_error:.4f} "
        f"D{_bar(target_effector_dist, 0.35)}{target_effector_dist:.4f} "
        f"C{_bar(gripper_contact, 1.0)}{gripper_contact:.2f} "
        f"CL{_bar(gripper_closure, 1.0)}{gripper_closure:.2f} "
        f"V{_bar(target_cube_speed, 0.02)}{target_cube_speed:.4f} "
        f"grasp={target_grasp:+.1f} strict={target_grasp_strict:+.1f}"
        f"{reward_suffix}"
    )


def _dense_metric_value(
    *,
    info: Optional[dict],
    reward_components: Optional[dict],
    fallback_reward: float,
) -> float:
    # Canonical dense/debug scalar from the cube reward tracker output.
    if isinstance(reward_components, dict):
        for key in (
            "mode_total_pre_intervention",
            "dense_phase_reward",
            "dense_target_distance_reward",
        ):
            if key in reward_components:
                return _component_scalar(reward_components, key, float(fallback_reward))
    # Fallback to wrapper-reported dense channel.
    for key in ("dense_reward", "dense_reward_raw"):
        value = _info_scalar(info, key, default=float("nan"))
        if np.isfinite(value):
            return float(value)
    return float(fallback_reward)


def _compact_step_line(
    *,
    step: int,
    step_reward: float,
    episode_reward: float,
    target_block: int,
    dense_value: float,
    goal_reached: bool,
    success: bool,
    intervened: bool,
    reason: Optional[str],
    reward_suffix: str,
) -> str:
    reason_txt = str(reason) if (intervened and reason is not None) else "-"
    return (
        f"step={int(step):04d} "
        f"r_step={float(step_reward):+.4f} "
        f"r_ep={float(episode_reward):+.4f} "
        f"target={int(target_block)} "
        f"dense={float(dense_value):+.4f} "
        f"goal={int(bool(goal_reached))} "
        f"success={int(bool(success))} "
        f"intv={int(bool(intervened))} "
        f"reason={reason_txt}"
        f"{reward_suffix}"
    )


def create_env(args: argparse.Namespace) -> gym.Env:
    if not args.headless and args.render_mode == "human":
        if args.mujoco_gl == "auto":
            os.environ["MUJOCO_GL"] = "glfw"
        else:
            os.environ["MUJOCO_GL"] = args.mujoco_gl
    elif args.mujoco_gl != "auto":
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    # Import after MUJOCO_GL is configured so MuJoCo backend selection is respected.
    import ogbench  # noqa: F401

    # Manip env ignores render_mode and always returns rgb arrays from render().
    # Use None for interactive mode and native passive viewer instead.
    render_mode = "rgb_array" if args.headless else (None if args.render_mode == "human" else args.render_mode)
    env = gym.make(args.env_name, max_episode_steps=args.max_episode_steps, render_mode=render_mode)
    env = build_ogbench_manip_wrapper(
        env_name=args.env_name,
        obs_mode=args.obs_mode,
        include_goal=args.include_goal,
        include_distance=args.include_distance,
        include_direction=args.include_direction,
        include_velocity=args.include_velocity,
        include_relative_cube_features=args.include_relative_cube_features,
        reward_type=args.reward_type,
        dense_reward_scale=args.dense_reward_scale,
        step_penalty=args.step_penalty,
        cube_reward_mode=args.cube_reward_mode,
        intervention_mode=args.intervention_mode,
        teacher_type=args.teacher_type,
        tolerance_type=args.tolerance_type,
        tolerance_value=args.tolerance_value,
        tolerance_channel_weights=args.tolerance_channel_weights,
        binary_gripper_actions=args.binary_gripper_actions,
        binary_gripper_threshold=args.binary_gripper_threshold,
        hard_gripper_intervention=args.hard_gripper_intervention,
        gripper_intervene_pick_radius=args.gripper_intervene_pick_radius,
        gripper_intervene_place_radius=args.gripper_intervene_place_radius,
        gripper_intervene_contact_threshold=args.gripper_intervene_contact_threshold,
        hard_block_lethal=args.hard_block_lethal,
        intervention_enable_after_steps=args.intervention_enable_after_steps,
        intervention_reward_patience_steps=args.intervention_reward_patience_steps,
        intervention_reward_improvement_epsilon=args.intervention_reward_improvement_epsilon,
        teacher_target_mode=args.teacher_target_mode,
        cube_success_tolerance=args.cube_success_tolerance,
        static_reset_seed=args.static_reset_seed,
    )(env)
    return env


def _flatten_preview(x: Any, max_dims: int) -> str:
    if isinstance(x, tuple):
        x = x[0]
    if isinstance(x, dict):
        # Prefer explicit policy-style key if present.
        for key in ("policy", "observation", "state", "obs"):
            if key in x:
                x = x[key]
                break
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    shown = arr[: max(1, int(max_dims))]
    return np.array2string(shown, precision=4, suppress_small=False)


def _extract_obs_vector(x: Any) -> np.ndarray:
    if isinstance(x, tuple):
        x = x[0]
    if isinstance(x, dict):
        for key in ("policy", "observation", "state", "obs"):
            if key in x:
                x = x[key]
                break
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _extract_goal_vector(info: Any) -> Optional[np.ndarray]:
    if not isinstance(info, dict):
        return None
    if "goal" not in info:
        return None
    try:
        return np.asarray(info["goal"], dtype=np.float32).reshape(-1)
    except Exception:
        return None


def _split_obs_goal_vectors(obs_vec: np.ndarray, goal_vec: Optional[np.ndarray]) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (obs_core, goal_core) for diagnostics.

    Supports both native manip observations and goal-conditioned wrapper output
    where observation is `[obs, goal]`.
    """
    obs_core = np.asarray(obs_vec, dtype=np.float32).reshape(-1)
    goal_core = None if goal_vec is None else np.asarray(goal_vec, dtype=np.float32).reshape(-1)
    if goal_core is not None:
        if obs_core.shape[0] == 2 * goal_core.shape[0]:
            obs_core = obs_core[: goal_core.shape[0]]
        return obs_core, goal_core
    if obs_core.shape[0] % 2 == 0:
        half = obs_core.shape[0] // 2
        # Heuristic: when wrapper appends goal, second half typically differs from first half.
        if np.linalg.norm(obs_core[:half] - obs_core[half:]) > 1e-6:
            return obs_core[:half], obs_core[half:]
    return obs_core, None


def _infer_manip_obs_layout(obs_dim: int, num_cubes: int) -> Optional[dict[str, Any]]:
    # cube env observation layout in state mode:
    # [joint_pos(n), joint_vel(n), eff_pos(3), cos(yaw), sin(yaw), grip_open(1), grip_contact(1), blocks(num_cubes*9)]
    block_width = 9
    base = int(obs_dim) - int(num_cubes) * block_width
    if base < 7:
        return None
    rem = base - 7
    if rem < 0 or rem % 2 != 0:
        return None
    n_joint = rem // 2
    i = 0
    joint_pos = (i, i + n_joint)
    i += n_joint
    joint_vel = (i, i + n_joint)
    i += n_joint
    eff_pos = (i, i + 3)
    i += 3
    eff_yaw_cos = (i, i + 1)
    i += 1
    eff_yaw_sin = (i, i + 1)
    i += 1
    gripper_open = (i, i + 1)
    i += 1
    gripper_contact = (i, i + 1)
    i += 1
    blocks: list[dict[str, tuple[int, int]]] = []
    for _ in range(num_cubes):
        blocks.append(
            {
                "pos": (i, i + 3),
                "quat": (i + 3, i + 7),
                "yaw_cos": (i + 7, i + 8),
                "yaw_sin": (i + 8, i + 9),
            }
        )
        i += block_width
    if i != obs_dim:
        return None
    return {
        "n_joint": n_joint,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "eff_pos": eff_pos,
        "eff_yaw_cos": eff_yaw_cos,
        "eff_yaw_sin": eff_yaw_sin,
        "gripper_open": gripper_open,
        "gripper_contact": gripper_contact,
        "blocks": blocks,
    }


def _slice(arr: np.ndarray, sl: tuple[int, int]) -> np.ndarray:
    return arr[int(sl[0]) : int(sl[1])]


def _obs_block_position_distances(obs_vec: np.ndarray, goal_vec: Optional[np.ndarray], layout: Optional[dict[str, Any]]) -> Optional[np.ndarray]:
    if layout is None or goal_vec is None or goal_vec.shape[0] != obs_vec.shape[0]:
        return None
    dists = []
    for blk in layout["blocks"]:
        cur = _slice(obs_vec, blk["pos"])
        goal = _slice(goal_vec, blk["pos"])
        dists.append(float(np.linalg.norm(cur - goal)))
    return np.asarray(dists, dtype=np.float32)


def _print_obs_breakdown(
    *,
    obs_vec: np.ndarray,
    goal_vec: Optional[np.ndarray],
    layout: Optional[dict[str, Any]],
    info: Any,
    header: str,
) -> None:
    print(header)
    print(f"obs_dim={obs_vec.shape[0]} goal_present={int(goal_vec is not None)}")
    if goal_vec is not None:
        print(
            f"goal_dim={goal_vec.shape[0]} goal_dim_match_obs={int(goal_vec.shape[0] == obs_vec.shape[0])} "
            f"obs_goal_l2={float(np.linalg.norm(obs_vec - goal_vec)):.4f}"
        )
    if layout is None:
        print(f"obs_preview={_flatten_preview(obs_vec, 24)}")
        return
    eff_pos = _slice(obs_vec, layout["eff_pos"])
    grip_open = float(_slice(obs_vec, layout["gripper_open"])[0])
    grip_contact = float(_slice(obs_vec, layout["gripper_contact"])[0])
    print(
        f"n_joint={layout['n_joint']} eff_pos=({eff_pos[0]:+.3f},{eff_pos[1]:+.3f},{eff_pos[2]:+.3f}) "
        f"grip_open={grip_open:+.3f} grip_contact={grip_contact:+.3f}"
    )
    for bi, blk in enumerate(layout["blocks"]):
        cur = _slice(obs_vec, blk["pos"])
        if goal_vec is not None and goal_vec.shape[0] == obs_vec.shape[0]:
            goal = _slice(goal_vec, blk["pos"])
            dist = float(np.linalg.norm(cur - goal))
            print(
                f"cube[{bi}] pos=({cur[0]:+.3f},{cur[1]:+.3f},{cur[2]:+.3f}) "
                f"goal=({goal[0]:+.3f},{goal[1]:+.3f},{goal[2]:+.3f}) dist={dist:.4f}"
            )
        else:
            print(f"cube[{bi}] pos=({cur[0]:+.3f},{cur[1]:+.3f},{cur[2]:+.3f})")
    if isinstance(info, dict):
        print(f"reset_info_has_goal={int('goal' in info)}")


def run(args: argparse.Namespace) -> None:
    if not args.headless and args.render_mode != "human":
        print("For interactive manipulation debugging, forcing native render_mode=human (avoids flicker).")
        args.render_mode = "human"
    if args.step_through and args.headless:
        raise ValueError("--step_through requires a control window; run without --headless.")
    if args.step_through and args.print_every > 1:
        print("step_through: forcing --print_every=1 so each stepped transition prints metrics.")
        args.print_every = 1
    if args.controller in ("keyboard", "human") and args.print_every > 1:
        print("human/keyboard control: forcing --print_every=1 to print per-step reward channels.")
        args.print_every = 1

    env = create_env(args)
    action_space = env.action_space
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"Expected Box action space, got {type(action_space)}")

    action_dim = int(np.prod(action_space.shape))
    if args.controller == "policy":
        if not args.model_path:
            raise ValueError("--model_path is required when --controller policy")
        device = select_device(args.device)
        controller = FastSACPolicyController(model_path=args.model_path, env=env, device=device)
    elif args.controller == "random":
        controller = RandomController(action_space)
    elif args.controller == "idle":
        controller = IdleController(action_dim)
    else:
        # "human" is an alias for keyboard teleop.
        controller = KeyboardController(action_dim, magnitude=float(args.keyboard_scale))
    native_viewer = NativeViewer()
    control_panel = None
    use_panel = (
        (not args.headless)
        and (args.controller not in ("keyboard", "human"))
        and ((not args.disable_control_panel) or args.step_through)
    )
    if use_panel:
        control_panel = EpisodeControlPanel()

    print("=== Manip Eval ===")
    print(f"env={args.env_name}")
    print(f"controller={args.controller}")
    if args.controller == "policy":
        print(f"model={args.model_path}")
    print(f"teacher={args.teacher_type}, mode={args.intervention_mode}")
    print(f"tolerance={args.tolerance_type}:{args.tolerance_value}")
    if str(args.tolerance_channel_weights).strip():
        print(f"tolerance_channel_weights={args.tolerance_channel_weights}")
    if args.binary_gripper_actions:
        print(f"binary_gripper=true threshold={args.binary_gripper_threshold}")
    if args.hard_gripper_intervention:
        print(
            "hard_gripper_intervention=true "
            f"pick_r={args.gripper_intervene_pick_radius} "
            f"place_r={args.gripper_intervene_place_radius} "
            f"contact_th={args.gripper_intervene_contact_threshold}"
        )
    print(f"target_mode={args.teacher_target_mode}, cube_success_tol={args.cube_success_tolerance}")
    print(f"action_dim={int(np.prod(action_space.shape))}")
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '<unset>')}")
    print(f"DISPLAY={os.environ.get('DISPLAY', '<unset>')}")
    if args.intervention_mode != "none":
        print("note: teacher interventions are enabled; agent movement may occur without keyboard input.")
    reward_tracker = _build_cube_reward_tracker(args)
    if reward_tracker is not None:
        print(f"cube_reward_mode={args.cube_reward_mode}")
    if args.step_through:
        print("step_through=true (press ENTER or '.' to execute exactly one env step)")
    if args.focus_reward_metrics:
        print(
            "focus_reward_metrics=true (printing core+extra: "
            "mode_dense, mode_dense_cum, mode_pre, target_err, target_eff_dist, target_grasp, "
            "gripper_contact, gripper_closure, target_cube_speed)"
        )
    print("")

    episode_idx = 0
    episode_visits = 0
    quit_requested = False
    target_fps = max(1.0, float(args.fps))
    frame_dt = 1.0 / target_fps
    total_candidate = 0
    total_intervened = 0
    total_success_episodes = 0
    episode_tcp_final_positions: list[np.ndarray] = []
    obs_layout: Optional[dict[str, Any]] = None
    wandb_run = None
    wandb_step = 0

    try:
        if args.wandb_enable and args.wandb_mode != "disabled":
            try:
                import wandb  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "WANDB logging was requested (--wandb_enable), but wandb import failed."
                ) from exc
            wandb_tags = [tag.strip() for tag in str(args.wandb_tags).split(",") if tag.strip()]
            wandb_run = wandb.init(
                project=str(args.wandb_project),
                entity=(str(args.wandb_entity).strip() or None),
                name=(str(args.wandb_run_name).strip() or None),
                group=(str(args.wandb_group).strip() or None),
                tags=(wandb_tags or None),
                mode=str(args.wandb_mode),
                config=dict(vars(args)),
            )
            print(
                "WANDB enabled "
                f"(project={args.wandb_project}, mode={args.wandb_mode}, run={wandb_run.name})"
            )

        while (args.num_episodes == 0 or episode_visits < args.num_episodes) and not quit_requested:
            obs, info = env.reset(seed=args.seed + episode_idx)
            obs_vec_reset_full = _extract_obs_vector(obs)
            goal_vec_reset_info = _extract_goal_vector(info)
            obs_vec_reset, goal_vec_reset = _split_obs_goal_vectors(obs_vec_reset_full, goal_vec_reset_info)
            if obs_layout is None:
                num_cubes = int(getattr(env.unwrapped, "_num_cubes", 0))
                obs_layout = _infer_manip_obs_layout(obs_vec_reset.shape[0], num_cubes)
            if args.print_obs_structure_once and episode_visits == 0:
                _print_obs_breakdown(
                    obs_vec=obs_vec_reset,
                    goal_vec=goal_vec_reset,
                    layout=obs_layout,
                    info=info,
                    header="\n[Obs Breakdown @ Reset]",
                )
            if reward_tracker is not None:
                reward_tracker.reset()
            if not args.headless and args.render_mode == "human":
                if not native_viewer.enabled:
                    native_viewer.maybe_launch(env)
                native_viewer.sync(env)
            episode_visits += 1
            ep_reward = 0.0
            ep_len = 0
            ep_candidate = 0
            ep_intervened = 0
            ep_last_success = False
            ep_effector_positions: list[np.ndarray] = []
            ep_dense_phase_sum = 0.0
            ep_dense_phase_cum_last = 0.0
            prev_obs_vec = obs_vec_reset
            prev_intervention_reason: Optional[str] = None
            done = False
            episode_nav_delta = 1
            print(f"\n[Episode {episode_idx + 1} | visit {episode_visits}]", flush=True)
            if control_panel is not None:
                control_panel.set_status(active=False, fps=target_fps, rewards=None)

            while not done and not quit_requested:
                t0 = time.perf_counter()
                _pump_pygame_events()
                obs_before = obs
                panel_prev = False
                panel_next = False
                panel_quit = False
                panel_advance = False
                panel_fps_delta = 0.0
                if control_panel is not None:
                    panel_prev, panel_next, panel_quit, panel_advance, panel_fps_delta = control_panel.poll()
                    if panel_quit:
                        quit_requested = True
                    if panel_fps_delta != 0.0:
                        target_fps = min(240.0, max(1.0, target_fps + float(panel_fps_delta)))
                        frame_dt = 1.0 / target_fps
                step_advance = (not args.step_through)
                if args.controller == "policy":
                    prev_requested = panel_prev
                    next_requested = panel_next
                    if args.step_through:
                        step_advance = bool(panel_advance)
                    if not step_advance:
                        if not args.headless and args.render_mode == "human":
                            native_viewer.sync(env)
                        if frame_dt > 0:
                            elapsed = time.perf_counter() - t0
                            if elapsed < frame_dt:
                                time.sleep(frame_dt - elapsed)
                        continue
                    action = controller.action(obs)
                elif args.controller in ("random", "idle"):
                    prev_requested = panel_prev
                    next_requested = panel_next
                    if args.step_through:
                        step_advance = bool(panel_advance)
                    if not step_advance:
                        if not args.headless and args.render_mode == "human":
                            native_viewer.sync(env)
                        if frame_dt > 0:
                            elapsed = time.perf_counter() - t0
                            if elapsed < frame_dt:
                                time.sleep(frame_dt - elapsed)
                        continue
                    action = controller.action()
                else:
                    action, prev_requested, next_requested, quit_requested, kb_advance, kb_fps_delta = controller.action()
                    prev_requested = bool(prev_requested or panel_prev)
                    next_requested = bool(next_requested or panel_next)
                    quit_requested = bool(quit_requested or panel_quit)
                    if kb_fps_delta != 0.0:
                        target_fps = min(240.0, max(1.0, target_fps + float(kb_fps_delta)))
                        frame_dt = 1.0 / target_fps
                    if args.step_through:
                        step_advance = bool(kb_advance or panel_advance)
                    if not step_advance:
                        if not args.headless and args.render_mode == "human":
                            native_viewer.sync(env)
                        if frame_dt > 0:
                            elapsed = time.perf_counter() - t0
                            if elapsed < frame_dt:
                                time.sleep(frame_dt - elapsed)
                        continue

                if prev_requested:
                    print("previous requested: moving to previous episode")
                    episode_nav_delta = -1
                    done = True
                    continue
                if next_requested:
                    print("next requested: moving to next episode")
                    episode_nav_delta = 1
                    done = True
                    continue

                action = clip_action(
                    action * float(args.action_scale),
                    clip_l2=(bool(args.clip_action_l2) and not bool(args.no_clip_action_l2)),
                    binary_gripper_actions=bool(args.binary_gripper_actions),
                    binary_gripper_threshold=float(args.binary_gripper_threshold),
                )

                # Optional hard freeze for manual control: when no key input is active,
                # keep the scene state fixed by not advancing the environment.
                if (
                    args.freeze_when_idle
                    and not args.step_through
                    and args.controller in ("keyboard", "human", "idle")
                    and float(np.linalg.norm(action)) <= 1e-8
                ):
                    if not args.headless and args.render_mode == "human":
                        native_viewer.sync(env)
                    if frame_dt > 0:
                        elapsed = time.perf_counter() - t0
                        if elapsed < frame_dt:
                            time.sleep(frame_dt - elapsed)
                    continue

                obs, reward, terminated, truncated, info = env.step(action)
                base_reward = float(reward)
                done = bool(terminated or truncated)
                reward_components = None
                if reward_tracker is not None:
                    shaped_reward_tensor, reward_components = reward_tracker.compute(
                        base_rewards=torch.tensor([base_reward], dtype=torch.float32),
                        infos=info if isinstance(info, dict) else {},
                        dones=torch.tensor([done], dtype=torch.bool),
                    )
                    reward = float(shaped_reward_tensor.item())
                    ep_dense_phase_sum += _component_scalar(reward_components, "dense_phase_reward", 0.0)
                    ep_dense_phase_cum_last = _component_scalar(reward_components, "dense_phase_cumulative", 0.0)
                reward_suffix = _reward_channels_suffix(
                    info=info if isinstance(info, dict) else None,
                    base_reward=base_reward,
                    final_reward=float(reward),
                    reward_components=reward_components,
                )

                if not args.headless and args.render_mode == "human":
                    native_viewer.sync(env)

                ep_reward += float(reward)
                ep_len += 1

                diag = _extract_diag(info if isinstance(info, dict) else {}, args.tolerance_value)
                if control_panel is not None:
                    control_panel.set_status(
                        active=diag.intervened,
                        fps=target_fps,
                        rewards=_reward_panel_values(
                            info=info if isinstance(info, dict) else None,
                            reward_components=reward_components,
                            final_reward=float(reward),
                        ),
                    )
                success_val = False
                if isinstance(info, dict):
                    raw_success = info.get("success", False)
                    if isinstance(raw_success, (np.ndarray, list, tuple)):
                        raw_arr = np.asarray(raw_success).reshape(-1)
                        success_val = bool(raw_arr[0]) if raw_arr.size else False
                    else:
                        success_val = bool(raw_success)
                ep_last_success = success_val
                if diag.candidate:
                    ep_candidate += 1
                if diag.intervened:
                    ep_intervened += 1
                if isinstance(info, dict) and "proprio/effector_pos" in info:
                    try:
                        ep_effector_positions.append(np.asarray(info["proprio/effector_pos"], dtype=np.float32).reshape(-1)[:3])
                    except Exception:
                        pass

                should_print_step = (ep_len % max(1, args.print_every) == 0)
                reason_changed = bool(diag.intervened and (diag.reason != prev_intervention_reason))
                if args.focus_reward_metrics:
                    if should_print_step:
                        print(
                            _reward_focus_line(
                                step=ep_len,
                                step_reward=float(reward),
                                episode_reward=float(ep_reward),
                                reward_components=reward_components,
                                reward_suffix=reward_suffix,
                            ),
                            flush=True,
                        )
                else:
                    dense_value = _dense_metric_value(
                        info=info if isinstance(info, dict) else None,
                        reward_components=reward_components,
                        fallback_reward=float(reward),
                    )
                    if should_print_step or (args.print_interventions and diag.intervened and reason_changed):
                        print(
                            _compact_step_line(
                                step=ep_len,
                                step_reward=float(reward),
                                episode_reward=float(ep_reward),
                                target_block=diag.target_block,
                                dense_value=dense_value,
                                goal_reached=diag.goal_reached,
                                success=success_val,
                                intervened=diag.intervened,
                                reason=diag.reason,
                                reward_suffix=reward_suffix,
                            ),
                            flush=True,
                        )
                if args.print_action_obs_every > 0 and ep_len % int(args.print_action_obs_every) == 0:
                    act_preview = _flatten_preview(action, args.print_action_dims)
                    obs_before_preview = _flatten_preview(obs_before, args.print_obs_dims)
                    obs_after_preview = _flatten_preview(obs, args.print_obs_dims)
                    raw_piece = ""
                    if args.print_raw_obs_in_debug:
                        try:
                            raw_obs = env.unwrapped.compute_observation()
                            raw_piece = f" raw_obs={_flatten_preview(raw_obs, args.print_obs_dims)}"
                        except Exception:
                            raw_piece = " raw_obs=<unavailable>"
                    print(
                        f"step={ep_len:04d} debug action={act_preview} "
                        f"obs_before={obs_before_preview} obs_after={obs_after_preview}{raw_piece}",
                        flush=True,
                    )
                if args.print_obs_delta_every > 0 and ep_len % int(args.print_obs_delta_every) == 0:
                    obs_vec_now_full = _extract_obs_vector(obs)
                    goal_vec_now_info = _extract_goal_vector(info)
                    obs_vec_now, goal_vec_now = _split_obs_goal_vectors(obs_vec_now_full, goal_vec_now_info)
                    delta_l2 = float(np.linalg.norm(obs_vec_now - prev_obs_vec))
                    msg = f"step={ep_len:04d} obs_delta_l2={delta_l2:.5f}"
                    block_dists = _obs_block_position_distances(obs_vec_now, goal_vec_now, obs_layout)
                    if block_dists is not None and block_dists.size > 0:
                        msg += (
                            f" cube_goal_dist_min={float(block_dists.min()):.4f}"
                            f" cube_goal_dist_mean={float(block_dists.mean()):.4f}"
                            f" cube_goal_dist_max={float(block_dists.max()):.4f}"
                        )
                    print(msg, flush=True)
                    prev_obs_vec = obs_vec_now
                if args.print_info_keys_once and ep_len == 1 and isinstance(info, dict):
                    print(f"info_keys={sorted(info.keys())}", flush=True)
                prev_intervention_reason = diag.reason if diag.intervened else None

                if wandb_run is not None:
                    dense_value = _dense_metric_value(
                        info=info if isinstance(info, dict) else None,
                        reward_components=reward_components,
                        fallback_reward=float(reward),
                    )
                    wandb_payload = {
                        "eval/episode_idx": float(episode_idx + 1),
                        "eval/step_in_episode": float(ep_len),
                        "eval/done": float(done),
                        "reward/dense_metric": float(dense_value),
                        "diag/target_block": float(diag.target_block),
                        "diag/goal_reached": float(diag.goal_reached),
                        "diag/success": float(success_val),
                        "diag/teacher_intervened": float(diag.intervened),
                    }
                    wandb_run.log(wandb_payload, step=wandb_step)
                    wandb_step += 1

                elapsed = time.perf_counter() - t0
                if frame_dt > elapsed:
                    time.sleep(frame_dt - elapsed)

            total_candidate += ep_candidate
            total_intervened += ep_intervened
            total_success_episodes += int(ep_last_success)
            print(
                f"episode_reward={ep_reward:.3f} steps={ep_len} "
                f"success={int(ep_last_success)} "
                f"candidate_steps={ep_candidate} intervened_steps={ep_intervened} "
                f"dense_phase_sum={ep_dense_phase_sum:.3f} dense_phase_cum={ep_dense_phase_cum_last:.3f}",
                flush=True,
            )
            if ep_effector_positions:
                eff = np.asarray(ep_effector_positions, dtype=np.float32)
                final_pos = eff[-1]
                traj_std = np.std(eff, axis=0)
                traj_extent = np.ptp(eff, axis=0)
                episode_tcp_final_positions.append(final_pos.astype(np.float32))
                print(
                    "tcp_stats "
                    f"final=({final_pos[0]:+.3f},{final_pos[1]:+.3f},{final_pos[2]:+.3f}) "
                    f"std_l2={float(np.linalg.norm(traj_std)):.4f} "
                    f"extent_l2={float(np.linalg.norm(traj_extent)):.4f}",
                    flush=True,
                )
            if isinstance(info, dict) and "teacher_num_interventions" in info:
                print(
                    "teacher_summary "
                    f"num_interventions={int(info.get('teacher_num_interventions', 0))} "
                    f"intervention_steps={int(info.get('teacher_intervention_steps', 0))} "
                    f"fraction={float(info.get('teacher_fraction_steps', 0.0)):.3f}",
                    flush=True,
                )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "episode/reward_total": float(ep_reward),
                        "episode/steps": float(ep_len),
                        "episode/success": float(ep_last_success),
                        "episode/candidate_steps": float(ep_candidate),
                        "episode/intervened_steps": float(ep_intervened),
                        "episode/intervened_fraction": float(ep_intervened / max(1, ep_len)),
                        "episode/dense_phase_sum": float(ep_dense_phase_sum),
                        "episode/dense_phase_cumulative_final": float(ep_dense_phase_cum_last),
                    },
                    step=wandb_step,
                )

            if episode_nav_delta < 0:
                episode_idx = max(0, episode_idx - 1)
            else:
                episode_idx += 1

        print(
            f"\nDone. episode_visits={episode_visits} next_episode_index={episode_idx + 1} total_candidate_steps={total_candidate} "
            f"total_intervened_steps={total_intervened} "
            f"success_episodes={total_success_episodes}/{episode_visits if episode_visits > 0 else 1}",
            flush=True,
        )
        if len(episode_tcp_final_positions) >= 2:
            finals = np.asarray(episode_tcp_final_positions, dtype=np.float32)
            final_std = np.std(finals, axis=0)
            final_extent = np.ptp(finals, axis=0)
            print(
                "tcp_cross_episode "
                f"final_std_l2={float(np.linalg.norm(final_std)):.4f} "
                f"final_extent_l2={float(np.linalg.norm(final_extent)):.4f}",
                flush=True,
            )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "summary/episodes": float(episode_visits),
                    "summary/total_candidate_steps": float(total_candidate),
                    "summary/total_intervened_steps": float(total_intervened),
                    "summary/success_episodes": float(total_success_episodes),
                    "summary/success_fraction": float(total_success_episodes / max(1, episode_visits)),
                },
                step=wandb_step,
            )
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass
        controller.close()
        if control_panel is not None:
            control_panel.close()
        native_viewer.close(env)
        env.close()


def main() -> None:
    args = parse_args()
    if args.model_path:
        apply_model_config_defaults(args)
    run(args)


if __name__ == "__main__":
    main()
