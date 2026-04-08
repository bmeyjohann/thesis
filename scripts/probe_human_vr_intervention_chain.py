#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FASTTD3_ROOT = REPO_ROOT / "fasttd3"
if str(FASTTD3_ROOT) not in sys.path:
    sys.path.insert(0, str(FASTTD3_ROOT))
FASTSAC_ROOT = FASTTD3_ROOT / "fast_sac"
if str(FASTSAC_ROOT) not in sys.path:
    sys.path.insert(0, str(FASTSAC_ROOT))

from fasttd3.fast_sac.fast_sac_utils import SimpleReplayBuffer  # noqa: E402
from ogbench_utils.fastsac_ogbench_manip_env import build_manip_environment  # noqa: E402
from ogbench_utils.fastsac_ogbench_manip_train import _maybe_build_train_vr_teleop  # noqa: E402
from ogbench_utils.vr_teleop import VRPublisherServer  # noqa: E402
from scripts.recover_offline_actor_manip_from_checkpoint import (  # noqa: E402
    _coerce_args_dict,
    _namespace_from_checkpoint_args,
)


def _record_progress(_: str) -> None:
    return None


class ProbeBackend:
    """Deterministic fake VR backend with alternating gated and ungated motion."""

    def __init__(self, *, hand: str = "left", rate_hz: float = 60.0):
        self.hand = str(hand)
        self.rate_hz = float(rate_hz)

    def close(self) -> None:
        return None

    def sample(self, seq: int) -> dict[str, Any]:
        t = float(seq) / max(1.0, self.rate_hz)
        cycle = int(seq // 20) % 4
        gate_active = cycle in {0, 1}
        # Keep moving even when gate is off so we can detect hidden non-gated full overrides.
        pos_scale = 0.05 if gate_active else 0.035
        pos_x = pos_scale * math.sin(t * 2.1)
        pos_y = 0.04 * math.cos(t * 1.7)
        pos_z = 0.03 * math.sin(t * 1.3)
        yaw = 0.35 * math.sin(t * 0.9)
        trigger = 0.85 if (int(seq // 12) % 2 == 0) else 0.05
        quat_w = math.cos(yaw / 2.0)
        quat_z = math.sin(yaw / 2.0)
        controller = {
            "role": self.hand,
            "connected": True,
            "tracked": True,
            "device_index": 1,
            "pose": {
                "position": [pos_x, pos_y, pos_z],
                "quaternion_wxyz": [quat_w, 0.0, 0.0, quat_z],
            },
            "linear_velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
            "axes": [
                {"x": 0.0, "y": 0.0},
                {"x": trigger, "y": 0.0},
                {"x": 0.0, "y": 0.0},
                {"x": 0.0, "y": 0.0},
                {"x": 0.0, "y": 0.0},
            ],
            "trigger_value": trigger,
            "pressed_button_ids": [2] if gate_active else [],
            "buttons": {"grip": bool(gate_active)},
        }
        devices = {"left": None, "right": None}
        devices[self.hand] = controller
        return {
            "version": 1,
            "backend": "probe_demo",
            "seq": int(seq),
            "monotonic_time": float(time.perf_counter()),
            "devices": devices,
        }


def _load_args_from_checkpoint(checkpoint_path: Path) -> SimpleNamespace:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args_dict = _coerce_args_dict(checkpoint.get("args"))
    args_ns = _namespace_from_checkpoint_args(args_dict)
    return args_ns


def _raw_flag(info: dict[str, Any] | None) -> bool:
    return bool(info.get("teacher_intervened", False)) if isinstance(info, dict) else False


def _raw_gate(diag: dict[str, Any] | None) -> bool:
    return bool(diag.get("gate_pressed", False)) if isinstance(diag, dict) else False


def _action_arr(value: Any, *, expected_dim: int) -> np.ndarray:
    if value is None:
        return np.zeros((expected_dim,), dtype=np.float32)
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = arr.astype(np.float32).reshape(-1)
    if arr.shape[0] != expected_dim:
        raise ValueError(f"Expected action dim {expected_dim}, got shape={tuple(arr.shape)}")
    return arr


def _diff_mask(lhs: np.ndarray, rhs: np.ndarray) -> bool:
    return bool(np.abs(lhs - rhs).sum() > 1e-6)


def _xyz_diff_mask(lhs: np.ndarray, rhs: np.ndarray) -> bool:
    if lhs.shape[0] < 3 or rhs.shape[0] < 3:
        return False
    return bool(np.abs(lhs[:3] - rhs[:3]).sum() > 1e-6)


def _as_tensor_batch(value: Any, *, device: torch.device, dtype: torch.dtype, num_envs: int) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.to(device=device, dtype=dtype)
    else:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.view(1).expand(num_envs)
    if tensor.shape[0] != num_envs:
        tensor = tensor.view(num_envs, *tensor.shape[1:])
    return tensor


def _stress_replay_bool_write(device: torch.device, loops: int) -> dict[str, Any]:
    rb = SimpleReplayBuffer(
        n_env=1,
        buffer_size=max(64, loops + 2),
        n_obs=4,
        n_act=4,
        n_critic_obs=4,
        device=device,
    )
    mismatches = 0
    first_examples: list[dict[str, Any]] = []
    for idx in range(int(loops)):
        expected = bool(idx % 3 == 0)
        td = TensorDict(
            {
                "observations": torch.randn(1, 4, device=device),
                "actions": torch.randn(1, 4, device=device),
                "student_actions": torch.randn(1, 4, device=device),
                "teacher_intervened": torch.tensor([expected], device=device, dtype=torch.bool),
                "next": {
                    "observations": torch.randn(1, 4, device=device),
                    "rewards": torch.randn(1, device=device),
                    "dones": torch.zeros(1, device=device, dtype=torch.long),
                    "truncations": torch.zeros(1, device=device, dtype=torch.long),
                },
            },
            batch_size=(1,),
            device=device,
        )
        rb.extend(td)
        cap = int(rb.env_capacities[0])
        last_slot = (int(rb.env_ptr[0].item()) - 1) % cap
        stored = bool(rb.teacher_intervened[0, last_slot].item())
        if stored != expected:
            mismatches += 1
            if len(first_examples) < 5:
                first_examples.append({"idx": idx, "expected": expected, "stored": stored})
    return {
        "device": str(device),
        "loops": int(loops),
        "mismatches": int(mismatches),
        "examples": first_examples,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe the manipulation human VR intervention chain with a fake VR publisher.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            REPO_ROOT
            / "models"
            / "fast_sac"
            / "cube_single_task1_human_collect_norot_fixedalpha1e3_20260408_143217"
            / "cube_single_singletask_task1_v0_step25000.pt"
        ),
    )
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--port", type=int, default=18765)
    p.add_argument("--rate_hz", type=float, default=60.0)
    p.add_argument("--disable_saved_mapping", action="store_true", default=False)
    p.add_argument(
        "--output_json",
        type=str,
        default=str(REPO_ROOT / "local" / "reports" / "human_vr_intervention_probe.json"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    base_args = _load_args_from_checkpoint(ckpt_path)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")

    # Configure args for a no-render, single-env diagnostic with fake VR input.
    base_args.num_envs = 1
    base_args.total_timesteps = int(args.steps)
    base_args.max_episode_steps = 200
    base_args.device = str(device)
    base_args.train_render_mode = "none"
    base_args.visualize_intervention_colors = False
    base_args.use_intervention = True
    base_args.intervention_mode = "human"
    base_args.human_input_device = "vr"
    base_args.vr_mode = "connect"
    base_args.vr_host = "127.0.0.1"
    base_args.vr_port = int(args.port)
    base_args.vr_reconnect_seconds = 0.25
    base_args.vr_cache_path = str(REPO_ROOT / "local" / "vr" / "probe_last_endpoint.json")
    base_args.demo_prefill_episodes = 0
    base_args.demo_prefill_num_envs = 0
    base_args.static_reset_seed = 0
    base_args.vr_use_saved_mapping = not bool(args.disable_saved_mapping)
    base_args.allow_simultaneous_train_eval_render = False

    backend = ProbeBackend(hand="left", rate_hz=float(args.rate_hz))
    publisher = VRPublisherServer(backend, host="127.0.0.1", port=int(args.port), rate_hz=float(args.rate_hz))
    publisher.start()

    teleop_interface = None
    vr_source = None
    envs = None
    summary: dict[str, Any] = {}
    try:
        teleop_interface, vr_source = _maybe_build_train_vr_teleop(base_args)
        if teleop_interface is None:
            raise RuntimeError("Failed to construct VR teleop interface for probe.")

        # Wait for the client to receive at least one sample.
        sample_deadline = time.time() + 5.0
        while time.time() < sample_deadline:
            sample = vr_source.latest_sample() if hasattr(vr_source, "latest_sample") else None
            if isinstance(sample, dict):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("VR probe client did not receive any samples from the fake publisher.")

        envs, _, _, _, n_obs, n_act, obs = build_manip_environment(
            base_args,
            device,
            _record_progress,
            teleop_interface=teleop_interface,
        )
        rb = SimpleReplayBuffer(
            n_env=1,
            buffer_size=max(512, int(args.steps) + 4),
            n_obs=n_obs,
            n_act=n_act,
            n_critic_obs=n_obs,
            device=device,
        )

        stress = _stress_replay_bool_write(device, loops=256)

        counts: dict[str, int] = {
            "steps": 0,
            "gate_pressed_steps": 0,
            "raw_teacher_steps": 0,
            "raw_override_steps": 0,
            "raw_xyz_override_steps": 0,
            "extra_teacher_mask_steps": 0,
            "extra_override_steps": 0,
            "stored_teacher_steps": 0,
            "raw_teacher_vs_override_mismatch": 0,
            "raw_override_vs_extra_mask_mismatch": 0,
            "extra_mask_vs_stored_mismatch": 0,
            "gate_off_raw_teacher_steps": 0,
            "gate_off_xyz_override_steps": 0,
            "gate_off_extra_teacher_steps": 0,
            "gate_off_stored_teacher_steps": 0,
        }
        examples: list[dict[str, Any]] = []
        reward_sum = 0.0
        done_count = 0

        for step_idx in range(int(args.steps)):
            policy_action = torch.zeros((1, n_act), dtype=torch.float32, device=device)
            obs_before = obs
            next_obs, rewards, dones, infos = envs.step(policy_action)
            raw_info = getattr(envs._env, "_passive_viewer_info", None)
            teleop_diag = teleop_interface.get_last_diag()

            raw_student = _action_arr(raw_info.get("student_action") if isinstance(raw_info, dict) else None, expected_dim=n_act)
            raw_applied = _action_arr(raw_info.get("applied_action") if isinstance(raw_info, dict) else None, expected_dim=n_act)
            raw_teacher = _raw_flag(raw_info)
            gate_pressed = _raw_gate(teleop_diag)
            raw_override = _diff_mask(raw_applied, raw_student)
            raw_xyz_override = _xyz_diff_mask(raw_applied, raw_student)

            applied_actions = _as_tensor_batch(infos.get("applied_actions"), device=device, dtype=torch.float32, num_envs=1)
            student_actions = _as_tensor_batch(infos.get("student_actions"), device=device, dtype=torch.float32, num_envs=1)
            teacher_mask = _as_tensor_batch(
                infos.get("teacher_intervened_mask", torch.zeros(1, device=device, dtype=torch.bool)),
                device=device,
                dtype=torch.bool,
                num_envs=1,
            )
            extra_teacher = bool(teacher_mask[0].item())
            extra_override = bool((torch.abs(applied_actions - student_actions).sum(dim=-1) > 1e-6)[0].item())

            truncations = _as_tensor_batch(
                infos.get("time_outs", torch.zeros(1, dtype=torch.long, device=device)),
                device=device,
                dtype=torch.bool,
                num_envs=1,
            )
            transition = TensorDict(
                {
                    "observations": obs_before.detach().to(device=device, dtype=torch.float32),
                    "actions": applied_actions.detach(),
                    "student_actions": student_actions.detach(),
                    "teacher_intervened": teacher_mask.detach(),
                    "next": {
                        "observations": next_obs.detach().to(device=device, dtype=torch.float32),
                        "rewards": rewards.detach().to(device=device, dtype=torch.float32),
                        "dones": _as_tensor_batch(dones, device=device, dtype=torch.bool, num_envs=1).long(),
                        "truncations": truncations.long(),
                    },
                },
                batch_size=(1,),
                device=device,
            )
            rb.extend(transition)
            cap = int(rb.env_capacities[0])
            last_slot = (int(rb.env_ptr[0].item()) - 1) % cap
            stored_teacher = bool(rb.teacher_intervened[0, last_slot].item())

            counts["steps"] += 1
            counts["gate_pressed_steps"] += int(gate_pressed)
            counts["raw_teacher_steps"] += int(raw_teacher)
            counts["raw_override_steps"] += int(raw_override)
            counts["raw_xyz_override_steps"] += int(raw_xyz_override)
            counts["extra_teacher_mask_steps"] += int(extra_teacher)
            counts["extra_override_steps"] += int(extra_override)
            counts["stored_teacher_steps"] += int(stored_teacher)
            counts["raw_teacher_vs_override_mismatch"] += int(raw_teacher != raw_override)
            counts["raw_override_vs_extra_mask_mismatch"] += int(raw_override != extra_teacher)
            counts["extra_mask_vs_stored_mismatch"] += int(extra_teacher != stored_teacher)
            if not gate_pressed:
                counts["gate_off_raw_teacher_steps"] += int(raw_teacher)
                counts["gate_off_xyz_override_steps"] += int(raw_xyz_override)
                counts["gate_off_extra_teacher_steps"] += int(extra_teacher)
                counts["gate_off_stored_teacher_steps"] += int(stored_teacher)

            if (
                raw_teacher != raw_override
                or raw_override != extra_teacher
                or extra_teacher != stored_teacher
            ) and len(examples) < 12:
                examples.append(
                    {
                        "step": step_idx,
                        "gate_pressed": gate_pressed,
                        "teacher_reason": raw_info.get("teacher_reason") if isinstance(raw_info, dict) else None,
                        "raw_teacher": raw_teacher,
                        "raw_override": raw_override,
                        "raw_xyz_override": raw_xyz_override,
                        "extra_teacher": extra_teacher,
                        "extra_override": extra_override,
                        "stored_teacher": stored_teacher,
                        "teleop_diag": {
                            "connected": bool(teleop_diag.get("connected", False)),
                            "tracked": bool(teleop_diag.get("tracked", False)),
                            "gate_pressed": bool(teleop_diag.get("gate_pressed", False)),
                            "motion_active": bool(teleop_diag.get("motion_active", False)),
                            "reset_gate_latched": bool(teleop_diag.get("reset_gate_latched", False)),
                            "mapped_action": list(teleop_diag.get("mapped_action", [])),
                        },
                        "raw_student_action": raw_student.tolist(),
                        "raw_applied_action": raw_applied.tolist(),
                        "extra_student_action": student_actions[0].detach().cpu().tolist(),
                        "extra_applied_action": applied_actions[0].detach().cpu().tolist(),
                    }
                )

            reward_sum += float(rewards.reshape(-1)[0].item())
            done_count += int(bool(_as_tensor_batch(dones, device=device, dtype=torch.bool, num_envs=1)[0].item()))
            obs = next_obs

        summary = {
            "device": str(device),
            "checkpoint": str(ckpt_path),
            "saved_mapping_enabled": bool(base_args.vr_use_saved_mapping),
            "mapping_path": str(getattr(base_args, "vr_mapping_path", "")),
            "stress_replay_bool_write": stress,
            "counts": counts,
            "reward_sum": reward_sum,
            "done_count": done_count,
            "examples": examples,
        }
    finally:
        try:
            if envs is not None:
                envs.close()
        except Exception:
            pass
        try:
            if vr_source is not None and hasattr(vr_source, "close"):
                vr_source.close()
        except Exception:
            pass
        try:
            publisher.close()
        except Exception:
            pass

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[VRProbeDone] json={output_path}", flush=True)


if __name__ == "__main__":
    main()
