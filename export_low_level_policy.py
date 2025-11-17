#!/usr/bin/env python
"""
Offline exporter for trained RSL-RL policies.

This script reads a saved checkpoint (model_*.pt) produced by safe-locomotion /
IsaacLab training, reconstructs the ActorCritic network purely from the stored
weights, and emits TorchScript + ONNX artifacts that can be used as the
low-level policy for navigation.

It is designed to run on the login node without launching Isaac Sim or
requiring a GPU.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import copy
from typing import Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent


def _resolve_activation(name: str) -> torch.nn.Module:
    table = {
        "elu": torch.nn.ELU,
        "selu": torch.nn.SELU,
        "relu": torch.nn.ReLU,
        "leaky_relu": torch.nn.LeakyReLU,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
        "softplus": torch.nn.Softplus,
        "gelu": torch.nn.GELU,
        "silu": torch.nn.SiLU,
        "mish": torch.nn.Mish,
        "identity": torch.nn.Identity,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f"Unsupported activation '{name}'. Valid options: {sorted(table.keys())}")
    return table[key]


class SimpleActorCritic(torch.nn.Module):
    """Minimal Actor-Critic container that mirrors the RSL-RL parameter layout."""

    is_recurrent = False

    def __init__(
        self,
        actor_in: int,
        actor_hidden: List[int],
        action_dim: int,
        critic_in: int,
        critic_hidden: List[int],
        activation: str = "elu",
        noise_std_type: str = "scalar",
    ):
        super().__init__()
        act_cls = _resolve_activation(activation)
        self.actor = self._build_mlp(actor_in, actor_hidden, action_dim, act_cls)
        self.critic = self._build_mlp(critic_in, critic_hidden, 1, act_cls)
        self.actor_obs_normalizer = torch.nn.Identity()
        self.normalizer = torch.nn.Identity()
        self.adaptation_module = None
        self.noise_std_type = noise_std_type
        if noise_std_type == "log":
            self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
        else:
            self.std = torch.nn.Parameter(torch.ones(action_dim))

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims: List[int], output_dim: int, act_cls: type[torch.nn.Module]):
        layers: List[torch.nn.Module] = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for idx in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(act_cls())
        return torch.nn.Sequential(*layers)


def export_policy_as_jit(actor_critic: torch.nn.Module, normalizer, path: str, filename: str = "policy.pt"):
    exporter = _TorchPolicyExporter(actor_critic, normalizer)
    exporter.export(path, filename)


def export_policy_as_onnx(
    actor_critic: torch.nn.Module,
    path: str,
    normalizer,
    filename: str = "policy.onnx",
    verbose: bool = False,
):
    exporter = _OnnxPolicyExporter(actor_critic, normalizer, verbose)
    exporter.export(path, filename)


class _TorchPolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic, normalizer=None):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = getattr(actor_critic, "is_recurrent", False)
        if self.is_recurrent:
            raise NotImplementedError("Recurrent policies not supported by this exporter.")
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, obs):
        obs = self.normalizer(obs)
        return self.actor(obs)

    def export(self, path: str, filename: str):
        os.makedirs(path, exist_ok=True)
        scripted = torch.jit.script(self.cpu())
        scripted.save(os.path.join(path, filename))


class _OnnxPolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(actor_critic.actor)
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, obs):
        obs = self.normalizer(obs)
        return self.actor(obs)

    def export(self, path: str, filename: str):
        os.makedirs(path, exist_ok=True)
        obs = torch.zeros(1, self.actor[0].weight.shape[1])
        torch.onnx.export(
            self.cpu(),
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
        )


def _infer_mlp_dims(state_dict: Dict[str, torch.Tensor], prefix: str) -> Tuple[int, List[int], int]:
    """Return (input_dim, hidden_dims, output_dim) for the MLP with the given prefix."""
    layer_indices = sorted(
        {
            int(key.split(".")[1])
            for key in state_dict.keys()
            if key.startswith(prefix) and key.endswith(".weight")
        }
    )
    if not layer_indices:
        raise RuntimeError(f"No linear layers detected for prefix '{prefix}'.")
    first_key = f"{prefix}.{layer_indices[0]}.weight"
    input_dim = state_dict[first_key].shape[1]
    hidden_dims = [state_dict[f"{prefix}.{idx}.weight"].shape[0] for idx in layer_indices[:-1]]
    output_dim = state_dict[f"{prefix}.{layer_indices[-1]}.weight"].shape[0]
    return input_dim, hidden_dims, output_dim


def _build_actor_critic(
    state_dict: Dict[str, torch.Tensor],
    activation: str = "elu",
) -> SimpleActorCritic:
    """Instantiate ActorCritic with shapes inferred from the checkpoint weights."""
    actor_in, actor_hidden, action_dim = _infer_mlp_dims(state_dict, "actor")
    critic_in, critic_hidden, _ = _infer_mlp_dims(state_dict, "critic")
    noise_std_type = "log" if any(key.startswith("log_std") for key in state_dict) else "scalar"

    policy = SimpleActorCritic(
        actor_in=actor_in,
        actor_hidden=actor_hidden or [action_dim],
        action_dim=action_dim,
        critic_in=critic_in,
        critic_hidden=critic_hidden or [1],
        activation=activation,
        noise_std_type=noise_std_type,
    )
    policy.load_state_dict(state_dict, strict=True)
    policy.to("cpu")
    policy.eval()
    return policy


def main():
    parser = argparse.ArgumentParser(
        description="Convert an RSL-RL checkpoint into TorchScript/ONNX artifacts for navigation."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model_*.pt produced by training (e.g., logs/rsl_rl/unitree_go2_flat/.../model_150.pt).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to place the exported policy files (defaults to <run>/exported next to the checkpoint).",
    )
    parser.add_argument("--jit-name", default="policy.pt", help="Filename for the TorchScript export.")
    parser.add_argument("--onnx-name", default="policy.onnx", help="Filename for the ONNX export.")
    parser.add_argument("--activation", default="elu", help="MLP activation to assume when reconstructing the model.")
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX export (useful when only TorchScript is required).",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise KeyError(f"{checkpoint_path} is missing 'model_state_dict'.")
    model_state = ckpt["model_state_dict"]

    actor_critic = _build_actor_critic(model_state, activation=args.activation)

    if args.output_dir is None:
        output_dir = checkpoint_path.parent / "exported"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Exporter] Writing TorchScript policy to {output_dir / args.jit_name}")
    export_policy_as_jit(actor_critic, None, path=str(output_dir), filename=args.jit_name)

    if not args.skip_onnx:
        print(f"[Exporter] Writing ONNX policy to {output_dir / args.onnx_name}")
        export_policy_as_onnx(actor_critic, path=str(output_dir), filename=args.onnx_name)
    else:
        print("[Exporter] Skipping ONNX export as requested.")

    print("[Exporter] Done.")


if __name__ == "__main__":
    main()
