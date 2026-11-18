"""Online thesis-method updates for interactive Unitree human intervention."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch


ONLINE_LEARNING_DEFAULTS = {
    "online_alpha": 0.01,
    "online_alpha_min": 1e-5,
    "online_alpha_max": 0.1,
    "online_alpha_lr": 3e-4,
    "online_alpha_state_scope": "student",
    "online_target_entropy": -3.0,
    "online_pref_sampling_mode": "separate",
    "online_pref_batch_size": 256,
    "online_pref_replay_capacity": 100000,
}


def add_online_learning_args(parser):
    """Share the selected alpha/replay contract with the interactive entrypoint."""
    for name in ("online_alpha", "online_alpha_min", "online_alpha_max", "online_alpha_lr",
                 "online_target_entropy"):
        parser.add_argument("--" + name.replace("_", "-"), type=float,
                            default=ONLINE_LEARNING_DEFAULTS[name])
    parser.add_argument("--online-alpha-state-scope", choices=["all", "student"], default="student",
                        help="States used for alpha updates; student excludes executed human takeovers.")
    parser.add_argument("--online-pref-sampling-mode", choices=["linked", "separate"], default="separate")
    for name in ("online_pref_batch_size", "online_pref_replay_capacity"):
        parser.add_argument("--" + name.replace("_", "-"), type=int,
                            default=ONLINE_LEARNING_DEFAULTS[name])


def _resolve_learning_config(args):
    for name, default in ONLINE_LEARNING_DEFAULTS.items():
        if not hasattr(args, name):
            setattr(args, name, default)
    for name in ("online_alpha", "online_alpha_min", "online_alpha_max", "online_alpha_lr",
                 "online_target_entropy"):
        if not math.isfinite(float(getattr(args, name))):
            raise ValueError(f"{name} must be finite")
    if args.online_alpha <= 0 or args.online_alpha_lr <= 0:
        raise ValueError("Online alpha initial value and learning rate must be positive")
    if (args.online_alpha_min < 0 or args.online_alpha_max < 0
            or (args.online_alpha_max > 0 and args.online_alpha_min > args.online_alpha_max)):
        raise ValueError("Invalid online alpha bounds")
    if (args.online_alpha < args.online_alpha_min
            or (args.online_alpha_max > 0 and args.online_alpha > args.online_alpha_max)):
        raise ValueError("Online alpha initial value must lie within its bounds")
    if args.online_alpha_state_scope not in {"all", "student"}:
        raise ValueError("Online alpha state scope must be all or student")
    if args.online_pref_sampling_mode not in {"linked", "separate"}:
        raise ValueError("Online preference sampling must be linked or separate")
    if args.online_pref_batch_size < 1 or args.online_pref_replay_capacity < 1:
        raise ValueError("Online preference batch size and replay capacity must be positive")
    return {name: getattr(args, name) for name in ONLINE_LEARNING_DEFAULTS}


def _unique_run_dir(output_dir: Path, requested_name: str) -> tuple[Path, str]:
    """Allocate an immutable run directory without failing on stale shell state."""
    base_name = str(requested_name).strip() or "unitree_human"
    candidate = output_dir / base_name
    suffix = 1
    while candidate.exists():
        candidate = output_dir / f"{base_name}_retry{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate, candidate.name


class UnitreeOnlineHumanLearner:
    def __init__(self, args: argparse.Namespace, *, obs_dim: int, act_dim: int) -> None:
        from safetygym_utils.sac import build_sac
        from train_unitree_nav_thesis import UnitreeNStepAccumulator, UnitreeReplayBuffer

        if int(args.num_envs) != 1:
            raise ValueError("Online human intervention training requires --num-envs 1")
        self.args = args
        self.learning_config = _resolve_learning_config(args)
        self.objective = self._resolve_objective(str(args.online_objective_mode))
        initial_pref_lambda = float(args.online_pref_lambda_init)
        pref_lambda_max = float(args.online_pref_lambda_max)
        pref_lambda_lr = float(args.online_pref_lambda_lr)
        if not math.isfinite(pref_lambda_lr) or pref_lambda_lr < 0.0:
            raise ValueError("--online-pref-lambda-lr must be finite and nonnegative")
        if not math.isfinite(initial_pref_lambda) or initial_pref_lambda < 0.0:
            raise ValueError("--online-pref-lambda-init must be finite and nonnegative")
        if pref_lambda_max > 0.0 and initial_pref_lambda > pref_lambda_max:
            raise ValueError("--online-pref-lambda-init must not exceed --online-pref-lambda-max")
        self.device = torch.device(args.device)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        output_dir = Path(args.online_output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir, resolved_name = _unique_run_dir(output_dir, str(args.online_run_name))
        if resolved_name != str(args.online_run_name):
            print(
                f"[online-human] requested run name already exists; using {resolved_name}",
                flush=True,
            )
        args.online_run_name = resolved_name
        print(f"[online-human] run directory: {self.run_dir}", flush=True)
        print(f"[online-human] objective: {json.dumps(self.objective, sort_keys=True)}", flush=True)
        print(f"[online-human] learning settings: {json.dumps(self.learning_config, sort_keys=True)}", flush=True)
        (self.run_dir / "args.json").write_text(
            json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.sac = build_sac(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_actor=int(args.hidden_dim),
            hidden_critic=int(args.hidden_dim),
            num_critics=2,
            use_layer_norm=bool(args.use_layer_norm),
            layer_norm_eps=1e-5,
            init_scale=0.01,
            lr_actor=float(args.online_lr_actor),
            lr_critic=float(args.online_lr_critic),
            weight_decay=0.0,
            num_envs=1,
            device=self.device,
            alpha_init=float(args.online_alpha),
            temporal_encoder="unitree_scan_cnn" if args.policy_encoder == "scan_cnn" else "none",
            obs_frame_stack=int(args.scan_history),
            unitree_action_history=int(args.action_history),
        )
        self.sac.pref_lambda = initial_pref_lambda
        if str(args.model_path).strip():
            checkpoint = torch.load(args.model_path, map_location=self.device, weights_only=False)
            if int(checkpoint.get("obs_dim", self.obs_dim)) != self.obs_dim:
                raise ValueError("Checkpoint observation dimension does not match online human environment")
            self.sac.actor.load_state_dict(checkpoint["actor_state_dict"])
            if bool(args.online_restore_full_state) and "critic_state_dict" in checkpoint:
                self.sac.critic.load_state_dict(checkpoint["critic_state_dict"])
                self.sac.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
                with torch.no_grad():
                    self.sac.log_alpha.copy_(torch.as_tensor(checkpoint["log_alpha"], device=self.device))
                self.sac.pref_lambda = float(checkpoint.get("pref_lambda", self.sac.pref_lambda))
                self.sac.pref_violation_ema = float(
                    checkpoint.get("pref_violation_ema", self.sac.pref_violation_ema)
                )
                print(f"[online-human] restored actor, critics, alpha, preference state from {args.model_path}")
            else:
                print(f"[online-human] restored actor only from {args.model_path}")
        else:
            print("[online-human] initialized actor and critics from scratch")
        if pref_lambda_lr == 0.0:
            self.sac.pref_lambda = initial_pref_lambda
        self.sac.target_entropy = float(args.online_target_entropy)
        for group in self.sac.alpha_optimizer.param_groups:
            group["lr"] = float(args.online_alpha_lr)
        with torch.no_grad():
            lower = math.log(args.online_alpha_min) if args.online_alpha_min > 0 else -math.inf
            upper = math.log(args.online_alpha_max) if args.online_alpha_max > 0 else math.inf
            self.sac.log_alpha.clamp_(min=lower, max=upper)
        print(f"[online-human] alpha: {self.sac.log_alpha.exp().item():.6g}; "
              f"scope={args.online_alpha_state_scope}; target_entropy={self.sac.target_entropy:g}; "
              f"bounds=[{args.online_alpha_min:g},{args.online_alpha_max:g}]", flush=True)
        print(f"[online-human] preference lambda: {self.sac.pref_lambda:.6g}; "
              f"mode={'fixed' if pref_lambda_lr == 0.0 else 'adaptive'}; lr={pref_lambda_lr:g}", flush=True)
        self.buffer = UnitreeReplayBuffer(
            capacity=int(args.online_replay_capacity),
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            device=self.device,
        )
        from unitree_nav_preference import UnitreePreferenceBuffer
        self.preference_buffer = (
            UnitreePreferenceBuffer(capacity=int(args.online_pref_replay_capacity),
                                    obs_dim=self.obs_dim, act_dim=self.act_dim, device=self.device)
            if self.objective["pref"] and args.online_pref_rank_weight > 0
            and args.online_pref_sampling_mode == "separate" else None
        )
        self.nstep = UnitreeNStepAccumulator(num_envs=1, n_step=int(args.online_n_step), gamma=float(args.online_gamma))
        self.step = 0
        self.updates = 0
        self.interventions = 0
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.wandb_run = None
        if str(args.online_wandb_mode) != "disabled":
            import wandb

            self.wandb_run = wandb.init(
                project=str(args.online_wandb_project),
                name=str(args.online_run_name),
                group=str(args.online_wandb_group) or None,
                mode=str(args.online_wandb_mode),
                config=vars(args),
                dir=str(self.run_dir),
            )

    @property
    def actor(self):
        return self.sac.actor

    @staticmethod
    def _resolve_objective(mode: str) -> dict[str, object]:
        objectives = {
            "pref_bc_rl": dict(td_weight=1.0, pref=True, bc=True, bc_only=False, proxy=False, q_only=False),
            "pref_only": dict(td_weight=0.0, pref=True, bc=False, bc_only=False, proxy=True, q_only=True),
            "pref_rl": dict(td_weight=1.0, pref=True, bc=False, bc_only=False, proxy=False, q_only=False),
            "bc_rl": dict(td_weight=1.0, pref=False, bc=True, bc_only=False, proxy=False, q_only=False),
            "bc_only": dict(td_weight=0.0, pref=False, bc=True, bc_only=True, proxy=False, q_only=False),
            "rl_only": dict(td_weight=1.0, pref=False, bc=False, bc_only=False, proxy=False, q_only=False),
        }
        try:
            return {"mode": mode, **objectives[mode]}
        except KeyError as exc:
            raise ValueError(f"Unsupported online objective mode: {mode}") from exc

    def observe(
        self,
        *,
        obs: torch.Tensor,
        student_action: torch.Tensor,
        executed_action: torch.Tensor,
        next_obs: torch.Tensor,
        done: bool,
        terminal_success: bool,
        intervened: bool,
        intervention_start: bool,
        goal_distance: float,
        next_goal_distance: float,
        cost: float,
    ) -> None:
        from safetygym_utils.sac import sac_update_step
        from unitree_nav_control import project_navigation_action

        self.step += 1
        self.interventions += int(intervened)
        if (self.preference_buffer is not None and intervened
                and (executed_action - student_action).norm(dim=-1).item()
                >= float(self.args.online_pref_action_delta_min)):
            # Store the actual decision pair before n-step aggregation changes TD routing.
            self.preference_buffer.add(obs, executed_action, student_action, step=self.step)
        post_distance = float(goal_distance) if done else float(next_goal_distance)
        reward_value = float(self.args.online_dense_progress_scale) * (float(goal_distance) - post_distance)
        if terminal_success:
            reward_value += float(self.args.online_success_bonus)
        elif done:
            reward_value += float(self.args.online_failure_penalty)
        done_tensor = torch.tensor([done], dtype=torch.bool, device=self.device)
        replay_rows = self.nstep.add(
            obs=obs.detach(),
            actions=executed_action.detach(),
            student_actions=student_action.detach(),
            next_obs=next_obs.detach(),
            rewards=torch.tensor([reward_value], dtype=torch.float32, device=self.device),
            dones=done_tensor,
            truncations=torch.zeros_like(done_tensor),
            teacher_intervened=torch.tensor([intervened], dtype=torch.bool, device=self.device),
            intervention_start=torch.tensor([intervention_start], dtype=torch.bool, device=self.device),
            env_ids=torch.zeros(1, dtype=torch.long, device=self.device),
        )
        if replay_rows is not None:
            self.buffer.add(**replay_rows)
        latest = None
        if self.buffer.size >= int(self.args.online_learning_starts):
            for _ in range(int(self.args.online_updates_per_step)):
                preference_batch = (
                    self.preference_buffer.sample(int(self.args.online_pref_batch_size), step=self.step)
                    if self.preference_buffer is not None else None
                )
                metrics = sac_update_step(
                    sac=self.sac,
                    batch=self.buffer.sample(int(self.args.online_batch_size)),
                    gamma=float(self.args.online_gamma),
                    tau=float(self.args.online_tau),
                    max_grad_norm=float(self.args.online_max_grad_norm),
                    actor_action_transform=(project_navigation_action
                        if self.args.online_project_actor_actions else None),
                    pref_batch=preference_batch,
                    pref_sampling_mode=self.args.online_pref_sampling_mode,
                    pref_rank_weight=(
                        float(self.args.online_pref_rank_weight) if bool(self.objective["pref"]) else 0.0
                    ),
                    pref_rank_margin=float(self.args.online_pref_rank_margin),
                    pref_loss_type="lagrangian",
                    pref_stopgrad_positive=bool(self.args.online_pref_stopgrad_positive),
                    pref_lambda_lr=float(self.args.online_pref_lambda_lr),
                    pref_lambda_max=float(self.args.online_pref_lambda_max),
                    pref_action_delta_min=float(self.args.online_pref_action_delta_min),
                    algo_variant="pvp" if bool(self.objective["proxy"]) else "plain",
                    pvp_proxy_value_bound=float(self.args.online_pref_proxy_value_bound),
                    critic_td_weight=float(self.objective["td_weight"]),
                    actor_q_only=bool(self.objective["q_only"]),
                    actor_bc_weight=(
                        float(self.args.online_actor_bc_weight) if bool(self.objective["bc"]) else 0.0
                    ),
                    actor_bc_teacher_only=True,
                    actor_bc_only=bool(self.objective["bc_only"]),
                    alpha_min=float(self.args.online_alpha_min),
                    alpha_max=float(self.args.online_alpha_max),
                    alpha_state_scope=self.args.online_alpha_state_scope,
                    update_actor=(self.step % int(self.args.online_policy_frequency) == 0),
                )
                latest = asdict(metrics)
                if preference_batch is not None:
                    latest.update({f"pref_sample_{key}": float(preference_batch[key])
                                   for key in ("unique_rows", "age_ticks_mean", "reuse_mean")})
                self.updates += 1
        if self.step == 1 or self.step % int(self.args.online_log_interval) == 0:
            row = {
                "step": self.step,
                "updates": self.updates,
                "replay_size": self.buffer.size,
                "intervention_fraction": self.interventions / self.step,
                "last_intervened": float(intervened),
                "last_cost": float(cost),
                "last_reward": reward_value,
                "objective_mode": str(self.objective["mode"]),
                "environmental_td_enabled": float(float(self.objective["td_weight"]) > 0.0),
                "preference_enabled": float(bool(self.objective["pref"])),
                "behavior_cloning_enabled": float(bool(self.objective["bc"])),
                "pref_lambda": float(self.sac.pref_lambda),
                "pref_lambda_fixed": float(self.args.online_pref_lambda_lr == 0.0),
                "actor_action_projection_enabled": float(self.args.online_project_actor_actions),
                "alpha_state_scope": str(self.args.online_alpha_state_scope),
                "alpha_min": float(self.args.online_alpha_min),
                "alpha_max": float(self.args.online_alpha_max),
                "alpha_lr": float(self.args.online_alpha_lr),
                "target_entropy": float(self.sac.target_entropy),
                "pref_sampling_mode": str(self.args.online_pref_sampling_mode),
                "preference_replay_size": self.preference_buffer.size if self.preference_buffer is not None else 0,
                "preference_pairs_inserted": self.preference_buffer.inserted if self.preference_buffer is not None else 0,
            }
            if latest is not None:
                row.update({key: float(value) for key, value in latest.items()})
            print(f"[online-human] {json.dumps(row, sort_keys=True)}", flush=True)
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            if self.wandb_run is not None:
                self.wandb_run.log({f"train/{key}": value for key, value in row.items()}, step=self.step)
        if self.step % int(self.args.online_checkpoint_interval) == 0:
            self.save(self.run_dir / f"step_{self.step}.pt")

    def save(self, path: Path | None = None) -> Path:
        from train_unitree_nav_thesis import save_checkpoint

        target = path or (self.run_dir / "final.pt")
        save_checkpoint(
            target,
            sac=self.sac,
            args=self.args,
            step=self.step,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
        )
        vision = getattr(self, "vision_encoder", None)
        if vision is not None:
            payload = torch.load(target, map_location="cpu", weights_only=False)
            payload["navigation_vision"] = vision.checkpoint_payload()
            temporary = target.with_suffix(".vision.tmp")
            torch.save(payload, temporary)
            temporary.replace(target)
        print(f"[online-human] saved checkpoint {target}", flush=True)
        return target

    def close(self) -> Path:
        target = self.save()
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None
        return target
