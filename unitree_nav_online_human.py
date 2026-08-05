"""Online thesis-method updates for interactive Unitree human intervention."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch


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
        self.buffer = UnitreeReplayBuffer(
            capacity=int(args.online_replay_capacity),
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            device=self.device,
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

        self.step += 1
        self.interventions += int(intervened)
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
                metrics = sac_update_step(
                    sac=self.sac,
                    batch=self.buffer.sample(int(self.args.online_batch_size)),
                    gamma=float(self.args.online_gamma),
                    tau=float(self.args.online_tau),
                    max_grad_norm=float(self.args.online_max_grad_norm),
                    pref_sampling_mode="linked",
                    pref_rank_weight=float(self.args.online_pref_rank_weight),
                    pref_rank_margin=float(self.args.online_pref_rank_margin),
                    pref_loss_type="lagrangian",
                    pref_stopgrad_positive=bool(self.args.online_pref_stopgrad_positive),
                    pref_lambda_lr=float(self.args.online_pref_lambda_lr),
                    pref_lambda_max=float(self.args.online_pref_lambda_max),
                    pref_action_delta_min=float(self.args.online_pref_action_delta_min),
                    actor_bc_weight=float(self.args.online_actor_bc_weight),
                    actor_bc_teacher_only=True,
                    actor_bc_only=False,
                    alpha_min=float(self.args.online_alpha),
                    alpha_max=float(self.args.online_alpha),
                    update_actor=(self.step % int(self.args.online_policy_frequency) == 0),
                )
                latest = asdict(metrics)
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
        print(f"[online-human] saved checkpoint {target}", flush=True)
        return target

    def close(self) -> Path:
        target = self.save()
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None
        return target
