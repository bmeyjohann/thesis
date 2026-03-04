from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tensordict import TensorDict

from .buffers import PreferencePairBuffer


class FastSACUpdater:
    """Encapsulate critic/actor/alpha updates plus auxiliary buffer losses."""

    def __init__(
        self,
        *,
        args,
        actor_backbone: torch.nn.Module,
        actor_head: torch.nn.Module,
        critic_backbone: torch.nn.Module | None,
        critic_heads: torch.nn.Module,
        critic_target_backbone: torch.nn.Module,
        critic_target_heads: torch.nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        trunk_optimizer: torch.optim.Optimizer | None,
        alpha_optimizer: torch.optim.Optimizer,
        actor_params,
        critic_params,
        trunk_params,
        log_alpha: torch.Tensor,
        reshape_obs_fn,
        normalize_obs_fn,
        device: torch.device,
        amp_enabled: bool,
        amp_dtype: torch.dtype,
        amp_device_type: str,
        scaler: GradScaler,
        target_entropy: float,
        pref_buffer: PreferencePairBuffer | None = None,
    ):
        self.args = args
        self.actor_backbone = actor_backbone
        self.actor_head = actor_head
        self.critic_backbone = critic_backbone
        self.critic_heads = critic_heads
        self.critic_target_backbone = critic_target_backbone
        self.critic_target_heads = critic_target_heads
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.trunk_optimizer = trunk_optimizer
        self.alpha_optimizer = alpha_optimizer
        self.actor_params = actor_params
        self.critic_params = critic_params
        self.trunk_params = trunk_params
        self.log_alpha = log_alpha
        self.reshape_obs = reshape_obs_fn
        self.normalize_obs = normalize_obs_fn
        self.device = device
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype
        self.amp_device_type = amp_device_type
        self.scaler = scaler
        self.target_entropy = target_entropy
        self.pref_buffer = pref_buffer
        self.pref_lambda = float(getattr(args, "pref_lambda_init", 0.0))
        self.pref_lambda_lr = float(getattr(args, "pref_lambda_lr", 1e-3))
        self.pref_lambda_max = float(getattr(args, "pref_lambda_max", 10.0))
        self.pref_lambda_ema = float(getattr(args, "pref_lambda_ema", 0.9))
        self.pref_violation_clip = float(getattr(args, "pref_violation_clip", 10.0))
        self._pref_violation_ema = 0.0

    def actor_forward(self, obs_flat: torch.Tensor):
        obs_in = self.reshape_obs(obs_flat)
        features = self.actor_backbone(obs_in)
        action, log_pi, mean = self.actor_head(features)
        return action, log_pi, mean, features

    def critic_forward(self, backbone_module, heads_module, obs_flat: torch.Tensor, actions: torch.Tensor):
        obs_in = self.reshape_obs(obs_flat)
        features = backbone_module(obs_in)
        q_values = heads_module(features, actions)
        return features, q_values

    def update(
        self,
        *,
        replay_buffer,
        demo_buffer=None,
        total_env_steps: int,
        main_batch: int,
        base_batch: int,
        b_pref: int,
        b_demo: int,
    ) -> Tuple[Dict[str, float], int]:
        args = self.args
        metrics_accumulator = {
            "critic_loss": 0.0,
            "critic_loss_replay": 0.0,
            "critic_loss_pref": 0.0,
            "critic_loss_pref_weighted": 0.0,
            "critic_loss_total": 0.0,
            "pref_linked_rows": 0.0,
            "pref_lambda": 0.0,
            "pref_lambda_delta": 0.0,
            "pref_dual_violation": 0.0,
            "pref_violation": 0.0,
            "pref_violation_ema": 0.0,
            "pref_lagrangian_loss": 0.0,
            "actor_loss": 0.0,
            "actor_loss_sac": 0.0,
            "actor_bc_loss_demo": 0.0,
            "actor_bc_loss_pref": 0.0,
            "alpha_loss": 0.0,
            "entropy": 0.0,
            "action_norm": 0.0,
            "target_q": 0.0,
            "q_min_pi": 0.0,
            "q_disagreement_data": 0.0,
            "q_disagreement_pi": 0.0,
            "reward": 0.0,
            "reward_abs": 0.0,
            "alpha_value": 0.0,
            "timing_sample_s": 0.0,
            "timing_opt_s": 0.0,
        }
        updates_count = 0

        for _ in range(args.num_updates):
            sample_t0 = time.perf_counter()
            batch = replay_buffer.sample(main_batch)
            demo_batch = None
            demo_bc_obs = None
            demo_bc_actions = None
            pref_states_for_actor = None
            pref_teacher_actions_for_actor = None
            if demo_buffer is not None and b_demo > 0:
                try:
                    demo_size = getattr(demo_buffer, "size", 0)
                except Exception:
                    demo_size = 0
                if demo_size >= b_demo:
                    demo_batch = demo_buffer.sample(b_demo)
                    batch = TensorDict.cat([batch, demo_batch], dim=0)
                    demo_bc_obs = demo_batch["observations"]
                    demo_bc_actions = demo_batch["actions"]
            sample_elapsed = time.perf_counter() - sample_t0
            metrics_accumulator["timing_sample_s"] += float(sample_elapsed)

            opt_t0 = time.perf_counter()
            obs_batch = batch["observations"]
            next_obs_batch = batch["next"]["observations"]
            actions_batch = batch["actions"]
            rewards_batch = batch["next"]["rewards"].unsqueeze(-1)
            dones_batch = batch["next"]["dones"].float().unsqueeze(-1)

            obs_batch = self.normalize_obs(obs_batch)
            next_obs_batch = self.normalize_obs(next_obs_batch)

            if args.arch_shared_trunk and self.trunk_optimizer is not None:
                self.trunk_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.amp_device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                with torch.no_grad():
                    next_actions, next_log_pi, _, _ = self.actor_forward(next_obs_batch)
                    next_features_target = self.critic_target_backbone(self.reshape_obs(next_obs_batch))
                    target_q_list = self.critic_target_heads(next_features_target, next_actions)
                    min_next_q = torch.min(torch.stack(target_q_list, dim=0), dim=0).values
                    min_next_q = min_next_q - self.log_alpha.exp() * next_log_pi
                    target_q = rewards_batch + (1.0 - dones_batch) * (args.gamma * min_next_q)

                current_backbone = self.actor_backbone if args.arch_shared_trunk else self.critic_backbone
                current_features = current_backbone(self.reshape_obs(obs_batch))
                current_q_list = self.critic_heads(current_features, actions_batch)
                q_stack_data = torch.stack(current_q_list, dim=0).squeeze(-1)
                qf_loss_replay_sum = torch.tensor(0.0, device=self.device)
                for q_pred in current_q_list:
                    qf_loss_replay_sum = qf_loss_replay_sum + F.mse_loss(q_pred, target_q)
                num_critics = max(1, len(current_q_list))
                critic_loss_replay_tensor = qf_loss_replay_sum / num_critics
                qf_loss = qf_loss_replay_sum
                rank_loss = torch.tensor(0.0, device=self.device)
                rank_loss_weighted = torch.tensor(0.0, device=self.device)
                pref_violation_value = 0.0
                pref_dual_violation_value = 0.0
                pref_violation_ema_value = float(self._pref_violation_ema)
                pref_lambda_value = float(self.pref_lambda)
                pref_lambda_delta_value = 0.0

                critic_loss_replay_value = float(critic_loss_replay_tensor.detach().cpu().item())
                q_disagreement_data = torch.max(q_stack_data, dim=0).values - torch.min(q_stack_data, dim=0).values
                q_disagreement_data_value = float(q_disagreement_data.detach().mean().cpu().item())

                # Preference ranking loss
                pref_sampling_mode = str(getattr(args, "pref_sampling_mode", "independent")).strip().lower()
                if float(args.pref_rank_weight) > 0.0:
                    pref_states = None
                    pref_teacher_actions = None
                    pref_student_actions = None

                    if pref_sampling_mode == "linked":
                        has_student_actions = False
                        try:
                            has_student_actions = "student_actions" in batch.keys(include_nested=False)
                        except TypeError:
                            has_student_actions = "student_actions" in batch.keys()
                        except Exception:
                            has_student_actions = "student_actions" in batch
                        if has_student_actions:
                            linked_student_actions = batch["student_actions"]
                            has_teacher_intervened = False
                            try:
                                has_teacher_intervened = "teacher_intervened" in batch.keys(include_nested=False)
                            except TypeError:
                                has_teacher_intervened = "teacher_intervened" in batch.keys()
                            except Exception:
                                has_teacher_intervened = "teacher_intervened" in batch
                            if has_teacher_intervened:
                                linked_mask = batch["teacher_intervened"].to(torch.bool)
                                linked_rows = int(linked_mask.sum().item())
                                metrics_accumulator["pref_linked_rows"] += float(linked_rows)
                                if linked_rows > 0:
                                    pref_states = obs_batch[linked_mask]
                                    pref_teacher_actions = actions_batch[linked_mask]
                                    pref_student_actions = linked_student_actions[linked_mask]
                    elif (
                        self.pref_buffer is not None
                        and b_pref > 0
                        and self.pref_buffer.size > 0
                    ):
                        pref_t0 = time.perf_counter()
                        pref_sample = self.pref_buffer.sample(b_pref)
                        metrics_accumulator["timing_sample_s"] += float(time.perf_counter() - pref_t0)
                        if pref_sample is not None:
                            pref_states = pref_sample.states
                            pref_teacher_actions = pref_sample.teacher_actions
                            pref_student_actions = pref_sample.student_actions

                    if (
                        pref_states is not None
                        and pref_teacher_actions is not None
                        and pref_student_actions is not None
                        and pref_states.shape[0] > 0
                    ):
                        pref_features = current_backbone(self.reshape_obs(pref_states))
                        q_pos = self.critic_heads(pref_features, pref_teacher_actions)
                        q_neg = self.critic_heads(pref_features, pref_student_actions)
                        # Per-head preference ranking: every critic is trained to rank teacher > student.
                        q_pos_stack = torch.stack(q_pos, dim=0)
                        q_neg_stack = torch.stack(q_neg, dim=0)
                        if bool(getattr(args, "pref_stopgrad_positive", False)):
                            q_pos_term = q_pos_stack.detach()
                        else:
                            q_pos_term = q_pos_stack
                        margin = float(args.pref_rank_margin)
                        delta = q_pos_term - q_neg_stack
                        pref_loss_type = str(getattr(args, "pref_loss_type", "margin")).strip().lower()
                        if pref_loss_type == "bradley_terry":
                            rank_loss = F.softplus(-delta).mean()
                            rank_loss_weighted = float(args.pref_rank_weight) * rank_loss
                        elif pref_loss_type == "lagrangian":
                            prev_pref_lambda = float(self.pref_lambda)
                            violation = torch.clamp(margin - delta, min=0.0)
                            pref_violation_value = float(violation.detach().mean().cpu().item())
                            if self.pref_violation_clip > 0.0:
                                violation = torch.clamp(violation, max=self.pref_violation_clip)
                            if self.pref_lambda_ema > 0.0:
                                self._pref_violation_ema = (
                                    self.pref_lambda_ema * self._pref_violation_ema
                                    + (1.0 - self.pref_lambda_ema) * float(pref_violation_value)
                                )
                                dual_violation = float(self._pref_violation_ema)
                            else:
                                dual_violation = float(pref_violation_value)
                            pref_dual_violation_value = float(dual_violation)
                            if self.pref_lambda_lr > 0.0:
                                self.pref_lambda = max(0.0, self.pref_lambda + self.pref_lambda_lr * dual_violation)
                                if self.pref_lambda_max > 0.0:
                                    self.pref_lambda = min(self.pref_lambda, self.pref_lambda_max)
                            pref_lambda_value = float(self.pref_lambda)
                            pref_lambda_delta_value = pref_lambda_value - prev_pref_lambda
                            pref_violation_ema_value = float(self._pref_violation_ema)
                            rank_loss = (pref_lambda_value * violation).mean()
                            # Lagrangian already scales by lambda; keep extra rank weight neutral.
                            rank_loss_weighted = rank_loss
                        else:
                            rank_loss = F.softplus(margin - delta).mean()
                            rank_loss_weighted = float(args.pref_rank_weight) * rank_loss
                        qf_loss = qf_loss + rank_loss_weighted
                        pref_states_for_actor = pref_states
                        pref_teacher_actions_for_actor = pref_teacher_actions
                critic_loss_pref_value = float(rank_loss.detach().cpu().item())
                critic_loss_pref_weighted_value = float(rank_loss_weighted.detach().cpu().item())
                critic_loss_total_value = critic_loss_replay_value + critic_loss_pref_weighted_value

            self.scaler.scale(qf_loss).backward()
            self.scaler.unscale_(self.critic_optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.critic_params, max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf")
            )
            self.scaler.step(self.critic_optimizer)

            # Actor update
            self.actor_optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.amp_device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                pi_actions, log_pi, _, _ = self.actor_forward(obs_batch)
                actor_backbone_eval = current_backbone
                q_pi_list = self.critic_heads(actor_backbone_eval(self.reshape_obs(obs_batch)), pi_actions)
                q_stack_pi = torch.stack(q_pi_list, dim=0)
                min_q_pi = torch.min(q_stack_pi, dim=0).values
                actor_loss_sac = (self.log_alpha.exp().detach() * log_pi - min_q_pi).mean()
                actor_loss = actor_loss_sac
                bc_loss_demo = torch.tensor(0.0, device=self.device)
                bc_loss_pref = torch.tensor(0.0, device=self.device)

                if (
                    float(getattr(args, "actor_bc_weight_demo", 0.0)) > 0.0
                    and demo_bc_obs is not None
                    and demo_bc_actions is not None
                ):
                    demo_obs_norm = self.normalize_obs(demo_bc_obs)
                    demo_pi, _, _, _ = self.actor_forward(demo_obs_norm)
                    bc_loss_demo = F.mse_loss(demo_pi, demo_bc_actions)
                    actor_loss = actor_loss + float(args.actor_bc_weight_demo) * bc_loss_demo

                if (
                    float(getattr(args, "actor_bc_weight_pref", 0.0)) > 0.0
                    and pref_states_for_actor is not None
                    and pref_teacher_actions_for_actor is not None
                ):
                    pref_pi, _, _, _ = self.actor_forward(pref_states_for_actor)
                    bc_loss_pref = F.mse_loss(pref_pi, pref_teacher_actions_for_actor)
                    actor_loss = actor_loss + float(args.actor_bc_weight_pref) * bc_loss_pref

            actor_loss_value = float(actor_loss.detach().cpu().item())
            actor_loss_sac_value = float(actor_loss_sac.detach().cpu().item())
            bc_demo_loss_value = float(bc_loss_demo.detach().cpu().item())
            bc_pref_loss_value = float(bc_loss_pref.detach().cpu().item())
            entropy_value = float((-log_pi).detach().mean().cpu().item())
            action_norm_value = float(pi_actions.detach().norm(dim=-1).mean().cpu().item())
            target_q_mean = float(target_q.detach().mean().cpu().item())
            min_q_pi_mean = float(min_q_pi.detach().mean().cpu().item())
            q_disagreement_pi = torch.max(q_stack_pi, dim=0).values - torch.min(q_stack_pi, dim=0).values
            q_disagreement_pi_value = float(q_disagreement_pi.detach().mean().cpu().item())
            reward_mean = float(rewards_batch.detach().mean().cpu().item())

            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            if args.arch_shared_trunk and self.trunk_optimizer is not None:
                self.scaler.unscale_(self.trunk_optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.actor_params, max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf")
            )
            if args.arch_shared_trunk and self.trunk_optimizer is not None and self.trunk_params is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.trunk_params, max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf")
                )
            self.scaler.step(self.actor_optimizer)
            if args.arch_shared_trunk and self.trunk_optimizer is not None:
                self.scaler.step(self.trunk_optimizer)
            self.scaler.update()

            # Alpha update
            if total_env_steps < int(getattr(args, "alpha_freeze_steps", 0)):
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss_value = 0.0
                alpha_value = float(self.log_alpha.exp().detach().cpu().item())
            else:
                self.alpha_optimizer.zero_grad(set_to_none=True)
                _, log_pi_curr, _, _ = self.actor_forward(obs_batch)
                log_pi_detached = log_pi_curr.detach()
                alpha_loss = (-self.log_alpha.exp() * (log_pi_detached + self.target_entropy)).mean()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                with torch.no_grad():
                    min_log_alpha = None
                    if float(args.alpha_min) > 0.0:
                        min_log_alpha = np.log(max(1e-6, float(args.alpha_min)))
                    max_log_alpha = None
                    if float(args.alpha_max) > 0.0:
                        max_log_alpha = np.log(float(args.alpha_max))
                    lower = min_log_alpha if min_log_alpha is not None else -torch.inf
                    upper = max_log_alpha if max_log_alpha is not None else torch.inf
                    if not np.isinf(lower) or not np.isinf(upper):
                        self.log_alpha.clamp_(min=lower, max=upper)
                alpha_loss_value = float(alpha_loss.detach().cpu().item())
                alpha_value = float(self.log_alpha.exp().detach().cpu().item())

            # Soft update targets
            source_backbone = self.actor_backbone if args.arch_shared_trunk else self.critic_backbone
            if source_backbone is None:
                raise RuntimeError("Critic backbone is None while required for target update")
            for src_param, tgt_param in zip(source_backbone.parameters(), self.critic_target_backbone.parameters()):
                tgt_param.data.copy_(args.tau * src_param.data + (1 - args.tau) * tgt_param.data)
            for src_param, tgt_param in zip(self.critic_heads.parameters(), self.critic_target_heads.parameters()):
                tgt_param.data.copy_(args.tau * src_param.data + (1 - args.tau) * tgt_param.data)

            metrics_accumulator["critic_loss"] += critic_loss_total_value
            metrics_accumulator["critic_loss_replay"] += critic_loss_replay_value
            metrics_accumulator["critic_loss_pref"] += critic_loss_pref_value
            metrics_accumulator["critic_loss_pref_weighted"] += critic_loss_pref_weighted_value
            metrics_accumulator["critic_loss_total"] += critic_loss_total_value
            metrics_accumulator["pref_lambda"] += pref_lambda_value
            metrics_accumulator["pref_lambda_delta"] += pref_lambda_delta_value
            metrics_accumulator["pref_dual_violation"] += pref_dual_violation_value
            metrics_accumulator["pref_violation"] += pref_violation_value
            metrics_accumulator["pref_violation_ema"] += pref_violation_ema_value
            metrics_accumulator["pref_lagrangian_loss"] += (
                critic_loss_pref_weighted_value if str(getattr(args, "pref_loss_type", "margin")).strip().lower() == "lagrangian" else 0.0
            )
            metrics_accumulator["actor_loss"] += actor_loss_value
            metrics_accumulator["actor_loss_sac"] += actor_loss_sac_value
            metrics_accumulator["actor_bc_loss_demo"] += bc_demo_loss_value
            metrics_accumulator["actor_bc_loss_pref"] += bc_pref_loss_value
            metrics_accumulator["alpha_loss"] += alpha_loss_value
            metrics_accumulator["entropy"] += entropy_value
            metrics_accumulator["action_norm"] += action_norm_value
            metrics_accumulator["target_q"] += target_q_mean
            metrics_accumulator["q_min_pi"] += min_q_pi_mean
            metrics_accumulator["q_disagreement_data"] += q_disagreement_data_value
            metrics_accumulator["q_disagreement_pi"] += q_disagreement_pi_value
            metrics_accumulator["reward"] += reward_mean
            metrics_accumulator["reward_abs"] += float(rewards_batch.detach().abs().mean().cpu().item())
            metrics_accumulator["alpha_value"] += alpha_value
            metrics_accumulator["timing_opt_s"] += float(time.perf_counter() - opt_t0)
            updates_count += 1

        return metrics_accumulator, updates_count
