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
        self.pref_violation_target = float(getattr(args, "pref_violation_target", 0.0))
        self.fixed_alpha = float(getattr(args, "fixed_alpha", -1.0))
        self._pref_violation_ema = 0.0
        self._critic_update_count = 0
        self._apply_alpha_bounds_()

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

    def current_alpha_tensor(self) -> torch.Tensor:
        if self.fixed_alpha >= 0.0:
            return torch.tensor(self.fixed_alpha, device=self.device, dtype=self.log_alpha.dtype)
        lower, upper = self._alpha_log_bounds()
        bounded_log_alpha = self.log_alpha
        if lower is not None or upper is not None:
            lo = lower if lower is not None else -float("inf")
            hi = upper if upper is not None else float("inf")
            bounded_log_alpha = torch.clamp(bounded_log_alpha, min=lo, max=hi)
        return bounded_log_alpha.exp()

    def algo_variant(self) -> str:
        return str(getattr(self.args, "algo_variant", "own") or "own").strip().lower()

    def _alpha_log_bounds(self) -> tuple[float | None, float | None]:
        min_log_alpha = None
        if float(getattr(self.args, "alpha_min", 0.0)) > 0.0:
            min_log_alpha = float(np.log(max(1e-6, float(self.args.alpha_min))))
        max_log_alpha = None
        if float(getattr(self.args, "alpha_max", 0.0)) > 0.0:
            max_log_alpha = float(np.log(float(self.args.alpha_max)))
        return min_log_alpha, max_log_alpha

    def _apply_alpha_bounds_(self) -> None:
        if self.fixed_alpha >= 0.0:
            with torch.no_grad():
                self.log_alpha.data.copy_(
                    torch.tensor(
                        [np.log(max(self.fixed_alpha, 1e-12))],
                        device=self.log_alpha.device,
                        dtype=self.log_alpha.dtype,
                    )
                )
            return
        lower, upper = self._alpha_log_bounds()
        if lower is None and upper is None:
            return
        lo = lower if lower is not None else -float("inf")
        hi = upper if upper is not None else float("inf")
        with torch.no_grad():
            self.log_alpha.data.clamp_(min=lo, max=hi)

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
            "pref_linked_effective_rows": 0.0,
            "pref_linked_fraction": 0.0,
            "pref_linked_effective_fraction": 0.0,
            "pref_lambda": 0.0,
            "pref_lambda_delta": 0.0,
            "pref_lambda_min": 0.0,
            "pref_lambda_max_value": 0.0,
            "pref_lambda_std": 0.0,
            "pref_lambda_active_fraction": 0.0,
            "pref_lambda_per_linked_enabled": 0.0,
            "pref_dual_violation": 0.0,
            "pref_dual_signal": 0.0,
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
            "q_min_data": 0.0,
            "q_min_teacher_data_sum": 0.0,
            "q_min_teacher_data_count": 0.0,
            "q_min_non_teacher_data_sum": 0.0,
            "q_min_non_teacher_data_count": 0.0,
            "pref_q_delta_sum": 0.0,
            "pref_q_delta_count": 0.0,
            "pref_action_delta_sum": 0.0,
            "pref_action_delta_count": 0.0,
            "pref_action_weight_sum": 0.0,
            "pref_action_weight_count": 0.0,
            "reward": 0.0,
            "reward_abs": 0.0,
            "alpha_value": 0.0,
            "alpha_metric_count": 0.0,
            "alpha_optimizer_step_count": 0.0,
            "alpha_student_only_rows": 0.0,
            "alpha_student_only_fraction": 0.0,
            "alpha_student_only_skipped_updates": 0.0,
            "pvp_td_reward": 0.0,
            "pvp_proxy_teacher_loss": 0.0,
            "pvp_proxy_student_loss": 0.0,
            "intervened_batch_fraction": 0.0,
            "action_override_batch_fraction": 0.0,
            "intervention_override_mismatch_fraction": 0.0,
            "intervention_override_false_negative_fraction": 0.0,
            "intervention_override_false_positive_fraction": 0.0,
            "pvp_intervened_batch_fraction": 0.0,
            "eil_good_loss": 0.0,
            "eil_bad_loss": 0.0,
            "eil_pair_loss": 0.0,
            "eil_good_batch_fraction": 0.0,
            "eil_bad_batch_fraction": 0.0,
            "eil_intervened_batch_fraction": 0.0,
            "timing_sample_s": 0.0,
            "timing_opt_s": 0.0,
            "actor_update_count": 0.0,
            "alpha_update_count": 0.0,
            "demo_rows_requested": 0.0,
            "demo_rows_sampled": 0.0,
            "demo_fallback_updates": 0.0,
            "demo_buffer_nonempty_updates": 0.0,
        }
        updates_count = 0
        cta_ratio = max(1, int(getattr(args, "cta_ratio", 1)))
        algo_variant = self.algo_variant()
        pvp_enabled = algo_variant == "pvp"
        eil_enabled = algo_variant == "eil"
        deterministic_variant = pvp_enabled or eil_enabled

        for _ in range(args.num_updates):
            sample_t0 = time.perf_counter()
            if pvp_enabled:
                replay_ready = int(getattr(replay_buffer, "size", 0)) > 0
                human_ready = demo_buffer is not None and int(getattr(demo_buffer, "size", 0)) > 0
                if replay_ready and human_ready and b_demo > 0:
                    novice_batch = replay_buffer.sample(max(1, base_batch - b_demo))
                    human_batch = demo_buffer.sample(max(1, b_demo))
                    batch = TensorDict.cat([novice_batch, human_batch], dim=0)
                elif human_ready and demo_buffer is not None:
                    batch = demo_buffer.sample(base_batch)
                elif replay_ready:
                    batch = replay_buffer.sample(base_batch)
                else:
                    break
            else:
                batch = replay_buffer.sample(main_batch)
                batch_device = batch["actions"].device
                batch["pref_lagrangian_buffer_id"] = torch.zeros(
                    batch.batch_size, device=batch_device, dtype=torch.long
                )
            demo_batch = None
            demo_bc_obs = None
            demo_bc_actions = None
            pref_states_for_actor = None
            pref_teacher_actions_for_actor = None
            if (not pvp_enabled) and demo_buffer is not None and b_demo > 0:
                try:
                    demo_size = getattr(demo_buffer, "size", 0)
                except Exception:
                    demo_size = 0
                metrics_accumulator["demo_rows_requested"] += float(b_demo)
                if demo_size > 0:
                    metrics_accumulator["demo_buffer_nonempty_updates"] += 1.0
                    demo_batch = demo_buffer.sample(b_demo)
                    demo_batch_device = demo_batch["actions"].device
                    demo_batch["pref_lagrangian_buffer_id"] = torch.ones(
                        demo_batch.batch_size, device=demo_batch_device, dtype=torch.long
                    )
                    batch = TensorDict.cat([batch, demo_batch], dim=0)
                    demo_bc_obs = demo_batch["observations"]
                    demo_bc_actions = demo_batch["actions"]
                    metrics_accumulator["demo_rows_sampled"] += float(b_demo)
                else:
                    metrics_accumulator["demo_fallback_updates"] += 1.0
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
                    alpha_tensor = torch.zeros_like(self.current_alpha_tensor()) if deterministic_variant else self.current_alpha_tensor()
                    next_actions, next_log_pi, next_mean_actions, _ = self.actor_forward(next_obs_batch)
                    if deterministic_variant:
                        next_actions = next_mean_actions
                    next_features_target = self.critic_target_backbone(self.reshape_obs(next_obs_batch))
                    target_q_list = self.critic_target_heads(next_features_target, next_actions)
                    min_next_q = torch.min(torch.stack(target_q_list, dim=0), dim=0).values
                    if not deterministic_variant:
                        min_next_q = min_next_q - alpha_tensor * next_log_pi
                    td_rewards = (
                        rewards_batch
                        if (
                            (not deterministic_variant)
                            or (pvp_enabled and bool(getattr(args, "pvp_include_env_reward_in_td", False)))
                        )
                        else torch.zeros_like(rewards_batch)
                    )
                    target_q = td_rewards + (1.0 - dones_batch) * (args.gamma * min_next_q)

                current_backbone = self.actor_backbone if args.arch_shared_trunk else self.critic_backbone
                current_features = current_backbone(self.reshape_obs(obs_batch))
                current_q_list = self.critic_heads(current_features, actions_batch)
                q_stack_data = torch.stack(current_q_list, dim=0).squeeze(-1)
                min_q_data = torch.min(q_stack_data, dim=0).values
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
                pref_dual_signal_value = 0.0
                pref_violation_ema_value = float(self._pref_violation_ema)
                pref_lambda_value = float(self.pref_lambda)
                pref_lambda_delta_value = 0.0
                pref_lambda_min_value = float(self.pref_lambda)
                pref_lambda_max_value = float(self.pref_lambda)
                pref_lambda_std_value = 0.0
                pref_lambda_active_fraction_value = 1.0 if self.pref_lambda > 1e-8 else 0.0

                critic_loss_replay_value = float(critic_loss_replay_tensor.detach().cpu().item())
                q_disagreement_data = torch.max(q_stack_data, dim=0).values - torch.min(q_stack_data, dim=0).values
                q_disagreement_data_value = float(q_disagreement_data.detach().mean().cpu().item())
                q_min_data_value = float(min_q_data.detach().mean().cpu().item())
                pvp_td_reward_value = float(td_rewards.detach().mean().cpu().item())
                pvp_proxy_teacher_loss_value = 0.0
                pvp_proxy_student_loss_value = 0.0
                pvp_intervened_batch_fraction_value = 0.0
                eil_good_loss_value = 0.0
                eil_bad_loss_value = 0.0
                eil_pair_loss_value = 0.0
                eil_good_batch_fraction_value = 0.0
                eil_bad_batch_fraction_value = 0.0
                eil_intervened_batch_fraction_value = 0.0
                critic_loss_pref_value = 0.0
                critic_loss_pref_weighted_value = 0.0
                critic_loss_total_value = critic_loss_replay_value

                has_teacher_intervened = False
                teacher_mask = None
                linked_action_epsilon = float(getattr(args, "pref_linked_action_epsilon", 1e-6))
                try:
                    has_teacher_intervened = "teacher_intervened" in batch.keys(include_nested=False)
                except TypeError:
                    has_teacher_intervened = "teacher_intervened" in batch.keys()
                except Exception:
                    has_teacher_intervened = "teacher_intervened" in batch
                if has_teacher_intervened:
                    teacher_mask = batch["teacher_intervened"].to(torch.bool)
                    pvp_intervened_batch_fraction_value = float(teacher_mask.float().mean().detach().cpu().item())
                    eil_intervened_batch_fraction_value = pvp_intervened_batch_fraction_value
                    metrics_accumulator["intervened_batch_fraction"] += pvp_intervened_batch_fraction_value

                has_student_actions = False
                try:
                    has_student_actions = "student_actions" in batch.keys(include_nested=False)
                except TypeError:
                    has_student_actions = "student_actions" in batch.keys()
                except Exception:
                    has_student_actions = "student_actions" in batch
                if has_student_actions:
                    student_actions_batch_diag = batch["student_actions"].to(torch.float32)
                    action_override_mask = (
                        torch.abs(actions_batch - student_actions_batch_diag).sum(dim=-1) > linked_action_epsilon
                    )
                    action_override_fraction_value = float(action_override_mask.float().mean().detach().cpu().item())
                    metrics_accumulator["action_override_batch_fraction"] += action_override_fraction_value
                    if teacher_mask is not None:
                        mismatch_mask = teacher_mask != action_override_mask
                        metrics_accumulator["intervention_override_mismatch_fraction"] += float(
                            mismatch_mask.float().mean().detach().cpu().item()
                        )
                        metrics_accumulator["intervention_override_false_negative_fraction"] += float(
                            ((~teacher_mask) & action_override_mask).float().mean().detach().cpu().item()
                        )
                        metrics_accumulator["intervention_override_false_positive_fraction"] += float(
                            (teacher_mask & (~action_override_mask)).float().mean().detach().cpu().item()
                        )

                # Preference ranking loss
                pref_sampling_mode = str(getattr(args, "pref_sampling_mode", "independent")).strip().lower()
                if pvp_enabled:
                    has_student_actions = False
                    try:
                        has_student_actions = "student_actions" in batch.keys(include_nested=False)
                    except TypeError:
                        has_student_actions = "student_actions" in batch.keys()
                    except Exception:
                        has_student_actions = "student_actions" in batch
                    student_actions_batch = batch["student_actions"] if has_student_actions else actions_batch
                    if teacher_mask is not None and bool(teacher_mask.any().item()):
                        q_novice_list = self.critic_heads(current_features, student_actions_batch)
                        teacher_mask_f = teacher_mask.to(dtype=torch.float32).unsqueeze(-1)
                        proxy_teacher_terms = []
                        proxy_student_terms = []
                        q_value_bound = float(getattr(args, "pvp_proxy_value_bound", 1.0))
                        for q_behavior, q_novice in zip(current_q_list, q_novice_list):
                            proxy_teacher_terms.append(
                                (teacher_mask_f * F.mse_loss(
                                    q_behavior,
                                    q_value_bound * torch.ones_like(q_behavior),
                                    reduction="none",
                                )).mean()
                            )
                            proxy_student_terms.append(
                                (teacher_mask_f * F.mse_loss(
                                    q_novice,
                                    -q_value_bound * torch.ones_like(q_novice),
                                    reduction="none",
                                )).mean()
                            )
                        proxy_teacher_loss = torch.stack(proxy_teacher_terms).mean()
                        proxy_student_loss = torch.stack(proxy_student_terms).mean()
                        qf_loss = qf_loss + proxy_teacher_loss + proxy_student_loss
                        pvp_proxy_teacher_loss_value = float(proxy_teacher_loss.detach().cpu().item())
                        pvp_proxy_student_loss_value = float(proxy_student_loss.detach().cpu().item())
                        critic_loss_total_value = (
                            critic_loss_replay_value
                            + pvp_proxy_teacher_loss_value
                            + pvp_proxy_student_loss_value
                        )
                    else:
                        critic_loss_total_value = critic_loss_replay_value
                elif eil_enabled:
                    has_student_actions = False
                    try:
                        has_student_actions = "student_actions" in batch.keys(include_nested=False)
                    except TypeError:
                        has_student_actions = "student_actions" in batch.keys()
                    except Exception:
                        has_student_actions = "student_actions" in batch
                    student_actions_batch = batch["student_actions"] if has_student_actions else actions_batch
                    has_eil_good = False
                    has_eil_bad = False
                    try:
                        has_eil_good = "eil_good" in batch.keys(include_nested=False)
                        has_eil_bad = "eil_bad" in batch.keys(include_nested=False)
                    except TypeError:
                        has_eil_good = "eil_good" in batch.keys()
                        has_eil_bad = "eil_bad" in batch.keys()
                    except Exception:
                        has_eil_good = "eil_good" in batch
                        has_eil_bad = "eil_bad" in batch
                    eil_good_mask = batch["eil_good"].to(torch.bool) if has_eil_good else None
                    eil_bad_mask = batch["eil_bad"].to(torch.bool) if has_eil_bad else None
                    if eil_good_mask is None:
                        eil_good_mask = torch.zeros(actions_batch.shape[0], dtype=torch.bool, device=self.device)
                    if eil_bad_mask is None:
                        eil_bad_mask = torch.zeros(actions_batch.shape[0], dtype=torch.bool, device=self.device)
                    teacher_mask_local = teacher_mask
                    if teacher_mask_local is None:
                        teacher_mask_local = torch.zeros(actions_batch.shape[0], dtype=torch.bool, device=self.device)

                    good_mask = eil_good_mask | teacher_mask_local
                    bad_exec_mask = eil_bad_mask
                    bad_student_mask = teacher_mask_local
                    eil_good_batch_fraction_value = float(good_mask.float().mean().detach().cpu().item())
                    eil_bad_batch_fraction_value = float(
                        (bad_exec_mask | bad_student_mask).float().mean().detach().cpu().item()
                    )

                    q_student_list = self.critic_heads(current_features, student_actions_batch)
                    threshold = float(getattr(args, "eil_threshold", 0.0))
                    good_margin = float(getattr(args, "eil_good_margin", 0.0))
                    bad_margin = float(getattr(args, "eil_bad_margin", 0.01))
                    pair_margin = float(getattr(args, "eil_pair_margin", 0.01))
                    good_target = threshold + good_margin
                    bad_target = threshold - bad_margin
                    good_mask_f = good_mask.to(dtype=torch.float32).unsqueeze(-1)
                    bad_exec_mask_f = bad_exec_mask.to(dtype=torch.float32).unsqueeze(-1)
                    bad_student_mask_f = bad_student_mask.to(dtype=torch.float32).unsqueeze(-1)
                    good_terms = []
                    bad_terms = []
                    pair_terms = []
                    for q_behavior, q_student in zip(current_q_list, q_student_list):
                        good_terms.append(
                            (good_mask_f * torch.clamp(good_target - q_behavior, min=0.0)).mean()
                        )
                        bad_terms.append(
                            (bad_exec_mask_f * torch.clamp(q_behavior - bad_target, min=0.0)).mean()
                            + (bad_student_mask_f * torch.clamp(q_student - bad_target, min=0.0)).mean()
                        )
                        pair_terms.append(
                            (bad_student_mask_f * torch.clamp(pair_margin - (q_behavior - q_student), min=0.0)).mean()
                        )
                    eil_good_loss = torch.stack(good_terms).mean()
                    eil_bad_loss = torch.stack(bad_terms).mean()
                    eil_pair_loss = torch.stack(pair_terms).mean()
                    qf_loss = qf_loss + eil_good_loss + eil_bad_loss + eil_pair_loss
                    eil_good_loss_value = float(eil_good_loss.detach().cpu().item())
                    eil_bad_loss_value = float(eil_bad_loss.detach().cpu().item())
                    eil_pair_loss_value = float(eil_pair_loss.detach().cpu().item())
                    critic_loss_total_value = (
                        critic_loss_replay_value
                        + eil_good_loss_value
                        + eil_bad_loss_value
                        + eil_pair_loss_value
                    )
                elif float(args.pref_rank_weight) > 0.0:
                    pref_states = None
                    pref_teacher_actions = None
                    pref_student_actions = None
                    pref_pair_weights = None
                    pref_replay_env_indices = None
                    pref_replay_buffer_indices = None
                    pref_action_delta_mean_value = 0.0
                    pref_action_weight_mean_value = 0.0

                    def _weighted_mean(loss_tensor: torch.Tensor, sample_weights: torch.Tensor | None) -> torch.Tensor:
                        if sample_weights is None:
                            return loss_tensor.mean()
                        weights = sample_weights.to(device=loss_tensor.device, dtype=loss_tensor.dtype)
                        if loss_tensor.ndim == 1:
                            weight_view = weights
                        elif loss_tensor.ndim == 2:
                            weight_view = weights.unsqueeze(-1)
                        else:
                            weight_view = weights.unsqueeze(0).unsqueeze(-1)
                        denom = torch.clamp(weight_view.sum(), min=1e-8)
                        return (loss_tensor * weight_view).sum() / denom

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
                                metrics_accumulator["pref_linked_fraction"] += (
                                    float(linked_rows) / max(1.0, float(linked_mask.numel()))
                                )
                                if linked_rows > 0:
                                    linked_delta_abs = torch.abs(actions_batch - linked_student_actions)
                                    linked_delta_l1 = linked_delta_abs.sum(dim=-1)
                                    linked_delta_mean_abs = linked_delta_abs.mean(dim=-1)
                                    effective_linked_mask = linked_mask & (linked_delta_l1 > linked_action_epsilon)
                                    effective_rows = int(effective_linked_mask.sum().item())
                                    metrics_accumulator["pref_linked_effective_rows"] += float(effective_rows)
                                    metrics_accumulator["pref_linked_effective_fraction"] += (
                                        float(effective_rows) / max(1.0, float(linked_mask.numel()))
                                    )
                                    if effective_rows > 0:
                                        pref_states = obs_batch[effective_linked_mask]
                                        pref_teacher_actions = actions_batch[effective_linked_mask]
                                        pref_student_actions = linked_student_actions[effective_linked_mask]
                                        try:
                                            has_replay_indices = (
                                                "replay_env_indices" in batch.keys(include_nested=False)
                                                and "replay_buffer_indices" in batch.keys(include_nested=False)
                                            )
                                        except TypeError:
                                            has_replay_indices = (
                                                "replay_env_indices" in batch.keys()
                                                and "replay_buffer_indices" in batch.keys()
                                            )
                                        except Exception:
                                            has_replay_indices = (
                                                "replay_env_indices" in batch
                                                and "replay_buffer_indices" in batch
                                            )
                                        if has_replay_indices:
                                            pref_replay_env_indices = batch["replay_env_indices"][effective_linked_mask]
                                            pref_replay_buffer_indices = batch["replay_buffer_indices"][effective_linked_mask]
                                            try:
                                                has_buffer_ids = "pref_lagrangian_buffer_id" in batch.keys(include_nested=False)
                                            except TypeError:
                                                has_buffer_ids = "pref_lagrangian_buffer_id" in batch.keys()
                                            except Exception:
                                                has_buffer_ids = "pref_lagrangian_buffer_id" in batch
                                            if has_buffer_ids:
                                                pref_replay_source_mask = (
                                                    batch["pref_lagrangian_buffer_id"][effective_linked_mask].to(torch.long) == 0
                                                )
                                            else:
                                                pref_replay_source_mask = torch.ones(
                                                    effective_rows,
                                                    device=self.device,
                                                    dtype=torch.bool,
                                                )
                                        else:
                                            pref_replay_source_mask = None
                                        pref_pair_weights = torch.ones(
                                            effective_rows, device=self.device, dtype=torch.float32
                                        )
                                        weight_scale = float(
                                            getattr(args, "pref_linked_action_weight_scale", 0.0)
                                        )
                                        effective_delta_mean_abs = linked_delta_mean_abs[effective_linked_mask]
                                        if weight_scale > 0.0:
                                            pref_pair_weights = torch.clamp(
                                                effective_delta_mean_abs / weight_scale,
                                                min=0.0,
                                                max=1.0,
                                            )
                                        pref_action_delta_mean_value = float(
                                            effective_delta_mean_abs.detach().mean().cpu().item()
                                        )
                                        pref_action_weight_mean_value = float(
                                            pref_pair_weights.detach().mean().cpu().item()
                                        )
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
                            pref_pair_weights = None

                    if (
                        pref_states is not None
                        and pref_teacher_actions is not None
                        and pref_student_actions is not None
                        and pref_states.shape[0] > 0
                    ):
                        pref_features = current_backbone(self.reshape_obs(pref_states))
                        q_pos = self.critic_heads(pref_features, pref_teacher_actions)
                        q_neg = self.critic_heads(pref_features, pref_student_actions)
                        q_pos_stack = torch.stack(q_pos, dim=0)
                        q_neg_stack = torch.stack(q_neg, dim=0)
                        pref_critic_scope = str(getattr(args, "pref_critic_scope", "all")).strip().lower()
                        if pref_critic_scope == "min":
                            q_pos_base = torch.min(q_pos_stack, dim=0).values
                            q_neg_base = torch.min(q_neg_stack, dim=0).values
                        else:
                            # Per-head preference ranking: every critic is trained to rank teacher > student.
                            q_pos_base = q_pos_stack
                            q_neg_base = q_neg_stack
                        if bool(getattr(args, "pref_stopgrad_positive", False)):
                            q_pos_term = q_pos_base.detach()
                        else:
                            q_pos_term = q_pos_base
                        margin = float(args.pref_rank_margin)
                        delta = q_pos_term - q_neg_base
                        pref_loss_type = str(getattr(args, "pref_loss_type", "margin")).strip().lower()
                        pref_lagrangian_scope = str(
                            getattr(args, "pref_lagrangian_scope", "global")
                        ).strip().lower()
                        if pref_loss_type == "bradley_terry":
                            rank_terms = F.softplus(-delta)
                            rank_loss = _weighted_mean(rank_terms, pref_pair_weights)
                            rank_loss_weighted = float(args.pref_rank_weight) * rank_loss
                        elif pref_loss_type == "hinge":
                            rank_terms = torch.clamp(margin - delta, min=0.0)
                            rank_loss = _weighted_mean(rank_terms, pref_pair_weights)
                            rank_loss_weighted = float(args.pref_rank_weight) * rank_loss
                        elif pref_loss_type == "lagrangian":
                            pref_lagrangian_violation_type = str(
                                getattr(args, "pref_lagrangian_violation_type", "hinge")
                            ).strip().lower()
                            if pref_lagrangian_violation_type == "smooth":
                                violation = F.softplus(margin - delta)
                            else:
                                violation = torch.clamp(margin - delta, min=0.0)
                            pref_violation_value = float(violation.detach().mean().cpu().item())
                            if self.pref_violation_clip > 0.0:
                                violation = torch.clamp(violation, max=self.pref_violation_clip)
                            use_per_linked_lambda = (
                                pref_sampling_mode == "linked"
                                and pref_lagrangian_scope == "per_linked"
                                and pref_replay_env_indices is not None
                                and pref_replay_buffer_indices is not None
                                and pref_replay_source_mask is not None
                                and bool(pref_replay_source_mask.any().item())
                                and hasattr(replay_buffer, "gather_pref_lagrangian_state")
                                and hasattr(replay_buffer, "update_pref_lagrangian_state")
                            )
                            if use_per_linked_lambda:
                                replay_row_mask = pref_replay_source_mask
                                per_lambda, per_ema, _ = replay_buffer.gather_pref_lagrangian_state(
                                    pref_replay_env_indices[replay_row_mask],
                                    pref_replay_buffer_indices[replay_row_mask],
                                    init_lambda=float(getattr(args, "pref_lambda_init", 0.0)),
                                    device=self.device,
                                )
                                lambda_all = torch.full(
                                    (violation.shape[-2],),
                                    float(self.pref_lambda),
                                    device=self.device,
                                    dtype=violation.dtype,
                                )
                                lambda_all[replay_row_mask] = per_lambda.to(dtype=violation.dtype)
                                prev_per_lambda = per_lambda
                                violation_for_dual = violation.detach()
                                if violation_for_dual.ndim == 3:
                                    sample_violation_all = violation_for_dual.mean(dim=(0, 2))
                                elif violation_for_dual.ndim == 2:
                                    sample_violation_all = violation_for_dual.mean(dim=-1)
                                else:
                                    sample_violation_all = violation_for_dual
                                sample_violation = sample_violation_all[replay_row_mask]
                                if self.pref_lambda_ema > 0.0:
                                    per_ema = (
                                        self.pref_lambda_ema * per_ema
                                        + (1.0 - self.pref_lambda_ema) * sample_violation
                                    )
                                    dual_violation_tensor = per_ema
                                else:
                                    dual_violation_tensor = sample_violation
                                dual_signal = dual_violation_tensor - float(self.pref_violation_target)
                                if self.pref_lambda_lr > 0.0:
                                    per_lambda = torch.clamp(
                                        per_lambda + self.pref_lambda_lr * dual_signal,
                                        min=0.0,
                                    )
                                    if self.pref_lambda_max > 0.0:
                                        per_lambda = torch.clamp(per_lambda, max=self.pref_lambda_max)
                                replay_buffer.update_pref_lagrangian_state(
                                    pref_replay_env_indices[replay_row_mask],
                                    pref_replay_buffer_indices[replay_row_mask],
                                    lambdas=per_lambda,
                                    violation_emas=per_ema,
                                )
                                pref_lambda_value = float(per_lambda.detach().mean().cpu().item())
                                pref_lambda_delta_value = float(
                                    (per_lambda - prev_per_lambda).detach().mean().cpu().item()
                                )
                                pref_lambda_min_value = float(per_lambda.detach().min().cpu().item())
                                pref_lambda_max_value = float(per_lambda.detach().max().cpu().item())
                                pref_lambda_std_value = float(
                                    per_lambda.detach().std(unbiased=False).cpu().item()
                                )
                                pref_lambda_active_fraction_value = float(
                                    (per_lambda.detach() > 1e-8).float().mean().cpu().item()
                                )
                                pref_violation_ema_value = float(per_ema.detach().mean().cpu().item())
                                pref_dual_violation_value = float(
                                    dual_violation_tensor.detach().mean().cpu().item()
                                )
                                pref_dual_signal_value = float(dual_signal.detach().mean().cpu().item())
                                lambda_view = lambda_all
                                if violation.ndim == 3:
                                    lambda_view = lambda_view.view(1, -1, 1)
                                elif violation.ndim == 2:
                                    lambda_view = lambda_view.view(-1, 1)
                                rank_terms = lambda_view * violation
                            else:
                                prev_pref_lambda = float(self.pref_lambda)
                                if self.pref_lambda_ema > 0.0:
                                    self._pref_violation_ema = (
                                        self.pref_lambda_ema * self._pref_violation_ema
                                        + (1.0 - self.pref_lambda_ema) * float(pref_violation_value)
                                    )
                                    dual_violation = float(self._pref_violation_ema)
                                else:
                                    dual_violation = float(pref_violation_value)
                                pref_dual_violation_value = float(dual_violation)
                                pref_dual_signal_value = float(dual_violation - self.pref_violation_target)
                                if self.pref_lambda_lr > 0.0:
                                    self.pref_lambda = max(0.0, self.pref_lambda + self.pref_lambda_lr * pref_dual_signal_value)
                                    if self.pref_lambda_max > 0.0:
                                        self.pref_lambda = min(self.pref_lambda, self.pref_lambda_max)
                                pref_lambda_value = float(self.pref_lambda)
                                pref_lambda_delta_value = pref_lambda_value - prev_pref_lambda
                                pref_lambda_min_value = pref_lambda_value
                                pref_lambda_max_value = pref_lambda_value
                                pref_lambda_std_value = 0.0
                                pref_lambda_active_fraction_value = 1.0 if pref_lambda_value > 1e-8 else 0.0
                                pref_violation_ema_value = float(self._pref_violation_ema)
                                rank_terms = pref_lambda_value * violation
                            rank_loss = _weighted_mean(rank_terms, pref_pair_weights)
                            # Lagrangian already scales by lambda; keep extra rank weight neutral.
                            rank_loss_weighted = rank_loss
                        else:
                            rank_terms = F.softplus(margin - delta)
                            rank_loss = _weighted_mean(rank_terms, pref_pair_weights)
                            rank_loss_weighted = float(args.pref_rank_weight) * rank_loss
                        pref_q_delta_value = float(delta.detach().mean().cpu().item())
                        metrics_accumulator["pref_q_delta_sum"] += pref_q_delta_value
                        metrics_accumulator["pref_q_delta_count"] += 1.0
                        if pref_sampling_mode == "linked":
                            metrics_accumulator["pref_action_delta_sum"] += pref_action_delta_mean_value
                            metrics_accumulator["pref_action_delta_count"] += 1.0
                            metrics_accumulator["pref_action_weight_sum"] += pref_action_weight_mean_value
                            metrics_accumulator["pref_action_weight_count"] += 1.0
                        qf_loss = qf_loss + rank_loss_weighted
                        pref_states_for_actor = pref_states
                        pref_teacher_actions_for_actor = pref_teacher_actions
                critic_loss_pref_value = float(rank_loss.detach().cpu().item())
                critic_loss_pref_weighted_value = float(rank_loss_weighted.detach().cpu().item())
                if (not pvp_enabled) and (not eil_enabled):
                    critic_loss_total_value = critic_loss_replay_value + critic_loss_pref_weighted_value

            self.scaler.scale(qf_loss).backward()
            self.scaler.unscale_(self.critic_optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.critic_params, max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf")
            )
            self.scaler.step(self.critic_optimizer)
            self._critic_update_count += 1

            actor_loss_value = 0.0
            actor_loss_sac_value = 0.0
            bc_demo_loss_value = 0.0
            bc_pref_loss_value = 0.0
            entropy_value = 0.0
            action_norm_value = 0.0
            min_q_pi_mean = 0.0
            q_disagreement_pi_value = 0.0
            reward_mean = float(rewards_batch.detach().mean().cpu().item())
            target_q_mean = float(target_q.detach().mean().cpu().item())
            alpha_loss_value = 0.0
            alpha_value = float(self.current_alpha_tensor().detach().cpu().item())

            should_update_actor = (self._critic_update_count % cta_ratio) == 0
            if should_update_actor:
                # Actor update
                self.actor_optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=self.amp_device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                    alpha_tensor = torch.zeros_like(self.current_alpha_tensor()).detach() if deterministic_variant else self.current_alpha_tensor().detach()
                    pi_actions, log_pi, mean_actions, _ = self.actor_forward(obs_batch)
                    if deterministic_variant:
                        pi_actions = mean_actions
                    actor_backbone_eval = current_backbone
                    q_pi_list = self.critic_heads(actor_backbone_eval(self.reshape_obs(obs_batch)), pi_actions)
                    q_stack_pi = torch.stack(q_pi_list, dim=0)
                    min_q_pi = torch.min(q_stack_pi, dim=0).values
                    actor_loss_sac = (-min_q_pi).mean() if deterministic_variant else (alpha_tensor * log_pi - min_q_pi).mean()
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
                entropy_value = 0.0 if deterministic_variant else float((-log_pi).detach().mean().cpu().item())
                action_norm_value = float(pi_actions.detach().norm(dim=-1).mean().cpu().item())
                min_q_pi_mean = float(min_q_pi.detach().mean().cpu().item())
                q_disagreement_pi = torch.max(q_stack_pi, dim=0).values - torch.min(q_stack_pi, dim=0).values
                q_disagreement_pi_value = float(q_disagreement_pi.detach().mean().cpu().item())

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
                metrics_accumulator["actor_update_count"] += 1.0

                # Alpha update
                if deterministic_variant:
                    self.alpha_optimizer.zero_grad(set_to_none=True)
                    alpha_loss_value = 0.0
                    alpha_value = 0.0
                elif self.fixed_alpha >= 0.0:
                    self.alpha_optimizer.zero_grad(set_to_none=True)
                    alpha_loss_value = 0.0
                    alpha_value = float(self.fixed_alpha)
                elif total_env_steps < int(getattr(args, "alpha_freeze_steps", 0)):
                    self.alpha_optimizer.zero_grad(set_to_none=True)
                    alpha_loss_value = 0.0
                    alpha_value = float(self.log_alpha.exp().detach().cpu().item())
                else:
                    self.alpha_optimizer.zero_grad(set_to_none=True)
                    _, log_pi_curr, _, _ = self.actor_forward(obs_batch)
                    log_pi_detached = log_pi_curr.detach()
                    alpha_rows_total = int(log_pi_detached.shape[0])
                    alpha_mask = None
                    if bool(getattr(args, "alpha_update_student_only", False)) and teacher_mask is not None:
                        alpha_mask = ~teacher_mask.reshape(-1)
                    if alpha_mask is None:
                        alpha_rows_used = alpha_rows_total
                        alpha_log_pi = log_pi_detached
                    else:
                        alpha_rows_used = int(alpha_mask.sum().item())
                        alpha_log_pi = log_pi_detached[alpha_mask]
                    metrics_accumulator["alpha_student_only_rows"] += float(alpha_rows_used)
                    if alpha_rows_total > 0:
                        metrics_accumulator["alpha_student_only_fraction"] += float(alpha_rows_used) / float(alpha_rows_total)
                    if alpha_rows_used > 0:
                        alpha_loss = (-self.log_alpha.exp() * (alpha_log_pi + self.target_entropy)).mean()
                        alpha_loss.backward()
                        self.alpha_optimizer.step()
                        self._apply_alpha_bounds_()
                        alpha_loss_value = float(alpha_loss.detach().cpu().item())
                        alpha_value = float(self.current_alpha_tensor().detach().cpu().item())
                        metrics_accumulator["alpha_optimizer_step_count"] += 1.0
                    else:
                        alpha_loss_value = 0.0
                        alpha_value = float(self.current_alpha_tensor().detach().cpu().item())
                        metrics_accumulator["alpha_student_only_skipped_updates"] += 1.0
                metrics_accumulator["alpha_metric_count"] += 1.0
                metrics_accumulator["alpha_update_count"] += 1.0
                metrics_accumulator["alpha_value"] += alpha_value

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
            metrics_accumulator["pref_lambda_min"] += pref_lambda_min_value
            metrics_accumulator["pref_lambda_max_value"] += pref_lambda_max_value
            metrics_accumulator["pref_lambda_std"] += pref_lambda_std_value
            metrics_accumulator["pref_lambda_active_fraction"] += pref_lambda_active_fraction_value
            metrics_accumulator["pref_lambda_per_linked_enabled"] += (
                1.0 if str(getattr(args, "pref_lagrangian_scope", "global")).strip().lower() == "per_linked" else 0.0
            )
            metrics_accumulator["pref_dual_violation"] += pref_dual_violation_value
            metrics_accumulator["pref_dual_signal"] += pref_dual_signal_value
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
            metrics_accumulator["q_min_data"] += q_min_data_value
            if teacher_mask is not None:
                teacher_count = float(teacher_mask.sum().item())
                if teacher_count > 0.0:
                    metrics_accumulator["q_min_teacher_data_sum"] += float(
                        min_q_data[teacher_mask].detach().sum().cpu().item()
                    )
                    metrics_accumulator["q_min_teacher_data_count"] += teacher_count
                non_teacher_mask = ~teacher_mask
                non_teacher_count = float(non_teacher_mask.sum().item())
                if non_teacher_count > 0.0:
                    metrics_accumulator["q_min_non_teacher_data_sum"] += float(
                        min_q_data[non_teacher_mask].detach().sum().cpu().item()
                    )
                    metrics_accumulator["q_min_non_teacher_data_count"] += non_teacher_count
            metrics_accumulator["reward"] += reward_mean
            metrics_accumulator["reward_abs"] += float(rewards_batch.detach().abs().mean().cpu().item())
            metrics_accumulator["pvp_td_reward"] += pvp_td_reward_value
            metrics_accumulator["pvp_proxy_teacher_loss"] += pvp_proxy_teacher_loss_value
            metrics_accumulator["pvp_proxy_student_loss"] += pvp_proxy_student_loss_value
            metrics_accumulator["pvp_intervened_batch_fraction"] += pvp_intervened_batch_fraction_value
            metrics_accumulator["eil_good_loss"] += eil_good_loss_value
            metrics_accumulator["eil_bad_loss"] += eil_bad_loss_value
            metrics_accumulator["eil_pair_loss"] += eil_pair_loss_value
            metrics_accumulator["eil_good_batch_fraction"] += eil_good_batch_fraction_value
            metrics_accumulator["eil_bad_batch_fraction"] += eil_bad_batch_fraction_value
            metrics_accumulator["eil_intervened_batch_fraction"] += eil_intervened_batch_fraction_value
            metrics_accumulator["timing_opt_s"] += float(time.perf_counter() - opt_t0)
            updates_count += 1

        return metrics_accumulator, updates_count
