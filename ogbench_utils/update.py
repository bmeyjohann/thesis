from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tensordict import TensorDict

from .buffers import CounterfactualBuffer, PreferencePairBuffer, PreferenceTDBuffer


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
        cf_buffer: CounterfactualBuffer | None = None,
        pref_buffer: PreferencePairBuffer | None = None,
        pref_td_buffer: PreferenceTDBuffer | None = None,
        pixel_shape: Optional[Tuple[int, int, int]] = None,
        pixel_random_shift_pad: int = 0,
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
        self.cf_buffer = cf_buffer
        self.pref_buffer = pref_buffer
        self.pref_td_buffer = pref_td_buffer
        self.pixel_shape = pixel_shape
        self.pixel_random_shift_pad = int(max(0, pixel_random_shift_pad))
        self._random_shift_base_grid: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}
        if (
            getattr(args, "obs_mode", "state") == "pixels"
            and self.pixel_shape is not None
            and self.pixel_random_shift_pad > 0
        ):
            self.random_shift_enabled = True
            c, h, w = self.pixel_shape
            self._pixel_flat_dim = c * h * w
        else:
            self.random_shift_enabled = False
            self._pixel_flat_dim = None

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

    def _apply_random_shift_flat(self, obs_flat: torch.Tensor) -> torch.Tensor:
        if not self.random_shift_enabled:
            return obs_flat
        if obs_flat is None or obs_flat.ndim != 2:
            return obs_flat
        if self._pixel_flat_dim is None or obs_flat.shape[1] != self._pixel_flat_dim:
            return obs_flat
        obs_view = self.reshape_obs(obs_flat)
        shifted = self._random_shift(obs_view)
        return shifted.view(obs_flat.shape[0], -1)

    def _random_shift(self, obs: torch.Tensor) -> torch.Tensor:
        pad = self.pixel_random_shift_pad
        if pad <= 0 or self.pixel_shape is None:
            return obs
        obs_padded = F.pad(obs, (pad, pad, pad, pad), mode="replicate")
        n = obs_padded.shape[0]
        base_grid = self._get_random_shift_base_grid(obs_padded.device, obs_padded.dtype)
        base_grid = base_grid.expand(n, -1, -1, -1)
        offsets = torch.randint(-pad, pad + 1, size=(n, 2), device=obs_padded.device).to(base_grid.dtype)
        height = self.pixel_shape[1] + 2 * pad
        width = self.pixel_shape[2] + 2 * pad
        shift_y = offsets[:, 0] * (2.0 / max(1, height))
        shift_x = offsets[:, 1] * (2.0 / max(1, width))
        shift = torch.stack((shift_x, shift_y), dim=-1).view(n, 1, 1, 2)
        grid = base_grid + shift
        return F.grid_sample(obs_padded, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def _get_random_shift_base_grid(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        cached = self._random_shift_base_grid.get(key)
        if cached is not None:
            return cached
        if self.pixel_shape is None:
            raise RuntimeError("Pixel shape must be provided for random shift augmentation")
        pad = self.pixel_random_shift_pad
        _, h, w = self.pixel_shape
        total_h = h + 2 * pad
        total_w = w + 2 * pad
        eps_y = 1.0 / max(1, total_h)
        eps_x = 1.0 / max(1, total_w)
        y_coords = torch.linspace(-1.0 + eps_y, 1.0 - eps_y, total_h, device=device, dtype=dtype)
        x_coords = torch.linspace(-1.0 + eps_x, 1.0 - eps_x, total_w, device=device, dtype=dtype)
        try:
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        except TypeError:  # PyTorch < 1.10 fallback
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords)
        base_grid = torch.stack((grid_x, grid_y), dim=-1)
        if pad > 0:
            base_grid = base_grid[pad:-pad, pad:-pad, :]
        base_grid = base_grid.unsqueeze(0)
        self._random_shift_base_grid[key] = base_grid
        return base_grid

    def update(
        self,
        *,
        replay_buffer,
        demo_buffer=None,
        total_env_steps: int,
        main_batch: int,
        base_batch: int,
        b_pref: int,
        b_pref_td: int,
        b_demo: int,
    ) -> Tuple[Dict[str, float], int]:
        args = self.args
        metrics_accumulator = {
            "critic_loss": 0.0,
            "actor_loss": 0.0,
            "alpha_loss": 0.0,
            "entropy": 0.0,
            "action_norm": 0.0,
            "target_q": 0.0,
            "q_min_pi": 0.0,
            "reward": 0.0,
            "reward_abs": 0.0,
            "alpha_value": 0.0,
        }
        updates_count = 0

        for _ in range(args.num_updates):
            batch = replay_buffer.sample(main_batch)
            if demo_buffer is not None and b_demo > 0:
                try:
                    demo_size = getattr(demo_buffer, "size", 0)
                except Exception:
                    demo_size = 0
                if demo_size >= b_demo:
                    demo_batch = demo_buffer.sample(b_demo)
                    batch = TensorDict.cat([batch, demo_batch], dim=0)
            obs_batch = batch["observations"]
            next_obs_batch = batch["next"]["observations"]
            actions_batch = batch["actions"]
            rewards_batch = batch["next"]["rewards"].unsqueeze(-1)
            dones_batch = batch["next"]["dones"].float().unsqueeze(-1)

            obs_batch = self.normalize_obs(obs_batch)
            next_obs_batch = self.normalize_obs(next_obs_batch)

            obs_batch = self._apply_random_shift_flat(obs_batch)
            next_obs_batch = self._apply_random_shift_flat(next_obs_batch)

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
                qf_loss = torch.tensor(0.0, device=self.device)
                for q_pred in current_q_list:
                    qf_loss = qf_loss + F.mse_loss(q_pred, target_q)

                critic_loss_value = float((qf_loss / max(1, len(current_q_list))).detach().cpu().item())

                # Counterfactual critic penalty
                if (
                    self.cf_buffer is not None
                    and args.cf_q_weight > 0.0
                    and args.cf_sample_ratio > 0.0
                    and self.cf_buffer.size > 0
                ):
                    cf_b = max(1, int(base_batch * args.cf_sample_ratio))
                    cf_sample = self.cf_buffer.sample(cf_b)
                    if cf_sample is not None:
                        cf_states = self._apply_random_shift_flat(cf_sample.states)
                        cf_features = current_backbone(self.reshape_obs(cf_states))
                        cf_q_list = self.critic_heads(cf_features, cf_sample.actions)
                        for q_cf in cf_q_list:
                            qf_loss = qf_loss + args.cf_q_weight * F.mse_loss(
                                q_cf, torch.full_like(q_cf, args.cf_penalty_target)
                            )

                # Preference ranking loss
                if (
                    self.pref_buffer is not None
                    and args.pref_rank_weight > 0.0
                    and b_pref > 0
                    and self.pref_buffer.size > 0
                ):
                    pref_sample = self.pref_buffer.sample(b_pref)
                    if pref_sample is not None:
                        pref_states = self._apply_random_shift_flat(pref_sample.states)
                        pref_features = current_backbone(self.reshape_obs(pref_states))
                        q_pos = self.critic_heads(pref_features, pref_sample.teacher_actions)
                        q_neg = self.critic_heads(pref_features, pref_sample.student_actions)
                        qpos_min = torch.min(torch.stack(q_pos, dim=0), dim=0).values
                        qneg_min = torch.min(torch.stack(q_neg, dim=0), dim=0).values
                        margin = float(args.pref_rank_margin)
                        rank_loss = F.softplus(margin - (qpos_min - qneg_min)).mean()
                        qf_loss = qf_loss + float(args.pref_rank_weight) * rank_loss

                # Preference TD loss
                if (
                    self.pref_td_buffer is not None
                    and args.pref_td_q_weight > 0.0
                    and b_pref_td > 0
                    and self.pref_td_buffer.teacher_size > 0
                    and self.pref_td_buffer.student_size > 0
                ):
                    td_batch = self.pref_td_buffer.sample(b_pref_td)
                    if td_batch is not None:
                        t_s = td_batch["t_states"]
                        t_a = td_batch["t_actions"]
                        t_r = td_batch["t_rewards"]
                        t_next_s = td_batch["t_next_states"]
                        t_done = td_batch["t_dones"]
                        s_s = td_batch["s_states"]
                        s_a = td_batch["s_actions"]
                        s_r = td_batch["s_rewards"]

                        t_s = self._apply_random_shift_flat(t_s)
                        t_next_s = self._apply_random_shift_flat(t_next_s)
                        s_s = self._apply_random_shift_flat(s_s)

                        with torch.no_grad():
                            next_actions_td, next_log_pi_td, _, _ = self.actor_forward(t_next_s)
                            next_features_td = self.critic_target_backbone(self.reshape_obs(t_next_s))
                            q_td_list = self.critic_target_heads(next_features_td, next_actions_td)
                            min_q_td = torch.min(torch.stack(q_td_list, dim=0), dim=0).values
                            min_q_td = min_q_td - self.log_alpha.exp() * next_log_pi_td
                            target_teacher = t_r + (1.0 - t_done) * (args.gamma * min_q_td)

                        teacher_features = current_backbone(self.reshape_obs(t_s))
                        teacher_q_list = self.critic_heads(teacher_features, t_a)
                        loss_teacher = sum(F.mse_loss(q_t, target_teacher) for q_t in teacher_q_list)

                        student_features = current_backbone(self.reshape_obs(s_s))
                        student_q_list = self.critic_heads(student_features, s_a)
                        loss_student = sum(F.mse_loss(q_s, s_r) for q_s in student_q_list)
                        qf_loss = qf_loss + float(args.pref_td_q_weight) * (loss_teacher + loss_student)

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
                min_q_pi = torch.min(torch.stack(q_pi_list, dim=0), dim=0).values
                actor_loss = (self.log_alpha.exp().detach() * log_pi - min_q_pi).mean()
            actor_loss_value = float(actor_loss.detach().cpu().item())
            entropy_value = float((-log_pi).detach().mean().cpu().item())
            action_norm_value = float(pi_actions.detach().norm(dim=-1).mean().cpu().item())
            target_q_mean = float(target_q.detach().mean().cpu().item())
            min_q_pi_mean = float(min_q_pi.detach().mean().cpu().item())
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

            metrics_accumulator["critic_loss"] += critic_loss_value
            metrics_accumulator["actor_loss"] += actor_loss_value
            metrics_accumulator["alpha_loss"] += alpha_loss_value
            metrics_accumulator["entropy"] += entropy_value
            metrics_accumulator["action_norm"] += action_norm_value
            metrics_accumulator["target_q"] += target_q_mean
            metrics_accumulator["q_min_pi"] += min_q_pi_mean
            metrics_accumulator["reward"] += reward_mean
            metrics_accumulator["reward_abs"] += float(rewards_batch.detach().abs().mean().cpu().item())
            metrics_accumulator["alpha_value"] += alpha_value
            updates_count += 1

        return metrics_accumulator, updates_count
