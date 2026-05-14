from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


@dataclass
class TeacherSnapshot:
    mean_disagreement_teacher: float | None
    mean_disagreement_non: float | None
    mean_disagreement_all: float | None
    corr_value: float | None
    hist_teacher: Dict[str, float]
    hist_non_teacher: Dict[str, float]
    threshold_percentages: Dict[float, float]
    qmin_all: float | None
    qmin_teacher: float | None
    qmin_non: float | None


@dataclass
class TimingWindow:
    steps: int = 0
    action_s: float = 0.0
    env_s: float = 0.0
    info_s: float = 0.0
    replay_s: float = 0.0
    sample_s: float = 0.0
    update_s: float = 0.0
    misc_s: float = 0.0
    total_s: float = 0.0

    def add(
        self,
        *,
        action_s: float,
        env_s: float,
        info_s: float,
        replay_s: float,
        sample_s: float,
        update_s: float,
        misc_s: float,
        total_s: float,
    ) -> None:
        self.steps += 1
        self.action_s += float(action_s)
        self.env_s += float(env_s)
        self.info_s += float(info_s)
        self.replay_s += float(replay_s)
        self.sample_s += float(sample_s)
        self.update_s += float(update_s)
        self.misc_s += float(misc_s)
        self.total_s += float(total_s)

    def summary(self, prefix: str = "Perf/timing") -> Dict[str, float]:
        if self.steps <= 0:
            return {}
        out: Dict[str, float] = {
            f"{prefix}/steps": float(self.steps),
            f"{prefix}/step_ms_mean": float((self.total_s / self.steps) * 1e3),
            f"{prefix}/action_ms_mean": float((self.action_s / self.steps) * 1e3),
            f"{prefix}/env_step_ms_mean": float((self.env_s / self.steps) * 1e3),
            f"{prefix}/info_ms_mean": float((self.info_s / self.steps) * 1e3),
            f"{prefix}/replay_add_ms_mean": float((self.replay_s / self.steps) * 1e3),
            f"{prefix}/sample_ms_mean": float((self.sample_s / self.steps) * 1e3),
            f"{prefix}/update_ms_mean": float((self.update_s / self.steps) * 1e3),
            f"{prefix}/misc_ms_mean": float((self.misc_s / self.steps) * 1e3),
            f"{prefix}/update_total_ms_mean": float(((self.sample_s + self.update_s) / self.steps) * 1e3),
        }
        denom = max(1e-9, self.total_s)
        out[f"{prefix}/pct_action"] = float(100.0 * self.action_s / denom)
        out[f"{prefix}/pct_env_step"] = float(100.0 * self.env_s / denom)
        out[f"{prefix}/pct_info"] = float(100.0 * self.info_s / denom)
        out[f"{prefix}/pct_replay_add"] = float(100.0 * self.replay_s / denom)
        out[f"{prefix}/pct_sample"] = float(100.0 * self.sample_s / denom)
        out[f"{prefix}/pct_update"] = float(100.0 * self.update_s / denom)
        out[f"{prefix}/pct_misc"] = float(100.0 * self.misc_s / denom)
        out[f"{prefix}/pct_update_total"] = float(100.0 * (self.sample_s + self.update_s) / denom)
        return out

    def reset(self) -> None:
        self.steps = 0
        self.action_s = 0.0
        self.env_s = 0.0
        self.info_s = 0.0
        self.replay_s = 0.0
        self.sample_s = 0.0
        self.update_s = 0.0
        self.misc_s = 0.0
        self.total_s = 0.0


class TeacherMetricsAccumulator:
    def __init__(
        self,
        *,
        device: torch.device,
        hist_edges_tensor: torch.Tensor | None,
        hist_labels: list[str],
        thresh_tensor: torch.Tensor | None,
        thresh_values: list[float],
    ):
        self.device = device
        self.hist_edges_tensor = hist_edges_tensor
        self.hist_labels = hist_labels
        self.thresh_tensor = thresh_tensor
        self.thresh_values = thresh_values
        self.teacher_hist_counts = (
            torch.zeros(len(hist_labels), device=device) if hist_edges_tensor is not None else None
        )
        self.non_teacher_hist_counts = (
            torch.zeros(len(hist_labels), device=device) if hist_edges_tensor is not None else None
        )
        self.teacher_above_counts = (
            torch.zeros(len(thresh_values), device=device) if thresh_tensor is not None else None
        )
        self.reset_running_stats()

    def reset_running_stats(self) -> None:
        self.teacher_disagreement_sum = 0.0
        self.teacher_disagreement_steps = 0.0
        self.non_teacher_disagreement_sum = 0.0
        self.non_teacher_disagreement_steps = 0.0
        self.corr_total_steps = 0.0
        self.corr_sum_mask = 0.0
        self.corr_sum_dis = 0.0
        self.corr_sum_mask_sq = 0.0
        self.corr_sum_dis_sq = 0.0
        self.corr_sum_mask_dis = 0.0
        self.qmin_sum_all = 0.0
        self.qmin_steps_all = 0.0
        self.qmin_sum_teacher = 0.0
        self.qmin_steps_teacher = 0.0
        self.qmin_sum_non = 0.0
        self.qmin_steps_non = 0.0
        self.teacher_steps_window = 0.0
        if self.teacher_hist_counts is not None:
            self.teacher_hist_counts.zero_()
        if self.non_teacher_hist_counts is not None:
            self.non_teacher_hist_counts.zero_()
        if self.teacher_above_counts is not None:
            self.teacher_above_counts.zero_()

    def update(
        self,
        disagreement_step: torch.Tensor,
        teacher_mask_float: torch.Tensor,
        non_teacher_mask_float: torch.Tensor,
        qmin_step: torch.Tensor,
    ) -> None:
        self.teacher_disagreement_sum += float((disagreement_step * teacher_mask_float).sum().item())
        self.non_teacher_disagreement_sum += float((disagreement_step * non_teacher_mask_float).sum().item())
        self.teacher_disagreement_steps += float(teacher_mask_float.sum().item())
        self.non_teacher_disagreement_steps += float(non_teacher_mask_float.sum().item())

        self.corr_total_steps += float(disagreement_step.numel())
        self.corr_sum_mask += float(teacher_mask_float.sum().item())
        self.corr_sum_dis += float(disagreement_step.sum().item())
        self.corr_sum_mask_sq += float((teacher_mask_float ** 2).sum().item())
        self.corr_sum_dis_sq += float((disagreement_step ** 2).sum().item())
        self.corr_sum_mask_dis += float((teacher_mask_float * disagreement_step).sum().item())

        if self.hist_edges_tensor is not None and self.teacher_hist_counts is not None:
            bin_idx = torch.bucketize(disagreement_step, self.hist_edges_tensor)
            self.teacher_hist_counts.scatter_add_(0, bin_idx, teacher_mask_float)
            self.non_teacher_hist_counts.scatter_add_(0, bin_idx, non_teacher_mask_float)

        if self.thresh_tensor is not None and self.teacher_above_counts is not None:
            comp = (disagreement_step.unsqueeze(1) >= self.thresh_tensor.unsqueeze(0)).float()
            comp_teacher = comp * teacher_mask_float.unsqueeze(1)
            self.teacher_above_counts += comp_teacher.sum(dim=0)
            self.teacher_steps_window += float(teacher_mask_float.sum().item())

        self.qmin_sum_all += float(qmin_step.sum().item())
        self.qmin_steps_all += float(qmin_step.numel())
        self.qmin_sum_teacher += float((qmin_step * teacher_mask_float).sum().item())
        self.qmin_steps_teacher += float(teacher_mask_float.sum().item())
        self.qmin_sum_non += float((qmin_step * non_teacher_mask_float).sum().item())
        self.qmin_steps_non += float(non_teacher_mask_float.sum().item())

    def snapshot(self) -> TeacherSnapshot:
        teacher_mean = (
            self.teacher_disagreement_sum / max(1e-8, self.teacher_disagreement_steps)
            if self.teacher_disagreement_steps > 0
            else None
        )
        non_teacher_mean = (
            self.non_teacher_disagreement_sum / max(1e-8, self.non_teacher_disagreement_steps)
            if self.non_teacher_disagreement_steps > 0
            else None
        )
        total_dis_sum = self.teacher_disagreement_sum + self.non_teacher_disagreement_sum
        total_dis_steps = self.teacher_disagreement_steps + self.non_teacher_disagreement_steps
        mean_all = total_dis_sum / max(1e-8, total_dis_steps) if total_dis_steps > 0 else None

        corr_value = None
        denom_part_x = self.corr_total_steps * self.corr_sum_mask_sq - (self.corr_sum_mask ** 2)
        denom_part_y = self.corr_total_steps * self.corr_sum_dis_sq - (self.corr_sum_dis ** 2)
        if self.corr_total_steps > 0 and denom_part_x > 1e-8 and denom_part_y > 1e-8:
            numer = self.corr_total_steps * self.corr_sum_mask_dis - self.corr_sum_mask * self.corr_sum_dis
            corr_value = numer / np.sqrt(denom_part_x * denom_part_y)

        hist_teacher: Dict[str, float] = {}
        hist_non_teacher: Dict[str, float] = {}
        if self.teacher_hist_counts is not None:
            teacher_hist_cpu = self.teacher_hist_counts.detach().cpu()
            non_teacher_hist_cpu = self.non_teacher_hist_counts.detach().cpu()
            for idx, label in enumerate(self.hist_labels):
                hist_teacher[label] = float(teacher_hist_cpu[idx].item())
                hist_non_teacher[label] = float(non_teacher_hist_cpu[idx].item())

        threshold_percentages: Dict[float, float] = {}
        if self.teacher_above_counts is not None and self.teacher_steps_window > 0:
            pct = (self.teacher_above_counts / max(1.0, self.teacher_steps_window)).detach().cpu().numpy()
            for i, thr in enumerate(self.thresh_values):
                threshold_percentages[thr] = float(pct[i])

        qmin_all = self.qmin_sum_all / max(1e-8, self.qmin_steps_all) if self.qmin_steps_all > 0 else None
        qmin_teacher = self.qmin_sum_teacher / max(1e-8, self.qmin_steps_teacher) if self.qmin_steps_teacher > 0 else None
        qmin_non = self.qmin_sum_non / max(1e-8, self.qmin_steps_non) if self.qmin_steps_non > 0 else None

        return TeacherSnapshot(
            mean_disagreement_teacher=teacher_mean,
            mean_disagreement_non=non_teacher_mean,
            mean_disagreement_all=mean_all,
            corr_value=corr_value,
            hist_teacher=hist_teacher,
            hist_non_teacher=hist_non_teacher,
            threshold_percentages=threshold_percentages,
            qmin_all=qmin_all,
            qmin_teacher=qmin_teacher,
            qmin_non=qmin_non,
        )

    def reset_after_log(self) -> None:
        if self.teacher_hist_counts is not None:
            self.teacher_hist_counts.zero_()
        if self.non_teacher_hist_counts is not None:
            self.non_teacher_hist_counts.zero_()
        if self.teacher_above_counts is not None:
            self.teacher_above_counts.zero_()
        self.teacher_steps_window = 0.0
        self.qmin_sum_all = self.qmin_steps_all = 0.0
        self.qmin_sum_teacher = self.qmin_steps_teacher = 0.0
        self.qmin_sum_non = self.qmin_steps_non = 0.0
        self.teacher_disagreement_sum = self.teacher_disagreement_steps = 0.0
        self.non_teacher_disagreement_sum = self.non_teacher_disagreement_steps = 0.0
        self.corr_total_steps = (
            self.corr_sum_mask
        ) = self.corr_sum_dis = self.corr_sum_mask_sq = self.corr_sum_dis_sq = self.corr_sum_mask_dis = 0.0


class TrainingLogger:
    def __init__(self, *, args, record_progress, teacher_metrics: TeacherMetricsAccumulator):
        self.args = args
        self.record_progress = record_progress
        self.teacher_metrics = teacher_metrics
        self.next_log_step = args.log_interval if args.log_interval > 0 else None
        self.wandb_run = None

    def ensure_wandb_run(self):
        if not self.args.use_wandb:
            return None
        if self.wandb_run is None:
            import wandb

            try:
                self.wandb_run = wandb.init(
                    project=self.args.project,
                    entity=(str(getattr(self.args, "wandb_entity", "")).strip() or None),
                    group=(str(getattr(self.args, "wandb_group", "")).strip() or None),
                    name=self.args.exp_name,
                    id=self.args.exp_name,
                    config=vars(self.args),
                    reinit=True,
                    resume="allow",
                )
            except Exception as exc:
                msg = f"[WandB] init failed; continuing without wandb logging: {exc}"
                print(msg, flush=True)
                self.record_progress(msg)
                self.args.use_wandb = False
                self.wandb_run = None
                return None
        return self.wandb_run

    def should_log(self, total_env_steps: int, force: bool = False) -> bool:
        if force:
            return True
        if self.next_log_step is None:
            return False
        if total_env_steps >= self.next_log_step:
            self.next_log_step += self.args.log_interval
            return True
        return False

    def log(
        self,
        *,
        total_env_steps: int,
        total_timesteps: int,
        iteration_idx: int,
        collection_time: float,
        rewbuffer: list,
        lenbuffer: list,
        last_update_metrics: tuple[Dict[str, float], int] | None,
        infos,
        log_alpha: torch.Tensor,
        last_denied_samples: int,
        pref_size: int,
        pref_capacity: int,
        replay_size: int,
        replay_capacity: int,
        demo_size: int,
        demo_capacity: int,
        timing_summary: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        fps = int(total_env_steps / max(1e-6, collection_time))
        logs = {
            "Perf/total_fps": fps,
            "Perf/collection_time_sec": collection_time,
            "Perf/env_steps": total_env_steps,
            "Perf/iterations": iteration_idx,
        }

        if rewbuffer:
            logs["Train/mean_reward"] = float(np.mean(rewbuffer[-100:]))
        if lenbuffer:
            logs["Train/mean_episode_length"] = float(np.mean(lenbuffer[-100:]))

        if last_update_metrics is not None:
            metrics_accumulator, updates_count = last_update_metrics
            denom = float(max(1, updates_count))
            actor_denom = float(max(1.0, metrics_accumulator.get("actor_update_count", float(updates_count))))
            alpha_metric_denom = float(max(1.0, metrics_accumulator.get("alpha_metric_count", float(updates_count))))
            alpha_step_denom = float(max(1.0, metrics_accumulator.get("alpha_optimizer_step_count", 0.0)))
            pref_loss_type = str(getattr(self.args, "pref_loss_type", "margin")).strip().lower()
            lagrangian_enabled = 1.0 if pref_loss_type == "lagrangian" else 0.0
            algo_variant = str(getattr(self.args, "algo_variant", "own") or "own").strip().lower()
            logs["Train/critic_loss"] = metrics_accumulator["critic_loss"] / denom
            logs["Train/critic_loss_replay"] = metrics_accumulator["critic_loss_replay"] / denom
            logs["Train/critic_loss_pref"] = metrics_accumulator["critic_loss_pref"] / denom
            logs["Train/critic_loss_pref_weighted"] = metrics_accumulator["critic_loss_pref_weighted"] / denom
            logs["Train/critic_loss_total"] = metrics_accumulator["critic_loss_total"] / denom
            logs["Train/pref_lambda"] = metrics_accumulator["pref_lambda"] / denom
            logs["Train/pref_lambda_delta"] = metrics_accumulator["pref_lambda_delta"] / denom
            logs["Train/pref_lambda_min"] = metrics_accumulator.get("pref_lambda_min", 0.0) / denom
            logs["Train/pref_lambda_max"] = metrics_accumulator.get("pref_lambda_max_value", 0.0) / denom
            logs["Train/pref_lambda_std"] = metrics_accumulator.get("pref_lambda_std", 0.0) / denom
            logs["Train/pref_lambda_active_fraction"] = (
                metrics_accumulator.get("pref_lambda_active_fraction", 0.0) / denom
            )
            logs["Train/pref_lambda_per_linked_enabled"] = (
                metrics_accumulator.get("pref_lambda_per_linked_enabled", 0.0) / denom
            )
            logs["Train/pref_dual_violation"] = metrics_accumulator["pref_dual_violation"] / denom
            logs["Train/pref_dual_signal"] = metrics_accumulator["pref_dual_signal"] / denom
            logs["Train/pref_violation"] = metrics_accumulator["pref_violation"] / denom
            logs["Train/pref_violation_ema"] = metrics_accumulator["pref_violation_ema"] / denom
            logs["Train/pref_lagrangian_loss"] = metrics_accumulator["pref_lagrangian_loss"] / denom
            logs["Train/pref_lagrangian_enabled"] = lagrangian_enabled
            pref_lagrangian_violation_type = str(
                getattr(self.args, "pref_lagrangian_violation_type", "hinge")
            ).strip().lower()
            logs["Train/pref_lagrangian_violation_is_smooth"] = (
                1.0 if pref_lagrangian_violation_type == "smooth" else 0.0
            )
            logs["Train/pref_lambda_lr"] = float(getattr(self.args, "pref_lambda_lr", 0.0))
            logs["Train/pref_lambda_max_cfg"] = float(getattr(self.args, "pref_lambda_max", 0.0))
            logs["Train/pref_lagrangian_scope_is_per_linked"] = (
                1.0 if str(getattr(self.args, "pref_lagrangian_scope", "global")).strip().lower() == "per_linked" else 0.0
            )
            logs["Train/pref_lambda_ema_cfg"] = float(getattr(self.args, "pref_lambda_ema", 0.0))
            logs["Train/pref_violation_clip"] = float(getattr(self.args, "pref_violation_clip", 0.0))
            logs["Train/pref_violation_target"] = float(getattr(self.args, "pref_violation_target", 0.0))
            logs["Train/actor_loss"] = metrics_accumulator["actor_loss"] / actor_denom
            logs["Train/actor_loss_sac"] = metrics_accumulator["actor_loss_sac"] / actor_denom
            logs["Train/actor_bc_loss_demo"] = metrics_accumulator["actor_bc_loss_demo"] / actor_denom
            logs["Train/actor_bc_loss_pref"] = metrics_accumulator["actor_bc_loss_pref"] / actor_denom
            logs["Train/alpha_loss"] = metrics_accumulator["alpha_loss"] / alpha_step_denom
            logs["Train/policy_entropy"] = metrics_accumulator["entropy"] / actor_denom
            logs["Train/action_l2"] = metrics_accumulator["action_norm"] / actor_denom
            logs["Train/target_q_mean"] = metrics_accumulator["target_q"] / denom
            logs["Train/q_min_pi_mean"] = metrics_accumulator["q_min_pi"] / actor_denom
            logs["Train/q_min_data_mean"] = metrics_accumulator["q_min_data"] / denom
            logs["Train/q_disagreement_data_mean"] = metrics_accumulator["q_disagreement_data"] / denom
            logs["Train/q_disagreement_pi_mean"] = metrics_accumulator["q_disagreement_pi"] / actor_denom
            teacher_q_count = float(metrics_accumulator.get("q_min_teacher_data_count", 0.0))
            if teacher_q_count > 0.0:
                logs["Train/q_min_teacher_action_mean"] = (
                    float(metrics_accumulator["q_min_teacher_data_sum"]) / teacher_q_count
                )
            non_teacher_q_count = float(metrics_accumulator.get("q_min_non_teacher_data_count", 0.0))
            if non_teacher_q_count > 0.0:
                logs["Train/q_min_non_teacher_action_mean"] = (
                    float(metrics_accumulator["q_min_non_teacher_data_sum"]) / non_teacher_q_count
                )
            pref_q_delta_count = float(metrics_accumulator.get("pref_q_delta_count", 0.0))
            if pref_q_delta_count > 0.0:
                logs["Train/pref_q_delta_mean"] = (
                    float(metrics_accumulator["pref_q_delta_sum"]) / pref_q_delta_count
                )
            logs["Train/replay_reward_mean"] = metrics_accumulator["reward"] / denom
            logs["Train/replay_reward_abs_mean"] = metrics_accumulator["reward_abs"] / denom
            logs["Train/alpha"] = metrics_accumulator["alpha_value"] / alpha_metric_denom
            logs["Train/updates_per_iter"] = updates_count
            logs["Train/actor_updates_per_iter"] = float(metrics_accumulator.get("actor_update_count", 0.0))
            logs["Train/alpha_updates_per_iter"] = float(metrics_accumulator.get("alpha_optimizer_step_count", 0.0))
            logs["Train/intervened_batch_fraction"] = (
                float(metrics_accumulator.get("intervened_batch_fraction", 0.0)) / denom
            )
            logs["Train/action_override_batch_fraction"] = (
                float(metrics_accumulator.get("action_override_batch_fraction", 0.0)) / denom
            )
            logs["Train/intervention_override_mismatch_fraction"] = (
                float(metrics_accumulator.get("intervention_override_mismatch_fraction", 0.0)) / denom
            )
            logs["Train/intervention_override_false_negative_fraction"] = (
                float(metrics_accumulator.get("intervention_override_false_negative_fraction", 0.0)) / denom
            )
            logs["Train/intervention_override_false_positive_fraction"] = (
                float(metrics_accumulator.get("intervention_override_false_positive_fraction", 0.0)) / denom
            )
            logs["Train/pref_linked_rows_per_iter"] = float(metrics_accumulator.get("pref_linked_rows", 0.0)) / denom
            logs["Train/pref_linked_effective_rows_per_iter"] = (
                float(metrics_accumulator.get("pref_linked_effective_rows", 0.0)) / denom
            )
            logs["Train/pref_linked_fraction"] = float(metrics_accumulator.get("pref_linked_fraction", 0.0)) / denom
            logs["Train/pref_linked_effective_fraction"] = (
                float(metrics_accumulator.get("pref_linked_effective_fraction", 0.0)) / denom
            )
            pref_action_delta_count = float(metrics_accumulator.get("pref_action_delta_count", 0.0))
            if pref_action_delta_count > 0.0:
                logs["Train/pref_action_delta_mean"] = (
                    float(metrics_accumulator.get("pref_action_delta_sum", 0.0)) / pref_action_delta_count
                )
            pref_action_weight_count = float(metrics_accumulator.get("pref_action_weight_count", 0.0))
            if pref_action_weight_count > 0.0:
                logs["Train/pref_action_weight_mean"] = (
                    float(metrics_accumulator.get("pref_action_weight_sum", 0.0)) / pref_action_weight_count
                )
            logs["Train/demo_rows_requested_per_iter"] = float(metrics_accumulator.get("demo_rows_requested", 0.0)) / denom
            logs["Train/demo_rows_sampled_per_iter"] = float(metrics_accumulator.get("demo_rows_sampled", 0.0)) / denom
            logs["Train/demo_sampling_fraction"] = (
                float(metrics_accumulator.get("demo_rows_sampled", 0.0))
                / max(1.0, float(metrics_accumulator.get("demo_rows_requested", 0.0)))
            )
            logs["Train/demo_fallback_updates"] = float(metrics_accumulator.get("demo_fallback_updates", 0.0))
            logs["Train/demo_buffer_nonempty_updates"] = float(
                metrics_accumulator.get("demo_buffer_nonempty_updates", 0.0)
            )
            if algo_variant == "pvp":
                logs["Train/pvp_td_reward_mean"] = metrics_accumulator["pvp_td_reward"] / denom
                logs["Train/pvp_proxy_teacher_loss"] = metrics_accumulator["pvp_proxy_teacher_loss"] / denom
                logs["Train/pvp_proxy_student_loss"] = metrics_accumulator["pvp_proxy_student_loss"] / denom
                logs["Train/pvp_intervened_batch_fraction"] = (
                    metrics_accumulator["pvp_intervened_batch_fraction"] / denom
                )
                logs["Train/pvp_include_env_reward_in_td"] = (
                    1.0 if bool(getattr(self.args, "pvp_include_env_reward_in_td", False)) else 0.0
                )
                logs["Train/pvp_proxy_value_bound"] = float(getattr(self.args, "pvp_proxy_value_bound", 1.0))
            elif algo_variant == "eil":
                logs["Train/eil_good_loss"] = metrics_accumulator["eil_good_loss"] / denom
                logs["Train/eil_bad_loss"] = metrics_accumulator["eil_bad_loss"] / denom
                logs["Train/eil_pair_loss"] = metrics_accumulator["eil_pair_loss"] / denom
                logs["Train/eil_good_batch_fraction"] = metrics_accumulator["eil_good_batch_fraction"] / denom
                logs["Train/eil_bad_batch_fraction"] = metrics_accumulator["eil_bad_batch_fraction"] / denom
                logs["Train/eil_intervened_batch_fraction"] = (
                    metrics_accumulator["eil_intervened_batch_fraction"] / denom
                )
                logs["Train/eil_threshold"] = float(getattr(self.args, "eil_threshold", 0.0))
                logs["Train/eil_good_margin"] = float(getattr(self.args, "eil_good_margin", 0.0))
                logs["Train/eil_bad_margin"] = float(getattr(self.args, "eil_bad_margin", 0.01))
                logs["Train/eil_pair_margin"] = float(getattr(self.args, "eil_pair_margin", 0.01))
                logs["Train/eil_bad_pre_steps"] = float(getattr(self.args, "eil_bad_pre_steps", 8))
            if bool(getattr(self.args, "alpha_update_student_only", False)):
                logs["Train/alpha_student_only_fraction"] = (
                    float(metrics_accumulator.get("alpha_student_only_fraction", 0.0)) / alpha_metric_denom
                )
                logs["Train/alpha_student_only_rows_per_update"] = (
                    float(metrics_accumulator.get("alpha_student_only_rows", 0.0)) / alpha_metric_denom
                )
                logs["Train/alpha_student_only_skipped_updates"] = float(
                    metrics_accumulator.get("alpha_student_only_skipped_updates", 0.0)
                )
        else:
            fixed_alpha = float(getattr(self.args, "fixed_alpha", -1.0))
            if fixed_alpha >= 0.0:
                logs["Train/alpha"] = fixed_alpha
            else:
                logs["Train/alpha"] = float(log_alpha.exp().detach().cpu().item())

        if "log" in infos and isinstance(infos["log"], dict):
            for k, v in infos["log"].items():
                try:
                    logs[k] = float(v.float().mean().item())
                except Exception:
                    pass

        if self.args.store_denied_actions:
            logs["/Teacher/denied_transition_samples"] = float(last_denied_samples)

        teacher_snapshot = self.teacher_metrics.snapshot()
        if teacher_snapshot.qmin_all is not None:
            logs["/Critic/mean_q_min_all"] = teacher_snapshot.qmin_all
        if teacher_snapshot.qmin_teacher is not None:
            logs["/Critic/mean_q_min_intervened"] = teacher_snapshot.qmin_teacher
        if teacher_snapshot.qmin_non is not None:
            logs["/Critic/mean_q_min_no_intervention"] = teacher_snapshot.qmin_non

        pref_size_f = float(max(0, int(pref_size)))
        pref_capacity_f = float(max(0, int(pref_capacity)))
        replay_size_f = float(max(0, int(replay_size)))
        replay_capacity_f = float(max(0, int(replay_capacity)))
        demo_size_f = float(max(0, int(demo_size)))
        demo_capacity_f = float(max(0, int(demo_capacity)))

        # Always log buffer state, even when individual buffers are disabled.
        logs["/Buffers/pref_pairs"] = pref_size_f
        logs["/Buffers/pref_capacity"] = pref_capacity_f
        logs["/Buffers/replay_size"] = replay_size_f
        logs["/Buffers/replay_capacity"] = replay_capacity_f
        logs["/Buffers/demo_size"] = demo_size_f
        logs["/Buffers/demo_capacity"] = demo_capacity_f

        # Duplicated under Train/* so panels can be grouped with loss curves.
        logs["Train/buffer_pref_size"] = pref_size_f
        logs["Train/buffer_pref_capacity"] = pref_capacity_f
        logs["Train/buffer_replay_size"] = replay_size_f
        logs["Train/buffer_replay_capacity"] = replay_capacity_f
        logs["Train/buffer_demo_size"] = demo_size_f
        logs["Train/buffer_demo_capacity"] = demo_capacity_f
        algo_variant_final = str(getattr(self.args, "algo_variant", "own") or "own").strip().lower()
        pref_sampling_mode_final = str(getattr(self.args, "pref_sampling_mode", "independent") or "independent").strip().lower()
        if pref_sampling_mode_final == "linked":
            replay_intervened_fraction_estimate = pref_size_f / max(1.0, replay_size_f)
            logs["Train/replay_linked_intervened_fraction"] = replay_intervened_fraction_estimate
            if "Train/total_interventions" in logs and total_env_steps > 0:
                cumulative_intervention_fraction = float(logs["Train/total_interventions"]) / max(1.0, float(total_env_steps))
                logs["Train/cumulative_intervention_fraction"] = cumulative_intervention_fraction
                logs["Train/intervention_storage_gap_fraction"] = (
                    cumulative_intervention_fraction - replay_intervened_fraction_estimate
                )
        if algo_variant_final == "pvp":
            logs["/Buffers/novice_size"] = replay_size_f
            logs["/Buffers/human_size"] = demo_size_f
            logs["Train/buffer_novice_size"] = replay_size_f
            logs["Train/buffer_human_size"] = demo_size_f
        if timing_summary:
            logs.update(timing_summary)

        for key in list(logs.keys()):
            lower_key = key.lower()
            if "frac_interventions_dis_ge_" in lower_key:
                logs.pop(key, None)

        log_line_parts = [
            f"env_steps {total_env_steps}/{total_timesteps}",
            f"iter {iteration_idx}",
            f"fps {fps}",
        ]
        if "Train/mean_reward" in logs:
            log_line_parts.append(f"mean_reward {logs['Train/mean_reward']:.2f}")
        if "/Episode/goal_success_rate" in logs:
            log_line_parts.append(f"success {logs['/Episode/goal_success_rate']:.2f}")
        if "/Teacher/teacher_fraction_steps" in logs:
            log_line_parts.append(f"teacher_frac {logs['/Teacher/teacher_fraction_steps']:.2f}")
        if "Train/critic_loss_total" in logs:
            log_line_parts.append(f"qloss {logs['Train/critic_loss_total']:.3f}")
        if "Train/critic_loss_pref_weighted" in logs:
            pref_loss = float(logs["Train/critic_loss_pref_weighted"])
            if abs(pref_loss) > 1e-8:
                log_line_parts.append(f"ploss {pref_loss:.3f}")
        if "Train/pref_lambda" in logs:
            pref_lambda = float(logs["Train/pref_lambda"])
            if pref_lambda > 1e-8:
                log_line_parts.append(f"lambda {pref_lambda:.2f}")
                if float(logs.get("Train/pref_lambda_per_linked_enabled", 0.0)) > 0.5:
                    lam_min = float(logs.get("Train/pref_lambda_min", pref_lambda))
                    lam_max = float(logs.get("Train/pref_lambda_max", pref_lambda))
                    lam_std = float(logs.get("Train/pref_lambda_std", 0.0))
                    log_line_parts.append(f"lambda_range {lam_min:.2f}-{lam_max:.2f}")
                    log_line_parts.append(f"lambda_std {lam_std:.3f}")
        if self.args.store_denied_actions and last_denied_samples > 0:
            log_line_parts.append(f"denied {last_denied_samples}")
        if timing_summary:
            step_ms = float(timing_summary.get("Perf/timing/step_ms_mean", 0.0))
            env_ms = float(timing_summary.get("Perf/timing/env_step_ms_mean", 0.0))
            sample_ms = float(timing_summary.get("Perf/timing/sample_ms_mean", 0.0))
            update_ms = float(timing_summary.get("Perf/timing/update_ms_mean", 0.0))
            log_line_parts.append(
                f"tms step={step_ms:.1f} env={env_ms:.1f} sample={sample_ms:.1f} update={update_ms:.1f}"
            )
        console_line = "[FastSAC] " + " | ".join(log_line_parts)
        print(console_line, flush=True)
        self.record_progress(console_line)

        wandb_run = self.ensure_wandb_run()
        if wandb_run is not None:
            wandb_run.log(logs, step=total_env_steps)

        self.teacher_metrics.reset_after_log()
        return logs

    def log_eval(self, *, total_env_steps: int, metrics: Dict[str, float]) -> None:
        payload = {f"Eval/{key}": float(value) for key, value in metrics.items()}
        msg = ", ".join(f"{k}={v:.3f}" for k, v in payload.items())
        console_line = f"[Eval] steps={total_env_steps} {msg}"
        print(console_line, flush=True)
        self.record_progress(console_line)
        wandb_run = self.ensure_wandb_run()
        if wandb_run is not None:
            wandb_run.log(payload, step=total_env_steps)

    def log_prefill(self, *, pseudo_step: int, payload: Dict[str, float]) -> None:
        console_metrics = ", ".join(f"{k}={v:.3f}" for k, v in sorted(payload.items()))
        console_line = f"[DemoPrefill] step={pseudo_step} {console_metrics}"
        print(console_line, flush=True)
        self.record_progress(console_line)
        wandb_run = self.ensure_wandb_run()
        if wandb_run is not None:
            wandb_run.log(payload, step=int(pseudo_step))

    def finish(self):
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception:
                pass


class CheckpointManager:
    def __init__(
        self,
        *,
        run_model_dir: Path,
        viz_output_dir: Path,
        viz_cache_path: Path,
        record_progress,
        args,
        generate_policy_map,
    ):
        self.run_model_dir = run_model_dir
        self.viz_output_dir = viz_output_dir
        self.viz_cache_path = viz_cache_path
        self.record_progress = record_progress
        self.args = args
        self.generate_policy_map = generate_policy_map

    def save(
        self,
        *,
        tag: str,
        step_value: int,
        run_prefix: str,
        actor_backbone: torch.nn.Module,
        actor_head: torch.nn.Module,
        critic_backbone: torch.nn.Module | None,
        shared_backbone: torch.nn.Module | None,
        critic_heads: torch.nn.Module,
        critic_target_backbone: torch.nn.Module,
        critic_target_heads: torch.nn.Module,
        obs_normalizer,
        critic_obs_normalizer,
        log_alpha: torch.Tensor,
        pixel_shape,
    ) -> Path:
        save_path = self.run_model_dir / f"{run_prefix}_{tag}.pt"
        checkpoint = {
            "step": step_value,
            "actor_backbone": actor_backbone.state_dict(),
            "actor_head": actor_head.state_dict(),
            "critic_backbone": (None if self.args.arch_shared_trunk else critic_backbone.state_dict()),
            "shared_backbone": shared_backbone.state_dict() if self.args.arch_shared_trunk else None,
            "critic_heads": critic_heads.state_dict(),
            "critic_target_backbone": critic_target_backbone.state_dict(),
            "critic_target_heads": critic_target_heads.state_dict(),
            "obs_normalizer_state": (obs_normalizer.state_dict() if hasattr(obs_normalizer, "state_dict") else None),
            "critic_obs_normalizer_state": (
                critic_obs_normalizer.state_dict() if hasattr(critic_obs_normalizer, "state_dict") else None
            ),
            "log_alpha": log_alpha.detach().cpu().item(),
            "pixel_shape": pixel_shape,
            "args": vars(self.args),
        }
        torch.save(checkpoint, save_path, _use_new_zipfile_serialization=True)
        self.record_progress(f"[Checkpoint] saved {save_path}")
        return save_path

    def maybe_render_policy_map(
        self,
        *,
        tag: str,
        step_value: int,
        checkpoint_path: Path,
        current_env_name: str,
        wandb_run,
    ) -> None:
        if not self.args.viz_on_checkpoint or self.generate_policy_map is None:
            return
        try:
            png_path, _, _ = self.generate_policy_map(
                model_path=checkpoint_path,
                output_dir=self.viz_output_dir,
                tag=tag,
                env_name=current_env_name,
                grid_resolution=self.args.viz_grid_resolution,
                quiver_stride=self.args.viz_quiver_stride,
                device=self.args.viz_device,
                seed=self.args.viz_seed,
                cache_path=self.viz_cache_path,
            )
            self.record_progress(f"[Viz] generated {png_path}")
            if self.args.use_wandb and wandb_run is not None:
                import wandb

                wandb_run.log(
                    {"viz/policy_map": wandb.Image(str(png_path), caption=tag)},
                    step=step_value,
                )
        except Exception as exc:
            self.record_progress(f"[Viz] failed: {exc}")
