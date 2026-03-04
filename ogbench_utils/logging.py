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
            pref_loss_type = str(getattr(self.args, "pref_loss_type", "margin")).strip().lower()
            lagrangian_enabled = 1.0 if pref_loss_type == "lagrangian" else 0.0
            logs["Train/critic_loss"] = metrics_accumulator["critic_loss"] / denom
            logs["Train/critic_loss_replay"] = metrics_accumulator["critic_loss_replay"] / denom
            logs["Train/critic_loss_pref"] = metrics_accumulator["critic_loss_pref"] / denom
            logs["Train/critic_loss_pref_weighted"] = metrics_accumulator["critic_loss_pref_weighted"] / denom
            logs["Train/critic_loss_total"] = metrics_accumulator["critic_loss_total"] / denom
            logs["Train/pref_lambda"] = metrics_accumulator["pref_lambda"] / denom
            logs["Train/pref_lambda_delta"] = metrics_accumulator["pref_lambda_delta"] / denom
            logs["Train/pref_dual_violation"] = metrics_accumulator["pref_dual_violation"] / denom
            logs["Train/pref_violation"] = metrics_accumulator["pref_violation"] / denom
            logs["Train/pref_violation_ema"] = metrics_accumulator["pref_violation_ema"] / denom
            logs["Train/pref_lagrangian_loss"] = metrics_accumulator["pref_lagrangian_loss"] / denom
            logs["Train/pref_lagrangian_enabled"] = lagrangian_enabled
            logs["Train/pref_lambda_lr"] = float(getattr(self.args, "pref_lambda_lr", 0.0))
            logs["Train/pref_lambda_max"] = float(getattr(self.args, "pref_lambda_max", 0.0))
            logs["Train/pref_lambda_ema_cfg"] = float(getattr(self.args, "pref_lambda_ema", 0.0))
            logs["Train/pref_violation_clip"] = float(getattr(self.args, "pref_violation_clip", 0.0))
            logs["Train/actor_loss"] = metrics_accumulator["actor_loss"] / denom
            logs["Train/actor_loss_sac"] = metrics_accumulator["actor_loss_sac"] / denom
            logs["Train/actor_bc_loss_demo"] = metrics_accumulator["actor_bc_loss_demo"] / denom
            logs["Train/actor_bc_loss_pref"] = metrics_accumulator["actor_bc_loss_pref"] / denom
            logs["Train/alpha_loss"] = metrics_accumulator["alpha_loss"] / denom
            logs["Train/policy_entropy"] = metrics_accumulator["entropy"] / denom
            logs["Train/action_l2"] = metrics_accumulator["action_norm"] / denom
            logs["Train/target_q_mean"] = metrics_accumulator["target_q"] / denom
            logs["Train/q_min_pi_mean"] = metrics_accumulator["q_min_pi"] / denom
            logs["Train/q_disagreement_data_mean"] = metrics_accumulator["q_disagreement_data"] / denom
            logs["Train/q_disagreement_pi_mean"] = metrics_accumulator["q_disagreement_pi"] / denom
            logs["Train/replay_reward_mean"] = metrics_accumulator["reward"] / denom
            logs["Train/replay_reward_abs_mean"] = metrics_accumulator["reward_abs"] / denom
            logs["Train/alpha"] = metrics_accumulator["alpha_value"] / denom
            logs["Train/updates_per_iter"] = updates_count
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

        if self.args.use_wandb:
            import wandb

            if self.wandb_run is None:
                self.wandb_run = wandb.init(
                    project=self.args.project,
                    name=self.args.exp_name,
                    id=self.args.exp_name,
                    config=vars(self.args),
                    reinit=True,
                    resume="allow",
                )
            self.wandb_run.log(logs, step=total_env_steps)

        self.teacher_metrics.reset_after_log()
        return logs

    def log_eval(self, *, total_env_steps: int, metrics: Dict[str, float]) -> None:
        payload = {f"Eval/{key}": float(value) for key, value in metrics.items()}
        msg = ", ".join(f"{k}={v:.3f}" for k, v in payload.items())
        console_line = f"[Eval] steps={total_env_steps} {msg}"
        print(console_line, flush=True)
        self.record_progress(console_line)
        if self.args.use_wandb:
            import wandb

            if self.wandb_run is None:
                self.wandb_run = wandb.init(
                    project=self.args.project,
                    name=self.args.exp_name,
                    id=self.args.exp_name,
                    config=vars(self.args),
                    reinit=True,
                    resume="allow",
                )
            self.wandb_run.log(payload, step=total_env_steps)

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
