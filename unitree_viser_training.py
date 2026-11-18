"""Online learner and raw recording lifecycle for the browser viewer."""
import time
import json

import numpy as np
import torch


class BrowserTrainingSession:
    def __init__(self, args, env, raw, obstacle_cells):
        from unitree_nav_observation import UnitreeScanHistory
        from eval_interactive_unitree_nav import _prepare_policy_obs
        self.args, self.env, self.cells = args, env, obstacle_cells
        self.timing_seconds = dict(physics=0.0, perception=0.0, learning=0.0)
        self.playback_events = []
        self.observation_frames = None
        if getattr(args, 'observation_only', False):
            from unitree_observation_view import ObservationFrames
            self.observation_frames = ObservationFrames(getattr(args, 'navigation_feature_lags', None))
        lags = getattr(args, 'navigation_feature_lags', None)
        if lags is not None and (args.scan_history != 1 or args.action_history != 0):
            raise ValueError('Fused feature lags require scan_history=1 and action_history=0; avoid double stacking')
        self.feature_history = UnitreeScanHistory(len(lags), scan_lags=lags) if lags is not None else None
        self.history = UnitreeScanHistory(args.scan_history, args.action_history, 3,
                                         history_stride=args.scan_history_stride)
        self.raw = raw
        from unitree_nav_camera_features import NavigationCameraFeatures, camera_enabled
        self.vision = NavigationCameraFeatures(args, env) if camera_enabled(args) else None
        self.obs = self._policy_input(self.history.reset(_prepare_policy_obs(raw, args)), reset=True)
        print(f'[navigation-history] feature_lags={lags}; ages_s='
              f'{[v / args.navigation_policy_hz for v in lags] if lags else None}; '
              f'obs_dim={self.obs.shape[1]}; context=current; reset_padding=initial_frame', flush=True)
        self.learner = None
        self.writer = None
        self.previous_human = False
        self.steps = self.episode = 0
        self.environment_ticks = 0
        self.pending = None
        self.cost_total = self.successes = self.interventions = 0
        self.held = torch.zeros((1, 3), device=args.device)
        from unitree_nav_control import navigation_action_repeat
        self.repeat = navigation_action_repeat(args.navigation_policy_hz)
        if args.online_train:
            from unitree_nav_online_human import UnitreeOnlineHumanLearner
            self.learner = UnitreeOnlineHumanLearner(args, obs_dim=self.obs.shape[1], act_dim=3)
            self.learner.vision_encoder = self.vision
            self.actor = self.learner.actor
        else:
            from eval_unitree_nav_baselines import _load_policy_actor
            self.actor = _load_policy_actor(args, obs_dim=self.obs.shape[1], act_dim=3) if args.controller == 'policy' else None
        if args.human_dataset_dir:
            from unitree_nav_human_dataset import UnitreeHumanDatasetWriter
            self.writer = UnitreeHumanDatasetWriter(args.human_dataset_dir,
                metadata={'args': vars(args), 'backend': 'viser', 'transition_hz': args.navigation_policy_hz,
                          'navigation_policy_hz': args.navigation_policy_hz},
                chunk_size=args.human_dataset_chunk_size)

    def _policy_input(self, stacked, *, reset=False, done=None):
        started = time.perf_counter()
        features = self.vision.transform(stacked) if self.vision is not None else stacked
        self.timing_seconds['perception'] += time.perf_counter() - started
        if getattr(self, 'observation_frames', None) is not None:
            self.observation_frames.capture(self.vision.display_frames,
                reset=reset or bool(done is not None and done.any()))
        if self.feature_history is None:
            return features
        # Freeze/encode each camera once per decision, then retain the compact features.
        if reset:
            return self.feature_history.reset(features)
        return self.feature_history.step(features, done=done)

    def reset(self, raw):
        from eval_interactive_unitree_nav import _prepare_policy_obs
        # User resets discard incomplete returns rather than linking two scenarios.
        if self.learner:
            self.learner.nstep.queues[0].clear()
        self.raw = raw
        self.pending = None
        self.obs = self._policy_input(self.history.reset(_prepare_policy_obs(raw, self.args)), reset=True)
        self.held.zero_()
        self.previous_human = False
        self.episode += 1
        self.decision_due = True

    def proposal(self):
        from eval_unitree_nav_baselines import controller_action, smooth_policy_action
        from unitree_nav_control import project_navigation_action
        if self.pending is None:
            with torch.no_grad():
                proposed = controller_action(self.obs, self.args, policy_actor=self.actor,
                                             env=self.env, obstacle_cells=self.cells)
                self.held = project_navigation_action(smooth_policy_action(
                    proposed, self.held, controller=self.args.controller,
                    smoothing=self.args.policy_action_smoothing))
            self.decision_due = False
        return self.held.clone()

    def step(self, executed, proposed, human, source, sample):
        if self.limit_reached:
            raise RuntimeError("Navigation decision budget already reached; no extra tick executed")
        from eval_interactive_unitree_nav import _prepare_policy_obs, _prepare_visual_obs
        from eval_unitree_nav_baselines import (_current_goal_distance, _extract_cost,
                                               _goal_termination_mask, _resample_continuous_goals)
        device = torch.device(self.args.device)
        distance = float(_current_goal_distance(self.env, 1, device)[0])
        before = self.obs.clone()
        base = _prepare_visual_obs(self.raw, self.args).clone()
        if self.pending is None:
            self.pending = dict(before=before, base=base, distance=distance,
                                executed=executed.clone(), proposed=proposed.clone(),
                                human=human, source=source, sample=sample,
                                frames=dict(self.vision.last_frames) if self.vision else {},
                                ticks=0, cost=0.0, reward=0.0)
        pending = self.pending
        executed, proposed = pending['executed'], pending['proposed']
        human, source, sample = pending['human'], pending['source'], pending['sample']
        before, base, distance = pending['before'], pending['base'], pending['distance']
        started = time.perf_counter()
        with torch.no_grad():
            raw, reward, done, extras = self.env.step(executed)
            self.environment_ticks += 1
            ended = bool(done.reshape(-1)[0])
            success = bool(_goal_termination_mask(self.env, 1, device)[0])
            after_distance = float(_current_goal_distance(self.env, 1, device)[0])
            success = success or (not ended and after_distance <= self.args.success_dist)
            boundary = ended or success
            cost = float(_extract_cost(extras, 1, device)[0])
        self.timing_seconds['physics'] += time.perf_counter() - started
        pending['ticks'] += 1
        pending['cost'] += cost
        pending['reward'] += float(reward.reshape(-1)[0])
        self.raw = raw
        if pending['ticks'] < self.repeat and not boundary:
            return False, False
        next_obs = self.history.step(_prepare_policy_obs(raw, self.args),
                                     done.reshape(-1).bool(), action=executed).clone()
        next_obs = self._policy_input(next_obs, done=done.reshape(-1).bool())
        cost = pending['cost']
        if self.learner:
            # MJLab's viewer normally wraps policy/physics in no_grad.
            started = time.perf_counter()
            with torch.enable_grad():
                self.learner.observe(obs=before, student_action=proposed,
                    executed_action=executed, next_obs=next_obs, done=boundary,
                    terminal_success=success, intervened=human,
                    intervention_start=human and not self.previous_human,
                    goal_distance=distance, next_goal_distance=after_distance, cost=cost)
            self.timing_seconds['learning'] += time.perf_counter() - started
        if self.writer:
            array = lambda value: value[0].detach().cpu().numpy()
            self.writer.add(observations=array(before), base_observations=array(base),
                student_actions=array(proposed), human_actions=array(executed if human else torch.zeros_like(executed)),
                executed_actions=array(executed), next_observations=array(next_obs),
                next_base_observations=array(_prepare_visual_obs(raw, self.args)),
                reward=pending['reward'], cost=cost, done=boundary,
                terminal_success=success, intervened=human,
                intervention_start=human and not self.previous_human,
                intervention_end=self.previous_human and not human, control_source=source,
                gamepad_connected=bool(sample and sample.connected),
                gamepad_stale=bool(sample is None or sample.stale),
                gamepad_command_norm=float(sample.command_norm if sample else 0),
                action_delta_to_policy=float(torch.linalg.vector_norm(executed-proposed)),
                goal_distance=distance, next_goal_distance=after_distance,
                episode_index=self.episode, step_index=self.steps, wall_time_unix_s=time.time(),
                **pending['frames'],
                **({f"next_{key}": value for key, value in self.vision.last_frames.items()} if self.vision else {}))
        self.steps += 1
        self.interventions += int(human)
        self.cost_total += cost
        self.successes += int(success)
        self.previous_human = human
        self.raw, self.obs = raw, next_obs
        self.pending = None
        if success and not ended and self.args.navigation_episode_mode == 'continuous_goals':
            raw, _, _ = _resample_continuous_goals(self.args, self.env, self.cells,
                                                 torch.zeros(1, dtype=torch.long, device=device))
            self.reset(raw)
        elif ended:
            self.reset(raw)
        return success, ended

    @property
    def limit_reached(self):
        limit = int(self.args.online_total_steps)
        return bool(self.args.online_train and limit > 0 and self.steps >= limit)

    def close(self):
        payload = None
        try:
            if self.learner:
                self.learner.close()
                summary = dict(navigation_decisions=self.steps, environment_ticks=self.environment_ticks,
                               decision_hz=self.args.navigation_policy_hz,
                               decision_limit=self.args.online_total_steps,
                               limit_reached=self.limit_reached, goals=self.successes,
                               intervention_decisions=self.interventions, cost=self.cost_total,
                               timing_seconds=self.timing_seconds, playback_events=self.playback_events)
                (self.learner.run_dir / 'browser_summary.json').write_text(json.dumps(summary, indent=2))
                receipt = getattr(self.args, 'online_session_receipt', '')
                if receipt:
                    from pathlib import Path
                    checkpoint = self.learner.run_dir / 'final.pt'
                    if not checkpoint.is_file():
                        raise RuntimeError('Final checkpoint is missing; cannot acknowledge completion')
                    payload = dict(summary, run_dir=str(self.learner.run_dir), checkpoint=str(checkpoint),
                                   mode=self.args.online_objective_mode,
                                   sensors=self.args.navigation_observation_mode, seed=self.args.seed,
                                   image_profile=getattr(self.args, 'navigation_image_profile', 'legacy'),
                                   run_name=self.args.online_run_name)
        finally:
            if self.writer:
                self.writer.close()
        if payload is not None:
            destination = Path(receipt)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload, indent=2))
            temporary.replace(destination)
