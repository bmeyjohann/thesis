"""Display-only load shedding for the single-owner Unitree simulator loop."""
import time


class LatestFramePacing:
    SPEED_MULTIPLIERS = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 1.5, 2.0, 4.0, 8.0]

    def init_pacing(self):
        self.display_fps = float(self.frame_rate)
        self.adaptive_display = True
        self.step_seconds = 0.0
        self.display_seconds = 0.0
        self._next_display = 0.0
        self._scene_future = None

    def _submit_scene_update_if_needed(self, sim, has_pending_updates):
        # One asynchronous scene update at a time. A busy worker drops this
        # display request; its successor reads the newest state, not a frame queue.
        if self._scene_future is not None:
            if not self._scene_future.done():
                return
            self._scene_future.result()
        if not self._should_submit_scene_update(self._counter, self._is_paused, has_pending_updates):
            return

        def update_scene():
            with self._sim_lock:
                started = time.perf_counter()
                with self._server.atomic():
                    self._scene.update(sim.data)
                    self._server.flush()
                self._scene_update_last_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        self._scene_future = self._threadpool.submit(update_scene)
        self._scene_submit_enqueue_last_ms = (time.perf_counter() - started) * 1000
        self._pending_update_reasons.clear()
        self._scene.needs_update = False

    def set_playback_speed(self, value):
        value = float(value)
        if value not in self.SPEED_MULTIPLIERS:
            raise ValueError(f"Unsupported playback speed: {value}")
        self._speed_index = self.SPEED_MULTIPLIERS.index(value)
        self._time_multiplier = value
        self._sim_budget = 0.0
        self._last_tick_time = time.perf_counter()

    def increase_speed(self):
        self.set_playback_speed(self.SPEED_MULTIPLIERS[min(self._speed_index + 1, len(self.SPEED_MULTIPLIERS) - 1)])

    def decrease_speed(self):
        self.set_playback_speed(self.SPEED_MULTIPLIERS[max(self._speed_index - 1, 0)])

    def reset_speed(self):
        self.set_playback_speed(1.0)

    def display_interval(self):
        interval = 1.0 / max(1.0, self.display_fps)
        if self.adaptive_display and not self._is_paused:
            # Spend at most roughly 10% on synchronous display work, with a
            # 5 Hz minimum target while overloaded so the human retains feedback.
            scene_seconds = getattr(self, '_scene_update_last_ms', 0.0) / 1000
            interval = max(interval, min(0.2, (self.display_seconds + scene_seconds) / 0.1))
            if self.step_seconds * self._time_multiplier > self.env.unwrapped.step_dt * 0.85:
                interval = max(interval, 0.2)
        return interval

    def tick(self):
        now = time.perf_counter()
        dt = max(0.0, now - self._last_tick_time)
        self._last_tick_time = now
        previous_pause, previous_speed = self._is_paused, self._time_multiplier
        self._process_actions()
        if not self.is_running():
            return False
        # A reset/resume/speed action starts a new pacing epoch. Do not repay
        # time spent paused as a burst of stale human actions.
        if (self._last_tick_time != now or previous_pause != self._is_paused
                or previous_speed != self._time_multiplier):
            dt = 0.0
        if not self._is_paused:
            step_dt = self.env.unwrapped.step_dt
            budget = self._sim_budget + dt * self._time_multiplier
            self._was_capped = budget > 2 * step_dt
            self._sim_budget = min(budget, 2 * step_dt)
            if self._sim_budget >= step_dt:
                self.sync_viewer_to_env()
                started = time.perf_counter()
                ok = self._execute_step()
                elapsed = time.perf_counter() - started
                self.step_seconds = elapsed if not self.step_seconds else 0.9 * self.step_seconds + 0.1 * elapsed
                self._sim_budget = max(0.0, self._sim_budget - step_dt) if ok else 0.0
        now = time.perf_counter()
        if now < self._next_display:
            return False
        if self._is_paused:
            self._forward_paused()
        started = time.perf_counter()
        self.sync_env_to_viewer()
        elapsed = time.perf_counter() - started
        self.display_seconds = elapsed if not self.display_seconds else 0.9 * self.display_seconds + 0.1 * elapsed
        # No catch-up render queue: publish only the latest state, then schedule
        # the next display from completion. Physics/sensor/learning steps stay intact.
        self._next_display = time.perf_counter() + self.display_interval()
        self._stats_frames += 1
        return True
