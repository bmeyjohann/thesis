from types import SimpleNamespace
from unittest.mock import patch
from concurrent.futures import Future
from contextlib import nullcontext

import pytest

from unitree_viewer_pacing import LatestFramePacing


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class Viewer(LatestFramePacing):
    def __init__(self, clock, step_cost=0.0, render_cost=0.0):
        self.clock = clock
        self.env = SimpleNamespace(unwrapped=SimpleNamespace(step_dt=0.05))
        self.frame_rate = 30
        self._is_paused = False
        self._time_multiplier = 1.0
        self._sim_budget = self._last_tick_time = 0.0
        self._stats_frames = self.steps = self.forwards = 0
        self.step_cost, self.render_cost = step_cost, render_cost
        self.actions = []
        self.init_pacing()

    def _process_actions(self):
        for action in self.actions:
            action()
        self.actions.clear()

    def is_running(self):
        return True

    def sync_viewer_to_env(self):
        pass

    def _execute_step(self):
        self.steps += 1
        self.clock.now += self.step_cost
        return True

    def _forward_paused(self):
        self.forwards += 1

    def sync_env_to_viewer(self):
        self.clock.now += self.render_cost


@pytest.mark.parametrize('speed,expected', [(0.5, 10), (1, 20), (1.5, 30), (2, 40)])
def test_speed_does_not_change_simulation_step(speed, expected):
    clock = Clock()
    viewer = Viewer(clock)
    with patch('unitree_viewer_pacing.time.perf_counter', clock):
        viewer.set_playback_speed(speed)
        for i in range(1, 1002):
            clock.now = i / 1000
            viewer.tick()
    assert viewer.steps == expected
    assert viewer.env.unwrapped.step_dt == 0.05
    assert viewer._stats_frames <= 31


def test_slow_render_never_accumulates_frame_backlog():
    clock = Clock()
    viewer = Viewer(clock, render_cost=0.3)
    with patch('unitree_viewer_pacing.time.perf_counter', clock):
        viewer.tick()
        assert viewer._next_display >= 0.5
        for _ in range(20):
            clock.now += 0.001
            viewer.tick()
    assert viewer._stats_frames == 1
    assert viewer.steps <= 2  # no stale-action catch-up burst


def test_slow_physics_sheds_display_not_transitions():
    clock = Clock()
    viewer = Viewer(clock, step_cost=0.08)
    with patch('unitree_viewer_pacing.time.perf_counter', clock):
        for _ in range(100):
            clock.now += 0.01
            viewer.tick()
    assert viewer.step_seconds == pytest.approx(0.08)
    assert viewer.display_interval() == 0.2
    assert viewer.steps > viewer._stats_frames


def test_paused_forward_only_at_display_rate_and_no_resume_debt():
    clock = Clock()
    viewer = Viewer(clock)
    viewer._is_paused = True
    with patch('unitree_viewer_pacing.time.perf_counter', clock):
        for i in range(1000):
            clock.now = i / 1000
            viewer.tick()
        assert viewer.steps == 0
        assert viewer.forwards <= 30
        clock.now = 100
        viewer.actions.append(lambda: viewer.set_playback_speed(2))
        viewer.actions.append(lambda: setattr(viewer, '_is_paused', False))
        viewer.tick()
        assert viewer.steps == 0


def test_invalid_speed_rejected():
    viewer = Viewer(Clock())
    with pytest.raises(ValueError):
        viewer.set_playback_speed(0)


def test_generic_speed_controls_use_same_presets():
    viewer = Viewer(Clock())
    viewer.set_playback_speed(1)
    viewer.increase_speed()
    assert viewer._time_multiplier == 1.5
    viewer.increase_speed()
    assert viewer._time_multiplier == 2
    viewer.decrease_speed()
    assert viewer._time_multiplier == 1.5
    viewer.reset_speed()
    assert viewer._time_multiplier == 1


def test_scene_worker_bounded_and_reads_latest_state():
    viewer = Viewer(Clock())
    submitted, observed = [], []
    future = Future()
    viewer._threadpool = SimpleNamespace(submit=lambda fn: (submitted.append(fn), future)[1])
    viewer._sim_lock = nullcontext()
    viewer._server = SimpleNamespace(atomic=nullcontext, flush=lambda: None)
    viewer._scene = SimpleNamespace(needs_update=True, update=lambda data: observed.append(data.value))
    viewer._pending_update_reasons = {'reset'}
    viewer._counter = 0
    viewer._should_submit_scene_update = lambda *args: True
    sim = SimpleNamespace(data=SimpleNamespace(value=1))
    viewer._submit_scene_update_if_needed(sim, True)
    for _ in range(100):
        viewer._submit_scene_update_if_needed(sim, True)
    assert len(submitted) == 1
    sim.data.value = 9
    submitted[0]()
    future.set_result(None)
    assert observed == [9]
    viewer._submit_scene_update_if_needed(sim, True)
    assert len(submitted) == 2
