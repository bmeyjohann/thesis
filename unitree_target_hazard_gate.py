"""Privileged, deterministic hazard/progress interventions for target navigation."""
from collections import deque
from dataclasses import dataclass, asdict
import heapq
import math
import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class HazardGateConfig:
    sand_engage: float = .70
    sand_release: float = 1.0
    obstacle_engage: float = .15
    obstacle_release: float = .35
    progress_window: float = 3.0
    min_progress: float = .15
    worsening_window: float = 1.5
    worsening_distance: float = .45
    minimum_hold: float = 2.0
    release_progress: float = .35
    success_distance: float = .40


class HazardProgressGate:
    def __init__(self, teacher, config=HazardGateConfig()):
        from unitree_target_terrain import _material_at
        self.teacher, self.config = teacher, config
        assert 0 < config.sand_engage < config.sand_release
        assert 0 <= config.obstacle_engage < config.obstacle_release
        material=np.array([_material_at(teacher.layout,*p) for p in teacher.xy.reshape(-1,2)]).reshape(teacher.hard.shape)
        self.sand=material=='sand'
        self.sand_distance=distance_transform_edt(~self.sand)*teacher.config.resolution if self.sand.any() else np.full(self.sand.shape,np.inf)
        self.obstacle_distance=distance_transform_edt(~teacher.hard)*teacher.config.resolution
        self.reset()

    def reset(self):
        self.history=deque()
        self.engaged=False
        self.reason=None
        self.since=0.
        self.start_distance=0.
        self.escape=False
        self.failures=0
        self.engagements={}

    def _escape_plan(self, position):
        """Shortest collision-free grid path to dry ground with release clearance."""
        t=self.teacher
        start=t.cell(position)
        if t.hard[start]:
            t.path=None;t.status='escape_start_blocked'
            return False
        targets=(self.sand_distance>=self.config.sand_release)&(~t.hard)
        heap=[(0.,start)]; distances={start:0.}; previous={}
        n=len(t.axis)
        while heap:
            distance,here=heapq.heappop(heap)
            if distance!=distances[here]:
                continue
            if targets[here]:
                cells=[here]
                while here!=start:
                    here=previous[here];cells.append(here)
                t.path=np.array([t.xy[cell] for cell in reversed(cells)])
                t.path[0]=position
                t.goal=t.path[-1].copy();t.status='sand_escape'
                return True
            r,c=here
            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
                rr,cc=r+dr,c+dc
                if not(0<=rr<n and 0<=cc<n) or t.hard[rr,cc]:
                    continue
                if dr and dc and (t.hard[r,cc] or t.hard[rr,c]):
                    continue
                other=(rr,cc);candidate=distance+math.hypot(dr,dc)*t.config.resolution
                if candidate<distances.get(other,math.inf):
                    distances[other]=candidate;previous[other]=r,c
                    heapq.heappush(heap,(candidate,other))
        t.path=None;t.status='no_sand_exit'
        return False

    def decide(self, position, yaw, goal, seconds):
        c=self.config;t=self.teacher
        distance=float(np.linalg.norm(goal-position))
        cell=t.cell(position)
        sand_distance=float(self.sand_distance[cell])
        obstacle_distance=float(self.obstacle_distance[cell])
        self.history.append((seconds,distance))
        while len(self.history)>1 and self.history[1][0]<=seconds-c.progress_window:
            self.history.popleft()
        elapsed=seconds-self.history[0][0]
        progress=self.history[0][1]-distance
        recent=next((v for v in reversed(self.history) if v[0]<=seconds-c.worsening_window),self.history[0])
        worsening=seconds-recent[0]>=c.worsening_window-.05 and distance-recent[1]>=c.worsening_distance
        stalled=elapsed>=c.progress_window-.05 and progress<c.min_progress
        reason=('on_sand' if self.sand[cell] else 'near_sand' if sand_distance<=c.sand_engage else
                'obstacle' if obstacle_distance<=c.obstacle_engage else 'worsening' if worsening else 'stalled' if stalled else None)
        if distance<=c.success_distance:
            self.engaged=False;self.reason=None;self.escape=False
        elif not self.engaged and reason:
            self.engaged=True;self.reason=reason;self.since=seconds;self.start_distance=distance
            self.engagements[reason]=self.engagements.get(reason,0)+1
            self.escape=bool(self.sand[cell])
            valid=self._escape_plan(position) if self.escape else t.plan(position,goal) is not None
            if not valid:
                self.failures+=1
        elif self.engaged:
            if self.sand[cell] and not self.escape:
                self.escape=True
                if not self._escape_plan(position):
                    self.failures+=1
            clear=sand_distance>=c.sand_release and obstacle_distance>=c.obstacle_release
            recovered=self.reason in ('near_sand','on_sand','obstacle') or self.start_distance-distance>=c.release_progress
            if clear and recovered and seconds-self.since>=c.minimum_hold:
                self.engaged=False;self.reason=None;self.escape=False;self.history.clear()
            elif self.escape and np.linalg.norm(position-t.goal)<=t.config.stop_distance:
                # A waypoint within stopping distance must not leave us parked at the sand edge.
                self.escape=False
                if t.plan(position,goal) is None:
                    self.failures+=1
        action=t.action(position,yaw) if self.engaged else None
        return action,dict(engaged=self.engaged,reason=self.reason,condition=reason,
            sand_distance=sand_distance,obstacle_distance=obstacle_distance,progress=progress,
            escape=self.escape,planning_failed=self.engaged and t.path is None)

    def metadata(self):
        return dict(config=asdict(self.config),engagements=self.engagements,planning_failures=self.failures)
