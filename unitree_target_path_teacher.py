"""Privileged target-arena A* teacher; never an observation-only controller."""
from dataclasses import asdict, dataclass
import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class PathTeacherConfig:
    resolution: float = .15
    clearance: float = .40
    sand_weight: float = 18.
    ice_weight: float = 8.
    feature_height_limit: float = .13
    lookahead: float = .65
    speed: float = .70
    yaw_gain: float = 1.5
    stop_distance: float = .35


class TargetPathTeacher:
    def __init__(self, layout, config=PathTeacherConfig()):
        from unitree_target_terrain import build_target_terrain_spec, _material_at
        import mujoco
        self.layout, self.config = layout, config
        half = layout.arena_size/2
        self.axis = np.arange(-half+config.resolution/2, half, config.resolution)
        xx, yy = np.meshgrid(self.axis, self.axis)
        self.xy = np.stack((xx, yy), axis=-1)
        hard = (np.abs(xx)>half-.15) | (np.abs(yy)>half-.15)
        model = build_target_terrain_spec(layout).compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for index in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ''
            if not name.startswith(('rough_', 'rubble_', 'stairs_', 'ramp_')):
                continue
            rotation = data.geom_xmat[index].reshape(3,3)
            size = model.geom_size[index]
            top = data.geom_xpos[index,2]+np.abs(rotation[2])@size
            if top < config.feature_height_limit:
                continue
            yaw = math.atan2(rotation[1,0],rotation[0,0])
            c,s = math.cos(yaw),math.sin(yaw)
            basis = np.array([[c,s],[-s,c]])
            local = (self.xy-data.geom_xpos[index,:2])@basis.T
            projected = np.abs(basis@rotation[:2])@size
            hard |= (np.abs(local[:,:,0])<=projected[0]) & (np.abs(local[:,:,1])<=projected[1])
        for obstacle in layout.obstacles:
            # Sweep exclusion is conservative and deterministic, not motion prediction.
            offsets = np.linspace(-1,1,31) if obstacle.moving else [0.]
            for offset in offsets:
                center = np.asarray(obstacle.center)+offset*obstacle.motion_amplitude*np.asarray(obstacle.motion_axis)
                delta = self.xy-center
                if obstacle.kind == 'cylinder':
                    hard |= np.linalg.norm(delta,axis=-1)<=obstacle.size[0]
                else:
                    c,s = math.cos(obstacle.yaw),math.sin(obstacle.yaw)
                    local = delta@np.array([[c,-s],[s,c]])
                    hard |= (np.abs(local[:,:,0])<=obstacle.size[0]) & (np.abs(local[:,:,1])<=obstacle.size[1])
        self.hard = distance_transform_edt(~hard)*config.resolution <= config.clearance
        materials = np.array([_material_at(layout,*point) for point in self.xy.reshape(-1,2)]).reshape(xx.shape)
        self.cost = np.ones(xx.shape)
        for name, weight in [('ice',config.ice_weight),('sand',config.sand_weight)]:
            mask = materials == name
            if mask.any():
                expanded = distance_transform_edt(~mask)*config.resolution <= config.clearance
                self.cost[expanded] = np.maximum(self.cost[expanded],weight)
        # Prefer a little extra separation without making traversable corridors impassable.
        clearance = distance_transform_edt(~self.hard)*config.resolution
        self.cost += .5*np.exp(-clearance/.4)
        self.path = None
        self.goal = None
        self.status = 'unplanned'

    def cell(self, xy):
        col,row = np.rint((np.asarray(xy)-self.axis[0])/self.config.resolution).astype(int)
        n = len(self.axis)
        return int(np.clip(row,0,n-1)),int(np.clip(col,0,n-1))

    def plan(self, start, goal):
        start_cell = self.cell(start)
        goal_cell = self.cell(goal)
        self.goal = np.asarray(goal).copy()
        self.path = None
        if self.hard[start_cell]:
            self.status = 'start_blocked'
            return None
        targets = [(goal_cell,0.)]
        if self.hard[goal_cell]:
            distances = np.linalg.norm(self.xy-self.goal,axis=-1)
            targets = [(tuple(cell),distances[tuple(cell)]) for cell in np.argwhere((distances<=self.config.stop_distance)&~self.hard)]
        targets = [cell for cell,_ in sorted(targets,key=lambda value:value[1])]
        if not targets:
            self.status = 'goal_blocked'
            return None
        target = targets[0]
        distances = {start_cell:0.}
        previous = {}
        heap = [(0.,0.,start_cell)]
        n = len(self.axis)
        while heap:
            _, distance, here = heapq.heappop(heap)
            if distance != distances[here]:
                continue
            if here == target:
                cells = [here]
                while here != start_cell:
                    here = previous[here]
                    cells.append(here)
                self.path = np.array([self.xy[cell] for cell in reversed(cells)])
                self.path[0] = start
                if not self.hard[goal_cell]:
                    self.path[-1] = goal
                self.status = 'planned'
                return self.path
            r,c = here
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                rr,cc = r+dr,c+dc
                if not (0<=rr<n and 0<=cc<n) or self.hard[rr,cc]:
                    continue
                if dr and dc and (self.hard[r,cc] or self.hard[rr,c]):
                    continue
                other = rr,cc
                step = math.hypot(dr,dc)*self.config.resolution
                proposed = distance+step*(self.cost[here]+self.cost[other])/2
                if proposed < distances.get(other,math.inf):
                    distances[other] = proposed
                    previous[other] = here
                    heuristic = math.hypot(rr-target[0],cc-target[1])*self.config.resolution
                    heapq.heappush(heap,(proposed+heuristic,proposed,other))
        self.status = 'no_path'
        return None

    def action(self, position, yaw):
        if self.path is None or np.linalg.norm(self.goal-position)<=self.config.stop_distance:
            return np.zeros(3,dtype=np.float32)
        distances = np.linalg.norm(self.path-position,axis=1)
        closest = int(distances.argmin())
        index = closest
        length = 0.
        while index+1<len(self.path) and length<self.config.lookahead:
            length += np.linalg.norm(self.path[index+1]-self.path[index])
            index += 1
        delta = self.path[index]-position
        angle = math.atan2(delta[1],delta[0])-yaw
        angle = math.atan2(math.sin(angle),math.cos(angle))
        goal_distance = np.linalg.norm(self.goal-position)
        speed = self.config.speed*min(1.,goal_distance/.8)*max(0.,math.cos(angle))
        if abs(angle)>.85:
            speed = 0.
        action = np.array([speed,0.,np.clip(self.config.yaw_gain*angle,-.9,.9)],dtype=np.float32)
        action /= max(1.,float(np.linalg.norm(action)))
        return action

    def metadata(self):
        return dict(config=asdict(self.config),status=self.status,privileged=True,
                    path=None if self.path is None else self.path.tolist())
