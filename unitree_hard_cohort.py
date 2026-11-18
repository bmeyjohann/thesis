"""CPU geometry admission checks for a physically blocked Unitree cohort."""
from __future__ import annotations

import hashlib
import heapq
import json
import math

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

CRITERIA = dict(min_components=2, min_distance=4.0, max_distance=7.0,
                start_clearance=1.0, goal_clearance=0.9,
                route_clearance=0.6, min_route_ratio=1.25, min_extra_route=1.0,
                grid_resolution=0.15)


class Geometry:
    def __init__(self, cells, half_size, pitch):
        self.cells = np.asarray(cells, dtype=float).reshape(-1, 2)
        self.half_size = np.asarray(half_size, dtype=float)
        self.pitch = float(pitch)
        if not len(self.cells) or self.pitch <= 0:
            raise ValueError('Nonempty heightfield geometry and positive pitch required')
        self.tree = cKDTree(self.cells)
        # Label physical raised samples, not inflated cells that can merge obstacles.
        ij = np.rint((self.cells + self.half_size) / self.pitch).astype(int)
        mask = np.zeros(tuple(ij.max(axis=0) + 1), dtype=bool)
        mask[ij[:, 0], ij[:, 1]] = True
        labels, _ = label(mask, structure=np.ones((3, 3), dtype=int))
        self.components = labels[ij[:, 0], ij[:, 1]]
        payload = dict(cells=np.round(self.cells, 4).tolist(),
                       half_size=np.round(self.half_size, 4).tolist(), pitch=round(self.pitch, 6))
        self.sha256 = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        self.resolution = CRITERIA['grid_resolution']
        # A diagonal half-cell bounds the interpolated heightfield footprint.
        self.cell_margin = self.pitch / math.sqrt(2)
        axes = [np.arange(-v + 0.75, v - 0.75, self.resolution) for v in self.half_size]
        self.xs, self.ys = axes
        xx, yy = np.meshgrid(*axes, indexing='ij')
        points = np.stack([xx, yy], axis=-1)
        self.distances = self.tree.query(points.reshape(-1, 2))[0].reshape(xx.shape) - self.cell_margin
        # Also reserve half a planning-cell diagonal so edges remain clear.
        self.free = self.distances >= CRITERIA['route_clearance'] + self.resolution / math.sqrt(2)

    def blocking_components(self, start, goal):
        start, goal = np.asarray(start), np.asarray(goal)
        distance = float(np.linalg.norm(goal - start))
        t = np.linspace(0, 1, max(2, int(math.ceil(distance / (self.pitch / 4))) + 1))
        points = start + t[:, None] * (goal - start)
        distances, indices = self.tree.query(points)
        # Inscribed sample disks certify crossing the raised interior, not a near miss.
        mask = (distances <= self.pitch * 0.45) & (t * distance > 0.75) & ((1-t) * distance > 0.75)
        return sorted(set(int(i) for i in self.components[indices[mask]]))

    def _node(self, point):
        return (int(np.argmin(abs(self.xs - point[0]))), int(np.argmin(abs(self.ys - point[1]))))

    def route(self, start, goal):
        start, goal = np.asarray(start), np.asarray(goal)
        source, target = self._node(start), self._node(goal)
        if not self.free[source] or not self.free[target]:
            return None
        best, previous = {source: 0.0}, {}
        queue = [(float(np.linalg.norm(np.subtract(source, target))) * self.resolution, 0.0, source)]
        while queue:
            _, cost, node = heapq.heappop(queue)
            if cost > best[node]:
                continue
            if node == target:
                path = [node]
                while path[-1] != source:
                    path.append(previous[path[-1]])
                path.reverse()
                points = np.array([[self.xs[i], self.ys[j]] for i, j in path])
                return np.vstack([start, points, goal])
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == dj == 0:
                        continue
                    new = node[0] + di, node[1] + dj
                    if not (0 <= new[0] < self.free.shape[0] and 0 <= new[1] < self.free.shape[1]):
                        continue
                    if not self.free[new] or not self.free[node[0]+di, node[1]] or not self.free[node[0], node[1]+dj]:
                        continue
                    candidate = cost + math.hypot(di, dj) * self.resolution
                    if candidate < best.get(new, math.inf):
                        best[new], previous[new] = candidate, node
                        heuristic = math.hypot(new[0]-target[0], new[1]-target[1]) * self.resolution
                        heapq.heappush(queue, (candidate + heuristic, candidate, new))
        return None

    def audit(self, start, goal, *, require=True):
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        distance = float(np.linalg.norm(goal-start))
        sc, gc = self.tree.query([start, goal])[0] - self.cell_margin
        basic = (np.all(abs(start) < self.half_size-0.8) and np.all(abs(goal) < self.half_size-0.8)
                 and CRITERIA['min_distance'] <= distance <= CRITERIA['max_distance']
                 and sc >= CRITERIA['start_clearance'] and gc >= CRITERIA['goal_clearance'])
        components = self.blocking_components(start, goal) if basic else []
        basic = basic and len(components) >= CRITERIA['min_components']
        route = self.route(start, goal) if basic else None
        length = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()) if route is not None else 0.0
        accepted = bool(basic and length >= CRITERIA['min_route_ratio'] * distance
                        and length-distance >= CRITERIA['min_extra_route'])
        result = dict(accepted=accepted, geometry_sha256=self.sha256,
                      start_xy=start.tolist(), goal_xy=goal.tolist(), half_size=self.half_size.tolist(),
                      physical_blocking_components=len(components), straight_distance=distance,
                      start_clearance=float(sc), goal_clearance=float(gc),
                      clearance_grid_route_length=length, clearance_grid_route_ratio=length/max(distance, 1e-9),
                      reference_route=[] if route is None else route.tolist())
        if require and not accepted:
            raise ValueError(f'Hard cohort admission failed: {result}')
        return result


_REPLAY_CACHE = {}


def audit_actual_replay(args, env, env_ids, entries, result):
    """Audit actual reset poses/goals and authoritative tile geometry on every replay."""
    import eval_unitree_nav_baselines as evaluation
    from unitree_nav_eval_manifest import capture_layout_entries
    unwrapped = env.env.unwrapped
    terrain = unwrapped.scene.terrain
    actual = capture_layout_entries(env, [int(i) for i in env_ids])
    for env_id, expected, observed in zip(env_ids, entries, actual):
        if 'hard_audit' not in expected:
            raise ValueError('Hard comparison must not replay an unaudited case')
        env_id = int(env_id)
        tile = int(expected['tile_index'])
        origin = evaluation._to_numpy(terrain.env_origins[env_id])[:2]
        cells = result[-1][env_id] - origin
        sizes = evaluation._to_numpy(unwrapped.sim.model.hfield_size)[tile, :2]
        ncol = int(evaluation._to_numpy(unwrapped.sim.model.hfield_ncol)[tile])
        geometry = Geometry(cells, sizes, 2*sizes[0]/(ncol-1))
        if geometry.sha256 != expected['hard_audit']['geometry_sha256']:
            raise ValueError('Actual replay geometry differs from frozen manifest')
        for field in ('root_pose_relative', 'goal_xy_relative'):
            if not np.allclose(observed[field], expected[field], atol=2e-4, rtol=0):
                raise ValueError(f'Actual reset {field} differs from manifest')
        if observed['tile_index'] != tile:
            raise ValueError('Actual tile differs from manifest')
        key = (geometry.sha256, tuple(np.round(observed['root_pose_relative'][:2], 4)),
               tuple(np.round(observed['goal_xy_relative'], 4)))
        if key not in _REPLAY_CACHE:
            _REPLAY_CACHE[key] = geometry.audit(observed['root_pose_relative'][:2], observed['goal_xy_relative'])
    return result
