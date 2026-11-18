"""Finite matched camera-policy evaluation; never trains or edits source runs."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parent


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blocked_by(start, goal, layout):
    points = np.linspace(start, goal, max(2, int(np.linalg.norm(goal-start)/.01)+1))
    hits = []
    for i, obstacle in enumerate(layout.obstacles):
        if obstacle.moving:
            continue
        d = points-np.array(obstacle.center)
        if obstacle.kind == 'cylinder':
            inside = np.linalg.norm(d, axis=1) < obstacle.size[0]
        else:
            c, s = math.cos(obstacle.yaw), math.sin(obstacle.yaw)
            q = d @ np.array([[c, -s], [s, c]])
            inside = (np.abs(q[:, 0]) < obstacle.size[0]) & (np.abs(q[:, 1]) < obstacle.size[1])
        if inside.any():
            hits.append(i)
    return hits


class ContactAudit:
    """Read-only physics-substep contacts and pre-autoreset terminal pose."""
    def __init__(self, manager, layout):
        self.u = manager
        self.robot = manager.scene['robot']
        model = manager.sim.mj_model
        names = [model.geom(i).name or '' for i in range(model.ngeom)]
        # MJLab renames generated terrain geoms. Match the compiled static
        # obstacle style and verify its complete count, rather than names alone.
        static_ids = [i for i in range(model.ngeom) if np.allclose(model.geom_rgba[i], [.86,.16,.12,1.], atol=1e-5)]
        moving_ids = [i for i,n in enumerate(names) if n.startswith('target_mover_')]
        assert len(static_ids) == sum(not o.moving for o in layout.obstacles), 'Static obstacle geometry coverage differs'
        assert len(moving_ids) == sum(o.moving for o in layout.obstacles), 'Moving obstacle geometry coverage differs'
        obstacle_ids = set(static_ids+moving_ids)
        self.names = dict(robot=[n for n in names if n.startswith('robot/')],
                          obstacle=[names[i] for i in sorted(obstacle_ids)], obstacle_geom_ids=sorted(obstacle_ids),
                          static_count=len(static_ids), moving_count=len(moving_ids))
        assert self.names['robot'] and self.names['obstacle']
        self.kind = torch.tensor([1 if n.startswith('robot/') else 2 if i in obstacle_ids else 0
                                  for i,n in enumerate(names)], device=manager.device)
        self.metrics = torch.zeros(2, device=manager.device)
        self.metrics_wp = wp.from_torch(self.metrics)
        self.kind_wp = wp.from_torch(self.kind)
        self.original_step = manager.sim.step
        self.original_term = manager.termination_manager.compute
        manager.sim.step = self.physics_step
        manager.termination_manager.compute = self.termination
        self.clear()

    def clear(self):
        self.metrics.zero_()
        self.last_position = None
        self.fall = False
        self.terminated = False
        self.timed_out = False

    def physics_step(self):
        self.original_step()
        from unitree_eval_contacts import accumulate_contacts
        data = self.u.sim.wp_data
        wp.launch(accumulate_contacts, dim=1,
                  inputs=[data.nacon, data.contact.geom, data.contact.dist, data.contact.worldid,
                          self.kind_wp, self.metrics_wp], device=data.contact.geom.device)

    @property
    def count(self):
        return self.metrics[0]

    @property
    def penetration(self):
        return self.metrics[1]

    def termination(self):
        result = self.original_term()
        self.last_position = self.robot.data.root_link_pos_w[0].detach().clone()
        self.fall = bool(self.u.termination_manager.get_term('fell_over')[0])
        self.terminated = bool(self.u.termination_manager.terminated[0])
        self.timed_out = bool(self.u.termination_manager.time_outs[0])
        return result

    def close(self):
        self.u.sim.step = self.original_step
        self.u.termination_manager.compute = self.original_term


def state_of(u):
    r = u.scene['robot'].data
    return torch.cat((r.root_link_pos_w[0], r.root_link_quat_w[0], r.root_link_lin_vel_w[0],
                      r.root_link_ang_vel_w[0], r.joint_pos[0], r.joint_vel[0])).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--curriculum', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', type=int, default=32)
    p.add_argument('--horizon', type=float, default=180.)
    p.add_argument('--resume', action='store_true')
    opt = p.parse_args()
    assert opt.cases > 0 and opt.cases % 2 == 0
    torch.set_num_threads(1)
    import eval_interactive_unitree_nav as interactive
    from eval_unitree_nav_baselines import _robot_positions_xy, _goal_positions_xy
    from unitree_target_terrain import generate_target_terrain, _material_at
    from unitree_target_navigation import _segment_feature_labels
    from unitree_target_path_teacher import TargetPathTeacher, PathTeacherConfig
    from unitree_viser_training import BrowserTrainingSession
    curriculum = json.loads(opt.curriculum.read_text(encoding='utf-8-sig'))
    selected = [r for r in curriculum['Runs'] if r['Status'] == 'complete']
    assert len(selected) == 3, 'Explicit three-completed-run scope required'
    checkpoints = {r['Id']: ROOT/'models/unitree_mjlab_nav_human'/r['RunName']/'final.pt' for r in selected}
    identities = {name: dict(path=str(path), sha256=sha(path)) for name, path in checkpoints.items()}
    opt.output.mkdir(parents=True, exist_ok=opt.resume)
    declaration = dict(checkpoints=identities, horizon=opt.horizon, count=opt.cases,
                       curriculum_sha256=sha(opt.curriculum), decision_hz=5,
                       safe_success='goal reached, no fall, no robot-static/moving-obstacle contact at physics substeps',
                       material_safe_success='safe success and no base-center ice/sand exposure',
                       sampling='half unfiltered reset draws; half static centerline-blocked draws with planner-feasible detour; no outcome filtering',
                       fresh_terrain=False, human_gate=False, learning=False)
    if (opt.output/'plan.json').exists():
        assert json.loads((opt.output/'plan.json').read_text()) == declaration
    else:
        save(opt.output/'plan.json', declaration)
    models = list(checkpoints.items())+[('direct_goal', next(iter(checkpoints.values())))]
    manifest_path = opt.output/'matched_manifest.json'
    cases = json.loads(manifest_path.read_text())['cases'] if manifest_path.exists() else None
    canonical_layout = None
    for name, checkpoint in models:
        destination = opt.output/name
        destination.mkdir(exist_ok=True)
        rows = json.loads((destination/'results.json').read_text())['episodes'] if (destination/'results.json').exists() else []
        if len(rows) == opt.cases:
            continue
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        sys.argv = ['matched-human-eval']
        args = interactive.parse_args()
        vars(args).update(saved['args'])
        args.model_path = str(checkpoint)
        args.online_train = False
        args.human_dataset_dir = ''
        args.observation_only = False
        args.online_total_steps = 0
        args.controller = 'direct_goal' if name == 'direct_goal' else 'policy'
        args.navigation_episode_mode = 'episodic'
        args.runtime_navigation_episode_mode = 'episodic'
        args.episode_length_s = opt.horizon
        assert args.navigation_policy_hz == 5 and args.navigation_feature_lags == [0, 1, 25]
        print('EVALUATING', name, str(checkpoint), flush=True)
        layout = generate_target_terrain(args.target_terrain_seed, preset=args.target_terrain_preset, arena_size=args.target_terrain_arena_size)
        layout_dict = json.loads(json.dumps(layout.to_dict()))
        if canonical_layout is None:
            canonical_layout = layout_dict
        assert layout_dict == canonical_layout
        env = interactive.make_env(args, num_envs=1, render=False)
        u = env.env.unwrapped
        robot = u.scene['robot']
        torso = int(robot.find_bodies('torso_link')[0][0])
        audit = ContactAudit(u, layout)
        save(destination/'contact_geometry.json', audit.names)
        def reset(seed):
            np.random.seed(seed)
            torch.manual_seed(seed)
            raw, _, _, _, cells = interactive._reset_until_feasible(args, env)
            origin = u.scene.env_origins[0, :2].cpu().numpy().copy()
            start, goal = _robot_positions_xy(env)[0]-origin, _goal_positions_xy(env)[0]-origin
            return raw, cells, origin, start, goal, state_of(u)
        try:
            if cases is None:
                cases = []
                planner = TargetPathTeacher(layout, PathTeacherConfig(clearance=.4))
                attempts = 0
                while len(cases) < opt.cases:
                    seed = 551000+attempts
                    attempts += 1
                    assert attempts <= 2000, 'Cannot construct blocked cohort; do not weaken it'
                    raw, cells, origin, start, goal, state = reset(seed)
                    blockers = blocked_by(start, goal, layout)
                    stratum = 'unfiltered' if len(cases) < opt.cases//2 else 'blocked_detour'
                    if stratum == 'blocked_detour':
                        if not blockers:
                            continue
                        planner.plan(start, goal)
                        if planner.path is None:
                            continue
                    cases.append(dict(seed=seed, stratum=stratum, start=start.tolist(), goal=goal.tolist(),
                                      state=state.tolist(), origin=origin.tolist(), blockers=blockers,
                                      route_features=list(_segment_feature_labels(start, goal, layout))))
                save(manifest_path, dict(layout=layout_dict, cases=cases, reset_candidates=attempts))
            assert json.loads(manifest_path.read_text())['layout'] == layout_dict
            if not (destination/'contact_probe.json').exists():
                reset(cases[0]['seed'])
                obstacle = next(o for o in layout.obstacles if not o.moving)
                probe = torch.as_tensor(state_of(u)[:13], device=args.device)[None].clone()
                probe[0, :2] = u.scene.env_origins[0, :2]+probe.new_tensor(obstacle.center)
                probe[0, 2] = u.scene.env_origins[0, 2]+obstacle.size[2]/2
                probe[:, 7:] = 0
                robot.write_root_state_to_sim(probe)
                u.sim.forward()
                audit.clear()
                env.step(torch.zeros((1, 3), device=args.device))
                assert int(audit.count) > 0, 'Positive obstacle-contact control failed'
                save(destination/'contact_probe.json', dict(positive_contact_substeps=int(audit.count),
                     excluded_from_evaluation=True, next_action='full matched reset before scoring'))
            done_seeds = {r['seed'] for r in rows}
            for case in cases:
                if case['seed'] in done_seeds:
                    continue
                raw, cells, origin, start, goal, state = reset(case['seed'])
                for key, value in [('start', start), ('goal', goal), ('state', state), ('origin', origin)]:
                    np.testing.assert_allclose(value, case[key], atol=1e-5, rtol=0, err_msg=f'{name} {case["seed"]} {key}')
                session = BrowserTrainingSession(args, env, raw, cells)
                # Encoder/head construction must not alter any later random draws.
                np.random.seed(case['seed']+10000000)
                torch.manual_seed(case['seed']+10000000)
                audit.clear()
                path = [start.tolist()]
                collisions, costs, commands, materials = [], [], [], []
                ice = sand = 0
                begin = time.perf_counter()
                success = done = False
                for tick in range(math.ceil(opt.horizon/.05)):
                    position = _robot_positions_xy(env)[0]-origin
                    material = _material_at(layout, *position)
                    ice += material == 'ice'
                    sand += material == 'sand'
                    force = torch.zeros((1, 1, 3), device=args.device)
                    if material == 'sand':
                        force[0, 0, :2] = -args.target_terrain_sand_linear_drag*robot.data.root_link_lin_vel_w[0, :2]
                    robot.write_external_wrench_to_sim(force, torch.zeros_like(force), body_ids=[torso])
                    if session.pending is None:
                        action = session.proposal()
                        commands.append(action[0].cpu().tolist())
                    old_contact = float(audit.count)
                    old_cost = session.cost_total
                    success, done = session.step(action, action, False, 0, None)
                    terminal = audit.last_position[:2].cpu().numpy()-origin
                    current = terminal if done else _robot_positions_xy(env)[0]-origin
                    path.append(current.tolist())
                    materials.append(material)
                    if float(audit.count) > old_contact:
                        collisions.append(dict(t=(tick+1)*.05, xy=current.tolist()))
                    if session.cost_total > old_cost:
                        costs.append(dict(t=(tick+1)*.05, xy=current.tolist()))
                    if success or done:
                        break
                ticks = tick+1
                fall = audit.fall
                count = int(audit.count)
                distances = np.linalg.norm(np.array(path)-goal, axis=1)
                safe = bool(success and not fall and count == 0)
                row = dict(seed=case['seed'], stratum=case['stratum'], start=start.tolist(), goal=goal.tolist(),
                           reset_state=state.tolist(), success=bool(success), safe_success=safe,
                           material_safe_success=bool(safe and not ice and not sand), fall=fall,
                           timeout=bool(not success and not fall and (audit.timed_out or not done)),
                           other_termination=bool(done and not success and not fall and not audit.timed_out),
                           seconds=ticks*.05, ticks=ticks, decisions=session.steps,
                           native_cost=float(session.cost_total), obstacle_contact_substeps=count,
                           obstacle_contact_seconds=count*float(u.physics_dt), max_penetration_m=float(audit.penetration),
                           ice_seconds=ice*.05, sand_seconds=sand*.05, final_distance=float(distances[-1]),
                           min_distance=float(distances.min()), progress_m=float(distances[0]-distances[-1]),
                           path_length_m=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
                           trajectory=path, contact_points=collisions, cost_points=costs, materials=materials,
                           commands=commands, wall_seconds=time.perf_counter()-begin)
                assert session.steps == math.ceil(ticks/4), 'Decision accounting differs from 5 Hz action hold'
                rows.append(row)
                save(destination/'results.json', dict(name=name, args=vars(args), layout=layout_dict,
                                                      manifest_sha256=sha(manifest_path), episodes=rows))
                print('RESULT', name, len(rows), json.dumps({k:v for k,v in row.items() if k not in ('trajectory','contact_points','cost_points','materials','commands','reset_state')}), flush=True)
                session.close()
        finally:
            audit.close()
            env.close()
    save(opt.output/'COMPLETE.json', dict(models=list(checkpoints)+['direct_goal'], cases=opt.cases))


if __name__ == '__main__':
    main()
