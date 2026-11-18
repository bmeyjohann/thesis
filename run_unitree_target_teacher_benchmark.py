"""Teacher rollout audit on a frozen target-arena goal manifest."""
import argparse
import json
import math
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=Path('artifacts/rgb_height_extended_20260913/manifest.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='*')
    parser.add_argument('--max-seconds',type=float,default=180)
    parser.add_argument('--clearance',type=float,default=.40)
    parser.add_argument('--speed',type=float,default=.70)
    parser.add_argument('--height-limit',type=float,default=.13)
    parser.add_argument('--lookahead',type=float,default=.65)
    parser.add_argument('--collect',action='store_true')
    parser.add_argument('--training-seed-start',type=int)
    parser.add_argument('--training-goals',type=int,default=32)
    parser.add_argument('--student',type=Path)
    parser.add_argument('--nav-checkpoint',type=Path)
    parser.add_argument('--direct-goal',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--hazard-progress',action='store_true')
    parser.add_argument('--probe-controller',choices=['random','stationary','away'])
    opt = parser.parse_args()
    assert sum(bool(value) for value in (opt.student,opt.nav_checkpoint,opt.direct_goal,opt.probe_controller))<=1
    assert not(opt.hazard_progress and opt.collect)
    assert not (opt.collect and (opt.student or opt.nav_checkpoint or opt.direct_goal))
    assert not (opt.resume and opt.collect),'Raw demonstration collection remains immutable'
    controller='navigation_policy' if opt.nav_checkpoint else 'student' if opt.student else 'direct_goal' if opt.direct_goal else 'privileged_path_teacher'
    print(f'[target-teacher] resolved {vars(opt)}; controller={controller}, no learning, no human gate',flush=True)
    import numpy as np
    import torch
    import eval_interactive_unitree_nav as interactive
    from eval_unitree_nav_baselines import _robot_positions_xy,_goal_positions_xy
    from unitree_target_terrain import generate_target_terrain,_material_at
    from unitree_target_path_teacher import TargetPathTeacher,PathTeacherConfig
    from unitree_viser_training import BrowserTrainingSession
    torch.set_num_threads(1)
    assert torch.cuda.is_available(), 'CUDA unavailable'
    print(f'[target-teacher] CUDA {torch.cuda.get_device_name()}',flush=True)
    manifest = json.loads(opt.manifest.read_text())
    sys.argv = ['teacher']
    args = interactive.parse_args()
    vars(args).update(manifest['args'])
    if opt.nav_checkpoint:
        nav_saved=torch.load(opt.nav_checkpoint,map_location='cpu',weights_only=False)
        vars(args).update(nav_saved['args'])
        for key in ('low_level_policy_path','goal_encoding','goal_distance_scale','velocity_scale','navigation_policy_hz',
                    'scan_history','action_history','target_terrain_sand_linear_drag','target_terrain_ice_friction'):
            assert vars(args).get(key)==manifest['args'].get(key),f'Navigation checkpoint differs from benchmark: {key}'
    args.model_path = str(opt.nav_checkpoint) if opt.nav_checkpoint else ''
    if not opt.nav_checkpoint:
        args.navigation_observation_mode = 'rgb_depth' if opt.collect or opt.student else 'height_scan'
    args.online_train = False
    args.human_dataset_dir = ''
    args.online_total_steps = 0
    args.controller = 'policy' if opt.nav_checkpoint else 'direct_goal'
    args.navigation_episode_mode = 'episodic'
    args.episode_length_s = opt.max_seconds
    opt.output.mkdir(parents=True,exist_ok=opt.resume)
    rows=[]
    if opt.resume and (opt.output/'results.json').exists():
        previous=json.loads((opt.output/'results.json').read_text())
        assert previous['layout']==manifest['layout']
        assert previous['args']['episode_length_s']==opt.max_seconds
        assert previous.get('benchmark',{}).get('hazard_progress',False)==opt.hazard_progress
        expected=str(opt.student or opt.nav_checkpoint) if opt.student or opt.nav_checkpoint else opt.probe_controller or ('direct_goal' if opt.direct_goal else 'path_teacher')
        assert all(row['controller']==expected for row in previous['episodes'])
        rows=previous['episodes']
    config = PathTeacherConfig(clearance=opt.clearance,speed=opt.speed,
        feature_height_limit=opt.height_limit,lookahead=opt.lookahead)
    layout = generate_target_terrain(args.target_terrain_seed,preset=args.target_terrain_preset,
                                    arena_size=args.target_terrain_arena_size)
    assert json.loads(json.dumps(layout.to_dict())) == manifest['layout']
    teacher = TargetPathTeacher(layout,config)
    from unitree_target_hazard_gate import HazardProgressGate
    gate=HazardProgressGate(teacher) if opt.hazard_progress else None
    student=None
    if opt.student:
        from unitree_target_encoder_screen import load_policy
        saved=torch.load(opt.student,map_location='cpu',weights_only=False)
        environment=saved.get('environment')
        if environment is not None:
            assert environment['layout']==manifest['layout'],'Student terrain differs from evaluation manifest'
            for key in ('scan_history','action_history','navigation_policy_hz','mask_goal_heading',
                        'target_terrain_sand_linear_drag','target_terrain_ice_friction'):
                assert environment['args'].get(key)==vars(args).get(key),f'Student/eval setting differs: {key}'
        student=load_policy(opt.student,args.device)
    if opt.resume and (opt.output/'planner_map.npz').exists():
        with np.load(opt.output/'planner_map.npz') as old_map:
            for name in ('axis','cost','hard'):
                np.testing.assert_array_equal(old_map[name],getattr(teacher,name))
    else:
        np.savez_compressed(opt.output/'planner_map.npz',axis=teacher.axis,cost=teacher.cost,hard=teacher.hard)
    env = interactive.make_env(args,num_envs=1,render=False)
    u=env.env.unwrapped
    robot=u.scene['robot']
    torso=int(robot.find_bodies('torso_link')[0][0])
    origin=u.scene.env_origins[0,:2].cpu().numpy().copy()
    cases=[case for case in manifest['cases'] if opt.seeds is None or case['seed'] in opt.seeds]
    if opt.training_seed_start is not None:
        seeds=range(opt.training_seed_start,opt.training_seed_start+opt.training_goals)
        assert not set(seeds).intersection(case['seed'] for case in manifest['cases'])
        cases=[dict(seed=seed,stratum='training_unfiltered') for seed in seeds]
    finished={row['seed'] for row in rows}
    assert len(finished)==len(rows) and finished.issubset({case['seed'] for case in cases})
    try:
        for case in cases:
            if case['seed'] in finished:
                print(f'[target-teacher] preserving completed seed={case["seed"]}',flush=True)
                continue
            np.random.seed(case['seed'])
            torch.manual_seed(case['seed'])
            raw,_,_,_,cells=interactive._reset_until_feasible(args,env)
            start=_robot_positions_xy(env)[0].copy()-origin
            goal=_goal_positions_xy(env)[0].copy()-origin
            state=torch.cat([robot.data.root_link_pos_w[0],robot.data.root_link_quat_w[0],
                robot.data.root_link_lin_vel_w[0],robot.data.root_link_ang_vel_w[0],
                robot.data.joint_pos[0],robot.data.joint_vel[0]]).cpu().numpy()
            for key, value in (('start',start),('goal',goal),
                               ('initial_orientation',robot.data.root_link_quat_w[0].cpu().numpy())):
                if key in case:
                    np.testing.assert_allclose(value,case[key],atol=1e-5,rtol=0)
            if 'state' in case:
                np.testing.assert_allclose(state,case['state'],atol=1e-5,rtol=0)
                np.testing.assert_allclose(start,case['start'],atol=1e-5,rtol=0)
                np.testing.assert_allclose(goal,case['goal'],atol=1e-5,rtol=0)
            teacher.plan(start,goal)
            if gate:
                gate.reset()
            rng=np.random.default_rng(case['seed'])
            session=BrowserTrainingSession(args,env,raw,cells)
            path=[start.tolist()]
            cost_points=[]
            sand_ticks=0
            started=time.perf_counter()
            success=done=False
            ticks=0
            action=torch.zeros((1,3),device=args.device)
            samples=[]
            commands=[]
            decision_positions=[]
            gate_events=[]
            teacher_ticks=[]
            teacher_cost=student_cost=0.
            intervened=False
            gate_abstained=False
            proposed=action
            if teacher.path is not None or student is not None or opt.direct_goal or opt.nav_checkpoint or opt.probe_controller:
                for tick in range(int(math.ceil(opt.max_seconds/.05))):
                    position=_robot_positions_xy(env)[0].copy()-origin
                    force=torch.zeros((1,1,3),device=args.device)
                    if _material_at(layout,*position)=='sand':
                        force[0,0,:2]=-args.target_terrain_sand_linear_drag*robot.data.root_link_lin_vel_w[0,:2]
                        sand_ticks+=1
                    robot.write_external_wrench_to_sim(force,torch.zeros_like(force),body_ids=[torso])
                    if session.pending is None:
                        w,x,y,z=robot.data.root_link_quat_w[0].cpu().numpy()
                        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
                        action=torch.tensor(teacher.action(position,yaw)[None],device=args.device)
                        if opt.direct_goal or opt.nav_checkpoint:
                            action=session.proposal()
                        if opt.probe_controller:
                            value=rng.uniform(-1,1,3) if opt.probe_controller=='random' else np.zeros(3)
                            if opt.probe_controller=='away':
                                angle=math.atan2(*(goal-position)[::-1])-yaw+math.pi
                                value=np.array([.7*max(0.,math.cos(angle)),0.,np.clip(math.atan2(math.sin(angle),math.cos(angle)),-.9,.9)])
                            value=value/max(1.,np.linalg.norm(value))
                            action=torch.tensor(value[None],device=args.device,dtype=torch.float32)
                        if opt.collect or student is not None:
                            base=interactive._prepare_policy_obs(session.raw,args)[0].detach().cpu().numpy().copy()
                            history=session.obs[0,-3*args.action_history:].detach().cpu().numpy().copy()
                            sample=dict(base=np.concatenate((base,history)),action=action[0].cpu().numpy().copy(),
                                **{key:value.copy() for key,value in session.vision.last_frames.items()})
                            if student is not None:
                                with torch.no_grad():
                                    action=student(*[torch.from_numpy(sample[key][None]).to(args.device)
                                        for key in ('base','vision_rgb','vision_depth')])
                            if opt.collect:
                                assert student is None,'Keep teacher demonstration labels unambiguous'
                                samples.append(sample)
                        proposed=action.clone()
                        if gate:
                            expert,event=gate.decide(position,yaw,goal,tick*.05)
                            intervened=expert is not None
                            gate_events.append(event)
                            if event['planning_failed']:
                                gate_abstained=True
                                break
                            if intervened:
                                action=torch.tensor(expert[None],device=args.device)
                        commands.append(action[0].detach().cpu().numpy().tolist())
                        decision_positions.append(position.tolist())
                    old_cost=session.cost_total
                    success,done=session.step(action,proposed,intervened,2 if intervened else 0,None)
                    teacher_ticks.append(intervened)
                    increment=session.cost_total-old_cost
                    teacher_cost+=increment if intervened else 0
                    student_cost+=increment if not intervened else 0
                    ticks=tick+1
                    if not done:
                        path.append((_robot_positions_xy(env)[0]-origin).tolist())
                    if session.cost_total>old_cost:
                        cost_points.append(path[-1])
                    if success or done:
                        break
                    if ticks%600==0:
                        print(f'[target-teacher] seed={case["seed"]} sim_seconds={ticks*.05}',flush=True)
            fall=bool(u.termination_manager.get_term('fell_over')[0]) if done else False
            row=dict(seed=case['seed'],stratum=case['stratum'],controller=str(opt.student or opt.nav_checkpoint) if student is not None or opt.nav_checkpoint else opt.probe_controller or ('direct_goal' if opt.direct_goal else 'path_teacher'),
                success=success,fall=fall,timeout=bool(not(success or fall or gate_abstained) and (teacher.path is not None or student is not None or opt.direct_goal or opt.nav_checkpoint or opt.probe_controller)),
                planning_failed=bool(gate_abstained or (teacher.path is None and student is None and not opt.direct_goal and not opt.nav_checkpoint and not opt.probe_controller)),seconds=ticks*.05,sand_seconds=sand_ticks*.05,
                cost=session.cost_total,decisions=session.steps,start=start.tolist(),goal=goal.tolist(),
                reset_state=state.tolist(),initial_orientation=state[3:7].tolist(),
                trajectory=path,cost_points=cost_points,planner=teacher.metadata(),
                commands=commands,decision_positions=decision_positions,
                gate=None if gate is None else gate.metadata(),gate_events=gate_events,
                teacher_ticks=teacher_ticks,teacher_executed_cost=teacher_cost,student_executed_cost=student_cost,
                intervention_fraction=float(np.mean(teacher_ticks)) if teacher_ticks else 0.,
                final_distance=float(np.linalg.norm(np.asarray(path[-1])-goal)),wall_seconds=time.perf_counter()-started)
            if commands:
                values=np.asarray(commands)
                norms=np.linalg.norm(values,axis=1)
                row['action_diagnostics']=dict(mean_norm=float(norms.mean()),low_norm_fraction=float(np.mean(norms<.1)),
                    forward_std=float(values[:,0].std()),yaw_std=float(values[:,2].std()))
            rows.append(row)
            if samples:
                np.savez_compressed(opt.output/f"episode_{case['seed']}.npz",
                    **{key:np.stack([sample[key] for sample in samples]) for key in samples[0]},
                    success=np.array(success),seed=np.array(case['seed']),reset_state=state)
            temporary=opt.output/'results.tmp'
            temporary.write_text(json.dumps(dict(args=vars(args),layout=layout.to_dict(),episodes=rows,
                benchmark={k:str(v) if isinstance(v,Path) else v for k,v in vars(opt).items()}),indent=2))
            temporary.replace(opt.output/'results.json')
            print('RESULT '+json.dumps({k:v for k,v in row.items() if k not in {'trajectory','cost_points','planner','commands','decision_positions','gate_events','teacher_ticks'}}),flush=True)
            session.close()
    finally:
        env.close()
    print('SUMMARY '+json.dumps(dict(episodes=len(rows),success=sum(r['success'] for r in rows),
        falls=sum(r['fall'] for r in rows),timeouts=sum(r['timeout'] for r in rows),
        planning_failed=sum(r['planning_failed'] for r in rows))),flush=True)


if __name__=='__main__':
    main()
