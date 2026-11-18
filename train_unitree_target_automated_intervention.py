"""Scripted-teacher diagnostic using the existing online preference/BC/RL learner."""
import argparse
import json
import math
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['height_scan','rgb','rgb_height','depth_height'],required=True)
    p.add_argument('--gate',choices=['disagreement','hazard_progress','none'],default='hazard_progress')
    p.add_argument('--manifest',type=Path,default=Path('artifacts/rgb_height_extended_20260913/manifest.json'))
    p.add_argument('--training-cases',type=Path,help='Teacher audit results with saved reset states; train on safe successful cases.')
    p.add_argument('--objective',choices=['pref_bc_rl','pref_rl','bc_rl','bc_only','pref_only','rl_only'],default='pref_rl')
    p.add_argument('--pref-lambda-init',type=float,default=1.0)
    p.add_argument('--pref-lambda-lr',type=float,default=0.0)
    p.add_argument('--pref-lambda-max',type=float,default=10.0)
    p.add_argument('--alpha',type=float,default=.01)
    p.add_argument('--bc-weight',type=float,default=.2)
    p.add_argument('--episode-seconds',type=float,default=90.)
    p.add_argument('--updates-per-step',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=4500)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--engage-delta',type=float,default=.35)
    p.add_argument('--release-delta',type=float,default=.20)
    p.add_argument('--teacher-warmup',type=int,default=500)
    p.add_argument('--max-falls',type=int,default=3)
    p.add_argument('--smoke',action='store_true')
    opt=p.parse_args()
    assert 0<=opt.release_delta<opt.engage_delta and opt.steps>0
    manifest=json.loads(opt.manifest.read_text())
    demonstrations=Path('artifacts/target_teacher_screen_20260913/demonstrations32')
    collection=json.loads((opt.training_cases or demonstrations/'results.json').read_text())
    cases=[case for case in collection['episodes'] if case['success'] and case.get('cost',0)==0]
    assert cases,'No safe successful teacher cases available'
    assert not set(case['seed'] for case in cases).intersection(case['seed'] for case in manifest['cases'])
    print('[automated-intervention] '+str(vars(opt))+'; privileged automated gate, not human input',flush=True)
    print('[automated-intervention] low_level='+manifest['args']['low_level_policy_path']+
          f'; same saved arena; episodic training cohort; objective={opt.objective}; progress=1 success=20 failure=-20; navigation=5Hz',flush=True)
    import numpy as np
    import torch
    import eval_interactive_unitree_nav as interactive
    from eval_unitree_nav_baselines import _robot_positions_xy,_goal_positions_xy
    from unitree_target_terrain import generate_target_terrain,_material_at
    from unitree_target_path_teacher import TargetPathTeacher
    from unitree_viser_training import BrowserTrainingSession
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    print('[automated-intervention] CUDA '+torch.cuda.get_device_name(),flush=True)
    sys.argv=['automated-intervention']
    args=interactive.parse_args()
    vars(args).update(manifest['args'])
    opt.output.mkdir(parents=True,exist_ok=False)
    args.model_path=''
    args.navigation_observation_mode=opt.mode
    args.controller='policy'
    args.online_train=True
    args.online_total_steps=opt.steps
    args.online_output_dir=str(opt.output.resolve())
    args.online_run_name='training'
    args.online_wandb_mode='disabled'
    args.online_objective_mode=opt.objective
    args.online_pref_lambda_init=opt.pref_lambda_init
    args.online_pref_lambda_lr=opt.pref_lambda_lr
    args.online_pref_lambda_max=opt.pref_lambda_max
    args.online_alpha=opt.alpha
    args.online_actor_bc_weight=opt.bc_weight
    args.online_updates_per_step=opt.updates_per_step
    args.online_checkpoint_interval=1500
    args.human_dataset_dir=''
    args.navigation_episode_mode='episodic'
    args.episode_length_s=opt.episode_seconds
    args.online_dense_progress_scale=1.
    args.online_success_bonus=20.
    args.online_failure_penalty=-20.
    args.automated_teacher_config={k:str(v) if isinstance(v,Path) else v for k,v in vars(opt).items()}
    if opt.smoke:
        args.online_learning_starts=2
        args.online_batch_size=2
        args.online_n_step=1
        args.online_log_interval=1
    layout=generate_target_terrain(args.target_terrain_seed,preset=args.target_terrain_preset,arena_size=args.target_terrain_arena_size)
    assert json.loads(json.dumps(layout.to_dict()))==manifest['layout']==collection['layout']
    from unitree_target_path_teacher import PathTeacherConfig
    teacher_config=cases[0].get('planner',{}).get('config',{})
    teacher=TargetPathTeacher(layout,PathTeacherConfig(**teacher_config))
    from unitree_target_hazard_gate import HazardProgressGate
    gate=HazardProgressGate(teacher) if opt.gate=='hazard_progress' else None
    env=interactive.make_env(args,num_envs=1,render=False)
    u=env.env.unwrapped
    robot=u.scene['robot']
    torso=int(robot.find_bodies('torso_link')[0][0])
    origin=u.scene.env_origins[0,:2].cpu().numpy().copy()
    session=None
    episodes=[]
    falls=0
    episode=0
    teacher_cost=student_cost=0.
    try:
        while session is None or not session.limit_reached:
            case=cases[episode%len(cases)]
            np.random.seed(case['seed']);torch.manual_seed(case['seed'])
            raw,_,_,_,cells=interactive._reset_until_feasible(args,env)
            position=_robot_positions_xy(env)[0].copy()-origin
            goal=_goal_positions_xy(env)[0].copy()-origin
            state=torch.cat([robot.data.root_link_pos_w[0],robot.data.root_link_quat_w[0],
                robot.data.root_link_lin_vel_w[0],robot.data.root_link_ang_vel_w[0],
                robot.data.joint_pos[0],robot.data.joint_vel[0]]).cpu().numpy()
            if 'reset_state' in case:
                np.testing.assert_allclose(state,case['reset_state'],atol=1e-5,rtol=0)
            else:
                with np.load(demonstrations/f"episode_{case['seed']}.npz") as saved:
                    np.testing.assert_allclose(state,saved['reset_state'],atol=1e-5,rtol=0)
            np.testing.assert_allclose(goal,case['goal'],atol=1e-5,rtol=0)
            assert teacher.plan(position,goal) is not None
            if session is None:
                torch.manual_seed(opt.seed)
                session=BrowserTrainingSession(args,env,raw,cells)
            else:
                session.cells=cells
                session.reset(raw)
            engaged=False
            if gate:
                gate.reset()
            start_steps=session.steps
            start_interventions=session.interventions
            success=done=False
            planning_failed=False
            while not session.limit_reached:
                position=_robot_positions_xy(env)[0].copy()-origin
                force=torch.zeros((1,1,3),device=args.device)
                if _material_at(layout,*position)=='sand':
                    force[0,0,:2]=-args.target_terrain_sand_linear_drag*robot.data.root_link_lin_vel_w[0,:2]
                robot.write_external_wrench_to_sim(force,torch.zeros_like(force),body_ids=[torso])
                if session.pending is None:
                    proposed=session.proposal()
                    w,x,y,z=robot.data.root_link_quat_w[0].cpu().numpy()
                    yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
                    if gate:
                        command,event=gate.decide(position,yaw,goal,(session.steps-start_steps)*.2)
                        if event['planning_failed']:
                            planning_failed=True
                            break
                        engaged=command is not None
                        expert=proposed if command is None else torch.tensor(command[None],device=args.device)
                    elif opt.gate=='none':
                        expert=proposed
                        engaged=False
                    else:
                        expert=torch.tensor(teacher.action(position,yaw)[None],device=args.device)
                        delta=float(torch.linalg.vector_norm(proposed-expert))
                        engaged=session.steps<opt.teacher_warmup or delta>(opt.release_delta if engaged else opt.engage_delta)
                    executed=expert if engaged else proposed
                before_cost=session.cost_total
                success,done=session.step(executed,proposed,engaged,2 if engaged else 0,None)
                increment=session.cost_total-before_cost
                teacher_cost+=increment if engaged else 0
                student_cost+=increment if not engaged else 0
                if success or done:
                    break
            fall=bool(u.termination_manager.get_term('fell_over')[0]) if done else False
            falls+=int(fall)
            episodes.append(dict(index=episode,seed=case['seed'],steps=session.steps-start_steps,
                interventions=session.interventions-start_interventions,success=success,fall=fall,
                planning_failed=planning_failed,
                budget_truncated=session.limit_reached and not(success or done)))
            if gate:
                episodes[-1]['gate']=gate.metadata()
            print('[automated-intervention] EPISODE '+json.dumps(episodes[-1]),flush=True)
            (opt.output/'episodes.json').write_text(json.dumps(episodes,indent=2))
            episode+=1
            if len(episodes)>=len(cases) and all(row['steps']==0 for row in episodes[-len(cases):]):
                raise RuntimeError('Teacher failed on an entire training cohort without progress')
            if falls>=opt.max_falls:
                print('[automated-intervention] safety budget reached; stopping with partial checkpoint',flush=True)
                break
        if opt.smoke:
            assert session.learner.updates>0,'Smoke must exercise gradient updates'
        assert all(torch.isfinite(parameter).all() for parameter in session.actor.parameters())
    finally:
        if session is not None:
            session.close()
            summary=dict(steps=session.steps,updates=session.learner.updates,interventions=session.interventions,
                goals=session.successes,falls=falls,teacher_executed_cost=teacher_cost,student_executed_cost=student_cost,
                checkpoint=str(session.learner.run_dir/'final.pt'),objective=opt.objective,
                pref_lambda_init=opt.pref_lambda_init,pref_lambda_lr=opt.pref_lambda_lr,
                final_pref_lambda=float(session.learner.sac.pref_lambda),
                planning_failed_episodes=sum(row['planning_failed'] for row in episodes),
                teacher_cases_total=len(collection['episodes']),teacher_cases_eligible=len(cases),
                gate=opt.gate,note='Scripted privileged intervention; training outcomes are not teacher-free policy quality.',
                training_seeds=[case['seed'] for case in cases])
            (opt.output/'summary.json').write_text(json.dumps(summary,indent=2))
            print('[automated-intervention] SUMMARY '+json.dumps(summary),flush=True)
        env.close()


if __name__=='__main__':
    main()
