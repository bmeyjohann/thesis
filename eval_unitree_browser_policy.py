"""Teacher-free checkpoint evaluation with the browser's exact decision/vision path."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--episode-seconds', type=float, default=60)
    parser.add_argument('--first-seed', type=int, default=101)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    if options.episodes < 1 or options.episode_seconds <= 0:
        raise ValueError('Positive episode count and duration required')
    options.output.mkdir(parents=True, exist_ok=False)
    import eval_interactive_unitree_nav as interactive
    from eval_unitree_nav_baselines import _robot_positions_xy, _goal_positions_xy
    from unitree_viser_training import BrowserTrainingSession
    from unitree_target_terrain import generate_target_terrain, _material_at
    saved = torch.load(options.checkpoint, map_location='cpu', weights_only=False)
    sys.argv = ['eval']
    args = interactive.parse_args()
    vars(args).update(saved['args'])
    args.model_path = str(options.checkpoint.resolve())
    args.online_train = False
    args.human_dataset_dir = ''
    args.controller = 'policy'
    args.navigation_episode_mode = 'episodic'
    args.episode_length_s = options.episode_seconds
    env = interactive.make_env(args, num_envs=1, render=False)
    u = env.env.unwrapped
    robot = u.scene['robot']
    torso = int(robot.find_bodies('torso_link')[0][0])
    layout = generate_target_terrain(args.target_terrain_seed, preset=args.target_terrain_preset,
                                     arena_size=args.target_terrain_arena_size)
    episodes = []
    session = None
    try:
        for seed in range(options.first_seed, options.first_seed+options.episodes):
            np.random.seed(seed)
            torch.manual_seed(seed)
            raw,_,_,_,cells = interactive._reset_until_feasible(args,env)
            if session is None:
                session = BrowserTrainingSession(args,env,raw,cells)
            else:
                session.reset(raw)
            start = _robot_positions_xy(env)[0].copy()
            goal = _goal_positions_xy(env)[0].copy()
            origin = u.scene.env_origins[0,:2].cpu().numpy().copy()
            orientation = robot.data.root_link_quat_w[0].cpu().numpy().tolist()
            path = [(start-origin).tolist()]
            cost_points = []
            started = time.perf_counter()
            old_cost = session.cost_total
            for tick in range(int(np.ceil(options.episode_seconds/.05))):
                force = torch.zeros((1,1,3), device=args.device)
                xy = robot.data.root_link_pos_w[0,:2].cpu().numpy()-origin
                if _material_at(layout,*xy) == 'sand':
                    force[0,0,:2] = -args.target_terrain_sand_linear_drag * robot.data.root_link_lin_vel_w[0,:2]
                robot.write_external_wrench_to_sim(force,torch.zeros_like(force),body_ids=[torso])
                action = session.proposal()
                before_cost = session.cost_total
                success,done = session.step(action,action,False,0,None)
                if not done:
                    path.append((_robot_positions_xy(env)[0]-origin).tolist())
                if session.cost_total > before_cost:
                    cost_points.append(path[-1])
                if success or done:
                    break
            fall = bool(u.termination_manager.get_term('fell_over')[0]) if done else False
            row = dict(seed=seed, start=(start-origin).tolist(), goal=(goal-origin).tolist(),
                       initial_orientation=orientation, trajectory=path, cost_points=cost_points,
                       success=success, fall=fall, timeout=not(success or fall),
                       seconds=(tick+1)*.05, wall_seconds=time.perf_counter()-started,
                       cost=session.cost_total-old_cost)
            episodes.append(row)
            (options.output/'results.json').write_text(json.dumps(dict(checkpoint=str(options.checkpoint),
                args=vars(args),layout=layout.to_dict(),episodes=episodes),indent=2))
            print(json.dumps({k:v for k,v in row.items() if k not in {'trajectory','cost_points'}}),flush=True)
    finally:
        if session is not None:
            session.close()
        env.close()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,len(episodes),figsize=(4*len(episodes),4),squeeze=False)
    for ax,row in zip(axes.flat,episodes):
        xy=np.asarray(row['trajectory'])
        ax.plot(xy[:,0],xy[:,1])
        ax.scatter(*row['start'],c='black')
        ax.scatter(*row['goal'],c='green',marker='*')
        if row['cost_points']:
            pts=np.asarray(row['cost_points'])
            ax.scatter(pts[:,0],pts[:,1],c='red',marker='x')
        ax.set_title(f"seed {row['seed']}: success={row['success']}")
        ax.set_aspect('equal',adjustable='datalim')
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(options.output/'trajectories.png',dpi=130)


if __name__ == '__main__':
    main()
