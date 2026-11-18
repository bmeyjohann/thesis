"""Small paired, teacher-free audit using the exact browser decision/history path."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eval_interactive_unitree_nav as interactive
from eval_unitree_nav_baselines import _robot_positions_xy, _goal_positions_xy
from unitree_viser_training import BrowserTrainingSession
from unitree_target_terrain import generate_target_terrain, _material_at


def main():
    run = ROOT / 'models/unitree_mjlab_nav_human/target_human_20260910_193717'
    out = ROOT / 'artifacts/browser_audit_20260910'
    out.mkdir(parents=True, exist_ok=False)
    sys.argv = ['audit']
    args = interactive.parse_args()
    vars(args).update(json.loads((run / 'args.json').read_text()))
    args.online_train = False
    args.human_dataset_dir = ''
    args.online_total_steps = 0
    args.navigation_episode_mode = 'episodic'
    args.episode_length_s = 60
    args.seed = 0
    env = interactive.make_env(args, num_envs=1, render=False)
    layout = generate_target_terrain(args.target_terrain_seed, preset=args.target_terrain_preset,
                                     arena_size=args.target_terrain_arena_size)
    robot = env.env.unwrapped.scene['robot']
    torso = int(robot.find_bodies('torso_link')[0][0])
    results, manifests = [], {}
    controllers = [('early_500', 'policy', run / 'step_500.pt'),
                   ('final_2542', 'policy', run / 'final.pt'),
                   ('direct_goal', 'direct_goal', None)]
    for label, controller, path in controllers:
        args.controller = controller
        args.model_path = str(path) if path else ''
        for seed in [101, 102, 103, 104]:
            np.random.seed(seed)
            torch.manual_seed(seed)
            raw, _, _, _, cells = interactive._reset_until_feasible(args, env)
            start, goal = _robot_positions_xy(env)[0].copy(), _goal_positions_xy(env)[0].copy()
            initial = np.concatenate([start, goal, robot.data.root_link_quat_w[0].cpu().numpy()])
            if seed in manifests:
                assert np.allclose(initial, manifests[seed], atol=1e-5), (seed, initial, manifests[seed])
            else:
                manifests[seed] = initial.tolist()
            session = BrowserTrainingSession(copy.copy(args), env, raw, cells)
            trajectory = [start.tolist()]
            cost_points = []
            minimum = float(np.linalg.norm(start-goal))
            success = ended = False
            ticks = 0
            while ticks < 1200:
                velocity = robot.data.root_link_lin_vel_w
                force = torch.zeros((1, 1, 3), device=args.device)
                xy = robot.data.root_link_pos_w[0, :2].cpu().numpy()
                origin = env.env.unwrapped.scene.env_origins[0, :2].cpu().numpy()
                if _material_at(layout, *(xy-origin)) == 'sand':
                    force[0, 0, :2] = -args.target_terrain_sand_linear_drag * velocity[0, :2]
                robot.write_external_wrench_to_sim(force, torch.zeros_like(force), body_ids=[torso])
                action = session.proposal()
                previous_cost = session.cost_total
                success, ended = session.step(action, action, False, 0, None)
                ticks += 1
                if not ended:
                    position = _robot_positions_xy(env)[0].copy()
                    trajectory.append(position.tolist())
                    minimum = min(minimum, float(np.linalg.norm(position-goal)))
                if session.cost_total > previous_cost:
                    cost_points.append(trajectory[-1])
                if success or ended:
                    break
            manager = env.env.unwrapped.termination_manager
            fall = bool(manager.get_term('fell_over')[0]) if ended else False
            result = dict(controller=label, seed=seed, success=success, fall=fall,
                          timeout=not success and not fall, seconds=ticks*0.05,
                          cost=session.cost_total, decisions=session.steps,
                          start=start.tolist(), goal=goal.tolist(), min_goal_distance=minimum,
                          final_goal_distance=float(np.linalg.norm(np.asarray(trajectory[-1])-goal)),
                          trajectory=trajectory, cost_points=cost_points)
            results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k not in ['trajectory','cost_points']}), flush=True)
            (out / 'results.json').write_text(json.dumps(dict(args=vars(args), layout=layout.to_dict(),
                manifest=manifests, episodes=results), indent=2))
            session.close()
    env.close()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for ax, row in zip(axes.flat, results):
        xy = np.asarray(row['trajectory'])
        ax.plot(xy[:,0], xy[:,1], linewidth=1)
        ax.scatter(*row['start'], c='black', marker='o', label='start')
        ax.scatter(*row['goal'], c='green', marker='*', s=90, label='goal')
        if row['cost_points']:
            points = np.asarray(row['cost_points'])
            ax.scatter(points[:,0], points[:,1], c='red', marker='x')
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title(f"{row['controller']} / seed {row['seed']}\nsuccess={row['success']} fall={row['fall']}")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / 'trajectories.png', dpi=130)
    metrics = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    for ax, key in zip(axes, ['intervention_fraction','batch_teacher_fraction','actor_loss_bc']):
        rows = [r for r in metrics if key in r]
        ax.plot([r['step'] for r in rows], [r[key] for r in rows])
        ax.set_title(key)
        ax.set_xlabel('5 Hz navigation decisions')
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / 'learning.png', dpi=130)


if __name__ == '__main__':
    main()
