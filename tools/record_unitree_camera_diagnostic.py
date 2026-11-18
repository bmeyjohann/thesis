"""Finite evaluation-only camera audit; never modifies policy images or checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--decisions', type=int, default=150)
    options = parser.parse_args()
    if options.decisions <= 0:
        parser.error('decisions must be positive')
    options.output.mkdir(parents=True, exist_ok=False)
    import eval_interactive_unitree_nav as interactive
    from unitree_viser_training import BrowserTrainingSession
    from types import SimpleNamespace
    sys.argv = [sys.argv[0]]
    defaults = vars(interactive.parse_args())
    saved = json.loads(options.checkpoint.with_name('args.json').read_text())
    if saved.get('navigation_image_profile', 'legacy') != 'legacy':
        raise ValueError('This same-pose renderer comparison is for legacy raw-camera inputs only. '
                         'Use export_unitree_observation_video.py on the saved dataset for transformed policy images.')
    args = SimpleNamespace(**{**defaults, **saved})
    args.model_path = str(options.checkpoint.resolve())
    args.online_train = False
    args.human_dataset_dir = ''
    args.human_input_device = 'none'
    args.controller = 'policy'
    args.observation_only = True
    args.online_session_receipt = ''
    env = session = renderer = None
    writers = []
    frames, records, inputs = [], [], []
    try:
        env = interactive.make_env(args, num_envs=1, render=False)
        raw, _, _, _, cells = interactive._reset_until_feasible(args, env)
        session = BrowserTrainingSession(args, env, raw, cells)
        assert session.learner is None and session.writer is None
        sim = env.env.unwrapped.sim
        model = sim.mj_model
        data = mujoco.MjData(model)
        camera_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                        for i in range(model.ncam)]
        matches = [i for i, name in enumerate(camera_names)
                   if args.navigation_camera_names[0] in (name or '')]
        if len(matches) != 1:
            raise RuntimeError(f'Ambiguous camera: {camera_names}, {args.navigation_camera_names}')
        ci = matches[0]
        renderer = mujoco.Renderer(model, height=480, width=640)
        def writer(name, width, height):
            process = subprocess.Popen(['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '5', '-i', '-',
                '-an', '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(options.output / name)], stdin=subprocess.PIPE)
            writers.append(process)
            return process
        raw_writer = writer('policy_rgb.mp4', 640, 480)
        paired_writer = writer('same_pose_goal_cylinder.mp4', 1280, 540)
        selected = np.linspace(0, options.decisions - 1, 8, dtype=int).tolist()
        sheet = np.zeros((4 * 270, 640, 3), dtype=np.uint8)
        for i in range(options.decisions):
            frame = session.vision.last_frames['vision_rgb'].copy()
            frames.append(frame)
            inputs.append(session.obs[0, :9].cpu().numpy().copy())
            for name in ('qpos', 'qvel', 'mocap_pos', 'mocap_quat'):
                if name.startswith('mocap') and not model.nmocap:
                    continue
                getattr(data, name)[:] = getattr(sim.data, name)[0].cpu().numpy()
            mujoco.mj_forward(model, data)
            goal = np.asarray(interactive._command_goal_w(env), dtype=float).reshape(-1)
            goal_xyz = np.array([goal[0], goal[1], .75])
            camera_pos = data.cam_xpos[ci].copy()
            rotation = data.cam_xmat[ci].reshape(3, 3).copy()
            direction = -rotation[:, 2]
            pitch = np.degrees(np.arcsin(np.clip(-direction[2], -1, 1)))
            roll = np.degrees(np.arctan2(rotation[2, 0], rotation[2, 1]))
            local = rotation.T @ (goal_xyz - camera_pos)
            focal = 240 / np.tan(np.radians(model.cam_fovy[ci] / 2))
            uv = [320 + focal * local[0] / max(-local[2], 1e-9),
                  240 - focal * local[1] / max(-local[2], 1e-9)]
            in_fov = bool(local[2] < 0 and 0 <= uv[0] < 640 and 0 <= uv[1] < 480)
            renderer.update_scene(data, camera=ci)
            scene = renderer.scene
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                np.array([args.success_dist, .75, 0.]), goal_xyz, np.eye(3).reshape(-1),
                np.array([.1, 1., .2, .75], dtype=np.float32))
            scene.ngeom += 1
            diagnostic = renderer.render().copy()
            actual = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
            canvas = np.full((540, 1280, 3), 22, dtype=np.uint8)
            canvas[40:520, :640] = actual
            canvas[40:520, 640:] = diagnostic
            cv2.putText(canvas, 'ACTUAL POLICY RGB: 80x60, no added goal', (8, 25), 0, .65, (255,255,255), 1)
            cv2.putText(canvas, 'SAME POSE: 640x480 + diagnostic goal cylinder', (648, 25), 0, .6, (255,255,255), 1)
            cv2.putText(canvas, f'{i/5:.1f}s | episode {session.episode} | pitch {pitch:.1f} roll {roll:.1f} deg | goal center in FOV: {in_fov} (occlusion not tested)',
                        (8, 536), 0, .5, (255,255,255), 1)
            raw_writer.stdin.write(actual.tobytes())
            paired_writer.stdin.write(canvas.tobytes())
            if i in selected:
                index = selected.index(i)
                # Each thumbnail preserves both synchronized panels.
                if index < 4:
                    sheet[index*270:(index+1)*270] = cv2.resize(canvas, (640, 270))
                cv2.imwrite(str(options.output / f'pair_{i:03d}.png'), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            records.append(dict(frame=i, episode=session.episode, environment_ticks=session.environment_ticks,
                camera_position=camera_pos.tolist(), camera_rotation=rotation.tolist(), goal=goal.tolist(),
                pitch_down_deg=float(pitch), roll_deg=float(roll), goal_center_in_fov=in_fov,
                goal_center_pixel=uv))
            start = session.steps
            while session.steps == start:
                action = session.proposal()
                session.step(action, action, False, 0, None)
            if i % 25 == 0:
                print(f'CAPTURE {i+1}/{options.decisions}', flush=True)
        cv2.imwrite(str(options.output / 'contact_sheet.png'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        np.savez_compressed(options.output / 'camera_observations.npz', rgb=np.stack(frames), context=np.stack(inputs))
        result = dict(checkpoint=str(options.checkpoint), checkpoint_sha256=hashlib.sha256(options.checkpoint.read_bytes()).hexdigest(),
            camera_config=args.navigation_camera_config, camera_name=camera_names[ci], fps=5,
            frames=len(frames), environment_ticks=session.environment_ticks, decisions=session.steps,
            goals=session.successes, cost=session.cost_total, records=records,
            policy_images_unchanged=True, training=False,
            diagnostic='Native MuJoCo re-render at matched camera pose, different renderer/resolution; goal cylinder display-only, never fed to policy. Not a quality benchmark. Episode resets retained.')
        (options.output / 'audit.json').write_text(json.dumps(result, indent=2))
    finally:
        for process in writers:
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError('Video encoding failed')
        if renderer:
            renderer.close()
        if session:
            session.close()
        if env:
            env.close()


if __name__ == '__main__':
    main()
