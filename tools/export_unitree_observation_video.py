"""Export immutable recorded simulator-camera observations at decision cadence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.dataset / 'manifest.json').read_text())
    hz = manifest['metadata']['transition_hz']
    profile = manifest['metadata']['args'].get('navigation_image_profile', 'legacy')
    frame_key = 'policy_image_mono_rgb' if profile in ('stabilized_goal_v1', 'stabilized_goal_v2', 'stabilized_goal_v3') else 'vision_rgb'
    frames, observations, episodes, sources, hashes = [], [], [], [], []
    for part in manifest['parts']:
        path = args.dataset / part['file']
        hashes.append(dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        with np.load(path) as data:
            pixels = data[frame_key].copy()
            if frame_key.startswith('policy_image_'):
                pixels = (np.clip(pixels, 0, 1)*255).round().astype(np.uint8)
            frames.extend(pixels)
            observations.extend(data['observations'][:, :9].copy())
            episodes.extend(data['episode_index'].tolist())
            sources.extend(data['intervened'].tolist())
    if not frames:
        raise RuntimeError('No completed RGB chunks available')
    def writer(path, width, height):
        return subprocess.Popen(['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}', '-r', str(hz), '-i', '-', '-an', '-c:v', 'libx264',
            '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
    video_height = int(round(640*frames[0].shape[0]/frames[0].shape[1]/2)*2)
    raw = writer(args.output / 'recorded_policy_rgb.mp4', 640, video_height)
    annotated = writer(args.output / 'recorded_policy_rgb_with_goal_compass.mp4', 960, video_height+60)
    scale = manifest['metadata']['args']['goal_distance_scale']
    sheet = np.zeros((3 * 210, 4 * 280, 3), dtype=np.uint8)
    selected = np.linspace(0, len(frames)-1, 12, dtype=int).tolist()
    for i, (frame, obs) in enumerate(zip(frames, observations)):
        enlarged = cv2.resize(frame, (640, video_height), interpolation=cv2.INTER_NEAREST)
        raw.stdin.write(enlarged.tobytes())
        canvas = np.full((video_height+60, 960, 3), 24, dtype=np.uint8)
        canvas[40:40+video_height, :640] = enlarged
        cv2.putText(canvas, 'SAVED ENCODER INPUT | '+profile, (12, 26), 0, .6, (240,240,240), 1)
        for j, line in enumerate([f'frame {i} | sim {i/hz:.1f}s', f'episode {episodes[i]}',
                'HUMAN' if sources[i] else 'POLICY', f'goal {float(obs[6])*scale:.2f}m',
                'Goal compass: display only', 'Playback: 5 Hz observations']):
            cv2.putText(canvas, line, (660, 50+j*28), 0, .55, (230,230,230), 1)
        cv2.circle(canvas, (800,360), 95, (90,90,90), 1)
        cv2.arrowedLine(canvas, (800,360), (int(800-obs[8]*80), int(360-obs[7]*80)), (50,230,90), 4)
        annotated.stdin.write(canvas.tobytes())
        if i in selected:
            k = selected.index(i)
            thumbnail = cv2.resize(frame, (280, 210), interpolation=cv2.INTER_NEAREST)
            cv2.putText(thumbnail, f'{i/hz:.1f}s ep{episodes[i]}', (4,16), 0, .45, (255,60,60), 1)
            sheet[(k//4)*210:(k//4+1)*210, (k%4)*280:(k%4+1)*280] = thumbnail
    for process in (raw, annotated):
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('ffmpeg failed')
    cv2.imwrite(str(args.output / 'contact_sheet.png'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    result = dict(dataset=str(args.dataset), frames=len(frames), fps=hz,
        duration_sim_s=len(frames)/hz, original_resolution=list(frames[0].shape[:2][::-1]),
        temporal_interpolation=False, source_frame_key=frame_key, image_profile=profile,
        policy_preprocessing_applied_during_export=False, source_parts=hashes,
        unique_episodes=len(set(episodes)), sheet_indices=selected,
        note='All rows in completed manifest chunks at export time; episode resets retained; paused wall time omitted.')
    (args.output / 'export.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
