"""Decode-check camera videos and summarize the saved camera-pose audit."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def decode_check(path):
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        frames += 1
    expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert frames == expected and frames > 0, (path, frames, expected)
    return dict(path=str(path), decoded_frames=frames, fps=fps, duration_s=frames/fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((args.directory / 'audit.json').read_text())
    records = audit['records']
    pitch = np.array([r['pitch_down_deg'] for r in records])
    roll = np.array([r['roll_deg'] for r in records])
    same_episode = np.diff([r['episode'] for r in records]) == 0
    rotation = np.array([r['camera_rotation'] for r in records])
    relative = np.einsum('tji,tjk->tik', rotation[:-1], rotation[1:])
    angles = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1, 1)))
    fovy = audit['camera_config']['fovy']
    height = np.array([r['camera_position'][2] for r in records])
    result = dict(videos=[decode_check(p) for p in sorted(args.directory.glob('*.mp4'))],
        frames=len(records), episodes=len(set(r['episode'] for r in records)),
        pitch_degrees=dict(min=float(pitch.min()), median=float(np.median(pitch)), max=float(pitch.max())),
        roll_degrees=dict(min=float(roll.min()), median=float(np.median(roll)), max=float(roll.max())),
        camera_rotation_per_200ms_degrees=dict(median=float(np.median(angles[same_episode])),
            p95=float(np.percentile(angles[same_episode], 95))),
        goal_center_in_frustum_fraction=float(np.mean([r['goal_center_in_fov'] for r in records])),
        camera_world_height_median=float(np.median(height)),
        approximate_flat_ground_nearest_visible_m=float(np.median(height)/np.tan(np.radians(np.median(pitch)+fovy/2))),
        caveats='Frustum excludes occlusion. Ground estimate assumes flat z=0 and ignores robot occlusion. Rotation includes intentional turns, not just shake. Short evaluation clip, not a performance benchmark.')
    data = np.load(args.directory / 'camera_observations.npz')
    for start in (0, 25, 75):
        sheet = np.zeros((3*180, 4*240, 3), dtype=np.uint8)
        for j, i in enumerate(range(start, min(start+12, len(records)))):
            tile = cv2.resize(data['rgb'][i], (240, 180), interpolation=cv2.INTER_NEAREST)
            cv2.putText(tile, f'{i/5:.1f}s ep{records[i]["episode"]}', (4, 16), 0, .45, (255,60,60), 1)
            sheet[j//4*180:(j//4+1)*180, j%4*240:(j%4+1)*240] = tile
        cv2.imwrite(str(args.directory / f'consecutive_{start:03d}.png'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    (args.directory / 'inspection.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
