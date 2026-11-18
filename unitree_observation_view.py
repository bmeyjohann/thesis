"""Cached policy inputs and a display-only HTTP interface; no simulator access."""
import base64
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image


class ObservationFrames:
    def __init__(self, lags):
        self.lags = list(lags or [0])
        self.history = deque(maxlen=max(self.lags) + 1)
        self.version = 0

    def capture(self, frames, *, reset=False):
        encoded = []
        for label, pixels in frames.items():
            pixels = np.asarray(pixels)
            if pixels.dtype != np.uint8:
                pixels = (255 * np.clip(pixels, 0, 1)).round().astype(np.uint8)
            buffer = BytesIO()
            Image.fromarray(pixels).save(buffer, format='PNG')
            encoded.append(dict(label=label, width=pixels.shape[1], height=pixels.shape[0],
                                image='data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')))
        if reset or not self.history:
            self.history.clear()
            self.history.extend([encoded] * self.history.maxlen)
        else:
            self.history.appendleft(encoded)
        self.version += 1

    def selected(self, hz):
        return [dict(age_s=lag / hz, frames=self.history[lag]) for lag in self.lags]


def observation_payload(session):
    """Read only the actor's prepared tensor and cached encoder input frames."""
    obs = session.obs[0].detach().cpu().numpy()
    args = session.args
    if args.goal_encoding == 'distance_bearing':
        distance = float(obs[6]) * args.goal_distance_scale
        forward, left = float(obs[7]), float(obs[8])
        clipped = bool(obs[6] >= 1)
    else:
        distance = float(np.linalg.norm(obs[6:8]))
        forward, left = (obs[6:8] / max(distance, 1e-8)).tolist()
        clipped = False
    scan = None
    if session.vision.include_scan:
        rows = round(args.height_scan_lateral_size / args.height_scan_resolution) + 1
        cols = round(args.height_scan_forward_size / args.height_scan_resolution) + 1
        scan = dict(rows=rows, cols=cols, values=obs[9:9 + rows * cols].tolist())
    return dict(version=session.observation_frames.version,
                sensor=args.navigation_observation_mode,
                image_profile=getattr(args, 'navigation_image_profile', 'legacy'),
                history=session.observation_frames.selected(args.navigation_policy_hz),
                context=obs[:9].tolist(), scan=scan,
                goal=dict(distance=distance, forward=forward, left=left, clipped=clipped),
                decision_hz=args.navigation_policy_hz)


class ObservationServer:
    """HTTP threads serve immutable snapshots and enqueue controls only."""
    def __init__(self, host, port, command):
        self.snapshot = b'{}'
        self.command = command
        page = (Path(__file__).parent / 'web' / 'unitree_observation.html').read_bytes()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, data, content_type):
                self.send_response(code)
                self.send_header('Content-Type', content_type)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == '/':
                    self.reply(200, page, 'text/html; charset=utf-8')
                elif self.path == '/state':
                    self.reply(200, owner.snapshot, 'application/json')
                else:
                    self.reply(404, b'not found', 'text/plain')

            def do_POST(self):
                # Same-origin JSON only: another website must not steer the robot.
                origin = self.headers.get('Origin')
                if (self.path != '/command' or self.headers.get('Content-Type') != 'application/json'
                        or (origin and origin != 'http://' + self.headers.get('Host', ''))):
                    self.reply(403, b'forbidden', 'text/plain')
                    return
                try:
                    length = int(self.headers.get('Content-Length', 0))
                    if not 0 < length <= 1024:
                        raise ValueError('Invalid request size')
                    payload = json.loads(self.rfile.read(length))
                    action = payload.get('action')
                    if action not in ('pause', 'step', 'reset', 'next', 'stop', 'speed', 'keyboard', 'release'):
                        raise ValueError('Unknown control')
                    if action == 'speed' and payload.get('value') not in (0.5, 1, 1.5, 2):
                        raise ValueError('Invalid speed')
                    if action == 'keyboard' and payload.get('key') not in ('w', 's', 'a', 'd', 'q', 'e'):
                        raise ValueError('Invalid key')
                    owner.command(payload)
                except (ValueError, TypeError, AttributeError):
                    self.reply(400, b'invalid command', 'text/plain')
                    return
                self.reply(200, b'{}', 'application/json')

        self.http = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)

    def start(self):
        self.thread.start()

    def publish(self, payload):
        self.snapshot = json.dumps(payload, allow_nan=False).encode('utf-8')

    def stop(self):
        if self.thread.is_alive():
            self.http.shutdown()
            self.thread.join(timeout=3)
        self.http.server_close()
