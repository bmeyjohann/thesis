import base64
from io import BytesIO
import json
from types import SimpleNamespace
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
from PIL import Image
import pytest
import torch

from unitree_observation_view import ObservationFrames, ObservationServer, observation_payload


def test_exact_history_padding_reset_and_original_pixels():
    history = ObservationFrames([0, 1, 25])
    for i in range(30):
        history.capture({'RGB': np.full((3, 4, 3), i, dtype=np.uint8)})
    selected = history.selected(5)
    assert [s['age_s'] for s in selected] == [0, .2, 5]
    for selected_frame, value in zip(selected, [29, 28, 4]):
        png = base64.b64decode(selected_frame['frames'][0]['image'].split(',')[1])
        assert (np.asarray(Image.open(BytesIO(png))) == value).all()
    history.capture({'RGB': np.zeros((3, 4, 3), dtype=np.uint8)}, reset=True)
    assert all(s['frames'] == history.selected(5)[0]['frames'] for s in history.selected(5))


@pytest.mark.parametrize('scan', [True, False])
def test_payload_only_uses_actor_observations(scan):
    args = SimpleNamespace(goal_encoding='distance_bearing', goal_distance_scale=14.,
        navigation_policy_hz=5, navigation_observation_mode='rgb_height' if scan else 'php_depth',
        height_scan_lateral_size=1., height_scan_forward_size=1.6, height_scan_resolution=.1)
    obs = torch.zeros((1, 762 if scan else 105))
    obs[0, 6:9] = torch.tensor([.5, 0., 1.])
    frames = ObservationFrames([0, 1, 25])
    frames.capture({'RGB': np.zeros((3, 4, 3), dtype=np.uint8)})
    session = SimpleNamespace(args=args, obs=obs, vision=SimpleNamespace(include_scan=scan), observation_frames=frames)
    payload = observation_payload(session)
    assert payload['goal'] == dict(distance=7., forward=0., left=1., clipped=False)
    assert (payload['scan'] is not None) == scan
    if scan:
        assert (payload['scan']['rows'], payload['scan']['cols']) == (11, 17)
        assert len(payload['scan']['values']) == 187
    assert not {'world_position', 'terrain', 'obstacles', 'cost', 'path'} & payload.keys()


def test_http_snapshot_and_control_validation():
    commands = []
    server = ObservationServer('127.0.0.1', 0, commands.append)
    server.publish({'version': 7})
    server.start()
    base = f'http://127.0.0.1:{server.http.server_port}'
    try:
        assert json.load(urlopen(base + '/state')) == {'version': 7}
        assert b'Policy view' in urlopen(base).read()
        def post(value, origin=base):
            return urlopen(Request(base + '/command', json.dumps(value).encode(),
                headers={'Content-Type': 'application/json', 'Origin': origin}))
        post({'action': 'speed', 'value': 1.5}).close()
        assert commands == [{'action': 'speed', 'value': 1.5}]
        with pytest.raises(HTTPError) as e:
            post({'action': 'speed', 'value': 9})
        assert e.value.code == 400
        with pytest.raises(HTTPError) as e:
            post({'action': 'pause'}, 'http://untrusted.example')
        assert e.value.code == 403
        assert len(commands) == 1
    finally:
        server.stop()
