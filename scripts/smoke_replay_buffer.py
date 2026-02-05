import os
import sys
from pathlib import Path

import numpy as np
from dm_env import specs

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drqv2.replay_buffer import ReplayBufferStorage, ReplayBuffer  # noqa: E402


def main() -> None:
    tmp = Path("/tmp/replay_smoke")
    tmp.mkdir(parents=True, exist_ok=True)

    data_specs = [
        specs.Array((4,), np.float32, "observation"),
        specs.Array((2,), np.float32, "action"),
        specs.Array((1,), np.float32, "reward"),
        specs.Array((1,), np.float32, "discount"),
    ]

    store = ReplayBufferStorage(data_specs, tmp)

    class TS(dict):
        def __init__(self, payload, is_last):
            super().__init__(payload)
            self._is_last = is_last

        def last(self):
            return self._is_last

    for t in range(2):
        payload = {
            "observation": np.zeros((4,), np.float32),
            "action": np.zeros((2,), np.float32),
            "reward": np.zeros((1,), np.float32),
            "discount": np.ones((1,), np.float32),
        }
        store.add(TS(payload, is_last=(t == 1)))

    buf = ReplayBuffer(
        tmp,
        max_size=100,
        num_workers=1,
        nstep=1,
        discount=0.99,
        fetch_every=1,
        save_snapshot=True,
    )

    it = iter(buf)
    sample = next(it)
    print("Replay sample ok. Tuple length:", len(sample))


if __name__ == "__main__":
    main()
