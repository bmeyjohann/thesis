import os
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = PROJECT_ROOT / "eval_interactive.py"


STATE_MODEL = PROJECT_ROOT / "models" / "fast_sac" / "L_pref_td_12520510" / "pointmaze_arena_danger_wall_v0_step800000.pt"
STATE_ENV = "pointmaze-arena-danger-wall-v0"

PIXEL_MODEL = PROJECT_ROOT / "models" / "fast_sac" / "P_pixels_mlp3_12576622" / "visual_pointmaze_arena_danger_wall_v0_step800000.pt"
PIXEL_ENV = "visual-pointmaze-arena-danger-wall-v0"


def _run_eval(model_path: Path, env_name: str, extra_args: list[str]) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--model_path",
        str(model_path),
        "--env_name",
        env_name,
        "--device",
        "cpu",
        "--num_episodes",
        "1",
        "--max_episode_steps",
        "16",
        "--fps",
        "5",
        "--headless",
        "--render_mode",
        "rgb_array",
    ] + extra_args
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class EvalSmokeTests(unittest.TestCase):
    def test_state_headless_eval(self) -> None:
        _run_eval(STATE_MODEL, STATE_ENV, [])

    def test_pixel_headless_eval(self) -> None:
        extra = [
            "--pixel_width",
            "64",
            "--pixel_height",
            "64",
        ]
        _run_eval(PIXEL_MODEL, PIXEL_ENV, extra)


if __name__ == "__main__":
    unittest.main()
