import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = PROJECT_ROOT / "logs" / "fast_sac"
MODEL_ROOT = PROJECT_ROOT / "models" / "fast_sac"


def _run_training(exp_name: str, extra_args: list[str]) -> None:
    """Execute the training script with a tiny configuration."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "train_fast_sac_ogbench.py"),
        # "--device",
        # "cpu",
        "--total_timesteps",
        "64",
        "--num_envs",
        "1",
        "--batch_size",
        "64",
        "--learning_starts",
        "16",
        "--save_interval",
        "0",
        "--log_interval",
        "128",
        "--exp_name",
        exp_name,
    ] + extra_args
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class TrainSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dirs: list[Path] = []

    def tearDown(self) -> None:
        for path in self._temp_dirs:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)

    def _cleanup_run(self, exp_name: str) -> None:
        log_dir = LOG_ROOT / exp_name
        model_dir = MODEL_ROOT / exp_name
        shutil.rmtree(log_dir, ignore_errors=True)
        shutil.rmtree(model_dir, ignore_errors=True)

    def test_state_observation_smoke(self) -> None:
        exp_name = "smoke_state_test"
        try:
            _run_training(
                exp_name,
                [
                    "--env_name",
                    "pointmaze-arena-danger-lethal-v0",
                    "--reward_type",
                    "sparse",
                    "--use_intervention",
                    "--intervention_mode",
                    "agent",
                    "--intervention_enable_after_steps",
                    "0",
                ],
            )
            self.assertTrue((LOG_ROOT / exp_name).exists())
            self.assertTrue((MODEL_ROOT / exp_name).exists())
        finally:
            self._cleanup_run(exp_name)

    def test_pixel_observation_smoke(self) -> None:
        exp_name = "smoke_pixel_test"
        try:
            _run_training(
                exp_name,
                [
                    "--env_name",
                    "visual-pointmaze-arena-v0",
                    "--obs_mode",
                    "pixels",
                    "--pixel_conv_channels",
                    "16,32",
                    "--pixel_kernel_sizes",
                    "5,3",
                    "--pixel_strides",
                    "2,1",
                    "--reward_type",
                    "dense",
                ],
            )
            self.assertTrue((LOG_ROOT / exp_name).exists())
            self.assertTrue((MODEL_ROOT / exp_name).exists())
        finally:
            self._cleanup_run(exp_name)


if __name__ == "__main__":
    unittest.main()
