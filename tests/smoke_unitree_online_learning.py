"""Bounded viewer/learning checks for both sensors using synthetic takeover input."""
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eval_interactive_unitree_nav as interactive
from unitree_viser_training import BrowserTrainingSession


def main():
    sys.argv = ["online-learning-smoke"]
    base = interactive.parse_args()
    base_args = ROOT / "models/unitree_mjlab_nav_human/target_human_rgb_height_20260925_175559/args.json"
    vars(base).update(json.loads(base_args.read_text()))
    out = ROOT / "local" / f"unitree_alpha_alignment_smoke_{time.time_ns()}"
    results = []
    for index, sensor in enumerate(("rgb_height", "php_depth")):
        args = copy.deepcopy(base)
        args.model_path = ""
        args.navigation_observation_mode = sensor
        if sensor == "php_depth":
            args.navigation_encoder_checkpoint = str(ROOT / "artifacts/unitree_depth_transfer_20261013/models/php_student/depth_backbone.onnx")
        args.online_train = True
        args.online_output_dir = str(out)
        args.online_run_name = sensor
        args.online_total_steps = 12
        args.online_learning_starts = 2
        args.online_batch_size = 2
        args.online_n_step = 1
        args.online_policy_frequency = 1
        args.online_log_interval = 1
        args.online_checkpoint_interval = 6
        args.online_wandb_mode = "disabled"
        args.online_objective_mode = "pref_rl"
        args.online_pref_lambda_init = 1.
        args.online_pref_lambda_lr = 0.
        args.online_pref_lambda_max = 10.
        args.human_dataset_dir = ""
        args.human_input_device = "gamepad"
        args.viewer_backend = "viser"
        args.viser_host = "127.0.0.1"
        args.viser_port = 8772 + index
        args.start_paused = False
        args.autopilot = True
        args.show_rgb = False
        args.controller = "policy"
        current = {}
        original = BrowserTrainingSession.__init__

        def initialize(session, *values):
            original(session, *values)
            current["session"] = session

        def sample(_):
            active = current["session"].steps < 4
            return SimpleNamespace(action=np.array([.2, .1, 0.], dtype=np.float32),
                intervening=active, connected=True, stale=False, command_norm=.224,
                state_age_s=0., state={"buttons": {}}, transport={"connected": True})

        with patch.object(BrowserTrainingSession, "__init__", initialize), \
             patch.object(interactive.UnitreeGamepadController, "sample", sample):
            interactive.run(args)
        run = out / args.online_run_name
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        saved = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
        summary = json.loads((run / "browser_summary.json").read_text())
        assert summary["navigation_decisions"] == 12 and summary["environment_ticks"] == 48
        assert summary["intervention_decisions"] == 4
        assert any(row.get("alpha_updates") == 0 and row.get("alpha_rows") == 0 for row in metrics)
        assert any(row.get("alpha_updates") == 1 for row in metrics)
        assert all(row["pref_sampling_mode"] == "separate" for row in metrics)
        assert all(row["alpha_state_scope"] == "student" for row in metrics)
        assert any(row.get("pref_source_rows") == 256 for row in metrics)
        assert all(row["pref_lambda"] == 1 for row in metrics)
        assert abs(saved["log_alpha"].exp().item() - .01) > 1e-8
        assert saved["target_entropy"] == -3
        assert saved["args"]["online_alpha_min"] == 1e-5
        assert saved["args"]["online_alpha_max"] == .1
        expected_dim = 275 if sensor == "rgb_height" else 56
        assert saved["obs_dim"] == expected_dim
        args.online_train = False
        args.human_input_device = "none"
        args.model_path = str(run / "final.pt")
        args.navigation_encoder_checkpoint = "/absent/donor.onnx"
        env = interactive.make_env(args, num_envs=1, render=False)
        try:
            raw, _, _, _, cells = interactive._reset_until_feasible(args, env)
            session = BrowserTrainingSession(args, env, raw, cells)
            assert session.obs.shape == (1, expected_dim) and torch.isfinite(session.proposal()).all()
            assert all(torch.equal(v, session.actor.state_dict()[k].cpu())
                       for k, v in saved["actor_state_dict"].items())
            session.close()
        finally:
            env.close()
        results.append(dict(sensor=sensor, status="PASS", decisions=12, ticks=48,
            synthetic_interventions=4, updates=metrics[-1]["updates"], obs_dim=expected_dim,
            final_alpha=saved["log_alpha"].exp().item(), pref_sampling="separate",
            alpha_scope="student", donor_free_policy_reload=True, run=str(run)))
    result = dict(results=results, human_hardware_tested=False)
    (out / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
