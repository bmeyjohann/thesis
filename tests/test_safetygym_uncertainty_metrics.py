import numpy as np

from safetygym_utils.train import _linear_slope, _summary_stats


def test_summary_stats_empty():
    out = _summary_stats([], "train/uncertainty_all")
    assert out["train/uncertainty_all_mean"] == 0.0
    assert out["train/uncertainty_all_min"] == 0.0
    assert out["train/uncertainty_all_max"] == 0.0
    assert out["train/uncertainty_all_p95"] == 0.0
    assert out["train/uncertainty_all_count"] == 0.0


def test_summary_stats_values():
    vals = [0.1, 0.2, 0.3, 0.4]
    out = _summary_stats(vals, "train/uncertainty_all")
    assert np.isclose(out["train/uncertainty_all_mean"], 0.25)
    assert np.isclose(out["train/uncertainty_all_min"], 0.1)
    assert np.isclose(out["train/uncertainty_all_max"], 0.4)
    assert out["train/uncertainty_all_count"] == 4.0
    assert out["train/uncertainty_all_p95"] >= 0.3


def test_linear_slope_direction():
    assert _linear_slope([1.0, 2.0, 3.0, 4.0]) > 0.0
    assert _linear_slope([4.0, 3.0, 2.0, 1.0]) < 0.0
    assert np.isclose(_linear_slope([2.0, 2.0, 2.0]), 0.0)
