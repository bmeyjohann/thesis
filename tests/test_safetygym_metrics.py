from safetygym_utils.metrics import classify_outcome, validate_episode_metrics


def _base_episode():
    return {
        "episode_return": 1.0,
        "episode_cost_sum": 0.0,
        "episode_cost_rate": 0.0,
        "episode_length": 10.0,
        "intervention_steps": 2.0,
        "intervention_fraction": 0.2,
        "intervention_num_bursts": 1.0,
        "intervention_avg_burst_len": 2.0,
        "goal_met": 1.0,
        "final_distance_to_goal": 0.0,
        "outcome_success": 1.0,
        "outcome_timeout": 0.0,
        "outcome_kill": 0.0,
        "outcome_other_failure": 0.0,
        "terminated": 1.0,
        "truncated": 0.0,
    }


def test_classify_outcome_success():
    assert classify_outcome(goal_met=True, episode_steps=10, max_episode_steps=100) == "success"


def test_classify_outcome_timeout():
    assert classify_outcome(goal_met=False, episode_steps=100, max_episode_steps=100) == "timeout"


def test_classify_outcome_kill():
    assert classify_outcome(goal_met=False, episode_steps=42, max_episode_steps=100) == "kill"


def test_validate_episode_metrics_passes_for_complete_schema():
    validate_episode_metrics(_base_episode())


def test_validate_episode_metrics_rejects_missing_key():
    payload = _base_episode()
    payload.pop("outcome_kill")
    try:
        validate_episode_metrics(payload)
    except ValueError:
        return
    raise AssertionError("expected ValueError for missing mandatory key")


def test_validate_episode_metrics_rejects_non_finite():
    payload = _base_episode()
    payload["final_distance_to_goal"] = float("nan")
    try:
        validate_episode_metrics(payload)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-finite mandatory metric")
