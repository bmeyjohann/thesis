import pytest

from unitree_nav_eval_manifest import ManifestEpisodeCursor


def test_case_order_is_independent_of_other_environment_resets():
    cursor = ManifestEpisodeCursor(2, 6)
    assert cursor.next([0, 1]) == [0, 1]
    assert cursor.next([1]) == [3]
    assert cursor.next([1]) == [5]
    assert cursor.next([0]) == [2]
    assert cursor.next([0]) == [4]
    assert cursor.next([1, 0]) == [1, 0]


def test_invalid_case_counts_and_duplicate_reset_ids_are_rejected():
    with pytest.raises(ValueError):
        ManifestEpisodeCursor(2, 3)
    cursor = ManifestEpisodeCursor(2, 4)
    with pytest.raises(ValueError):
        cursor.next([0, 0])
