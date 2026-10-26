"""Early terminal outcomes must come from measured goals, not elapsed time alone."""
from importlib import import_module
from pathlib import Path

import numpy as np


_TABLE_Z = .68
_GOOD = ((.5, .03, _TABLE_Z + .02), (.5, -.03, _TABLE_Z + .02),
         (.5, 0, _TABLE_Z + .06))
_BAD = (_GOOD[0], _GOOD[1], (.65, 0, _TABLE_Z + .06))
_FAR_HANDS = np.array(((0., 0., 2.), (0., 0., 2.)))
_NEAR_HANDS = np.array((_GOOD[0], _GOOD[0]))


def _scene():
    names = [f"letter_{i}" for i in range(3)]
    return {"_task": "pyramid", "_frame_dt_s": 1 / 30,
            "_hand_reference": "wrist_link", "_hand_reach_m": .25,
            "pose_node_ids": [f"body/{name}" for name in names],
            "sampled_parameters": {
                "scene_manifest.objects": [
                    {"class_name": "letter_block", "name": name, "size": [.04] * 3}
                    for name in names],
                "scene_manifest.table": {"top_z_m": _TABLE_Z,
                                         "x_range_m": [.2, .9], "y_range_m": [-.55, .55]},
                "scene_manifest.sampled_values": {"appearance": {"objects": {
                    name: {"letter": letter} for name, letter in zip(names, "ABC")}}}}}


def _pose(positions):
    result = np.zeros((3, 7), dtype=np.float64)
    result[:, :3] = positions
    result[:, 3] = 1
    return result


def _tracker(limit=70):
    return import_module("rollout_termination").TaskTermination(_scene(), Path("/unused"), limit)


def test_goal_requires_full_second_and_freezes_first_success():
    tracker = _tracker()
    for tick in range(30):
        assert tracker.update(tick, _pose(_GOOD), _FAR_HANDS) is None
    result = tracker.update(30, _pose(_GOOD), _FAR_HANDS)
    assert (result["status"], result["tick"], result["inference_count"]) == ("success", 30, 4)
    assert result["score"]["reason"] == "pyramid_stacked"
    assert tracker.update(31, _pose(_BAD), _NEAR_HANDS) == result


def test_wrong_goal_can_recover_and_finish_on_non_chunk_boundary():
    tracker = _tracker()
    for tick in range(31):
        assert tracker.update(tick, _pose(_BAD), _FAR_HANDS) is None
    for tick in range(31, 61):
        assert tracker.update(tick, _pose(_GOOD), _FAR_HANDS) is None
    result = tracker.update(61, _pose(_GOOD), _FAR_HANDS)
    assert (result["status"], result["tick"], result["inference_count"]) == ("success", 61, 8)


def test_deadline_distinguishes_failure_from_missing_release_evidence():
    wrong = _tracker(limit=35)
    unclear = _tracker(limit=35)
    for tick in range(35):
        assert wrong.update(tick, _pose(_BAD), _FAR_HANDS) is None
        assert unclear.update(tick, _pose(_GOOD), _NEAR_HANDS) is None
    failed = wrong.update(35, _pose(_BAD), _FAR_HANDS)
    unknown = unclear.update(35, _pose(_GOOD), _NEAR_HANDS)
    assert (failed["status"], failed["termination_reason"], failed["inference_count"]) == (
        "failure", "timeout_goal_unmet", 5)
    assert failed["score"]["status"] == "failure"
    assert unknown["status"] == "unsupported"
    assert unknown["score"]["reason"] == "hand_release_not_proven"
    assert unknown["tick"] == 35


def test_finished_slot_does_not_cut_off_another_slot():
    early, late = _tracker(limit=40), _tracker(limit=40)
    first = None
    for tick in range(41):
        first = early.update(tick, _pose(_GOOD), _FAR_HANDS) or first
        other = late.update(tick, _pose(_BAD), _FAR_HANDS)
        if tick < 40:
            assert other is None
    assert first["status"] == "success" and first["tick"] == 30
    assert other["status"] == "failure" and other["tick"] == 40
