"""Synthetic semantic arrangements, not raw demonstration endpoints."""
from pathlib import Path

import numpy as np

from success_blocks import score


SIZE = 0.04
TABLE = 0.68


def fixture(task, letters, instruction=None):
    names = [f"letter_{i}" for i in range(len(letters))]
    objects = [{"class_name": "letter_block", "name": name, "size": [SIZE] * 3}
               for name in names]
    appearance = {name: {"letter": letter} for name, letter in zip(names, letters)}
    scene = {
        "_task": task,
        # Synthetic certified upper bound for this fixture's modeled hands;
        # production data must obtain it from the articulated collision mesh.
        "_hand_reference": "wrist_link",
        "_hand_reach_m": .25,
        "_frame_dt_s": 1 / 30,
        "pose_node_ids": [f"body/{name}" for name in names],
        "sampled_parameters": {
            "scene_manifest.objects": objects,
            "scene_manifest.table": {"top_z_m": TABLE, "x_range_m": [0.2, 0.9], "y_range_m": [-0.55, 0.55]},
            "scene_manifest.sampled_values": {"appearance": {"objects": appearance}},
        },
    }
    if instruction is not None:
        scene["_task_instruction"] = instruction
    return scene


def poses(positions, frames=60, quaternion=(1, 0, 0, 0)):
    positions = np.asarray(positions, dtype=float)
    result = np.zeros((frames, len(positions), 7), dtype=float)
    result[..., :3] = positions
    result[..., 3:] = quaternion
    return result


def hands(frames=60, at=(0, 0, 2)):
    return np.tile(np.asarray(at, dtype=float), (frames, 2, 1))


def check(scene, positions, *, motion=None, near_hand=False):
    history = poses(positions)
    if motion is not None:
        history[-31:, 0, 0] += np.linspace(0, motion, 31)
    return score(scene, history, hands(at=positions[0] if near_hand else (0, 0, 2)), Path("/unused"))


def test_pyramid_supported_stable_and_not_toppled():
    scene = fixture("pyramid", "ABC")
    good = [(0.5, 0.03, TABLE + .02), (0.5, -.03, TABLE + .02),
            (0.5, 0, TABLE + .06)]
    assert check(scene, good)["status"] == "success"
    assert check(scene, good[:2] + [(0.65, 0, TABLE + .06)])["status"] == "failure"
    assert check(scene, good[:2] + [(0.5, 0, TABLE + .12)])["status"] == "failure"
    assert check(scene, good, motion=.02)["status"] == "failure"
    assert check(scene, good, near_hand=True)["status"] == "unsupported"


def test_spelling_order_and_table_support_are_independent_of_cube_letter_face():
    scene = fixture("spelling", "ECAF", "用字母积木拼出 CAFE。")
    correct = [(0.5, -.06, TABLE + .02), (0.5, .06, TABLE + .02),
               (0.5, .02, TABLE + .02), (0.5, -.02, TABLE + .02)]
    assert check(scene, correct)["status"] == "success"
    assert check(scene, [correct[1], correct[0], *correct[2:]])["status"] == "failure"
    assert check(scene, correct[:3] + [(0.5, -.02, TABLE + .08)])["status"] == "failure"
    assert check(scene, correct, motion=.02)["status"] == "failure"
    assert check(fixture("spelling", "ECAF"), correct)["status"] == "unsupported"
    face_down = poses(correct, quaternion=(0, 1, 0, 0))
    assert score(scene, face_down, hands(), Path("/unused"))["status"] == "success"
    side_face = poses(correct, quaternion=(np.sqrt(.5), np.sqrt(.5), 0, 0))
    assert score(scene, side_face, hands(), Path("/unused"))["status"] == "success"


def test_sorting_reverses_mapping_and_rejects_ambiguous_instruction():
    scene = fixture("vowel_consonant_sort", "BONA", "按桌面标注分类：左侧放辅音，右侧放元音。")
    positions = [(0.5, .15, TABLE + .02), (0.5, -.15, TABLE + .02),
                 (0.6, .15, TABLE + .02), (0.6, -.15, TABLE + .02)]
    assert check(scene, positions)["status"] == "success"
    scene["_task_instruction"] = "按桌面标注分类：左侧放元音，右侧放辅音。"
    assert check(scene, positions)["status"] == "failure"
    scene["_task_instruction"] = "将字母积木按元音和辅音分成两组。"
    assert check(scene, positions)["status"] == "unsupported"
    scene["_task_instruction"] = "按桌面标注分类：左侧放辅音，右侧放元音。"
    assert check(scene, positions[:3] + [(0.6, 0.001, TABLE + .02)])["status"] == "unsupported"


def test_bad_observability_is_unsupported_and_drawer_not_letter():
    scene = fixture("spelling", "CA", "用字母积木拼出 CA。")
    scene["sampled_parameters"]["scene_manifest.objects"].append(
        {"class_name": "drawer", "name": "drawer", "size": [0.2, 0.3, 0.08]})
    scene["pose_node_ids"].append("body/drawer")
    positions = [(0.5, .025, TABLE + .02), (0.5, -.025, TABLE + .02), (0.5, 0, .8)]
    assert check(scene, positions)["status"] == "success"
    del scene["_hand_reach_m"]
    assert check(scene, positions)["status"] == "unsupported"
    scene["_hand_reach_m"] = .25
    scene["_hand_reference"] = "unspecified_point"
    assert check(scene, positions)["status"] == "unsupported"
    scene["_hand_reference"] = "wrist_link"
    result = score(scene, poses(positions), np.full((60, 2, 3), np.nan), Path("/unused"))
    assert result["status"] == "unsupported"
    del scene["_frame_dt_s"]
    assert check(scene, positions)["status"] == "unsupported"
    scene["_frame_dt_s"] = 1 / 30
    assert score(scene, poses(positions, frames=30), hands(frames=30), Path("/unused"))["status"] == "unsupported"
    del scene["sampled_parameters"]["scene_manifest.table"]
    assert check(scene, positions)["status"] == "unsupported"
