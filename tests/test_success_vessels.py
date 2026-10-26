"""Synthetic physical configurations; never use any demonstration end-state as a label."""
import json
from pathlib import Path

import numpy as np

from success_vessels import score


IDENTITY = [1., 0., 0., 0.]


def history(*poses, hands=((3., 3., 3.), (4., 4., 4.))):
    return np.tile(np.asarray(poses, dtype=float), (31, 1, 1)), np.tile(np.asarray(hands), (31, 1, 1))


def scene(objects, affordances, static=(), table_top=0.7):
    for affordance in affordances.values():
        affordance["coordinate_frame"] = "object-local; transform by object pose"
    return {
        "_frame_dt_s": 1 / 30,
        # A synthetic verified upper bound exercises the API, not an assertion
        # about the actual robot (which must be measured from its collision model).
        "_hand_reference": "wrist_link",
        "_hand_reach_m": .2,
        "pose_node_ids": [f"body/{o['name']}" for o in objects if o['class_name'] != 'bin' and o['class_name'] != 'mug_tree'],
        "nodes": [{"id": f"body/{name}", "parent_id": "world", "driver": "static",
                   "translation": location, "rotation_wxyz": IDENTITY} for name, location in static],
        "sampled_parameters": {
            "scene_manifest.objects": objects,
            "scene_manifest.sampled_values": {"affordances": affordances},
            "scene_manifest.table": {"top_z_m": table_top, "center_xyz_m": [0., 0., table_top - .025],
                                     "size_xyz_m": [1., 1., .05]},
        },
    }


def obj(name, kind, size, asset_id=None, instance_id=1):
    return {"name": name, "class_name": kind, "size": size, "asset_id": asset_id or kind,
            "instance_id": instance_id}


def bottle_assets(tmp_path):
    base = tmp_path / "objects" / "abc_bottles"
    base.mkdir(parents=True)
    (base / "collision.obj").write_text("v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\n"
                                          "v -1 -1 1\nv 1 -1 1\nv 1 1 1\nv -1 1 1\n"
                                          "f 1 2 3 4\nf 5 6 7 8\nf 1 2 6 5\n")
    (base / "manifest.json").write_text(json.dumps({"variants": {"bottle_1": {
        "meshes": [{"role": "collision", "file": "collision.obj"}]}}}))
    return tmp_path


def test_multibottle_support_not_every_bottle_on_floor(tmp_path):
    assets = bottle_assets(tmp_path)
    objects = [obj("a", "bottle", [.025, .08], "bottle_1", 1),
               obj("b", "bottle", [.025, .08], "bottle_1", 2),
               obj("box", "bin", [.3, .3, .3], instance_id=3)]
    s = scene(objects, {"3": {"interior_floor_z_m": .008, "rim_top_z_m": .308,
                              "mouth_size_xy_m": [.3, .3]}}, [("box", [0, 0, .7])])
    poses, hands = history([0, 0, .708, *IDENTITY], [0, 0, .788, *IDENTITY])
    result = score(s, poses, hands, assets)
    assert result["status"] == "success", result
    assert result["thresholds"]["window_duration_s"] == 1.
    assert result["thresholds"]["window_intervals"] == 30
    del s["_hand_reach_m"]
    assert score(s, poses, hands, assets)["status"] == "unsupported"
    s["_hand_reach_m"] = .2
    s["_hand_reference"] = "unverified_palm"
    assert score(s, poses, hands, assets)["status"] == "unsupported"
    s["_hand_reference"] = "wrist_link"
    assert score(s, poses[:-1], hands[:-1], assets)["status"] == "unsupported"
    assert result["measurements"]["b"]["supported_by"] == "a"
    poses[:, 1, 0] = .20
    assert score(s, poses, hands, assets)["status"] == "failure"


def test_repeated_scoring_reuses_static_bottle_geometry(tmp_path, monkeypatch):
    assets = bottle_assets(tmp_path)
    objects = [obj("a", "bottle", [.025, .08], "bottle_1", 1),
               obj("box", "bin", [.3, .3, .3], instance_id=2)]
    sample = scene(objects, {"2": {"interior_floor_z_m": .008, "rim_top_z_m": .308,
                                  "mouth_size_xy_m": [.3, .3]}}, [("box", [0, 0, .7])])
    poses, hands = history([0, 0, .708, *IDENTITY])
    real_read = Path.read_text
    reads = []

    def count_reads(path, *args, **kwargs):
        if path.name in {"manifest.json", "collision.obj"}:
            reads.append(path.name)
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", count_reads)
    assert score(sample, poses, hands, assets)["status"] == "success"
    first = len(reads)
    assert first == 2
    assert score(sample, poses, hands, assets)["status"] == "success"
    assert len(reads) == first


def test_toss_requires_stability_and_actual_hand_clearance(tmp_path):
    assets = bottle_assets(tmp_path)
    objects = [obj("a", "bottle", [.025, .08], "bottle_1", 1), obj("box", "bin", [.3, .3, .3], instance_id=2)]
    s = scene(objects, {"2": {"interior_floor_z_m": .008, "rim_top_z_m": .308,
                              "mouth_size_xy_m": [.3, .3]}}, [("box", [0, 0, .7])])
    poses, hands = history([0, 0, .708, *IDENTITY])
    poses[:, 0, 0] = np.linspace(0, .03, 31)
    assert score(s, poses, hands, assets)["status"] == "failure"
    poses[:, 0, 0] = 0
    hands[:, 0] = [0, 0, .75]
    assert score(s, poses, hands, assets)["status"] == "unsupported"
    assert score(s, poses, np.full((31, 2, 3), np.nan), assets)["status"] == "unsupported"
    # 31 frames with a shorter actual interval span less than one second.
    s["_frame_dt_s"] = 1 / 30 - 1e-8
    assert score(s, poses, hands, assets)["status"] == "unsupported"
    del s["_frame_dt_s"]
    assert score(s, poses, hands, assets)["status"] == "unsupported"


def test_hang_requires_handle_around_hook_not_merely_near_tree(tmp_path):
    mug = obj("mug", "mug", [.04, .09, .003], instance_id=1)
    tree = obj("tree", "mug_tree", [.18, .18, .3], instance_id=2)
    s = scene([mug, tree], {"1": {"handle_center_xyz_m": [.055, 0, .0495],
                   "handle_inner_height_m": .042, "handle_inner_width_m": .03,
                   "handle_tube_radius_m": .004, "hook_diameter_clearance_m": .022},
              "2": {"hook_center_xyz_m": [[.085, 0, .185]],
                    "hook_tip_xyz_m": [[.085, .028, .199]], "hook_radius_m": .004,
                    "hook_length_m": .056}}, [("tree", [0, 0, .7])])
    poses, hands = history([.14, 0, .8185, 0, 0, 0, 1])
    assert score(s, poses, hands, tmp_path)["status"] == "success"
    # The recorded open handle can carry the hook through its middle; it
    # need not hold the shaft at one preselected height near the top rim.
    poses[:, 0, 2] = .8355
    assert score(s, poses, hands, tmp_path)["status"] == "success"
    # The horizontal shaft still passes through a handle near one end;
    # the upturned tip must not be mistaken for a sloped shaft endpoint.
    poses[:, 0, :3] = [.14, .02, .8185]
    assert score(s, poses, hands, tmp_path)["status"] == "success"
    poses[:, 0, 1] = .08
    assert score(s, poses, hands, tmp_path)["status"] == "failure"
    del s["sampled_parameters"]["scene_manifest.sampled_values"]["affordances"]["2"]["hook_tip_xyz_m"]
    assert score(s, poses, hands, tmp_path)["status"] == "unsupported"


def test_unstack_all_three_cups_independently_rest_on_table(tmp_path):
    cups = [obj(f"cup{i}", "cup", [.048, .096, .002, .029, .02], instance_id=i) for i in (1, 2, 3)]
    aff = {str(i): {"bottom_outer_radius_m": .029, "mouth_inner_radius_m": .046,
                    "nest_step_m": .02} for i in (1, 2, 3)}
    s = scene(cups, aff)
    poses, hands = history(*[[x, 0, .7, *IDENTITY] for x in (-.15, 0, .15)])
    assert score(s, poses, hands, tmp_path)["status"] == "success"
    poses[:, 1, :3] = poses[:, 0, :3] + [0, 0, .02]
    assert score(s, poses, hands, tmp_path)["status"] == "failure"
    del aff["2"]["bottom_outer_radius_m"]
    assert score(s, poses, hands, tmp_path)["status"] == "unsupported"
