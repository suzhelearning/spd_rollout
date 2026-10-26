"""Conservative geometry-based scoring of raw vessel scenes (never demo end poses)."""
from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np


# The raw controller and PhysX rollout both expose one sample per 30 Hz control tick.
WINDOW_TICKS = 31
STABLE_TRANSLATION_M = .01
STABLE_ROTATION_DEG = 10.
HAND_MARGIN_M = .06
CONTACT_TOLERANCE_M = .01
WALL_TOLERANCE_M = .003
UPRIGHT_DEG = 15.
TABLE_EDGE_MARGIN_M = .005
HANDLE_CONTACT_TOLERANCE_M = .003
MIN_HOOK_CROSSING_COSINE = .5


class Unavailable(ValueError):
    """The recorded geometry or the measured state cannot establish a label."""


def _array(value, shape, label):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise Unavailable(f"{label}: non-numeric data") from exc
    if a.shape != shape or not np.isfinite(a).all():
        raise Unavailable(f"{label}: expected finite {shape}, got {a.shape}")
    return a


def _number(value, label):
    a = _array(value, (), label)
    return float(a)


def _rotation(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
                     [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
                     [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)]])


def _pose(p):
    if np.linalg.norm(p[3:]) < 1e-8:
        raise Unavailable("object pose has a zero quaternion")
    return p[:3], _rotation(p[3:])


def _result(status, reason, measurements, thresholds):
    return {"status": status, "reason": reason, "measurements": measurements,
            "thresholds": thresholds}


def _static_pose(scene, name):
    nodes = [node for node in scene["nodes"] if node.get("id") == f"body/{name}"]
    if len(nodes) != 1 or nodes[0].get("driver") != "static" or nodes[0].get("parent_id") != "world":
        raise Unavailable(f"{name}: missing unambiguous world-fixed node")
    node = nodes[0]
    position = _array(node["translation"], (3,), f"{name} translation")
    rotation = _array(node["rotation_wxyz"], (4,), f"{name} quaternion")
    return _pose(np.r_[position, rotation])


def _scene(scene, poses, hands):
    if "_frame_dt_s" not in scene:
        raise Unavailable("actual PhysX control-frame interval missing")
    dt = _number(scene["_frame_dt_s"], "actual PhysX control-frame interval")
    if dt <= 0 or abs(dt - 1 / 30) > 1e-6:
        raise Unavailable("actual PhysX frame interval must be 1/30 s for this 31-sample (30-interval) stability window")
    params = scene["sampled_parameters"]
    objects = params["scene_manifest.objects"]
    if not isinstance(objects, list) or not isinstance(scene["pose_node_ids"], list):
        raise Unavailable("missing recorded objects or pose node ordering")
    nodes = scene["pose_node_ids"]
    if len(nodes) != len(set(nodes)):
        raise Unavailable("duplicate dynamic pose node")
    raw_poses = np.asarray(poses, dtype=float)
    if raw_poses.ndim != 3 or raw_poses.shape[1:] != (len(nodes), 7) or len(raw_poses) < WINDOW_TICKS:
        raise Unavailable("at least 31 consecutive 30 Hz poses (30 intervals = 1 s) aligned to pose_node_ids required")
    if (WINDOW_TICKS - 1) * dt < 1.0:
        raise Unavailable("final 31 measured control frames span less than one second")
    tail = raw_poses[-WINDOW_TICKS:]
    if not np.isfinite(tail).all() or np.any(np.linalg.norm(tail[:, :, 3:], axis=2) < 1e-8):
        raise Unavailable("missing, nonfinite, or zero-quaternion measured pose")
    hand_tail = _array(np.asarray(hands)[-WINDOW_TICKS:], (WINDOW_TICKS, 2, 3), "actual world hand positions")
    affordances = params["scene_manifest.sampled_values"]["affordances"]
    if not isinstance(affordances, dict):
        raise Unavailable("recorded object affordances missing")
    for obj in objects:
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            raise Unavailable("invalid recorded object manifest")
    return params, objects, nodes, tail, hand_tail, affordances


def _dynamic(obj, nodes, poses):
    node = f"body/{obj['name']}"
    if node not in nodes:
        raise Unavailable(f"{node}: missing dynamic pose")
    return poses[:, nodes.index(node), :]


def _hand_bound(scene):
    """Only a validated bound on *all* hand collision geometry can prove release."""
    if scene.get("_hand_reference") != "wrist_link":
        return None
    try:
        bound = float(_number(scene["_hand_reach_m"], "whole-hand collision reach from wrist"))
    except (KeyError, Unavailable):
        return None
    return bound if bound > 0 else None


def _stability_and_hands(poses, hands, objects, nodes, radii, measurements, hand_reach):
    problems = []
    released = hand_reach is not None
    for obj in objects:
        name = obj["name"]
        p = _dynamic(obj, nodes, poses)
        final_quat = p[-1, 3:] / np.linalg.norm(p[-1, 3:])
        all_quat = p[:, 3:] / np.linalg.norm(p[:, 3:], axis=1)[:, None]
        displacement = float(np.max(np.linalg.norm(p[:, :3] - p[-1, :3], axis=1)))
        angle = float(np.degrees(2 * np.arccos(np.clip(np.min(np.abs(all_quat @ final_quat)), 0, 1))))
        # Every hand-collision point is within the validated wrist reach.
        # Subtract the object's enclosing radius before checking clearance.
        wrist_clearance = float(np.min(np.linalg.norm(hands - p[:, None, :3], axis=2)) - radii[name])
        measurements.setdefault(name, {}).update({"max_final_displacement_m": displacement,
            "max_final_rotation_deg": angle, "min_wrist_to_object_sphere_clearance_m": wrist_clearance,
            "clearance_beyond_hand_reach_m": wrist_clearance - hand_reach if hand_reach is not None else None})
        if displacement > STABLE_TRANSLATION_M or angle > STABLE_ROTATION_DEG:
            problems.append(f"{name}:unstable")
        if hand_reach is None or wrist_clearance <= hand_reach + HAND_MARGIN_M:
            released = False
    return problems, released


def _finish(reasons, released, success_reason, measurements, thresholds, hand_reach):
    thresholds["whole_hand_collision_reach_from_wrist_m"] = hand_reach
    if reasons:
        return _result("failure", "; ".join(reasons), measurements, thresholds)
    if not released:
        return _result("unsupported", "release_unverifiable_from_wrist_only_or_insufficient_clearance", measurements, thresholds)
    return _result("success", success_reason, measurements, thresholds)


def _table(params):
    table = params["scene_manifest.table"]
    center = _array(table["center_xyz_m"], (3,), "table center")
    size = _array(table["size_xyz_m"], (3,), "table size")
    top = _number(table["top_z_m"], "table top")
    if np.any(size <= 0) or abs(center[2] + size[2] / 2 - top) > .002:
        raise Unavailable("table top and physical dimensions disagree")
    return center, size, top


def _affordance(affordances, obj, fields):
    a = affordances.get(str(obj["instance_id"]))
    if not isinstance(a, dict) or any(field not in a for field in fields):
        raise Unavailable(f"{obj['name']}: missing recorded affordance {fields}")
    if a.get("coordinate_frame") != "object-local; transform by object pose":
        raise Unavailable(f"{obj['name']}: unknown affordance coordinate frame")
    return a


@lru_cache(maxsize=64)
def _bottle_geometry(assets, variant):
    base = Path(assets) / "objects/abc_bottles"
    manifest = json.loads((base / "manifest.json").read_text())
    parts, edges = [], []
    for mesh in manifest["variants"][variant]["meshes"]:
        if mesh["role"] != "collision":
            continue
        vertices, local_edges = [], set()
        for line in (base / mesh["file"]).read_text().splitlines():
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v":
                vertices.append([float(v) for v in fields[1:]])
            elif fields[0] == "f":
                face = [int(part.split("/")[0]) - 1 for part in fields[1:]]
                local_edges.update(tuple(sorted((i, j))) for i, j in zip(face, face[1:] + face[:1]))
        verts = np.asarray(vertices, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) < 3 or not np.isfinite(verts).all() or not local_edges:
            raise Unavailable(f"{variant}: invalid collision mesh vertices/faces")
        if any(min(edge) < 0 or max(edge) >= len(verts) for edge in local_edges):
            raise Unavailable(f"{variant}: collision face references absent vertex")
        edges.extend((i + sum(len(part) for part in parts), j + sum(len(part) for part in parts)) for i, j in local_edges)
        parts.append(verts)
    if not parts:
        raise Unavailable(f"{variant}: collision vertices unavailable")
    return np.concatenate(parts), np.asarray(edges, dtype=int)


def _toss(scene, params, objects, nodes, poses, hands, aff, assets):
    bottles = [o for o in objects if o["class_name"] == "bottle"]
    bins = [o for o in objects if o["class_name"] == "bin"]
    if not bottles or len(bins) != 1 or len(bottles) + 1 != len(objects):
        raise Unavailable("toss scene must have all bottles and exactly one fixed bin")
    bin_obj = bins[0]
    origin, bin_rot = _static_pose(scene, bin_obj["name"])
    bin_aff = _affordance(aff, bin_obj, ("mouth_size_xy_m", "interior_floor_z_m", "rim_top_z_m"))
    mouth = _array(bin_aff["mouth_size_xy_m"], (2,), "bin inner mouth size")
    floor = _number(bin_aff["interior_floor_z_m"], "bin floor")
    rim = _number(bin_aff["rim_top_z_m"], "bin rim")
    bin_width, bin_depth, bin_height = _array(bin_obj["size"], (3,), "physical bin inner dimensions")
    if (np.any(mouth <= 0) or rim <= floor or min(bin_width, bin_depth, bin_height) <= 0
            or np.max(abs(mouth - [bin_width, bin_depth])) > .002
            or abs((rim - floor) - bin_height) > .002):
        raise Unavailable("bin affordance disagrees with physical bin dimensions")
    measures, thresholds = {}, {"window_ticks": WINDOW_TICKS, "window_intervals": WINDOW_TICKS - 1, "frame_dt_s": 1 / 30, "window_duration_s": 1., "first_last_sample_span_s": (WINDOW_TICKS - 1) / 30,
        "max_final_displacement_m": STABLE_TRANSLATION_M, "max_final_rotation_deg": STABLE_ROTATION_DEG,
        "min_clearance_beyond_whole_hand_reach_m": HAND_MARGIN_M, "wall_tolerance_m": WALL_TOLERANCE_M,
        "contact_tolerance_m": CONTACT_TOLERANCE_M, "inner_floor_z_m": floor,
        "rim_z_m": rim, "mouth_half_xy_m": (mouth / 2).tolist()}
    reasons, bounds, radii = [], {}, {}
    for bottle in bottles:
        name = bottle["name"]
        radius, height = _array(bottle["size"], (2,), f"{name} radius,height")
        if min(radius, height) <= 0:
            raise Unavailable(f"{name}: non-positive physical dimensions")
        canonical, edges = _bottle_geometry(assets, bottle["asset_id"])
        vertices_local = canonical * [radius, radius, height]
        radii[name] = float(np.max(np.linalg.norm(vertices_local - [0, 0, height / 2], axis=1)) + height / 2)
        location, rotation = _pose(_dynamic(bottle, nodes, poses)[-1])
        vertices = (vertices_local @ rotation.T - origin + location) @ bin_rot
        center = (location + rotation[:, 2] * height / 2 - origin) @ bin_rot
        bottom, top = float(vertices[:, 2].min()), float(vertices[:, 2].max())
        lower = vertices[vertices[:, 2] <= rim]
        starts, ends = vertices[edges[:, 0]], vertices[edges[:, 1]]
        crossing = (starts[:, 2] < rim) != (ends[:, 2] < rim)
        starts, ends = starts[crossing], ends[crossing]
        lower = np.concatenate((lower, starts + (ends - starts) * ((rim - starts[:, 2]) / (ends[:, 2] - starts[:, 2]))[:, None]))
        if len(lower):
            clearance = mouth / 2 - np.max(np.abs(lower[:, :2]), axis=0)
        else:
            clearance = np.array([-np.inf, -np.inf])
        measures[name] = {"center_bin_xyz_m": center.tolist(), "bottom_bin_z_m": bottom,
                          "top_bin_z_m": top, "inner_wall_clearance_xy_m": clearance.tolist() if len(lower) else None}
        bounds[name] = (vertices.min(axis=0), vertices.max(axis=0))
        if center[2] >= rim or bottom < floor - CONTACT_TOLERANCE_M or np.any(clearance < -WALL_TOLERANCE_M):
            reasons.append(f"{name}:outside_bin")
    # A stable upper bottle can be borne by another bottle. A directed graph
    # of possible *surface contacts* must connect every bottle to the floor.
    supported = {name for name, (lo, hi) in bounds.items() if abs(lo[2] - floor) <= CONTACT_TOLERANCE_M}
    for name in supported:
        measures[name]["supported_by"] = "bin_floor"
    remaining = set(bounds) - supported
    while remaining:
        added = []
        for name in sorted(remaining):
            low, high = bounds[name]
            for other in sorted(supported):
                other_low, other_high = bounds[other]
                lateral_gap = np.maximum(np.maximum(other_low[:2] - high[:2], low[:2] - other_high[:2]), 0)
                vertical_gap = max(other_low[2] - high[2], low[2] - other_high[2], 0)
                if np.max(lateral_gap) <= CONTACT_TOLERANCE_M and vertical_gap <= CONTACT_TOLERANCE_M and low[2] > other_low[2] + CONTACT_TOLERANCE_M:
                    measures[name]["supported_by"] = other
                    measures[name]["support_bbox_gap_xy_m"] = lateral_gap.tolist()
                    measures[name]["support_bbox_gap_z_m"] = vertical_gap
                    added.append(name)
                    break
        if not added:
            break
        supported.update(added)
        remaining.difference_update(added)
    if remaining:
        reasons.extend(f"{name}:no_floor_connected_support" for name in sorted(remaining))
    hand_reach = _hand_bound(scene)
    stability, released = _stability_and_hands(poses, hands, bottles, nodes, radii, measures, hand_reach)
    reasons += stability
    return _finish(reasons, released, "all_bottles_stably_in_bin_and_released", measures, thresholds, hand_reach)


def _hang(scene, params, objects, nodes, poses, hands, aff):
    mugs = [o for o in objects if o["class_name"] == "mug"]
    trees = [o for o in objects if o["class_name"] == "mug_tree"]
    if len(mugs) != 1 or len(trees) != 1 or len(objects) != 2:
        raise Unavailable("hang scene requires one mug and one fixed mug tree")
    mug, tree = mugs[0], trees[0]
    radius, height, thickness = _array(mug["size"], (3,), "mug radius,height,wall")
    if radius <= 0 or height <= 0 or thickness <= 0 or thickness >= radius:
        raise Unavailable("invalid mug dimensions")
    a = _affordance(aff, mug, ("handle_center_xyz_m", "handle_inner_height_m", "handle_inner_width_m", "handle_tube_radius_m", "hook_diameter_clearance_m"))
    b = _affordance(aff, tree, ("hook_center_xyz_m", "hook_tip_xyz_m", "hook_radius_m", "hook_length_m"))
    handle = _array(a["handle_center_xyz_m"], (3,), "mug handle center")
    opening_height = _number(a["handle_inner_height_m"], "handle inner height")
    opening_width = _number(a["handle_inner_width_m"], "handle inner width")
    tube_radius = _number(a["handle_tube_radius_m"], "handle tube radius")
    hook_radius = _number(b["hook_radius_m"], "hook radius")
    hook_length = _number(b["hook_length_m"], "hook length")
    recorded_clearance = _number(a["hook_diameter_clearance_m"], "hook aperture clearance")
    centers = np.asarray(b["hook_center_xyz_m"], dtype=float)
    tips = np.asarray(b["hook_tip_xyz_m"], dtype=float)
    if (centers.ndim != 2 or centers.shape[1] != 3 or not len(centers) or tips.shape != centers.shape
            or not np.isfinite(centers).all() or not np.isfinite(tips).all()
            or min(opening_height, opening_width, tube_radius, hook_radius, hook_length) <= 0
            or opening_width <= 2 * hook_radius or opening_height <= 2 * hook_radius
            or abs(recorded_clearance - (opening_width - 2 * hook_radius)) > .005):
        raise Unavailable("incomplete or inconsistent hook/handle physical geometry")
    tree_pos, tree_rot = _static_pose(scene, tree["name"])
    mug_pos, mug_rot = _pose(_dynamic(mug, nodes, poses)[-1])
    center_world = mug_pos + mug_rot @ handle
    tree_hook_centers = centers @ tree_rot.T + tree_pos
    relative = (tree_hook_centers - center_world) @ mug_rot
    tree_hook_tips = tips @ tree_rot.T + tree_pos
    shaft_endpoints = tips.copy()
    shaft_endpoints[:, 2] = centers[:, 2]
    tree_shaft_ends = shaft_endpoints @ tree_rot.T + tree_pos
    shaft_end_relative = (tree_shaft_ends - center_world) @ mug_rot
    # The tip rises 14 mm beyond the horizontal rod. Only its x/y position
    # identifies the rod endpoint; including its elevated z invents a slope.
    rod_start = 2 * relative - shaft_end_relative
    rod_delta = shaft_end_relative - rod_start
    rod_length = np.linalg.norm(rod_delta, axis=1)
    if np.any(rod_length < 1e-6) or np.any(np.abs(
            2 * np.linalg.norm(tree_shaft_ends - tree_hook_centers, axis=1) - hook_length) > .005):
        raise Unavailable("recorded horizontal hook shaft differs from physical hook length")
    transverse = np.abs(rod_delta[:, 1]) / rod_length
    crossing_t = -rod_start[:, 1] / np.where(np.abs(rod_delta[:, 1]) > 1e-9, rod_delta[:, 1], 1.)
    crossing = rod_start + crossing_t[:, None] * rod_delta
    half_width = opening_width / 2 - hook_radius + HANDLE_CONTACT_TOLERANCE_M
    half_height = opening_height / 2 - hook_radius + HANDLE_CONTACT_TOLERANCE_M
    if half_width <= 0 or half_height <= 0:
        raise Unavailable("hook does not fit inside the recorded handle opening")
    aperture_fraction = (crossing[:, 0] / half_width) ** 2 + (crossing[:, 2] / half_height) ** 2
    fits = ((crossing_t >= 0) & (crossing_t <= 1)
            & (transverse >= MIN_HOOK_CROSSING_COSINE) & (aperture_fraction <= 1))
    _, _, table_top = _table(params)
    bottom = float(mug_pos[2])  # procedural mug local origin is its bottom plane
    measures = {mug["name"]: {"handle_center_world_xyz_m": center_world.tolist(),
        "hook_center_in_handle_xyz_m": relative.tolist(),
        "horizontal_hook_shaft_end_in_handle_xyz_m": shaft_end_relative.tolist(),
        "upturned_hook_tip_in_handle_xyz_m": ((tree_hook_tips - center_world) @ mug_rot).tolist(),
        "hook_shaft_plane_crossing_in_handle_xyz_m": crossing.tolist(),
        "hook_shaft_crossing_t": crossing_t.tolist(),
        "hook_shaft_transverse_cosine": transverse.tolist(),
        "hook_aperture_fraction_squared": aperture_fraction.tolist(),
        "mug_bottom_above_table_m": bottom - table_top,
        "hook_index": int(np.argmin(aperture_fraction))}}
    thresholds = {"window_ticks": WINDOW_TICKS, "window_intervals": WINDOW_TICKS - 1,
        "frame_dt_s": 1 / 30, "window_duration_s": 1.,
        "max_final_displacement_m": STABLE_TRANSLATION_M,
        "max_final_rotation_deg": STABLE_ROTATION_DEG,
        "min_clearance_beyond_whole_hand_reach_m": HAND_MARGIN_M,
        "min_mug_bottom_above_table_m": CONTACT_TOLERANCE_M,
        "hook_contact_tolerance_m": HANDLE_CONTACT_TOLERANCE_M,
        "hook_aperture_half_width_m": half_width,
        "hook_aperture_half_height_m": half_height,
        "min_hook_plane_crossing_cosine": MIN_HOOK_CROSSING_COSINE}
    reasons = []
    if not np.any(fits):
        reasons.append("handle_not_around_hook")
    if bottom <= table_top + CONTACT_TOLERANCE_M:
        reasons.append("mug_not_off_table")
    enclosing_radius = math.hypot(radius + handle[0] + opening_width / 2 + tube_radius, height)
    hand_reach = _hand_bound(scene)
    stability, released = _stability_and_hands(poses, hands, [mug], nodes, {mug["name"]: enclosing_radius}, measures, hand_reach)
    reasons += stability
    return _finish(reasons, released, "mug_handle_stably_on_hook_and_released", measures, thresholds, hand_reach)


def _unstack(scene, params, objects, nodes, poses, hands, aff):
    if len(objects) != 3 or any(o["class_name"] != "cup" for o in objects):
        raise Unavailable("unstack scene requires exactly three procedural cups")
    center, table_size, table_top = _table(params)
    measures, radii, reasons = {}, {}, []
    thresholds = {"window_ticks": WINDOW_TICKS, "window_intervals": WINDOW_TICKS - 1, "frame_dt_s": 1 / 30, "window_duration_s": 1., "first_last_sample_span_s": (WINDOW_TICKS - 1) / 30, "max_final_displacement_m": STABLE_TRANSLATION_M,
        "max_final_rotation_deg": STABLE_ROTATION_DEG, "min_clearance_beyond_whole_hand_reach_m": HAND_MARGIN_M,
        "max_upright_tilt_deg": UPRIGHT_DEG, "table_contact_tolerance_m": CONTACT_TOLERANCE_M,
        "table_edge_margin_m": TABLE_EDGE_MARGIN_M}
    positions = []
    for cup in objects:
        name = cup["name"]
        outer, height, wall, base, step = _array(cup["size"], (5,), f"{name} dimensions")
        a = _affordance(aff, cup, ("bottom_outer_radius_m", "mouth_inner_radius_m", "nest_step_m"))
        base_aff = _number(a["bottom_outer_radius_m"], "cup base radius")
        inner = _number(a["mouth_inner_radius_m"], "cup mouth inner radius")
        if min(outer, height, wall, base, step, base_aff, inner) <= 0 or inner >= outer or abs(base_aff - base) > .002 or abs(step - _number(a["nest_step_m"], "nest step")) > .002:
            raise Unavailable(f"{name}: inconsistent cup geometry or nesting affordance")
        pos, rot = _pose(_dynamic(cup, nodes, poses)[-1])
        tilt = float(np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1))))
        # With an upright procedural cup, the local bottom plane is at z=0.
        bottom_gap = float(pos[2] - table_top)
        edge_clearance = table_size[:2] / 2 - abs(pos[:2] - center[:2]) - outer
        measures[name] = {"bottom_to_table_m": bottom_gap, "upright_tilt_deg": tilt,
                          "table_edge_clearance_xy_m": edge_clearance.tolist()}
        radii[name] = math.hypot(outer, height)
        positions.append(pos)
        if tilt > UPRIGHT_DEG:
            reasons.append(f"{name}:not_upright")
        if abs(bottom_gap) > CONTACT_TOLERANCE_M:
            reasons.append(f"{name}:not_independently_on_table")
        if np.any(edge_clearance < TABLE_EDGE_MARGIN_M):
            reasons.append(f"{name}:outside_table_support")
    for i, first in enumerate(objects):
        for j in range(i + 1, len(objects)):
            second = objects[j]
            separation = float(np.linalg.norm(positions[i][:2] - positions[j][:2]))
            required = float(first["size"][0] + second["size"][0] + CONTACT_TOLERANCE_M)
            key = f"{first['name']}|{second['name']}"
            measures[key] = {"center_xy_separation_m": separation, "min_nonoverlap_separation_m": required}
            if separation < required:
                reasons.append(f"{key}:nested_or_overlapping")
    hand_reach = _hand_bound(scene)
    stability, released = _stability_and_hands(poses, hands, objects, nodes, radii, measures, hand_reach)
    reasons += stability
    return _finish(reasons, released, "three_separate_upright_table_supported_cups_released", measures, thresholds, hand_reach)


def score(scene: dict, pose_history: np.ndarray, hand_positions: np.ndarray, assets: Path) -> dict:
    """Judge 31 final 30-Hz samples spanning one second; never score unknown timing."""
    try:
        params, objects, nodes, poses, hands, affordances = _scene(scene, pose_history, hand_positions)
        classes = [o["class_name"] for o in objects]
        if len(classes) == 2 and sorted(classes) == ["mug", "mug_tree"]:
            return _hang(scene, params, objects, nodes, poses, hands, affordances)
        if classes.count("bin") == 1 and classes.count("bottle") >= 1 and len(classes) == classes.count("bottle") + 1:
            return _toss(scene, params, objects, nodes, poses, hands, affordances, assets)
        if classes == ["cup", "cup", "cup"]:
            return _unstack(scene, params, objects, nodes, poses, hands, affordances)
        raise Unavailable(f"unsupported vessel scene object classes: {classes}")
    except (Unavailable, KeyError, IndexError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return _result("unsupported", str(exc), {}, {})
