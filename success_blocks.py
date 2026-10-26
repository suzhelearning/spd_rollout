"""Conservative pose-only scorers for three letter-block tasks.

Goals are from scene metadata/original instructions, never demonstration endpoints.
World +y is robot-left for the robot facing +x. All distances are in metres.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np


_VOWELS = frozenset("AEIOU")
_STABILITY_FRACTION = .15
_ROTATION_DRIFT_DEG = 8.
_TABLE_HEIGHT_FRACTION = .20
_HAND_CONTACT_MARGIN_SIDES = .25  # extra clearance beyond verified whole-hand reach
_WORD_GAP_SIDES = (.7, 1.75)
_ROW_DEPTH_SIDES = 1.
_PYRAMID_BASE_GAP_SIDES = (.85, 1.8)
_PYRAMID_OVERLAP_FRACTION = .12


def _result(status: str, reason: str, **measurements: object) -> dict:
    return {"status": status, "reason": reason, "measurements": measurements,
            "thresholds": {
                "stable_translation_side_fraction": _STABILITY_FRACTION,
                "stable_rotation_deg": _ROTATION_DRIFT_DEG,
                "table_height_side_fraction": _TABLE_HEIGHT_FRACTION,
                "hand_contact_margin_block_sides": _HAND_CONTACT_MARGIN_SIDES,
                "word_neighbour_gap_sides": list(_WORD_GAP_SIDES),
                "word_row_depth_sides": _ROW_DEPTH_SIDES,
                "pyramid_base_gap_sides": list(_PYRAMID_BASE_GAP_SIDES),
                "pyramid_min_support_overlap_fraction": _PYRAMID_OVERLAP_FRACTION}}


def _rotation(poses: np.ndarray) -> np.ndarray:
    """World-from-body rotations, for wxyz quaternions."""
    q = poses[..., 3:]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((np.stack((1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)), axis=-1),
                     np.stack((2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)), axis=-1),
                     np.stack((2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)), axis=-1)), axis=-2)


def _task(scene: dict, blocks: list[dict]) -> str | None:
    explicit = scene.get("_task") or scene.get("task")
    if explicit in {"pyramid", "spelling", "vowel_consonant_sort"}:
        return explicit
    names = [obj["name"] for obj in blocks]
    for task in ("pyramid", "spelling", "vowel_consonant_sort"):
        if names and all(f"_{task}_object_" in name for name in names):
            return task
    instruction = scene.get("_task_instruction", "")
    if "元音" in instruction or "辅音" in instruction:
        return "vowel_consonant_sort"
    if "拼出" in instruction:
        return "spelling"
    return "pyramid" if len(blocks) == 3 else None


def _inventory(scene: dict) -> tuple[list[dict], np.ndarray, dict]:
    sampled = scene["sampled_parameters"]
    blocks = [obj for obj in sampled["scene_manifest.objects"] if obj.get("class_name") == "letter_block"]
    names = [obj["name"] for obj in blocks]
    if not blocks or len(set(names)) != len(names):
        raise ValueError("missing or duplicate letter blocks")
    nodes = scene["pose_node_ids"]
    indices, sizes = [], []
    for obj in blocks:
        node = f"body/{obj['name']}"
        if nodes.count(node) != 1:
            raise ValueError(f"missing or duplicated pose node {node}")
        size = np.asarray(obj["size"], dtype=float)
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError(f"invalid cube size {node}")
        if np.max(size) / np.min(size) > 1.05:
            raise ValueError(f"non-cubic letter block {node}")
        indices.append(nodes.index(node))
        sizes.append(size)
    table = sampled["scene_manifest.table"]
    z = float(table["top_z_m"])
    x_range = np.asarray(table["x_range_m"], dtype=float)
    y_range = np.asarray(table["y_range_m"], dtype=float)
    if (not np.isfinite(z) or x_range.shape != (2,) or y_range.shape != (2,)
            or not np.isfinite(x_range).all() or not np.isfinite(y_range).all()
            or x_range[0] >= x_range[1] or y_range[0] >= y_range[1]):
        raise ValueError("missing valid table bounds")
    return blocks, np.asarray(indices), {"z": z, "x": x_range, "y": y_range,
                                          "sizes": np.asarray(sizes)}


def _letters(scene: dict, blocks: list[dict]) -> list[str]:
    appearance = scene["sampled_parameters"]["scene_manifest.sampled_values"]["appearance"]["objects"]
    letters = [appearance[obj["name"]]["letter"] for obj in blocks]
    if any(not isinstance(letter, str) or not re.fullmatch("[A-Z]", letter) for letter in letters):
        raise ValueError("missing unambiguous uppercase face letters")
    return letters


def _word(scene: dict, letters: list[str]) -> str | None:
    instruction = scene.get("_task_instruction")
    match = re.fullmatch(r"\s*用字母积木拼出\s*([A-Za-z]+)\s*[。.!！]?\s*", instruction) if isinstance(instruction, str) else None
    candidates = [match.group(1).upper()] if match else []
    sampled = scene["sampled_parameters"]
    for candidate in (scene.get("target_word"), sampled.get("target_word"),
                      sampled.get("scene_manifest.sampled_values", {}).get("target_word")):
        if candidate is not None:
            if not isinstance(candidate, str) or not re.fullmatch("[A-Za-z]+", candidate):
                return None
            candidates.append(candidate.upper())
    if not candidates or len(set(candidates)) != 1 or Counter(candidates[0]) != Counter(letters):
        return None
    return candidates[0]


def _orientation(scene: dict) -> str | None:
    instruction = scene.get("_task_instruction")
    if not isinstance(instruction, str):
        return None
    normalized = re.sub(r"\s+", "", instruction)
    left_vowel = bool(re.search(r"左侧放元音[^，,。；;]*[，,；;]?右侧放辅音", normalized))
    left_consonant = bool(re.search(r"左侧放辅音[^，,。；;]*[，,；;]?右侧放元音", normalized))
    if left_vowel == left_consonant:
        return None
    return "vowel" if left_vowel else "consonant"


def _table_pose(centers: np.ndarray, rotations: np.ndarray, sizes: np.ndarray,
                table: dict) -> tuple[list[str], dict]:
    # All six faces of a procedural letter block carry the same letter. Its
    # rotated collision box, rather than a chosen local face, determines support.
    half_vertical = np.sum(np.abs(rotations[:, 2, :]) * sizes, axis=1) / 2
    gaps = centers[:, 2] - half_vertical - table["z"]
    half_xy = np.einsum("nij,nj->ni", np.abs(rotations[:, :2, :]), sizes) / 2
    edges = np.stack((centers[:, 0] - half_xy[:, 0] - table["x"][0],
                      table["x"][1] - centers[:, 0] - half_xy[:, 0],
                      centers[:, 1] - half_xy[:, 1] - table["y"][0],
                      table["y"][1] - centers[:, 1] - half_xy[:, 1]), axis=-1)
    reasons = []
    if np.any(np.abs(gaps) > _TABLE_HEIGHT_FRACTION * sizes[:, 2]):
        reasons.append("airborne_or_below_table")
    if np.any(edges < -_TABLE_HEIGHT_FRACTION * sizes[:, :1]):
        reasons.append("outside_table")
    return reasons, {"table_bottom_gap_m": gaps.tolist(),
                     "min_table_edge_clearance_m": np.min(edges, axis=1).tolist()}


def _pyramid(centers: np.ndarray, rotations: np.ndarray, sizes: np.ndarray,
             table: dict) -> tuple[str, dict]:
    if len(centers) != 3 or not np.allclose(sizes, sizes[0], rtol=.05, atol=0):
        return "unsupported_pyramid_geometry", {}
    scale = float(np.mean(sizes[:, 0]))
    order = np.argsort(centers[:, 2])
    low, high = order[:2], order[2]
    bases, top = centers[low], centers[high]
    base_gap = float(np.linalg.norm(bases[0, :2] - bases[1, :2]))
    heights = (bases[:, 2] - table["z"]).tolist()
    top_gap = float(top[2] - np.max(bases[:, 2]) - scale)
    extents = np.einsum("nij,nj->ni", np.abs(rotations[:, :2, :]), sizes) / 2
    overlaps = []
    for base in low:
        intersection = np.maximum(0, np.minimum(centers[base, :2] + extents[base], top[:2] + extents[high]) -
                                    np.maximum(centers[base, :2] - extents[base], top[:2] - extents[high]))
        overlaps.append(float(np.prod(intersection) / np.prod(2 * extents[high])))
    evidence = {"support_block_indices": low.tolist(), "top_block_index": int(high),
                "base_center_separation_m": base_gap, "base_center_heights_m": heights,
                "top_to_support_gap_m": top_gap, "top_support_overlap_fraction": overlaps}
    if not (_PYRAMID_BASE_GAP_SIDES[0] * scale <= base_gap <= _PYRAMID_BASE_GAP_SIDES[1] * scale):
        return "incorrect_base_spacing", evidence
    if abs(heights[0] - heights[1]) > _TABLE_HEIGHT_FRACTION * scale:
        return "uneven_support_height", evidence
    if abs(top_gap) > _TABLE_HEIGHT_FRACTION * scale:
        return "unsupported_or_collapsed_top", evidence
    if min(overlaps) < _PYRAMID_OVERLAP_FRACTION:
        return "top_not_supported_by_both", evidence
    # Projection onto the segment joining the supports must be central.
    if abs(np.dot(top[:2] - bases.mean(axis=0)[:2],
                  (bases[0, :2] - bases[1, :2]) / base_gap)) > .25 * scale:
        return "unbalanced_top", evidence
    return "pyramid_stacked", evidence


def _spelling(centers: np.ndarray, sizes: np.ndarray, letters: list[str], word: str) -> tuple[str, dict]:
    # Facing +x: world +y to -y is the left-to-right reading direction.
    order = np.argsort(-centers[:, 1])
    written = "".join(letters[index] for index in order)
    scale = float(np.median(sizes[:, 0]))
    gaps = -np.diff(centers[order, 1])
    row_depth = float(np.ptp(centers[:, 0]))
    evidence = {"target_word": word, "observed_word": written,
                "left_to_right_block_indices": order.tolist(),
                "neighbour_center_gap_m": gaps.tolist(), "row_depth_m": row_depth}
    if written != word:
        return "wrong_letter_order", evidence
    if row_depth > _ROW_DEPTH_SIDES * scale:
        return "not_one_readable_row", evidence
    if np.any(gaps < _WORD_GAP_SIDES[0] * scale) or np.any(gaps > _WORD_GAP_SIDES[1] * scale):
        return "letters_not_spaced_as_word", evidence
    return "word_spelled", evidence


def _sorting(centers: np.ndarray, rotations: np.ndarray, sizes: np.ndarray,
             letters: list[str], table: dict, left: str) -> tuple[str, dict]:
    # Raw scenes do not supply sorting_regions or printed-patch geometry.
    # Each whole rotated block must lie in one open half of the table.
    center_y = float(np.mean(table["y"]))
    half_width_y = np.sum(np.abs(rotations[:, 1, :]) * sizes, axis=1) / 2
    margins = (centers[:, 1] - center_y) / half_width_y
    actual = ["left" if m > 1 else "right" if m < -1 else "border" for m in margins]
    expected = ["left" if (letter in _VOWELS) == (left == "vowel") else "right"
                for letter in letters]
    evidence = {"region_interpretation": "instruction_left_right_open_table_halves_not_printed_label_patches",
                "left_category": left, "letters": letters, "expected_sides": expected,
                "observed_sides": actual, "table_midline_y_m": center_y,
                "rotated_block_half_width_y_m": half_width_y.tolist(),
                "signed_midline_distance_in_rotated_half_widths": margins.tolist()}
    if "border" in actual:
        return "ambiguous_half_table_boundary", evidence
    if actual != expected:
        return "incorrect_vowel_consonant_side", evidence
    return "all_letters_classified", evidence


def score(scene: dict, pose_history: np.ndarray, hand_positions: np.ndarray, assets: Path) -> dict:
    """Score a final >=one-second observed window, never a demo endpoint.

    ``assets`` is retained for the common scorer API, not a source of answers.
    Unobservable goals, support, timing or hands yield unsupported, not failure.
    """
    del assets
    try:
        blocks, indices, table = _inventory(scene)
        task = _task(scene, blocks)
        if task is None:
            return _result("unsupported", "unknown_block_task")
        dt = float(scene["_frame_dt_s"])
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("invalid frame cadence")
        samples = int(math.ceil(1 / dt - 1e-9)) + 1
        pose = np.asarray(pose_history, dtype=float)
        hands = np.asarray(hand_positions, dtype=float)
        if (pose.ndim != 3 or pose.shape[1:] != (len(scene["pose_node_ids"]), 7)
                or hands.shape != (len(pose), 2, 3) or len(pose) < max(30, samples)
                or not np.isfinite(hands[-samples:]).all()
                or not np.isfinite(pose[-samples:, indices]).all()):
            raise ValueError("missing final-second pose or real hand observations")
        window = pose[-samples:, indices]
        quaternion_norms = np.linalg.norm(window[..., 3:], axis=-1)
        if np.any(quaternion_norms < .95) or np.any(quaternion_norms > 1.05):
            raise ValueError("nonunit or missing block quaternions")
        sizes = table["sizes"]
        centers = window[-1, :, :3]
        rotations = _rotation(window[-1])
        if task != "pyramid":
            letters = _letters(scene, blocks)
            if task == "spelling":
                word = _word(scene, letters)
                if word is None:
                    return _result("unsupported", "missing_or_conflicting_target_word")
            else:
                left = _orientation(scene)
                if left is None:
                    return _result("unsupported", "ambiguous_vowel_sort_orientation")
        displacement = np.max(np.linalg.norm(window[..., :3] - centers[None], axis=-1), axis=0)
        unit = window[..., 3:] / quaternion_norms[..., None]
        rotation_change = np.max(np.degrees(2 * np.arccos(np.clip(np.abs(
            np.einsum("tni,ni->tn", unit, unit[-1])), 0, 1))), axis=0)
        # Hand positions are WRIST links, not fingertips. Bound cube geometry
        # by its circumscribed sphere; all frames and both wrists must clear.
        hand_clearance = np.min(np.linalg.norm(
            hands[-samples:, :, None, :] - window[:, None, :, :3], axis=-1) -
            np.linalg.norm(sizes, axis=1)[None, None, :] / 2, axis=(0, 1))
        evidence = {"task": task, "final_window_frames": samples,
                    "frame_dt_s": dt, "window_duration_s": (samples - 1) * dt,
                    "max_displacement_m": displacement.tolist(),
                    "max_rotation_change_deg": rotation_change.tolist(),
                    "minimum_wrist_to_block_surface_lower_bound_m": hand_clearance.tolist(),
                    "block_side_lengths_m": sizes[:, 0].tolist()}
        if np.any(displacement > _STABILITY_FRACTION * sizes[:, 0]) or np.any(rotation_change > _ROTATION_DRIFT_DEG):
            return _result("failure", "not_stable_last_second", **evidence)
        table_errors, table_evidence = _table_pose(centers, rotations, sizes, table)
        evidence.update(table_evidence)
        if task == "pyramid":
            low = np.argsort(centers[:, 2])[:2]
            if np.any(np.asarray(table_evidence["min_table_edge_clearance_m"]) < -_TABLE_HEIGHT_FRACTION * sizes[:, 0]):
                table_errors = ["outside_table"]
            elif np.any(np.abs(np.asarray(table_evidence["table_bottom_gap_m"])[low]) > _TABLE_HEIGHT_FRACTION * sizes[low, 2]):
                table_errors = ["base_airborne_or_below_table"]
            else:
                table_errors = []  # top rests on blocks, not on tabletop
            reason, details = _pyramid(centers, rotations, sizes, table)
            evidence.update(details)
            if reason == "unsupported_pyramid_geometry":
                return _result("unsupported", reason, **evidence)
        elif task == "spelling":
            reason, details = _spelling(centers, sizes, letters, word)
            evidence.update(details)
        else:
            reason, details = _sorting(centers, rotations, sizes, letters, table, left)
            evidence.update(details)
            if reason == "ambiguous_half_table_boundary" and not table_errors:
                return _result("unsupported", reason, **evidence)
        if table_errors:
            return _result("failure", table_errors[0], **evidence)
        if reason not in {"pyramid_stacked", "word_spelled", "all_letters_classified"}:
            return _result("failure", reason, **evidence)
        if scene.get("_hand_reference") != "wrist_link":
            return _result("unsupported", "unverified_hand_reference", **evidence)
        try:
            reach = np.asarray(scene["_hand_reach_m"], dtype=float)
            if reach.shape not in {(), (2,)} or not np.isfinite(reach).all() or np.any(reach <= 0):
                raise ValueError("invalid whole-hand wrist reach bound")
        except (KeyError, TypeError, ValueError):
            return _result("unsupported", "missing_verified_whole_hand_reach", **evidence)
        # If per-hand bounds differ, the larger still safely bounds both.
        reach_m = float(np.max(reach))
        evidence["verified_hand_reach_m"] = reach_m
        evidence["required_wrist_to_surface_m"] = (reach_m + _HAND_CONTACT_MARGIN_SIDES * sizes[:, 0]).tolist()
        if np.any(hand_clearance <= reach_m + _HAND_CONTACT_MARGIN_SIDES * sizes[:, 0]):
            return _result("unsupported", "hand_release_not_proven", **evidence)
        return _result("success", reason, **evidence)
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
        return _result("unsupported", "missing_or_invalid_scene_evidence", detail=str(exc))
