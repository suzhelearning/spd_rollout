#!/usr/bin/env python3
"""Independently audit fixed-source GPU rollouts before publishing success rates."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

from aggregate_evaluation import legacy_baseline_identity, model_identity
from infer_recording import DINO_SHA256, MODELS, MODEL_CHECKPOINT_SHA256, MODEL_HISTORY_MASK_PROB
from baseline_raw_rollout import _raw_scene, _score
from eval_selection import FIVE_TASKS, build_supplemental_selection, select_fixed_sources
from five_task_experiment import _SOURCE_FILES as _LAUNCH_SOURCE_FILES
from rollout_policy import save_json
from rollout_termination import TASK_LIMIT_TICKS

_TERMINAL = {"success", "failure", "unsupported", "error"}
_FROZEN_SELECTION_DIGEST = "07600ec4e7805dab3bb08ae4b1df2a2d3870c97a7c9e46c5f7ca7d2b9897dcec"
_FROZEN_INVALID_COUNT = 36
_FROZEN_INVALID_DIGEST = "1f979f217972b141ab999a5a3264d2434bce9ce8168c1c0ddb6609b59dddadaf"
_FROZEN_BASE_SELECTION_SHA256 = "2327cdd2c4ef2a786e930fbfcaa58f19f6af82f5ff1ae7c8dbac2d204aacbd5f"
_REQUIRED_RUNTIME_HASHES = frozenset({
    "baseline_raw_rollout.py", "physx_scene.py", "physx_scene_builder.py",
    "eval_selection.py", "gpu_rollout.py", "rollout_termination.py",
    "success_blocks.py", "success_vessels.py", "gpu_physx_worker.py",
    "gpu_physx_world.py", "gpu_rgb_renderer.py", "rollout_policy.py",
    "abc/abc_minimal/spd.py", "abc/abc_minimal/policy.py",
    "checkpoints/baseline_30000.pt", "dino/model.safetensors"})
_SUPPLEMENT_RUNTIME_HASHES = frozenset(_LAUNCH_SOURCE_FILES)


def _raw_header_matches(row: dict) -> bool:
    with h5py.File(row["raw_h5"], "r") as source:
        return (str(source.attrs.get("episode_id")) == row["episode_id"]
                and str(source.attrs.get("task_id")) == row["task_id"]
                and int(source.attrs.get("schema_version", -1)) == 4
                and bool(source.attrs.get("recorded_action", False)))


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def result_evidence_sha256(selection: dict, run_root: Path, shards: int) -> str:
    """Bind every committed result, terminal window and physical metadata to an audit."""
    root = Path(run_root).resolve()
    digest = hashlib.sha256()
    for index, row in enumerate(selection["episodes"]):
        folder = root / f"shard{index % shards}" / row["split"] / row["task_id"] / row["episode_id"]
        result_path = folder / "result.json"
        paths = [result_path, folder / "final_window.npz"]
        if result_path.is_file():
            try:
                metadata = _read(result_path).get("physical_environment")
                if isinstance(metadata, str):
                    paths.append(Path(metadata))
            except (OSError, ValueError):
                pass
        digest.update(f'{index}:{row["source_id"]}\n'.encode())
        for path in paths:
            resolved = path.resolve()
            signature = _sha(resolved) if resolved.is_relative_to(root) and resolved.is_file() else "MISSING"
            digest.update(f"{resolved}:{signature}\n".encode())
    return digest.hexdigest()


def audit_run(selection_path: Path, run_root: Path, *, shards: int, assets: Path,
              model: str = "baseline") -> dict:
    selection_path, run_root = Path(selection_path), Path(run_root)
    selection = _read(selection_path)
    rows = selection["episodes"]
    failures: list[dict] = []
    if model not in MODELS:
        raise ValueError(f"Unknown evaluation model: {model}")
    expected_identity = (model, MODEL_CHECKPOINT_SHA256[model], MODEL_HISTORY_MASK_PROB[model])
    checkpoint_key = f"checkpoints/{MODELS[model]}"
    try:
        launch = _read(run_root / "launch.json")
    except (OSError, ValueError):
        launch = None
    legacy = (model == "baseline" and launch is not None
              and legacy_baseline_identity(launch) and "code_root" not in launch
              and selection.get("selection_method") != "supplemental_valid_toss")
    if shards <= 0 or len(rows) != selection.get("planned"):
        raise ValueError("Invalid shard count or frozen selection length")
    identities = [(row["split"], row["task_id"], row["episode_id"], row["source_id"]) for row in rows]
    if len(set(identities)) != len(rows):
        failures.append({"problem": "duplicate_source_identity"})
    supplemental = selection.get("selection_method") == "supplemental_valid_toss"
    if supplemental:
        if (selection.get("tasks") != ["toss_in_bin"]
                or selection.get("counts_per_task_split") != {"train": 29, "val": 7}
                or selection.get("by_split") != {"train": 29, "val": 7}
                or len(rows) != 36
                or Counter((row.get("split"), row.get("task_id")) for row in rows)
                != Counter({("train", "toss_in_bin"): 29, ("val", "toss_in_bin"): 7})):
            failures.append({"problem": "supplement_quota_mismatch"})
        base_path = selection.get("base_selection_path")
        if not isinstance(base_path, str) or not Path(base_path).is_absolute():
            failures.append({"problem": "supplement_base_selection_mismatch"})
        else:
            try:
                if (_sha(Path(base_path)) != _FROZEN_BASE_SELECTION_SHA256
                        or selection.get("base_selection_sha256") != _FROZEN_BASE_SELECTION_SHA256):
                    failures.append({"problem": "supplement_base_selection_mismatch"})
                expected_selection = build_supplemental_selection(Path(base_path))
                if selection != expected_selection:
                    failures.append({"problem": "supplement_selection_mismatch"})
            except (OSError, KeyError, ValueError, TypeError, RuntimeError) as exc:
                failures.append({"problem": "supplement_selection_mismatch", "error": str(exc)})
    if selection.get("selection_method") == "fixed_count_per_task_split":
        if (tuple(selection.get("tasks", ())) != FIVE_TASKS
                or selection.get("counts_per_task_split") != {"train": 100, "val": 40}
                or selection.get("by_split") != {"train": 500, "val": 200}
                or len(rows) != 700):
            failures.append({"problem": "wrong_five_task_quota"})
        else:
            recomputed = select_fixed_sources(Path(selection["source_split"]),
                Path(selection["raw_root"]), FIVE_TASKS, selection["selection_seed"], 100, 40)
            if rows != recomputed or selection["selection_digest"] != recomputed[0]["selection_digest"]:
                failures.append({"problem": "selection_source_mismatch"})

    counts, raw_invalid = Counter(), set()
    group_counts = defaultdict(Counter)
    worker_ids, wrapper_ids, stage_ids = defaultdict(set), defaultdict(set), defaultdict(set)
    products = defaultdict(dict)
    expected_paths = set()
    validated = 0
    for index, row in enumerate(rows):
        shard_index = index % shards
        group_counts[(row["split"], row["task_id"])]["planned"] += 1
        result_path = run_root / f"shard{shard_index}" / row["split"] / row["task_id"] / row["episode_id"] / "result.json"
        expected_paths.add(result_path.resolve())
        try:
            header_valid = _raw_header_matches(row)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            failures.append({"problem": "raw_header_unreadable", "source_id": row["source_id"], "error": str(exc)})
            continue
        if supplemental and not header_valid:
            failures.append({"problem": "supplement_raw_invalid", "source_id": row["source_id"]})
        if not header_valid:
            raw_invalid.add(row["source_id"])
        if not result_path.is_file():
            failures.append({"problem": "missing_result", "source_id": row["source_id"]})
            continue
        try:
            result = _read(result_path)
        except (OSError, ValueError) as exc:
            failures.append({"problem": "invalid_result", "source_id": row["source_id"], "error": str(exc)})
            continue
        status = result.get("status")
        if status not in _TERMINAL:
            failures.append({"problem": "nonterminal_result", "source_id": row["source_id"], "status": status})
            continue
        counts[status] += 1
        group_counts[(row["split"], row["task_id"])][status] += 1
        if supplemental and status == "error":
            failures.append({"problem": "supplement_result_error", "source_id": row["source_id"],
                             "reason": result.get("reason")})
        try:
            if model_identity(result, legacy_baseline=legacy) != expected_identity:
                raise ValueError("Result identity differs from requested evaluation model")
        except ValueError:
            failures.append({"problem": "source_seed_or_model_mismatch", "source_id": row["source_id"]})
        if (any(result.get(key) != row.get(key) for key in
                ("split", "task_id", "episode_id", "source_id", "selection_digest", "raw_h5"))
                or result.get("method") != "gpu_vector"
                or result.get("checkpoint_step") != 30000 or result.get("use_ema") is not True
                or result.get("planned_index") != (index if supplemental else index // shards)
                or result.get("seed") != 123 + index // shards
                or result.get("horizon_ticks") != TASK_LIMIT_TICKS.get(row["task_id"])):
            failures.append({"problem": "source_seed_or_model_mismatch", "source_id": row["source_id"]})
        if not header_valid:
            if status != "error":
                failures.append({"problem": "invalid_raw_not_error", "source_id": row["source_id"]})
            continue
        if status == "error":
            failures.append({"problem": "unexpected_error", "source_id": row["source_id"],
                             "reason": result.get("reason")})
            continue
        tick, inferences = result.get("tick"), result.get("inference_count")
        limit = TASK_LIMIT_TICKS.get(row["task_id"])
        if (type(tick) is not int or limit is None or tick < 0 or tick > limit):
            failures.append({"problem": "task_limit_exceeded", "source_id": row["source_id"], "tick": tick})
            continue
        if tick == 0 and status == "unsupported" and result.get("reason") == "raw_scene_lacks_goal_and_geometry_affordances":
            if supplemental:
                try:
                    scene, scene_id = _raw_scene(row)
                    if (result.get("scene_id") != scene_id
                            or isinstance(scene.get("sampled_parameters", {}).get(
                                "scene_manifest.sampled_values"), dict)):
                        failures.append({"problem": "false_unscored_precheck", "source_id": row["source_id"]})
                except (OSError, KeyError, ValueError, TypeError) as exc:
                    failures.append({"problem": "false_unscored_precheck",
                                     "source_id": row["source_id"], "error": str(exc)})
            continue
        if tick < 30:
            failures.append({"problem": "goal_window_incomplete", "source_id": row["source_id"], "tick": tick})
            continue
        if type(inferences) is not int or inferences != (tick + 7) // 8:
            failures.append({"problem": "inference_count_mismatch", "source_id": row["source_id"],
                             "tick": tick, "inference_count": inferences})
        required_reason = {"success": "goal_verified", "failure": "timeout_goal_unmet",
                           "unsupported": "goal_unverifiable"}[status]
        if (result.get("termination_reason") != required_reason
                or (status != "success" and tick != limit)):
            failures.append({"problem": "terminal_transition_mismatch", "source_id": row["source_id"]})
        metadata_path = Path(result.get("physical_environment", ""))
        try:
            if not metadata_path.resolve().is_relative_to(run_root.resolve()):
                raise ValueError("Physical evidence is outside the run root")
            metadata = _read(metadata_path)
            physics, rendering = metadata["physics"], metadata["rendering"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({"problem": "physical_evidence_missing", "source_id": row["source_id"],
                             "error": str(exc)})
            continue
        gpu_config = physics.get("gpu_configuration", {})
        if (physics.get("device") != "cuda:0" or physics.get("gpu_dynamics_enabled") is not True
                or physics.get("broadphase") != "GPU" or physics.get("fabric_enabled") is not False
                or not math.isclose(physics.get("timestep_s", 0), 1 / 480, abs_tol=1e-10)
                or physics.get("substeps_per_control") != 16
                or physics.get("global_step_calls_per_tick") != 1
                or any(gpu_config.get(name, 0) < 32768 * physics.get("environment_count", 0)
                       for name in ("gpu_found_lost_aggregate_pairs_capacity",
                                    "gpu_total_aggregate_pairs_capacity"))
                or rendering.get("batch_camera_count") != 3 * physics.get("environment_count", 0)
                or rendering.get("resident_camera_count", 0) < rendering.get("batch_camera_count", 0)
                or rendering.get("global_captures_per_image_tick") != 1
                or len(rendering.get("render_products", ())) != 3):
            failures.append({"problem": "physical_or_rendering_contract_mismatch", "source_id": row["source_id"]})
        isaac_pid, wrapper_pid = metadata.get("worker_pid"), result.get("worker_pid")
        if type(isaac_pid) is not int or isaac_pid <= 0 or type(wrapper_pid) is not int or wrapper_pid <= 0:
            failures.append({"problem": "worker_identity_missing", "source_id": row["source_id"]})
        else:
            worker_ids[shard_index].add(isaac_pid)
            wrapper_ids[shard_index].add(wrapper_pid)
        stage_ids[shard_index].add(rendering.get("stage_id"))
        if supplemental and (type(rendering.get("stage_id")) is not int
                             or rendering["stage_id"] <= 0):
            failures.append({"problem": "stage_identity_missing", "source_id": row["source_id"]})
        slot = rendering.get("scene_partition")
        previous = products[shard_index].setdefault(slot, rendering.get("render_products"))
        if previous != rendering.get("render_products"):
            failures.append({"problem": "render_product_changed_within_slot", "source_id": row["source_id"]})
        window_path = result_path.parent / "final_window.npz"
        try:
            with np.load(window_path, allow_pickle=False) as saved:
                poses, hands = saved["object_pose"], saved["wrist_positions"]
                if (len(poses) != 31 or hands.shape != (31, 2, 3)
                        or saved["state"].shape != (54,) or len(saved["joint_names"]) != 54):
                    raise ValueError("Final measured window is incomplete")
            scene, _ = _raw_scene(row)
            scene["_hand_reference"] = metadata["hand_reference"]
            scene["_hand_reach_m"] = metadata["hand_reach_m"]
            replay = _score(scene, deque(zip(poses, hands, strict=True), maxlen=31), assets)
            if (replay != result.get("score") or replay["status"] != status
                    or replay["reason"] != result.get("reason")):
                failures.append({"problem": "score_evidence_mismatch", "source_id": row["source_id"]})
            else:
                validated += 1
        except (OSError, KeyError, ValueError, TypeError, RuntimeError) as exc:
            failures.append({"problem": "score_evidence_unreadable", "source_id": row["source_id"],
                             "error": str(exc)})

    invalid_digest = hashlib.sha256("\n".join(sorted(raw_invalid)).encode()).hexdigest()
    if (selection.get("selection_digest") == _FROZEN_SELECTION_DIGEST
            and (len(raw_invalid) != _FROZEN_INVALID_COUNT or invalid_digest != _FROZEN_INVALID_DIGEST)):
        failures.append({"problem": "frozen_raw_error_set_changed",
                         "expected_count": _FROZEN_INVALID_COUNT, "actual_count": len(raw_invalid),
                         "expected_sha256": _FROZEN_INVALID_DIGEST, "actual_sha256": invalid_digest})
    actual_paths = {path.resolve() for split in ("train", "val")
                    for path in run_root.glob(f"shard*/{split}/*/*/result.json")}
    if actual_paths - expected_paths:
        failures.append({"problem": "unplanned_result", "count": len(actual_paths - expected_paths)})
    if launch is None and (selection.get("selection_method") in
                           ("fixed_count_per_task_split", "supplemental_valid_toss")
                           or model != "baseline"):
        failures.append({"problem": "missing_launch_evidence"})
    hardware = {}
    launched = {}
    if launch is not None:
        try:
            if model_identity(launch, legacy_baseline=legacy) != expected_identity:
                raise ValueError("Launch identity differs from requested evaluation model")
        except ValueError:
            failures.append({"problem": "launch_model_identity_mismatch"})
        code_root = launch.get("code_root")
        if not legacy and (not isinstance(code_root, str) or not Path(code_root).is_absolute()):
            failures.append({"problem": "launch_model_identity_mismatch"})
        if not legacy and isinstance(code_root, str) and Path(code_root).is_absolute():
            if (not isinstance(launch.get("checkpoint"), str)
                    or Path(launch["checkpoint"]).resolve()
                    != (Path(code_root) / checkpoint_key).resolve()):
                failures.append({"problem": "launch_model_identity_mismatch"})
        source_hashes = launch.get("source_hashes")
        if not isinstance(source_hashes, dict):
            source_hashes = {}
        if (launch.get("selection_digest") != selection.get("selection_digest")
                or not launch.get("source_hashes")
                or launch.get("source_hashes") != launch.get("source_hashes_after")):
            failures.append({"problem": "launch_source_changed"})
        if selection.get("selection_digest") == _FROZEN_SELECTION_DIGEST or supplemental:
            required = _SUPPLEMENT_RUNTIME_HASHES if supplemental else _REQUIRED_RUNTIME_HASHES
            missing_hashes = ((required - {"checkpoints/baseline_30000.pt"})
                              | {checkpoint_key}) - set(source_hashes)
            if missing_hashes:
                failures.append({"problem": "missing_runtime_source_hashes",
                                 "missing": sorted(missing_hashes)})
            if (source_hashes.get(checkpoint_key) != expected_identity[1]
                    or source_hashes.get("dino/model.safetensors") != DINO_SHA256):
                failures.append({"problem": "approved_checkpoint_mismatch"})
        if supplemental and isinstance(code_root, str) and Path(code_root).is_absolute():
            expected_sources = ((_SUPPLEMENT_RUNTIME_HASHES - {"checkpoints/baseline_30000.pt"})
                                | {checkpoint_key})
            if set(source_hashes) != expected_sources:
                failures.append({"problem": "runtime_source_set_mismatch",
                                 "missing": sorted(expected_sources - set(source_hashes)),
                                 "unexpected": sorted(set(source_hashes) - expected_sources)})
            for name in sorted(expected_sources):
                path = Path(code_root) / name
                try:
                    actual = _sha(path)
                except OSError:
                    actual = None
                if source_hashes.get(name) != actual:
                    failures.append({"problem": "runtime_source_hash_mismatch", "source": name})
        if not legacy and source_hashes.get(checkpoint_key) != expected_identity[1]:
            failures.append({"problem": "approved_checkpoint_mismatch"})
        hardware = {item["index"]: item for item in launch.get("hardware", [])}
        launched = {item["gpu"]: item for item in launch.get("shards", [])}
        if set(hardware) != set(range(shards)) or set(launched) != set(range(shards)):
            failures.append({"problem": "launch_hardware_or_shard_set_mismatch"})
    for shard in range(shards):
        base = run_root / f"shard{shard}"
        try:
            session, timing = _read(base / "session.json"), _read(base / "timing.json")
            expected = sum(i % shards == shard for i in range(len(rows)))
            if (session.get("selection_digest") != selection.get("selection_digest")
                    or session.get("shard_count") != shards or session.get("shard_index") != shard
                    or session.get("planned") != expected or session.get("model") != model
                    or session.get("gpu_index") != shard
                    or session.get("control_hz") != 30 or session.get("physics_substeps") != 16
                    or session.get("policy_action_chunk") != 8 or session.get("policy_image_stride") != 8
                    or session.get("termination_mode") != "early_task_goal"
                    or any(session.get("task_limit_ticks", {}).get(task) != TASK_LIMIT_TICKS[task]
                           for task in {row["task_id"] for i, row in enumerate(rows) if i % shards == shard})):
                failures.append({"problem": "session_contract_mismatch", "shard": shard})
            try:
                if model_identity(session, legacy_baseline=legacy) != expected_identity:
                    raise ValueError("Session identity differs from requested evaluation model")
            except ValueError:
                failures.append({"problem": "session_model_identity_mismatch", "shard": shard})
            expected_checkpoint = (Path(code_root) / checkpoint_key) if (
                launch is not None and isinstance(launch.get("code_root"), str)
                and Path(launch["code_root"]).is_absolute()) else (
                    Path("/root/sunyz/project/spd-rollout-gpu") / checkpoint_key)
            expected_checkpoint = expected_checkpoint.resolve()
            if (not isinstance(session.get("checkpoint"), str)
                    or Path(session["checkpoint"]).resolve() != expected_checkpoint) and (
                    not legacy or "checkpoint" in session):
                failures.append({"problem": "session_model_identity_mismatch", "shard": shard})
            loaded_model = _read(base / "model.json")
            if (loaded_model.get("checkpoint_step") != 30000 or loaded_model.get("use_ema") is not True
                    or not isinstance(loaded_model.get("checkpoint"), str)
                    or Path(loaded_model["checkpoint"]).resolve() != expected_checkpoint
                    or ((selection.get("selection_digest") == _FROZEN_SELECTION_DIGEST or not legacy)
                        and loaded_model.get("dino_sha256") != DINO_SHA256)):
                failures.append({"problem": "model_checkpoint_mismatch", "shard": shard})
            try:
                if model_identity(loaded_model, legacy_baseline=legacy) != expected_identity:
                    raise ValueError("Loaded model identity differs from requested evaluation model")
            except ValueError:
                failures.append({"problem": "model_checkpoint_mismatch", "shard": shard})
            if launch is not None:
                renderer = _read(base / "renderer.json")
                device = renderer.get("device", {})
                expected_gpu = hardware.get(shard, {})
                if (device.get("gpu_id") != shard
                        or device.get("uuid") != expected_gpu.get("uuid")
                        or device.get("driver_version") != expected_gpu.get("driver")
                        or launched.get(shard, {}).get("exit_code") not in (0, 1)):
                    failures.append({"problem": "gpu_identity_mismatch", "shard": shard})
                if worker_ids[shard] and renderer.get("pid") not in worker_ids[shard]:
                    failures.append({"problem": "isaac_worker_identity_mismatch", "shard": shard})
            pids = timing.get("worker_pids", [])
            if (timing.get("load_failures") or len(pids) > 1
                    or set(pids) != wrapper_ids[shard]
                    or len(worker_ids[shard]) != (1 if pids else 0)):
                failures.append({"problem": "worker_not_resident", "shard": shard,
                                 "wrapper_pids": pids, "isaac_pids": sorted(worker_ids[shard]),
                                 "load_failures": timing.get("load_failures")})
            if len(stage_ids[shard]) > 1:
                failures.append({"problem": "stage_not_resident", "shard": shard})
            if supplemental and (worker_ids[shard] or wrapper_ids[shard] or stage_ids[shard]) and (
                    len(worker_ids[shard]) != 1 or len(wrapper_ids[shard]) != 1
                    or len(stage_ids[shard]) != 1 or None in stage_ids[shard]):
                failures.append({"problem": "supplement_worker_or_stage_missing", "shard": shard})
            logs = sorted(base.glob("runs/*/worker*/isaacsim-gpu-physx.log"))
            if len(logs) != len(pids):
                failures.append({"problem": "missing_or_extra_native_worker_log", "shard": shard,
                                 "logs": len(logs), "workers": len(pids)})
            for log in logs:
                native_errors = [(line_no, line) for line_no, line in enumerate(log.read_text(errors="replace").splitlines(), 1)
                                 if "[Error]" in line or "[Fatal]" in line]
                if native_errors:
                    failures.append({"problem": "native_runtime_errors", "shard": shard,
                                     "log": str(log), "count": len(native_errors),
                                     "first_errors": native_errors[:20]})
        except (OSError, KeyError, ValueError, TypeError) as exc:
            failures.append({"problem": "missing_shard_evidence", "shard": shard, "error": str(exc)})
    task_counts = {"train": {}, "val": {}}
    for (split, task), group in group_counts.items():
        task_counts[split][task] = {
            **{key: group[key] for key in ("planned", "success", "failure", "unsupported", "error")},
            "pending": group["planned"] - sum(group[key] for key in _TERMINAL),
            "scored": group["success"] + group["failure"]}
    split_counts = {split: {
        key: sum(item[key] for item in tasks.values())
        for key in ("planned", "success", "failure", "unsupported", "error", "pending", "scored")}
        for split, tasks in task_counts.items()}
    return {"status": "passed" if not failures else "failed",
            "model": model, "checkpoint_sha256": expected_identity[1],
            "history_mask_prob": expected_identity[2],
            "selection_digest": selection.get("selection_digest"), "selection_sha256": _sha(selection_path),
            "run_root": str(run_root.resolve()), "planned": len(rows), "terminal": sum(counts.values()),
            "status_counts": dict(counts), "known_invalid_raw": len(raw_invalid),
            "counts_by_task": task_counts, "counts_by_split": split_counts,
            "result_evidence_sha256": result_evidence_sha256(selection, run_root, shards),
            "aggregate_sha256": _sha(run_root / "report.json") if (run_root / "report.json").is_file() else None,
            "launch_sha256": _sha(run_root / "launch.json") if launch is not None else None,
            "known_invalid_source_ids_sha256": invalid_digest,
            "replayed_physical_rollouts": validated,
            "worker_pids": {str(k): sorted(v) for k, v in worker_ids.items()},
            "wrapper_pids": {str(k): sorted(v) for k, v in wrapper_ids.items()},
            "stage_ids": {str(k): sorted(v) for k, v in stage_ids.items()},
            "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, default="baseline")
    args = parser.parse_args()
    audit = audit_run(args.selection, args.run_root, shards=args.shards, assets=args.assets, model=args.model)
    save_json(args.run_root / "audit.json", audit)
    print(json.dumps({key: value for key, value in audit.items() if key != "failures"}, ensure_ascii=False), flush=True)
    if audit["status"] != "passed":
        raise SystemExit(f"GPU source audit failed: {len(audit['failures'])} findings; inspect audit.json")


if __name__ == "__main__":
    main()
