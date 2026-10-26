"""A complete 29/7 supplemental run must audit, then reject changed recorded code."""
import hashlib
import json
from collections import deque

import h5py
import numpy as np
import pytest

from baseline_raw_rollout import _raw_scene, _score
from eval_selection import build_supplemental_selection
from five_task_audit import audit_run
from five_task_experiment import _SOURCE_FILES
from infer_recording import MODEL_CHECKPOINT_SHA256
from test_success_vessels import IDENTITY, bottle_assets, history, obj, scene
from test_supplement_selection import frozen_source


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n")


@pytest.fixture
def complete_supplement(frozen_source, tmp_path, monkeypatch):
    base, _, candidates = frozen_source
    assets = bottle_assets(tmp_path)
    vessel_scene = scene(
        [obj("a", "bottle", [.025, .08], "bottle_1", 1),
         obj("box", "bin", [.3, .3, .3], instance_id=2)],
        {"2": {"interior_floor_z_m": .008, "rim_top_z_m": .308,
               "mouth_size_xy_m": [.3, .3]}}, [("box", [0, 0, .7])],
    )
    for key in ("_frame_dt_s", "_hand_reference", "_hand_reach_m"):
        vessel_scene.pop(key)
    vessel_scene["schema_version"] = 4
    scene_bytes = json.dumps(vessel_scene, sort_keys=True).encode("utf-8")
    scene_id = hashlib.sha256(scene_bytes).hexdigest()
    scene_file = tmp_path / "raw/toss_in_bin/scenes" / f"{scene_id}.json"
    scene_file.write_bytes(scene_bytes)
    for row in candidates:
        with h5py.File(row["raw_h5"], "r+") as source:
            if int(source.attrs["schema_version"]) == 4:
                source.attrs["scene_id"] = scene_id
    selection_data = build_supplemental_selection(base)
    selection = tmp_path / "selection.json"
    _save(selection, selection_data)
    monkeypatch.setattr("five_task_audit._FROZEN_BASE_SELECTION_SHA256",
                        hashlib.sha256(base.read_bytes()).hexdigest())

    poses, hands = history([0, 0, .708, *IDENTITY])
    scored_scene, _ = _raw_scene(selection_data["episodes"][0])
    scored_scene["_hand_reference"] = "wrist_link"
    scored_scene["_hand_reach_m"] = .2
    score = _score(scored_scene, deque(zip(poses, hands, strict=True), maxlen=31), assets)
    assert score["status"] == "success"

    code_root = tmp_path / "code"
    hashes = {}
    for name in _SOURCE_FILES:
        source = code_root / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"verified {name}\n".encode())
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setitem(MODEL_CHECKPOINT_SHA256, "baseline",
                        hashes["checkpoints/baseline_30000.pt"])
    monkeypatch.setattr("five_task_audit.DINO_SHA256", hashes["dino/model.safetensors"])
    run = tmp_path / "gpu-full"
    hardware = [{"index": shard, "uuid": f"GPU-{shard}", "driver": "580.126.09"}
                for shard in range(8)]
    launch_shards = [{"gpu": shard, "exit_code": 0} for shard in range(8)]
    _save(run / "launch.json", {
        "model": "baseline", "checkpoint_sha256": MODEL_CHECKPOINT_SHA256["baseline"],
        "history_mask_prob": 0.0, "checkpoint": str(code_root / "checkpoints/baseline_30000.pt"),
        "code_root": str(code_root), "selection_digest": selection_data["selection_digest"],
        "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
        "selection_path": str(selection), "run_root": str(run.resolve()),
        "source_hashes": hashes, "source_hashes_after": dict(hashes),
        "hardware": hardware, "shards": launch_shards,
    })
    for shard in range(8):
        worker, wrapper = 1000 + shard, 2000 + shard
        folder = run / f"shard{shard}"
        _save(folder / "session.json", {
            "selection_digest": selection_data["selection_digest"],
            "model": "baseline", "checkpoint_sha256": MODEL_CHECKPOINT_SHA256["baseline"],
            "history_mask_prob": 0.0, "checkpoint": str(code_root / "checkpoints/baseline_30000.pt"),
            "shard_count": 8, "shard_index": shard, "planned": sum(i % 8 == shard for i in range(36)),
            "gpu_index": shard, "seed": 123, "control_hz": 30,
            "physics_substeps": 16, "policy_action_chunk": 8, "policy_image_stride": 8,
            "termination_mode": "early_task_goal", "task_limit_ticks": {"toss_in_bin": 1800},
            "horizon_ticks": None,
        })
        _save(folder / "model.json", {
            "model": "baseline", "checkpoint_sha256": MODEL_CHECKPOINT_SHA256["baseline"],
            "history_mask_prob": 0.0,
            "checkpoint": str(code_root / "checkpoints/baseline_30000.pt"),
            "checkpoint_step": 30000, "use_ema": True,
            "dino_sha256": hashes["dino/model.safetensors"],
        })
        _save(folder / "timing.json", {"worker_pids": [wrapper], "load_failures": []})
        _save(folder / "renderer.json", {
            "pid": worker,
            "device": {"gpu_id": shard, "uuid": f"GPU-{shard}", "driver_version": "580.126.09"},
        })
        log = folder / "runs/run0/worker0/isaacsim-gpu-physx.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("renderer startup and clean shutdown\n")

    for index, row in enumerate(selection_data["episodes"]):
        shard = index % 8
        output = run / f"shard{shard}" / row["split"] / row["task_id"] / row["episode_id"]
        metadata = output / "scene_metadata.json"
        _save(metadata, {
            "worker_pid": 1000 + shard, "hand_reference": "wrist_link", "hand_reach_m": .2,
            "physics": {
                "device": "cuda:0", "gpu_dynamics_enabled": True, "broadphase": "GPU",
                "fabric_enabled": False, "timestep_s": 1 / 480, "substeps_per_control": 16,
                "global_step_calls_per_tick": 1, "environment_count": 1,
                "gpu_configuration": {
                    "gpu_found_lost_aggregate_pairs_capacity": 32768,
                    "gpu_total_aggregate_pairs_capacity": 32768,
                },
            },
            "rendering": {
                "stage_id": 9223002, "scene_partition": "env_0000", "batch_camera_count": 3,
                "resident_camera_count": 3, "render_products": ["top", "left", "right"],
                "global_captures_per_image_tick": 1,
            },
        })
        np.savez_compressed(output / "final_window.npz", object_pose=poses,
                            wrist_positions=hands, state=np.zeros(54),
                            joint_names=np.asarray([str(i) for i in range(54)]))
        _save(output / "result.json", {
            **row, "method": "gpu_vector", "model": "baseline",
            "checkpoint_sha256": MODEL_CHECKPOINT_SHA256["baseline"], "history_mask_prob": 0.0,
            "checkpoint_step": 30000, "use_ema": True, "planned_index": index,
            "seed": 123 + index // 8, "horizon_ticks": 1800,
            "status": "success", "tick": 30, "inference_count": 4,
            "reason": score["reason"], "termination_reason": "goal_verified", "score": score,
            "worker_pid": 2000 + shard, "physical_environment": str(metadata),
            "scene_id": scene_id,
        })
    return selection, run, assets, code_root


def test_all_36_valid_toss_sources_have_auditable_physical_windows(complete_supplement):
    selection, run, assets, _ = complete_supplement
    audit = audit_run(selection, run, shards=8, assets=assets, model="baseline")
    assert audit["status"] == "passed", audit["failures"]
    assert audit["planned"] == audit["terminal"] == audit["replayed_physical_rollouts"] == 36
    assert audit["known_invalid_raw"] == 0 and audit["status_counts"]["success"] == 36


def test_post_launch_change_to_recorded_aggregate_code_blocks_audit(complete_supplement):
    selection, run, assets, code_root = complete_supplement
    assert audit_run(selection, run, shards=8, assets=assets)["status"] == "passed"
    source = code_root / "aggregate_evaluation.py"
    source.write_bytes(b"tampered aggregate code after launch")
    audited = audit_run(selection, run, shards=8, assets=assets)
    assert audited["status"] == "failed"
    assert any(failure["problem"] == "runtime_source_hash_mismatch"
               and failure["source"] == "aggregate_evaluation.py" for failure in audited["failures"])
