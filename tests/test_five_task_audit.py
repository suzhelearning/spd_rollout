"""A final report must reject false terminal, physics and source evidence."""
from importlib import import_module
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from success_blocks import score as score_blocks


_DIGEST = "a" * 64
_GOOD = np.array(((.5, .03, .70), (.5, -.03, .70), (.5, 0, .74)))


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _fixture(tmp_path):
    raw_root = tmp_path / "raw" / "pyramid"
    scene = {"schema_version": 4, "pose_node_ids": [f"body/letter_{i}" for i in range(3)],
             "sampled_parameters": {
                 "scene_manifest.objects": [
                     {"class_name": "letter_block", "name": f"letter_{i}", "size": [.04] * 3}
                     for i in range(3)],
                 "scene_manifest.table": {"top_z_m": .68,
                                          "x_range_m": [.2, .9], "y_range_m": [-.55, .55]},
                 "scene_manifest.sampled_values": {"appearance": {"objects": {
                     f"letter_{i}": {"letter": letter} for i, letter in enumerate("ABC")}}}}}
    scene_bytes = json.dumps(scene, sort_keys=True).encode()
    scene_id = hashlib.sha256(scene_bytes).hexdigest()
    (raw_root / "scenes").mkdir(parents=True)
    (raw_root / "scenes" / f"{scene_id}.json").write_bytes(scene_bytes)
    h5 = raw_root / "episodes" / "ep_0.h5"
    h5.parent.mkdir()
    with h5py.File(h5, "w") as source:
        source.attrs.update(episode_id="ep_0", task_id="pyramid", schema_version=4,
                            recorded_action=True, scene_id=scene_id)
        source.create_dataset("trajectory/state", data=np.zeros((1, 54)))
    scene.update(_task="pyramid", _frame_dt_s=1 / 30,
                 _hand_reference="wrist_link", _hand_reach_m=.25)
    poses = np.zeros((31, 3, 7))
    poses[..., :3], poses[..., 3] = _GOOD, 1
    hands = np.tile(np.array(((0., 0., 2.), (0., 0., 2.))), (31, 1, 1))
    score = score_blocks(scene, poses, hands, Path("/unused"))
    assert score["status"] == "success" and score["reason"] == "pyramid_stacked"
    row = {"split": "train", "task_id": "pyramid", "episode_id": "ep_0",
           "source_id": "/root/sunyz/data/spd-sim/pre-train/pyramid::ep_0",
           "raw_dataset": str(raw_root), "raw_h5": str(h5), "selection_digest": _DIGEST}
    selection = tmp_path / "selection.json"
    _save(selection, {"selection_digest": _DIGEST, "planned": 1, "episodes": [row]})
    run = tmp_path / "run"
    shard = run / "shard0"
    _save(shard / "session.json", {"selection_digest": _DIGEST, "model": "baseline",
                                   "shard_count": 1, "shard_index": 0, "planned": 1,
                                   "gpu_index": 0, "seed": 123, "control_hz": 30,
                                   "physics_substeps": 16, "policy_image_stride": 8,
                                   "policy_action_chunk": 8, "termination_mode": "early_task_goal",
                                   "task_limit_ticks": {"pyramid": 1500}})
    _save(shard / "timing.json", {"worker_pids": [321], "load_failures": []})
    _save(run / "launch.json", {
        "selection_digest": _DIGEST, "source_hashes": {"gpu_rollout.py": "a" * 64},
        "source_hashes_after": {"gpu_rollout.py": "a" * 64},
        "hardware": [{"index": 0, "uuid": "GPU-fixture", "driver": "580.126.09"}],
        "shards": [{"gpu": 0, "exit_code": 0}]})
    _save(shard / "renderer.json", {
        "pid": 322,
        "device": {"gpu_id": 0, "uuid": "GPU-fixture", "driver_version": "580.126.09"}})
    _save(shard / "model.json", {
        "checkpoint": "/root/sunyz/project/spd-rollout-gpu/checkpoints/baseline_30000.pt",
        "checkpoint_step": 30000, "use_ema": True, "dino_sha256": "a" * 64})
    log = shard / "runs" / "run0" / "worker0" / "isaacsim-gpu-physx.log"
    log.parent.mkdir(parents=True)
    log.write_text("renderer startup and clean shutdown\n")
    folder = shard / "train" / "pyramid" / "ep_0"
    metadata = folder / "scene_metadata.json"
    _save(metadata, {"worker_pid": 322, "hand_reference": "wrist_link", "hand_reach_m": .25,
                     "initial_error": {"robot_joint_rad": 0.},
                     "physics": {"device": "cuda:0", "gpu_dynamics_enabled": True,
                                 "broadphase": "GPU", "fabric_enabled": False,
                                 "timestep_s": 1 / 480, "substeps_per_control": 16,
                                 "global_step_calls_per_tick": 1, "environment_count": 1,
                                 "gpu_configuration": {"gpu_found_lost_aggregate_pairs_capacity": 32768,
                                                       "gpu_total_aggregate_pairs_capacity": 32768}},
                     "rendering": {"stage_id": 1, "scene_partition": "env_0000",
                                   "batch_camera_count": 3, "resident_camera_count": 3,
                                   "render_products": ["top", "left", "right"],
                                   "global_captures_per_image_tick": 1}})
    np.savez_compressed(folder / "final_window.npz", object_pose=poses,
                        wrist_positions=hands, state=np.zeros(54), joint_names=np.array([str(i) for i in range(54)]))
    result_path = folder / "result.json"
    result = {**row, "method": "gpu_vector", "seed": 123, "planned_index": 0,
              "checkpoint_step": 30000, "use_ema": True, "horizon_ticks": 1500,
              "tick": 30, "inference_count": 4, "status": "success",
              "reason": "pyramid_stacked", "termination_reason": "goal_verified",
              "score": score, "worker_pid": 321, "physical_environment": str(metadata)}
    _save(result_path, result)
    return selection, run, result_path, log


def _audit(selection, run):
    return import_module("five_task_audit").audit_run(selection, run, shards=1, assets=Path("/unused"))


def test_real_goal_window_and_one_worker_accept_verified_success(tmp_path):
    selection, run, _, _ = _fixture(tmp_path)
    report = _audit(selection, run)
    assert report["status"] == "passed" and report["failures"] == []
    assert (report["planned"], report["terminal"], report["status_counts"]["success"]) == (1, 1, 1)
    assert report["worker_pids"]["0"] == [322] and report["wrapper_pids"]["0"] == [321]


def test_bad_inference_count_or_task_limit_never_audits_as_final(tmp_path):
    selection, run, result_path, _ = _fixture(tmp_path)
    result = json.loads(result_path.read_text())
    result["inference_count"] = 3
    _save(result_path, result)
    assert "inference_count_mismatch" in {f["problem"] for f in _audit(selection, run)["failures"]}
    result["inference_count"], result["tick"] = 188, 1501
    _save(result_path, result)
    assert "task_limit_exceeded" in {f["problem"] for f in _audit(selection, run)["failures"]}


def test_false_success_or_native_error_never_audits_as_final(tmp_path):
    selection, run, result_path, log = _fixture(tmp_path)
    result = json.loads(result_path.read_text())
    result["score"]["reason"] = "not_really_stacked"
    _save(result_path, result)
    assert "score_evidence_mismatch" in {f["problem"] for f in _audit(selection, run)["failures"]}
    result["score"]["reason"] = "pyramid_stacked"
    _save(result_path, result)
    log.write_text("[Error] [omni.rtx] GPU pagefault\n")
    assert "native_runtime_errors" in {f["problem"] for f in _audit(selection, run)["failures"]}


def test_missing_result_or_worker_restart_cannot_be_final(tmp_path):
    selection, run, result_path, _ = _fixture(tmp_path)
    original = result_path.read_text()
    result_path.unlink()
    assert "missing_result" in {f["problem"] for f in _audit(selection, run)["failures"]}
    result_path.write_text(original)
    timing_path = run / "shard0" / "timing.json"
    timing = json.loads(timing_path.read_text())
    timing["worker_pids"].append(323)
    _save(timing_path, timing)
    assert "worker_not_resident" in {f["problem"] for f in _audit(selection, run)["failures"]}


def test_launch_source_drift_or_wrong_physical_gpu_blocks_final(tmp_path):
    selection, run, _, _ = _fixture(tmp_path)
    launch_path = run / "launch.json"
    launch = json.loads(launch_path.read_text())
    launch["source_hashes_after"]["gpu_rollout.py"] = "b" * 64
    _save(launch_path, launch)
    assert "launch_source_changed" in {f["problem"] for f in _audit(selection, run)["failures"]}
    launch["source_hashes_after"]["gpu_rollout.py"] = "a" * 64
    _save(launch_path, launch)
    renderer_path = run / "shard0" / "renderer.json"
    renderer = json.loads(renderer_path.read_text())
    renderer["device"]["uuid"] = "GPU-different"
    _save(renderer_path, renderer)
    assert "gpu_identity_mismatch" in {f["problem"] for f in _audit(selection, run)["failures"]}


def test_newly_corrupted_h5_cannot_expand_frozen_expected_error_set(tmp_path):
    selection, run, result_path, _ = _fixture(tmp_path)
    experiment_digest = "07600ec4e7805dab3bb08ae4b1df2a2d3870c97a7c9e46c5f7ca7d2b9897dcec"
    data = json.loads(selection.read_text())
    data["selection_digest"] = experiment_digest
    data["episodes"][0]["selection_digest"] = experiment_digest
    _save(selection, data)
    result = json.loads(result_path.read_text())
    result.update(selection_digest=experiment_digest, status="error",
                  reason="ValueError: Raw H5 identity mismatch", tick=0)
    _save(result_path, result)
    for path in (run / "shard0" / "session.json", run / "launch.json"):
        item = json.loads(path.read_text())
        item["selection_digest"] = experiment_digest
        _save(path, item)
    with h5py.File(data["episodes"][0]["raw_h5"], "r+") as source:
        source.attrs["schema_version"] = 3
    assert "frozen_raw_error_set_changed" in {
        finding["problem"] for finding in _audit(selection, run)["failures"]}


def test_loaded_checkpoint_step_must_match_approved_baseline(tmp_path):
    selection, run, _, _ = _fixture(tmp_path)
    model_path = run / "shard0" / "model.json"
    model = json.loads(model_path.read_text())
    model["checkpoint_step"] = 29000
    _save(model_path, model)
    assert "model_checkpoint_mismatch" in {
        finding["problem"] for finding in _audit(selection, run)["failures"]}


def test_prelaunch_checkpoint_hash_must_match_approved_weights(tmp_path):
    selection, run, result_path, _ = _fixture(tmp_path)
    frozen_digest = "07600ec4e7805dab3bb08ae4b1df2a2d3870c97a7c9e46c5f7ca7d2b9897dcec"
    data = json.loads(selection.read_text())
    data["selection_digest"] = frozen_digest
    data["episodes"][0]["selection_digest"] = frozen_digest
    _save(selection, data)
    result = json.loads(result_path.read_text())
    result["selection_digest"] = frozen_digest
    _save(result_path, result)
    session_path = run / "shard0" / "session.json"
    session = json.loads(session_path.read_text())
    session["selection_digest"] = frozen_digest
    _save(session_path, session)
    launch_path = run / "launch.json"
    launch = json.loads(launch_path.read_text())
    launch["selection_digest"] = frozen_digest
    launch["source_hashes"]["checkpoints/baseline_30000.pt"] = "b" * 64
    launch["source_hashes_after"]["checkpoints/baseline_30000.pt"] = "b" * 64
    _save(launch_path, launch)
    assert "approved_checkpoint_mismatch" in {
        finding["problem"] for finding in _audit(selection, run)["failures"]}


def test_result_change_after_audit_invalidates_evidence_digest(tmp_path):
    selection, run, result_path, _ = _fixture(tmp_path)
    checked = _audit(selection, run)
    assert checked["status"] == "passed"
    original_digest = checked["result_evidence_sha256"]
    changed = json.loads(result_path.read_text())
    changed["reason"] = "edited_after_audit"
    _save(result_path, changed)
    current = import_module("five_task_audit").result_evidence_sha256(
        json.loads(selection.read_text()), run, 1)
    assert current != original_digest


def test_frozen_run_without_physics_or_observation_hashes_cannot_pass(tmp_path):
    selection, run, result_path, _ = _fixture(tmp_path)
    frozen = "07600ec4e7805dab3bb08ae4b1df2a2d3870c97a7c9e46c5f7ca7d2b9897dcec"
    for path in (selection, result_path, run / "shard0/session.json", run / "launch.json"):
        value = json.loads(path.read_text())
        value["selection_digest"] = frozen
        if path == selection:
            value["episodes"][0]["selection_digest"] = frozen
        _save(path, value)
    assert "missing_runtime_source_hashes" in {
        finding["problem"] for finding in _audit(selection, run)["failures"]}

def test_mask10_audit_requires_matching_identity_on_every_evidence_record(tmp_path):
    from infer_recording import MODEL_CHECKPOINT_SHA256
    selection, run, result_path, _ = _fixture(tmp_path)
    identity = {"model": "mask10", "checkpoint_sha256": MODEL_CHECKPOINT_SHA256["mask10"],
                "history_mask_prob": 0.1}
    launch_path = run / "launch.json"
    launch = json.loads(launch_path.read_text())
    launch.update(identity, code_root="/root/sunyz/project/spd-rollout-gpu",
                  checkpoint="/root/sunyz/project/spd-rollout-gpu/checkpoints/mask10_30000.pt")
    launch["source_hashes"].pop("checkpoints/baseline_30000.pt", None)
    launch["source_hashes_after"].pop("checkpoints/baseline_30000.pt", None)
    launch["source_hashes"]["checkpoints/mask10_30000.pt"] = identity["checkpoint_sha256"]
    launch["source_hashes_after"]["checkpoints/mask10_30000.pt"] = identity["checkpoint_sha256"]
    _save(launch_path, launch)
    session_path = run / "shard0/session.json"
    session = json.loads(session_path.read_text())
    session.update(identity, checkpoint="/root/sunyz/project/spd-rollout-gpu/checkpoints/mask10_30000.pt")
    _save(session_path, session)
    model_path = run / "shard0/model.json"
    model = json.loads(model_path.read_text())
    model.update(identity, checkpoint=session["checkpoint"],
                 dino_sha256="9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b")
    _save(model_path, model)
    result = json.loads(result_path.read_text())
    result.update(identity)
    _save(result_path, result)
    assert _audit(selection, run)["status"] == "failed"
    assert import_module("five_task_audit").audit_run(
        selection, run, shards=1, assets=Path("/unused"), model="mask10")["status"] == "passed"
    for path in (launch_path, session_path, model_path, result_path):
        current = json.loads(path.read_text())
        current["checkpoint_sha256"] = "f" * 64
        _save(path, current)
        problems = {f["problem"] for f in import_module("five_task_audit").audit_run(
            selection, run, shards=1, assets=Path("/unused"), model="mask10")["failures"]}
        assert problems & {"launch_model_identity_mismatch", "session_model_identity_mismatch",
                           "model_checkpoint_mismatch", "source_seed_or_model_mismatch"}
        current.update(identity)
        _save(path, current)
    for path in (launch_path, session_path, model_path, result_path):
        current = json.loads(path.read_text())
        current["history_mask_prob"] = 0.0
        _save(path, current)
        assert import_module("five_task_audit").audit_run(
            selection, run, shards=1, assets=Path("/unused"), model="mask10")["status"] == "failed"
        current.update(identity)
        _save(path, current)
    session["checkpoint"] = "/root/sunyz/project/spd-rollout-gpu/checkpoints/baseline_30000.pt"
    _save(session_path, session)
    assert "session_model_identity_mismatch" in {f["problem"] for f in
        import_module("five_task_audit").audit_run(
            selection, run, shards=1, assets=Path("/unused"), model="mask10")["failures"]}
    _save(session_path, {**session, "checkpoint": model["checkpoint"]})
    launch["source_hashes"]["checkpoints/mask10_30000.pt"] = "f" * 64
    launch["source_hashes_after"]["checkpoints/mask10_30000.pt"] = "f" * 64
    _save(launch_path, launch)
    assert "approved_checkpoint_mismatch" in {f["problem"] for f in
        import_module("five_task_audit").audit_run(
            selection, run, shards=1, assets=Path("/unused"), model="mask10")["failures"]}


def test_mask10_missing_identity_rejected(tmp_path):
    selection, run, _, _ = _fixture(tmp_path)
    session_path = run / "shard0/session.json"
    session = json.loads(session_path.read_text())
    session["model"] = "mask10"
    _save(session_path, session)
    findings = import_module("five_task_audit").audit_run(
        selection, run, shards=1, assets=Path("/unused"), model="mask10")["failures"]
    assert any(f["problem"] == "session_model_identity_mismatch" for f in findings)
