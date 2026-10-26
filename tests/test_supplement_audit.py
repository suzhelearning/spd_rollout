"""Supplemental results must pass independently of the historical 700-source audit."""
import hashlib
import json
from pathlib import Path

import pytest

from aggregate_evaluation import aggregate
from five_task_audit import audit_run
from infer_recording import DINO_SHA256, MODEL_CHECKPOINT_SHA256
from test_five_task_audit import _fixture, _save


def _supplement(tmp_path, monkeypatch):
    selection_path, run, result_path, _ = _fixture(tmp_path)
    selection = json.loads(selection_path.read_text())
    selection.update(selection_method="supplemental_valid_toss", tasks=["toss_in_bin"],
                     counts_per_task_split={"train": 29, "val": 7},
                     by_split={"train": 29, "val": 7}, planned=1,
                     base_selection_path=str(tmp_path / "base.json"),
                     base_selection_sha256="a" * 64)
    _save(tmp_path / "base.json", {"episodes": [], "planned": 700})
    _save(selection_path, selection)
    monkeypatch.setattr("five_task_audit.build_supplemental_selection", lambda path: selection)
    return selection_path, run, result_path


def _problems(selection, run):
    return {entry["problem"] for entry in audit_run(
        selection, run, shards=1, assets=Path("/unused"))["failures"]}


def test_supplement_selection_must_be_recomputed_and_base_hash_bound(tmp_path, monkeypatch):
    selection, run, _ = _supplement(tmp_path, monkeypatch)
    problems = _problems(selection, run)
    assert "supplement_base_selection_mismatch" in problems
    saved = json.loads(selection.read_text())
    saved["episodes"][0]["source_id"] += "-forged"
    _save(selection, saved)
    assert "supplement_selection_mismatch" in _problems(selection, run)


def test_supplement_new_raw_error_is_not_historical_expected_error(tmp_path, monkeypatch):
    selection, run, result_path = _supplement(tmp_path, monkeypatch)
    result = json.loads(result_path.read_text())
    result.update(status="error", reason="raw_scene_schema_error")
    _save(result_path, result)
    assert "supplement_result_error" in _problems(selection, run)


def test_supplement_requires_pre_and_post_launch_source_and_weight_hashes(tmp_path, monkeypatch):
    selection, run, _ = _supplement(tmp_path, monkeypatch)
    launch_path = run / "launch.json"
    launch = json.loads(launch_path.read_text())
    launch.update(model="baseline", checkpoint_sha256=MODEL_CHECKPOINT_SHA256["baseline"],
                  history_mask_prob=0.0, code_root=str(tmp_path),
                  checkpoint=str(tmp_path / "checkpoints/baseline_30000.pt"))
    launch["source_hashes"] = launch["source_hashes_after"] = {
        "checkpoints/baseline_30000.pt": MODEL_CHECKPOINT_SHA256["baseline"],
        "dino/model.safetensors": DINO_SHA256}
    _save(launch_path, launch)
    assert "missing_runtime_source_hashes" in _problems(selection, run)


def test_aggregate_rejects_wrong_supplement_quota_and_unknown_selection_method(tmp_path, monkeypatch):
    selection, run, result_path = _supplement(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="quota"):
        aggregate(selection, run, 1)
    saved = json.loads(selection.read_text())
    saved.update(selection_method="unknown")
    _save(selection, saved)
    with pytest.raises(ValueError, match="selection method"):
        aggregate(selection, run, 1)


def test_supplement_result_with_other_model_cannot_be_accepted(tmp_path, monkeypatch):
    selection, run, result_path = _supplement(tmp_path, monkeypatch)
    result = json.loads(result_path.read_text())
    result.update(model="mask10", checkpoint_sha256=MODEL_CHECKPOINT_SHA256["mask10"],
                  history_mask_prob=0.1)
    _save(result_path, result)
    assert "source_seed_or_model_mismatch" in _problems(selection, run)


def test_aggregate_counts_only_29_7_and_rejects_other_model_or_cpu(tmp_path, monkeypatch):
    selection_path, run, _, _ = _fixture(tmp_path)
    base = tmp_path / "base.json"
    _save(base, {"planned": 700})
    base_sha = hashlib.sha256(base.read_bytes()).hexdigest()
    monkeypatch.setattr("aggregate_evaluation._FROZEN_BASE_SELECTION_SHA256", base_sha)
    original = json.loads(selection_path.read_text())["episodes"][0]
    rows = [{**original, "split": "train" if index < 29 else "val",
             "task_id": "toss_in_bin", "episode_id": f"new_{index}",
             "source_id": f"source_{index}", "selection_digest": "z" * 64}
            for index in range(36)]
    selection = {"selection_method": "supplemental_valid_toss", "tasks": ["toss_in_bin"],
                 "counts_per_task_split": {"train": 29, "val": 7},
                 "by_split": {"train": 29, "val": 7}, "planned": 36,
                 "base_selection_path": str(base), "base_selection_sha256": base_sha,
                 "source_split": "/source", "raw_root": "/raw",
                 "selection_digest": "z" * 64, "episodes": rows}
    _save(selection_path, selection)
    monkeypatch.setattr("aggregate_evaluation.build_supplemental_selection", lambda path: selection)
    session = json.loads((run / "shard0" / "session.json").read_text())
    session.update(selection_digest="z" * 64, planned=36, horizon_ticks=1800,
                   model="baseline", checkpoint_sha256=MODEL_CHECKPOINT_SHA256["baseline"],
                   history_mask_prob=0.0)
    _save(run / "shard0" / "session.json", session)
    for row in rows:
        result = {**row, "method": "gpu_vector", "status": "failure", "reason": "timeout_goal_unmet",
                  "model": "baseline", "checkpoint_sha256": MODEL_CHECKPOINT_SHA256["baseline"],
                  "history_mask_prob": 0.0}
        _save(run / "shard0" / row["split"] / row["task_id"] / row["episode_id"] / "result.json", result)
    report = aggregate(selection_path, run, 1)
    assert report["selection_method"] == "supplemental_valid_toss"
    assert report["counts_by_split"]["train"]["planned"] == 29
    assert report["counts_by_split"]["val"]["planned"] == 7
    result_path = run / "shard0" / "val" / "toss_in_bin" / "new_35" / "result.json"
    result = json.loads(result_path.read_text())
    result.update(model="mask10", checkpoint_sha256=MODEL_CHECKPOINT_SHA256["mask10"],
                  history_mask_prob=0.1)
    _save(result_path, result)
    with pytest.raises(ValueError, match="identity"):
        aggregate(selection_path, run, 1)
    result.update(model="baseline", checkpoint_sha256=MODEL_CHECKPOINT_SHA256["baseline"],
                  history_mask_prob=0.0, method="original_cpu")
    _save(result_path, result)
    with pytest.raises(ValueError, match="physical method"):
        aggregate(selection_path, run, 1)


def test_supplement_missing_physical_stage_blocks_pass(tmp_path, monkeypatch):
    selection, run, result_path = _supplement(tmp_path, monkeypatch)
    result = json.loads(result_path.read_text())
    metadata_path = Path(result["physical_environment"])
    metadata = json.loads(metadata_path.read_text())
    metadata["rendering"]["stage_id"] = None
    _save(metadata_path, metadata)
    assert "stage_identity_missing" in _problems(selection, run)

def test_changed_recorded_aggregate_source_cannot_pass_runtime_audit(tmp_path, monkeypatch):
    selection, run, _ = _supplement(tmp_path, monkeypatch)
    source = tmp_path / "aggregate_evaluation.py"
    source.write_bytes(b"aggregate source recorded at launch")
    launch_path = run / "launch.json"
    launch = json.loads(launch_path.read_text())
    launch.update(code_root=str(tmp_path), model="baseline",
                  checkpoint_sha256=MODEL_CHECKPOINT_SHA256["baseline"],
                  history_mask_prob=0.0,
                  checkpoint=str(tmp_path / "checkpoints/baseline_30000.pt"))
    launch["source_hashes"]["aggregate_evaluation.py"] = hashlib.sha256(source.read_bytes()).hexdigest()
    launch["source_hashes_after"] = dict(launch["source_hashes"])
    _save(launch_path, launch)
    source.write_bytes(b"aggregate source changed after launch")
    assert "runtime_source_hash_mismatch" in _problems(selection, run)
