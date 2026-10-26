"""Checkpoint selection and resume identity cannot silently mix trained policies."""
import json
from types import SimpleNamespace

import pytest

import five_task_experiment as experiment
import gpu_rollout as rollout
import infer_recording as models
import rollout_policy


def _args(model="mask10"):
    return SimpleNamespace(model=model, seed=123, selection_method="fixed_count_per_task_split",
                           horizon_ticks=720, retry_errors=False)


def _row():
    return {"split": "val", "task_id": "hang_mug", "episode_id": "episode",
            "source_id": "source", "selection_digest": "digest", "raw_h5": "raw.h5"}


def _result(row, model="mask10"):
    return {**row, "method": rollout.METHOD, "seed": 123, "planned_index": 0,
            "horizon_ticks": rollout.TASK_LIMIT_TICKS[row["task_id"]],
            "checkpoint_step": 30000, "use_ema": True, "status": "success",
            "model": model, "checkpoint_sha256": models.MODEL_CHECKPOINT_SHA256[model],
            "history_mask_prob": models.MODEL_HISTORY_MASK_PROB[model]}


def test_selected_checkpoint_must_match_actual_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "ROOT", tmp_path)
    checkpoint = tmp_path / "checkpoints" / models.MODELS["mask10"]
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"baseline checkpoint renamed to mask10")
    with pytest.raises(ValueError, match="SHA256"):
        models.verify_checkpoint("mask10")

def test_policy_loader_rejects_renamed_weights_before_torch_load(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "ROOT", tmp_path)
    checkpoint = tmp_path / "checkpoints" / models.MODELS["mask10"]
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"wrong bytes")
    monkeypatch.setattr(rollout_policy.torch, "load",
                        lambda *args, **kwargs: pytest.fail("Unverified checkpoint was deserialized"))
    with pytest.raises(ValueError, match="SHA256"):
        rollout_policy.load_policy(SimpleNamespace(model="mask10"), tmp_path)

def test_checkpoint_training_identity_rejects_wrong_mask_step_and_missing_ema(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "sha256_file", lambda path: models.DINO_SHA256)
    metadata = {"global_step": 30000, "train_config": {"spd_history_mask_prob": 0.0},
                "dino_sha256": models.DINO_SHA256, "ema": {"model": {"weight": object()}}}
    with pytest.raises(ValueError, match="mask probability"):
        models.verify_metadata("mask10", metadata, tmp_path / "dino")
    metadata["train_config"]["spd_history_mask_prob"] = 0.1
    metadata["global_step"] = 29999
    with pytest.raises(ValueError, match="step"):
        models.verify_metadata("mask10", metadata, tmp_path / "dino")
    metadata["global_step"] = 30000
    metadata["ema"] = {"model": {}}
    with pytest.raises(ValueError, match="EMA"):
        models.verify_metadata("mask10", metadata, tmp_path / "dino")


def test_mask_result_cannot_resume_with_baseline_identity(tmp_path):
    row = _row()
    folder = tmp_path / row["split"] / row["task_id"] / row["episode_id"]
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(json.dumps(_result(row, model="baseline")))
    with pytest.raises(ValueError, match="identity"):
        rollout._pending([row], tmp_path, {row["source_id"]: 0}, _args())


def test_mask_result_requires_explicit_identity_even_if_other_fields_match(tmp_path):
    row = _row()
    folder = tmp_path / row["split"] / row["task_id"] / row["episode_id"]
    folder.mkdir(parents=True)
    old = _result(row)
    for key in ("model", "checkpoint_sha256", "history_mask_prob"):
        old.pop(key)
    (folder / "result.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match="identity"):
        rollout._pending([row], tmp_path, {row["source_id"]: 0}, _args())


def test_old_baseline_result_remains_resumable(tmp_path):
    row = _row()
    folder = tmp_path / row["split"] / row["task_id"] / row["episode_id"]
    folder.mkdir(parents=True)
    old = _result(row, model="baseline")
    for key in ("model", "checkpoint_sha256", "history_mask_prob"):
        old.pop(key)
    (folder / "result.json").write_text(json.dumps(old))
    assert rollout._pending([row], tmp_path, {row["source_id"]: 0}, _args("baseline")) == []
    old["model"] = "baseline"
    old["checkpoint_sha256"] = "0" * 64
    (folder / "result.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match="identity"):
        rollout._pending([row], tmp_path, {row["source_id"]: 0}, _args("baseline"))
    assert not rollout._matches_session({"model": "mask10"}, {"model": "baseline"})

def test_only_exact_historical_baseline_session_can_resume():
    current = {"model": "baseline", "checkpoint_sha256": models.MODEL_CHECKPOINT_SHA256["baseline"],
               "history_mask_prob": 0.0, "checkpoint": "/root/checkpoints/baseline_30000.pt",
               "seed": 123}
    previous = {key: value for key, value in current.items()
                if key not in ("checkpoint_sha256", "history_mask_prob")}
    assert rollout._matches_session(previous, current)
    assert not rollout._matches_session({**previous, "checkpoint_sha256": "0" * 64}, current)
    assert not rollout._matches_session({**previous, "model": "mask10"}, current)
    assert not rollout._matches_session(previous, {**current, "model": "mask10"})


def test_selected_checkpoint_is_the_only_checkpoint_in_launcher_fingerprint(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / models.MODELS["mask10"]).write_bytes(b"mask")
    (tmp_path / "checkpoints" / models.MODELS["baseline"]).write_bytes(b"baseline")
    for name in experiment._SOURCE_FILES:
        if not name.startswith("checkpoints/"):
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"code")
    fingerprints = experiment._hashes(tmp_path, model="mask10")
    assert fingerprints["checkpoints/mask10_30000.pt"] == experiment._hash_file(
        tmp_path / "checkpoints/mask10_30000.pt")
    assert "checkpoints/baseline_30000.pt" not in fingerprints

def test_launcher_explicitly_passes_selected_model_to_each_shard(tmp_path, monkeypatch):
    called = {}
    monkeypatch.setattr(experiment.subprocess, "run",
                        lambda command, **kwargs: _capture_command(called, command))
    experiment._shard(0, tmp_path, tmp_path / "selection.json", tmp_path,
                      tmp_path / "python", "mask10")
    assert called["command"][called["command"].index("--model") + 1] == "mask10"


def _capture_command(called, command):
    called["command"] = command
    return SimpleNamespace(returncode=0)
