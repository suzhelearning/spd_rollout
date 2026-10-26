"""A comparison may only combine immutable audited evidence, not self-declared totals."""
import hashlib
import json
from pathlib import Path

import pytest

from eval_selection import FIVE_TASKS
from five_task_audit import result_evidence_sha256
from infer_recording import MODEL_CHECKPOINT_SHA256, MODEL_HISTORY_MASK_PROB
from supplement_compare import compare


def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _counts(planned, success=0, error=0):
    return dict(planned=planned, success=success, failure=planned - success - error,
                unsupported=0, error=error, pending=0, scored=planned - error)


def _fixture(tmp_path, monkeypatch):
    old_tasks = {split: {task: _counts(quota, error=bad if task == "toss_in_bin" else 0)
                         for task in FIVE_TASKS}
                 for split, quota, bad in (("train", 100, 29), ("val", 40, 7))}
    old_splits = {"train": _counts(500, error=29), "val": _counts(200, error=7)}
    base_rows = [
        {"split": split, "task_id": task, "episode_id": f"old_{split}_{task}_{index}",
         "source_id": f"old_{split}_{task}_{index}"}
        for split, quota in (("train", 100), ("val", 40))
        for task in FIVE_TASKS for index in range(quota)
    ]
    base = tmp_path / "base.json"
    _save(base, dict(selection_digest="old", planned=700, episodes=base_rows))
    base_sha = _sha(base)
    monkeypatch.setattr("supplement_compare._FROZEN_BASE_SELECTION_SHA256", base_sha)
    selection = tmp_path / "supplement.json"
    rows = [dict(split=split, task_id="toss_in_bin", episode_id=f"new_{i}",
                 source_id=f"source_{i}", raw_h5=f"raw_{i}", selection_digest="s" * 64)
            for i, split in enumerate(["train"] * 29 + ["val"] * 7)]
    _save(selection, dict(selection_method="supplemental_valid_toss", episodes=rows,
                          selection_digest="s" * 64, base_selection_path=str(base),
                          base_selection_sha256=base_sha, planned=36))
    monkeypatch.setattr("supplement_compare.build_supplemental_selection",
                        lambda base_path: json.loads(selection.read_text()))
    paths = {}
    for model in ("baseline", "mask10"):
        old_root = tmp_path / model / "historical"
        new_root = tmp_path / model / "supplement"
        historical_evidence_sha256 = result_evidence_sha256(
            json.loads(base.read_text()), old_root, 8)
        identity = dict(model=model, checkpoint_sha256=MODEL_CHECKPOINT_SHA256[model],
                        history_mask_prob=MODEL_HISTORY_MASK_PROB[model])
        old_aggregate_path = old_root / "aggregate.json"
        _save(old_aggregate_path, {**(identity if model == "mask10" else {"model": "baseline"}), "status": "complete", "method": "gpu_vector", "control_hz": 30,
                                   "selection_digest": "old", "selection_sha256": base_sha,
                                   "counts_by_task": old_tasks, "counts_by_split": old_splits})
        old_audit_path = old_root / "audit.json"
        _save(old_audit_path, {**(identity if model == "mask10" else {}), "status": "passed", "failures": [],
                               "planned": 700, "terminal": 700, "known_invalid_raw": 36,
                               "counts_by_split": old_splits, "selection_sha256": base_sha,
                               "selection_digest": "old", "run_root": str(old_root),
                               "aggregate_sha256": _sha(old_aggregate_path),
                               "result_evidence_sha256": historical_evidence_sha256})
        old_final_path = old_root / "final.json"
        _save(old_final_path, {**(identity if model == "mask10" else {}), "status": "final", "issues": [], "selection_digest": "old",
                               "generated_at": "2026-10-25T00:00:00+00:00",
                               "total": _counts(700, error=36),
                               "counts_by_split": old_splits, "counts_by_task": old_tasks,
                               "inputs": {"aggregate": {"sha256": _sha(old_aggregate_path)},
                                          "audit": {"sha256": _sha(old_audit_path)}}})
        new_splits = {"train": _counts(29, 8), "val": _counts(7, 2)}
        new_tasks = {split: {"toss_in_bin": data} for split, data in new_splits.items()}
        for i, row in enumerate(rows):
            status = "success" if i < 8 or i in (29, 30) else "failure"
            _save(new_root / f"shard{i % 8}" / row["split"] / "toss_in_bin" / row["episode_id"] / "result.json",
                  {**identity, **row, "method": "gpu_vector", "status": status})
        aggregate_path = new_root / "report.json"
        _save(aggregate_path, {**identity, "status": "complete", "selection_method": "supplemental_valid_toss",
                               "method": "gpu_vector", "control_hz": 30, "base_selection_sha256": base_sha,
                               "selection_sha256": _sha(selection), "selection_digest": "s" * 64,
                               "counts_by_split": new_splits, "counts_by_task": new_tasks})
        launch_path = new_root / "launch.json"
        _save(launch_path, {**identity, "selection_sha256": _sha(selection),
                            "selection_digest": "s" * 64, "run_root": str(new_root),
                            "started_at": "2026-10-26T00:00:00+00:00",
                            "finished_at": "2026-10-26T01:00:00+00:00",
                            "source_hashes": {"gpu_rollout.py": "a" * 64},
                            "source_hashes_after": {"gpu_rollout.py": "a" * 64}})
        audit_path = new_root / "audit.json"
        _save(audit_path, {**identity, "status": "passed", "failures": [], "planned": 36,
                          "terminal": 36, "known_invalid_raw": 0,
                          "status_counts": {"success": 10, "failure": 26},
                          "selection_sha256": _sha(selection), "selection_digest": "s" * 64,
                          "counts_by_split": new_splits, "counts_by_task": new_tasks,
                          "run_root": str(new_root), "aggregate_sha256": _sha(aggregate_path),
                          "launch_sha256": _sha(launch_path),
                          "result_evidence_sha256": result_evidence_sha256(json.loads(selection.read_text()), new_root, 8)})
        paths[model] = dict(old_final=old_final_path, old_audit=old_audit_path,
                            old_aggregate=old_aggregate_path, new_aggregate=aggregate_path,
                            new_audit=audit_path, new_launch=launch_path)
    return base, selection, paths


def test_keeps_raw_and_valid_denominators_and_source_paired_transitions(tmp_path, monkeypatch):
    base, selection, paths = _fixture(tmp_path, monkeypatch)
    report = compare(base, selection, paths["baseline"], paths["mask10"])
    assert report["status"] == "final"
    for model in ("baseline", "mask10"):
        totals = report["models"][model]["totals"]
        assert totals["historical_700"]["planned"] == 700
        assert totals["supplement_36"]["planned"] == 36
        assert totals["raw_736_including_original_h5_errors"]["planned"] == 736
        assert totals["valid_h5_700_excluding_original_h5_errors"]["planned"] == 700
        assert totals["valid_h5_700_excluding_original_h5_errors"]["scored"] == 700
    assert report["paired_transitions"]["train"] == {"success → success": 8, "failure → failure": 21}


def test_failed_new_audit_never_produces_final_comparison(tmp_path, monkeypatch):
    base, selection, paths = _fixture(tmp_path, monkeypatch)
    audit = paths["mask10"]["new_audit"]
    data = json.loads(audit.read_text())
    data["status"] = "failed"
    _save(audit, data)
    with pytest.raises(ValueError, match="audit"):
        compare(base, selection, paths["baseline"], paths["mask10"])


def test_new_error_even_under_passed_audit_is_rejected(tmp_path, monkeypatch):
    base, selection, paths = _fixture(tmp_path, monkeypatch)
    report = paths["baseline"]["new_aggregate"]
    data = json.loads(report.read_text())
    data["counts_by_split"]["train"] = _counts(29, 8, error=1)
    _save(report, data)
    with pytest.raises(ValueError, match="provenance|error|counts"):
        compare(base, selection, paths["baseline"], paths["mask10"])


def test_wrong_supplement_model_or_missing_launch_hashes_cannot_be_final(tmp_path, monkeypatch):
    base, selection, paths = _fixture(tmp_path, monkeypatch)
    launch_path = paths["mask10"]["new_launch"]
    launch = json.loads(launch_path.read_text())
    launch["model"] = "baseline"
    _save(launch_path, launch)
    with pytest.raises(ValueError, match="model|provenance"):
        compare(base, selection, paths["baseline"], paths["mask10"])
    launch["model"] = "mask10"
    launch["source_hashes"] = launch["source_hashes_after"] = {}
    _save(launch_path, launch)
    with pytest.raises(ValueError, match="provenance"):
        compare(base, selection, paths["baseline"], paths["mask10"])


def test_base_selection_sha_drift_blocks_comparison(tmp_path, monkeypatch):
    base, selection, paths = _fixture(tmp_path, monkeypatch)
    saved = json.loads(base.read_text())
    saved["episodes"].append({"source_id": "unplanned"})
    _save(base, saved)
    with pytest.raises(ValueError, match="base selection SHA"):
        compare(base, selection, paths["baseline"], paths["mask10"])


def test_scored_denominator_may_differ_on_same_valid_sources(tmp_path, monkeypatch):
    base, selection, paths = _fixture(tmp_path, monkeypatch)
    target = paths["mask10"]["new_aggregate"].parent
    result_path = target / "shard0" / "train" / "toss_in_bin" / "new_0" / "result.json"
    result = json.loads(result_path.read_text())
    result["status"] = "unsupported"
    _save(result_path, result)
    report_path = paths["mask10"]["new_aggregate"]
    report = json.loads(report_path.read_text())
    train = dict(_counts(29, 7), failure=21, unsupported=1, scored=28)
    report["counts_by_split"]["train"] = train
    report["counts_by_task"]["train"]["toss_in_bin"] = train
    _save(report_path, report)
    audit_path = paths["mask10"]["new_audit"]
    audit = json.loads(audit_path.read_text())
    audit["counts_by_split"]["train"] = train
    audit["counts_by_task"]["train"]["toss_in_bin"] = train
    audit["status_counts"] = {"success": 9, "failure": 26, "unsupported": 1}
    audit["aggregate_sha256"] = _sha(report_path)
    audit["result_evidence_sha256"] = result_evidence_sha256(
        json.loads(selection.read_text()), target, 8)
    _save(audit_path, audit)
    comparison = compare(base, selection, paths["baseline"], paths["mask10"])
    assert comparison["models"]["baseline"]["totals"]["valid_h5_700_excluding_original_h5_errors"]["scored"] == 700
    assert comparison["models"]["mask10"]["totals"]["valid_h5_700_excluding_original_h5_errors"]["scored"] == 699
    assert comparison["paired_transitions"]["train"]["success → unsupported"] == 1

def test_historical_result_changed_after_audit_cannot_be_reported_as_final(tmp_path, monkeypatch):
    base, selection, paths = _fixture(tmp_path, monkeypatch)
    first = json.loads(base.read_text())["episodes"][0]
    old_root = paths["baseline"]["old_audit"].parent
    result_path = old_root / "shard0" / first["split"] / first["task_id"] / first["episode_id"] / "result.json"
    _save(result_path, {"source_id": first["source_id"], "status": "failure"})
    with pytest.raises(ValueError, match="historical.*evidence|provenance"):
        compare(base, selection, paths["baseline"], paths["mask10"])
