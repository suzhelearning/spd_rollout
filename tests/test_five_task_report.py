"""The five-task document must show scored and planned denominators separately."""
from copy import deepcopy
from importlib import import_module

from eval_selection import FIVE_TASKS
from rollout_termination import TASK_LIMIT_TICKS


_DIGEST = "b" * 64


def _evidence():
    selection = {"selection_method": "fixed_count_per_task_split", "selection_digest": _DIGEST,
                 "planned": 700, "tasks": list(FIVE_TASKS),
                 "counts_per_task_split": {"train": 100, "val": 40},
                 "by_split": {"train": 500, "val": 200}}
    tasks = {split: {task: {"planned": planned, "success": 0, "failure": planned,
                            "unsupported": 0, "error": 0, "pending": 0, "scored": planned}
                     for task in FIVE_TASKS}
             for split, planned in (("train", 100), ("val", 40))}
    tasks["train"]["hang_mug"].update(success=5, failure=85, unsupported=10, scored=90)
    tasks["val"]["hang_mug"].update(success=2, failure=28, unsupported=10, scored=30)
    aggregate = {"selection_digest": _DIGEST, "status": "complete", "method": "gpu_vector",
                 "control_hz": 30, "counts_by_task": tasks,
                 "counts_by_split": {
                     "train": {"planned": 500, "success": 5, "failure": 485,
                               "unsupported": 10, "error": 0, "pending": 0, "scored": 490},
                     "val": {"planned": 200, "success": 2, "failure": 188,
                             "unsupported": 10, "error": 0, "pending": 0, "scored": 190}}}
    audit = {"status": "passed", "selection_digest": _DIGEST, "planned": 700,
             "terminal": 700, "status_counts": {"success": 7, "failure": 673,
                                                 "unsupported": 20, "error": 0},
             "counts_by_task": deepcopy(tasks), "counts_by_split": deepcopy(aggregate["counts_by_split"]),
             "run_root": "/experiment/run", "selection_sha256": "c" * 64,
             "failures": [], "known_invalid_raw": 0}
    calibration = {"source_splits": ["train", "val"],
                   "scope": "train_and_validation_source_demonstrations",
                   "tasks": {task: {"limit_ticks": limit} for task, limit in TASK_LIMIT_TICKS.items()}}
    launch = {"elapsed_s": 900, "run_root": "/experiment/run",
              "selection_sha256": "c" * 64,
              "shards": [{"gpu": i, "exit_code": 0} for i in range(8)]}
    return selection, aggregate, audit, calibration, launch


def _summary(evidence):
    return import_module("five_task_report").summarize_evidence(*evidence)


def test_report_preserves_scored_and_planned_success_denominators():
    report = _summary(_evidence())
    assert report["status"] == "final"
    train = report["counts_by_task"]["train"]["hang_mug"]
    val = report["counts_by_task"]["val"]["hang_mug"]
    assert (train["success_per_scored"], train["success_per_planned"], train["scored_coverage"]) == (
        5 / 90, 5 / 100, 90 / 100)
    assert (val["success_per_scored"], val["success_per_planned"]) == (2 / 30, 2 / 40)
    assert (report["total"]["success_per_scored"], report["total"]["success_per_planned"]) == (
        7 / 680, 7 / 700)
    from infer_recording import MODEL_CHECKPOINT_SHA256
    historical = list(_evidence())
    historical[2].update(model="baseline", checkpoint_sha256=MODEL_CHECKPOINT_SHA256["baseline"],
                         history_mask_prob=0.0)
    assert _summary(historical)["status"] == "final"


def test_failed_audit_or_inconsistent_counts_cannot_be_published_as_final():
    evidence = list(_evidence())
    evidence[2]["status"] = "failed"
    evidence[2]["failures"] = [{"problem": "native_runtime_errors"}]
    assert _summary(evidence)["status"] == "incomplete"
    evidence = list(_evidence())
    evidence[2]["status_counts"]["success"] = 8
    assert _summary(evidence)["status"] == "incomplete"
    evidence = list(_evidence())
    evidence[3]["tasks"]["pyramid"]["limit_ticks"] = 200
    assert _summary(evidence)["status"] == "incomplete"


def test_same_global_counts_from_different_task_distribution_or_run_are_not_final():
    evidence = list(_evidence())
    audited = evidence[2]["counts_by_task"]["train"]
    audited["hang_mug"]["success"] = 4
    audited["hang_mug"]["failure"] = 86
    audited["pyramid"]["success"] = 1
    audited["pyramid"]["failure"] = 99
    assert _summary(evidence)["status"] == "incomplete"
    evidence = list(_evidence())
    evidence[2]["run_root"] = "/different/run"
    assert _summary(evidence)["status"] == "incomplete"


def test_report_discloses_approved_train_and_validation_cap_calibration():
    evidence = list(_evidence())
    evidence[3]["source_splits"] = ["train"]
    assert _summary(evidence)["status"] == "incomplete"

def test_mask_report_requires_matching_aggregate_audit_and_launch_identity():
    from infer_recording import MODEL_CHECKPOINT_SHA256
    identity = {"model": "mask10", "checkpoint_sha256": MODEL_CHECKPOINT_SHA256["mask10"],
                "history_mask_prob": 0.1}
    evidence = list(_evidence())
    for record in (evidence[1], evidence[2], evidence[4]):
        record.update(identity)
    report = _summary(evidence)
    assert report["status"] == "final"
    assert all(report[key] == value for key, value in identity.items())
    evidence[4]["checkpoint_sha256"] = "f" * 64
    assert _summary(evidence)["status"] == "incomplete"
    evidence[4].update(identity)
    evidence[2]["history_mask_prob"] = 0.0
    assert _summary(evidence)["status"] == "incomplete"
    evidence[2].update(identity)
    evidence[1]["model"] = "baseline"
    assert _summary(evidence)["status"] == "incomplete"


