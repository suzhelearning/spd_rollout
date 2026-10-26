"""Selection operates on source episodes, never render fragments or demonstration endings."""
import json
from pathlib import Path

import pytest
import eval_selection

from eval_selection import select_sources, summarize


TASKS = ("hang_mug", "pyramid")
SOURCE_ROOT = "/root/sunyz/data/spd-sim/pre-train"


def _source(task, episode):
    return f"{SOURCE_ROOT}/{task}::{episode}"


def _fixture(tmp_path):
    raw = tmp_path / "raw"
    metadata = {"dataset_contract": {"train": {"episodes": []}, "val": {"episodes": []}}}
    for task in TASKS:
        dataset = raw / task
        (dataset / "episodes").mkdir(parents=True)
        episode_ids = [f"ep_{letter}" for letter in "abcdefghijk"]
        (dataset / "dataset.json").write_text(json.dumps({"schema_version": 4, "task_id": task,
                                                          "episodes": episode_ids}))
        for episode in episode_ids:
            (dataset / "episodes" / f"{episode}.h5").touch()
        for split, letters in (("train", "abcdef"), ("val", "ghijk")):
            for letter in letters:
                episode = f"ep_{letter}"
                # Two rendering/windows pointing to one original source are one candidate.
                metadata["dataset_contract"][split]["episodes"].extend(
                    {"source": {"source_episode": _source(task, episode)},
                     "path": f"rendered/{task}/{episode}_{fragment}.h5"}
                    for fragment in (0, 1)
                )
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    return path, raw, metadata


def test_select_sources_uses_source_groups_and_ceil_per_task_split(tmp_path):
    path, raw, _ = _fixture(tmp_path)
    selected = select_sources(path, raw, TASKS, selection_seed="fixture")
    assert [(row["split"], row["task_id"], row["episode_id"]) for row in selected] == [
        ("train", "hang_mug", "ep_f"), ("train", "hang_mug", "ep_c"),
        ("train", "pyramid", "ep_e"), ("train", "pyramid", "ep_b"),
        ("val", "hang_mug", "ep_h"), ("val", "pyramid", "ep_j"),
    ]
    assert all(row["raw_dataset"] == str(raw / row["task_id"]) for row in selected)
    assert all(row["raw_h5"] == str(raw / row["task_id"] / "episodes" / f'{row["episode_id"]}.h5')
               for row in selected)
    assert all(row["source_id"] == _source(row["task_id"], row["episode_id"])
               for row in selected)
    assert len({row["selection_digest"] for row in selected}) == 1
    assert len(selected[0]["selection_digest"]) == 64
    assert selected == select_sources(path, raw, tuple(reversed(TASKS)), selection_seed="fixture")


def test_fixed_quota_selects_distinct_sources_per_task_and_split(tmp_path):
    path, raw, metadata = _fixture(tmp_path)
    selected = eval_selection.select_fixed_sources(path, raw, TASKS, "fixture", 4, 2)
    assert [(row["split"], row["task_id"], row["episode_id"]) for row in selected] == [
        ("train", "hang_mug", "ep_f"), ("train", "hang_mug", "ep_c"),
        ("train", "hang_mug", "ep_d"), ("train", "hang_mug", "ep_b"),
        ("train", "pyramid", "ep_e"), ("train", "pyramid", "ep_b"),
        ("train", "pyramid", "ep_d"), ("train", "pyramid", "ep_f"),
        ("val", "hang_mug", "ep_h"), ("val", "hang_mug", "ep_g"),
        ("val", "pyramid", "ep_j"), ("val", "pyramid", "ep_h"),
    ]
    assert all(row["source_count"] == (6 if row["split"] == "train" else 5)
               for row in selected)
    assert all("selection_fraction" not in row for row in selected)
    assert {row["selection_digest"] for row in selected} != {
        select_sources(path, raw, TASKS, selection_seed="fixture", fraction=1)[0]["selection_digest"]}
    metadata["dataset_contract"]["train"]["episodes"].reverse()
    metadata["dataset_contract"]["val"]["episodes"].extend(
        [{"source": {"source_episode": _source("pyramid", "ep_g")}}] * 4)
    path.write_text(json.dumps(metadata))
    assert selected == eval_selection.select_fixed_sources(path, raw, tuple(reversed(TASKS)), "fixture", 4, 2)


def test_fixed_quota_rejects_short_or_overlapping_source_groups(tmp_path):
    path, raw, metadata = _fixture(tmp_path)
    with pytest.raises(ValueError, match="train|quota|available"):
        eval_selection.select_fixed_sources(path, raw, TASKS, "fixture", 7, 2)
    metadata["dataset_contract"]["val"]["episodes"].append(
        {"source": {"source_episode": _source("hang_mug", "ep_a")}})
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="train.*val|overlap"):
        eval_selection.select_fixed_sources(path, raw, TASKS, "fixture", 4, 2)


def test_select_sources_is_independent_of_render_fragment_count_and_order(tmp_path):
    path, raw, metadata = _fixture(tmp_path)
    before = select_sources(path, raw, TASKS, selection_seed="fixture")
    metadata["dataset_contract"]["train"]["episodes"].reverse()
    metadata["dataset_contract"]["train"]["episodes"].extend(
        [{"source": {"source_episode": _source("hang_mug", "ep_a")}}] * 20)
    path.write_text(json.dumps(metadata))
    assert before == select_sources(path, raw, TASKS, selection_seed="fixture")


def test_select_sources_rejects_overlap_even_if_not_selected(tmp_path):
    path, raw, metadata = _fixture(tmp_path)
    metadata["dataset_contract"]["val"]["episodes"].append(
        {"source": {"source_episode": _source("hang_mug", "ep_a")}})
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="train.*val|overlap"):
        select_sources(path, raw, TASKS)


@pytest.mark.parametrize("bad_source", [
    f"{SOURCE_ROOT}/hang_mug::../ep_x",
    f"{SOURCE_ROOT}/../pyramid::ep_x",
    "pre-train/hang_mug::ep_x/../../bad",
    f"{SOURCE_ROOT}/hang_mug::ep_x::extra",
])
def test_select_sources_rejects_unsafe_identity(tmp_path, bad_source):
    path, raw, metadata = _fixture(tmp_path)
    metadata["dataset_contract"]["train"]["episodes"][0]["source"]["source_episode"] = bad_source
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        select_sources(path, raw, TASKS)


@pytest.mark.parametrize("change", ["wrong_task", "missing_index", "missing_h5", "missing_split"])
def test_select_sources_fails_closed_on_raw_or_contract_mismatch(tmp_path, change):
    path, raw, metadata = _fixture(tmp_path)
    if change == "wrong_task":
        dataset = raw / "hang_mug" / "dataset.json"
        content = json.loads(dataset.read_text())
        content["task_id"] = "pyramid"
        dataset.write_text(json.dumps(content))
    elif change == "missing_index":
        dataset = raw / "hang_mug" / "dataset.json"
        content = json.loads(dataset.read_text())
        content["episodes"].remove("ep_f")
        dataset.write_text(json.dumps(content))
    elif change == "missing_h5":
        (raw / "hang_mug" / "episodes" / "ep_f.h5").unlink()
    else:
        del metadata["dataset_contract"]["val"]
        path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        select_sources(path, raw, TASKS, selection_seed="fixture")


def test_select_sources_rejects_non_six_task_and_invalid_fraction(tmp_path):
    path, raw, _ = _fixture(tmp_path)
    with pytest.raises(ValueError):
        select_sources(path, raw, ("../hang_mug",))
    with pytest.raises(ValueError):
        select_sources(path, raw, TASKS, fraction=0)


def _plan():
    return [{"split": split, "task_id": task, "episode_id": episode}
            for split, task, episodes in (("train", "hang_mug", ("a", "b", "c", "d", "e")),
                                          ("val", "pyramid", ("f", "g")))
            for episode in episodes]


def test_summarize_keeps_unsupported_errors_and_pending_out_of_scored_rate():
    plan = _plan()
    statuses = ("success", "failure", "unsupported", "error", "pending", "unsupported", "error")
    results = [{**row, "status": status} for row, status in zip(plan, statuses, strict=True)]
    summary = summarize(results, plan)
    assert summary["train"]["hang_mug"] == {
        "planned": 5, "success": 1, "failure": 1, "unsupported": 1, "error": 1,
        "pending": 1, "scored": 2, "success_rate": 0.5,
    }
    assert summary["val"]["pyramid"] == {
        "planned": 2, "success": 0, "failure": 0, "unsupported": 1, "error": 1,
        "pending": 0, "scored": 0, "success_rate": None,
    }


def test_summarize_rejects_unreported_missing_result_and_duplicate_identity():
    plan = _plan()
    with pytest.raises(ValueError, match="pending|missing"):
        summarize([{**row, "status": "success"} for row in plan[:-1]], plan)
    with pytest.raises(ValueError, match="(?i)duplicate"):
        summarize([{**row, "status": "success"} for row in plan] +
                  [{**plan[0], "status": "error"}], plan)


@pytest.mark.parametrize("invalid", ["unknown", "success_rate", None])
def test_summarize_rejects_invalid_status(invalid):
    plan = _plan()
    results = [{**row, "status": "failure"} for row in plan]
    results[0]["status"] = invalid
    with pytest.raises(ValueError):
        summarize(results, plan)


def test_summarize_rejects_unplanned_identity_and_repeated_planning():
    plan = _plan()
    with pytest.raises(ValueError):
        summarize([{**row, "status": "success"} for row in plan] +
                  [{"split": "val", "task_id": "hang_mug", "episode_id": "else", "status": "error"}], plan)
    with pytest.raises(ValueError):
        summarize([], plan + [plan[0]])

def test_aggregate_uses_session_model_and_rejects_mixed_result_identity(tmp_path):
    from aggregate_evaluation import aggregate
    from infer_recording import MODEL_CHECKPOINT_SHA256
    selection = tmp_path / "selection.json"
    row = {"split": "train", "task_id": "pyramid", "episode_id": "ep_0", "source_id": "source"}
    selection.write_text(json.dumps({
        "episodes": [row], "planned": 1, "selection_digest": "d" * 64,
        "selection_method": "fixed_count_per_task_split",
        "counts_per_task_split": {"train": 1, "val": 0}, "source_split": "/split",
        "raw_root": "/raw"}))
    run = tmp_path / "run"
    session_path = run / "shard0/session.json"
    session_path.parent.mkdir(parents=True)
    identity = {"model": "mask10", "checkpoint_sha256": MODEL_CHECKPOINT_SHA256["mask10"],
                "history_mask_prob": 0.1}
    session_path.write_text(json.dumps({**identity, "selection_digest": "d" * 64,
                                        "shard_count": 1, "shard_index": 0, "planned": 1,
                                        "horizon_ticks": 1500}))
    result_path = run / "shard0/train/pyramid/ep_0/result.json"
    result_path.parent.mkdir(parents=True)
    outcome = {**row, **identity, "selection_digest": "d" * 64, "method": "gpu_vector",
               "status": "failure"}
    result_path.write_text(json.dumps(outcome))
    report = aggregate(selection, run, 1)
    assert (report["model"], report["checkpoint_sha256"], report["history_mask_prob"]) == (
        "mask10", MODEL_CHECKPOINT_SHA256["mask10"], 0.1)
    outcome["model"] = "baseline"
    result_path.write_text(json.dumps(outcome))
    with pytest.raises(ValueError, match="model|identity|checkpoint"):
        aggregate(selection, run, 1)
    outcome.update(identity)
    outcome.pop("history_mask_prob")
    result_path.write_text(json.dumps(outcome))
    with pytest.raises(ValueError, match="model|identity|mask"):
        aggregate(selection, run, 1)


def test_aggregate_accepts_historical_baseline_without_new_identity_fields(tmp_path):
    from aggregate_evaluation import aggregate
    selection = tmp_path / "selection.json"
    row = {"split": "train", "task_id": "pyramid", "episode_id": "ep_0", "source_id": "source"}
    selection.write_text(json.dumps({
        "episodes": [row], "planned": 1, "selection_digest": "d" * 64,
        "selection_method": "fixed_count_per_task_split",
        "counts_per_task_split": {"train": 1, "val": 0}, "source_split": "/split",
        "raw_root": "/raw"}))
    run = tmp_path / "run"
    session_path = run / "shard0/session.json"
    session_path.parent.mkdir(parents=True)
    session_path.write_text(json.dumps({"model": "baseline", "selection_digest": "d" * 64,
                                        "shard_count": 1, "shard_index": 0, "planned": 1,
                                        "horizon_ticks": 1500}))
    result_path = run / "shard0/train/pyramid/ep_0/result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps({**row, "selection_digest": "d" * 64,
                                       "method": "gpu_vector", "status": "failure"}))
    report = aggregate(selection, run, 1)
    assert report["model"] == "baseline"
    assert report["checkpoint_sha256"] != "mask10"
