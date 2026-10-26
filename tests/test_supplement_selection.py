"""Supplemental toss selection stays paired, raw-valid, and separate from historical 700."""
import hashlib
import json
from types import SimpleNamespace

import h5py
import pytest

import eval_selection
import five_task_experiment
import gpu_rollout
import supplement_selection
from eval_selection import FIVE_TASKS, build_supplemental_selection, select_fixed_sources, select_sources


@pytest.fixture
def frozen_source(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    metadata = {"dataset_contract": {split: {"episodes": []} for split in ("train", "val")}}
    for task in FIVE_TASKS:
        dataset = raw / task
        (dataset / "episodes").mkdir(parents=True)
        episodes = []
        for split, count in (("train", 145 if task == "toss_in_bin" else 100),
                             ("val", 52 if task == "toss_in_bin" else 40)):
            for number in range(count):
                episode = f"{split}_{number:03}"
                episodes.append(episode)
                (dataset / "episodes" / f"{episode}.h5").touch()
                metadata["dataset_contract"][split]["episodes"].append({
                    "source": {"source_episode": f"/data/pre-train/{task}::{episode}"}})
        (dataset / "dataset.json").write_text(json.dumps({
            "schema_version": 4, "task_id": task, "episodes": episodes}))
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(metadata))
    seed = "frozen-test-seed"
    baseline_rows = select_fixed_sources(split_path, raw, FIVE_TASKS, seed, 100, 40)
    base = tmp_path / "frozen700.json"
    base.write_text(json.dumps({"model": "baseline_30000.pt", "source_split": str(split_path),
                                "raw_root": str(raw), "selection_method": "fixed_count_per_task_split",
                                "selection_seed": seed, "tasks": list(FIVE_TASKS),
                                "counts_per_task_split": {"train": 100, "val": 40},
                                "by_split": {"train": 500, "val": 200}, "planned": 700,
                                "selection_digest": baseline_rows[0]["selection_digest"],
                                "episodes": baseline_rows}))
    monkeypatch.setattr(eval_selection, "BASE_SELECTION_SHA256",
                        hashlib.sha256(base.read_bytes()).hexdigest())
    original = {row["source_id"] for row in baseline_rows}
    candidates = [row for row in select_sources(split_path, raw, ("toss_in_bin",), seed, fraction=1)
                  if row["source_id"] not in original]
    scene_bytes = b'{"schema_version": 4, "sampled_parameters": {"scene_manifest.sampled_values": {}}}'
    scene_id = hashlib.sha256(scene_bytes).hexdigest()
    scene_path = raw / "toss_in_bin" / "scenes" / f"{scene_id}.json"
    scene_path.parent.mkdir()
    scene_path.write_bytes(scene_bytes)
    for split, quota in (("train", 29), ("val", 7)):
        ordered = [row for row in candidates if row["split"] == split]
        for offset, row in enumerate(ordered):
            with h5py.File(row["raw_h5"], "w") as handle:
                handle.attrs.update(episode_id=row["episode_id"], task_id="toss_in_bin",
                                    schema_version=3 if offset < (16 if split == "train" else 5) else 4,
                                    recorded_action=True, scene_id=scene_id,
                                    task_instruction="toss the object into the bin")
        assert len(ordered) >= quota + (16 if split == "train" else 5)
    return base, original, candidates


def test_supplement_ignores_v3_and_preserves_frozen_source_order(frozen_source):
    base, original, candidates = frozen_source
    selection = build_supplemental_selection(base)
    rows = selection["episodes"]
    expected = ([row["source_id"] for row in candidates if row["split"] == "train"][16:45]
                + [row["source_id"] for row in candidates if row["split"] == "val"][5:12])
    assert [row["source_id"] for row in rows] == expected
    assert len(rows) == len(set(expected)) == 36
    assert not set(expected) & original
    assert selection["planned"] == 36
    assert selection["by_split"] == selection["counts_per_task_split"] == {"train": 29, "val": 7}
    assert selection["selection_method"] == "supplemental_valid_toss"
    assert selection["tasks"] == ["toss_in_bin"]
    assert selection["base_selection_path"] == str(base.resolve())
    assert selection["base_selection_sha256"] == hashlib.sha256(base.read_bytes()).hexdigest()
    assert all(row["selection_digest"] == selection["selection_digest"]
               and len(row["raw_h5_sha256"]) == 64 for row in rows)
    assert all(row["split"] == split for split, segment in
               (("train", rows[:29]), ("val", rows[29:])) for row in segment)
    assert all(row["task_id"] == "toss_in_bin" and row["selection_fraction"] == 1 for row in rows)
    for row in rows:
        with h5py.File(row["raw_h5"], "r") as source:
            assert source.attrs["schema_version"] == 4
    assert selection == build_supplemental_selection(base)


def test_supplement_rejects_exhausted_raw_valid_candidates(frozen_source):
    base, _, candidates = frozen_source
    training = [row for row in candidates if row["split"] == "train"]
    with h5py.File(training[16]["raw_h5"], "r+") as source:
        source.attrs["schema_version"] = 3
    with pytest.raises(ValueError, match="Not enough|raw-valid"):
        build_supplemental_selection(base)


def test_supplement_rejects_modified_frozen_700(frozen_source):
    base, _, _ = frozen_source
    saved = json.loads(base.read_text())
    saved["episodes"][0]["source_id"] = "/data/pre-train/hang_mug::changed"
    base.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="SHA256|frozen|baseline"):
        build_supplemental_selection(base)


def test_supplement_rejects_upstream_changed_valid_h5(frozen_source):
    base, _, _ = frozen_source
    frozen = build_supplemental_selection(base)
    source = frozen["episodes"][0]["raw_h5"]
    with h5py.File(source, "r+") as handle:
        handle.attrs["task_instruction"] = "a different yet valid instruction"
    updated = build_supplemental_selection(base)
    assert updated["selection_digest"] != frozen["selection_digest"]
    assert updated["episodes"][0]["raw_h5_sha256"] != frozen["episodes"][0]["raw_h5_sha256"]
    selection_path = base.parent / "supplement_selection.json"
    selection_path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="selection|manifest|differs"):
        gpu_rollout._selection(SimpleNamespace(selection=selection_path, shard_count=8,
                                               shard_index=0, source_ids=None))


def test_supplement_runner_refuses_tampered_manifest_even_with_same_digest(frozen_source):
    base, original, _ = frozen_source
    frozen = build_supplemental_selection(base)
    frozen["episodes"][0]["source_id"] = next(iter(original))
    selection_path = base.parent / "supplement_selection.json"
    selection_path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="selection|manifest|differs"):
        gpu_rollout._selection(SimpleNamespace(selection=selection_path, shard_count=8,
                                               shard_index=0, source_ids=None))


def test_supplement_uses_goal_limit_and_original_shard_seed(frozen_source, tmp_path):
    base, _, _ = frozen_source
    selection = build_supplemental_selection(base)
    selection_path = tmp_path / "supplement_selection.json"
    selection_path.write_text(json.dumps(selection))
    args = SimpleNamespace(selection=selection_path, shard_count=8, shard_index=0, source_ids=None,
                           selection_method="supplemental_valid_toss", model="mask10", seed=123,
                           horizon_ticks=720, retry_errors=False)
    _, planned, indices, _ = gpu_rollout._selection(args)
    assert [indices[row["source_id"]] for row in planned] == [0, 8, 16, 24, 32]
    runner = gpu_rollout.RolloutRunner(args, tmp_path / "out", planned, indices, tmp_path / "run", {})
    first = runner._prepare(planned[0])
    second = runner._prepare(planned[1])
    assert (first["result"]["seed"], first["result"]["planned_index"]) == (123, 0)
    assert (second["result"]["seed"], second["result"]["planned_index"]) == (124, 8)
    second["result"].update(status="failure", reason="timeout_goal_unmet")
    result_path = second["folder"] / "result.json"
    result_path.write_text(json.dumps(second["result"]))
    assert gpu_rollout._pending([planned[1]], tmp_path / "out", indices, args) == []
    second["result"]["seed"] = 123
    result_path.write_text(json.dumps(second["result"]))
    with pytest.raises(ValueError, match="identity"):
        gpu_rollout._pending([planned[1]], tmp_path / "out", indices, args)
    assert second["result"]["horizon_ticks"] == gpu_rollout.TASK_LIMIT_TICKS["toss_in_bin"] == 1800
    args.batch_size = 8
    args.dino_frame_batch_size = 24
    args.diffusion_steps = 10
    args.allow_diagnostic = True
    for name in ("isaac_python", "assets", "spd_source", "data_process_source", "cache_dir"):
        setattr(args, name, tmp_path / name)
    config = gpu_rollout._session_config(args, selection, planned, None, 0)
    assert config["termination_mode"] == "early_task_goal"
    assert config["task_limit_ticks"]["toss_in_bin"] == 1800
    assert (config["control_hz"], config["physics_substeps"], config["policy_image_stride"],
            config["policy_action_chunk"], config["scoring_window_ticks"]) == (30, 16, 8, 8, 31)


def test_cli_writes_new_selection_without_overwriting_frozen700(frozen_source, monkeypatch):
    base, _, _ = frozen_source
    frozen_bytes = base.read_bytes()
    output = base.parent / "supplement_selection.json"
    monkeypatch.setattr("sys.argv", ["supplement_selection.py", "--base-selection", str(base),
                                     "--output", str(output)])
    supplement_selection.main()
    saved = json.loads(output.read_text())
    assert saved["planned"] == 36
    assert base.read_bytes() == frozen_bytes
    five_task_experiment._validate_selection(saved)
    with pytest.raises(FileExistsError):
        supplement_selection.main()


def test_launcher_rejects_nonfrozen_supplement_before_hardware(tmp_path):
    invalid = {"selection_method": "supplemental_valid_toss", "planned": 35,
               "by_split": {"train": 29, "val": 6}, "episodes": []}
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    selection_path = evidence / "selection.json"
    selection_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="frozen|36|selection"):
        five_task_experiment._validate_selection(invalid)
