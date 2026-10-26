"""Safety caps come from unique valid successful source recordings, not fragments."""
from importlib import import_module
import json
from pathlib import Path

import h5py


def _fixture(tmp_path):
    raw = tmp_path / "raw"
    dataset = raw / "pyramid"
    (dataset / "episodes").mkdir(parents=True)
    names = [f"ep_{i}" for i in range(4)]
    (dataset / "dataset.json").write_text(json.dumps(
        {"schema_version": 4, "task_id": "pyramid", "episodes": names}))
    metadata = {"dataset_contract": {"train": {"episodes": []}, "val": {"episodes": []}}}
    for index, seconds in enumerate((10, 20, 30, 40)):
        episode = names[index]
        with h5py.File(dataset / "episodes" / f"{episode}.h5", "w") as source:
            source.attrs.update({"episode_id": episode, "task_id": "pyramid",
                                 "schema_version": 4, "recorded_action": True,
                                 "sample_rate_hz": 60, "success": True})
            source.create_dataset("trajectory/state", shape=(seconds * 60 + 1, 54), dtype="float32")
        split = "train" if index < 2 else "val"
        identity = f"/root/sunyz/data/spd-sim/pre-train/pyramid::{episode}"
        metadata["dataset_contract"][split]["episodes"].extend(
            {"source": {"source_episode": identity}, "path": f"rendered/{episode}_{window}.h5"}
            for window in (0, 1))
    path = tmp_path / "source-metadata.json"
    path.write_text(json.dumps(metadata))
    return path, raw


def _calibrate(path, raw):
    return import_module("five_task_calibration").calibrate_lengths(path, raw, ("pyramid",))


def test_cap_uses_source_duration_at_60hz_then_p95_and_safety_margin(tmp_path):
    path, raw = _fixture(tmp_path)
    item = _calibrate(path, raw)["pyramid"]
    assert item["source_count"] == item["valid_success_demos"] == 4
    assert item["invalid_headers"] == 0
    assert (item["p95_s"], item["limit_seconds"], item["limit_ticks"]) == (38.5, 50, 1500)


def test_invalid_h5_is_not_used_to_calibrate_the_deadline(tmp_path):
    path, raw = _fixture(tmp_path)
    with h5py.File(raw / "pyramid" / "episodes" / "ep_3.h5", "r+") as source:
        source.attrs["schema_version"] = 3
    item = _calibrate(path, raw)["pyramid"]
    assert item["source_count"] == 4 and item["valid_success_demos"] == 3
    assert item["invalid_headers"] == 1
    assert (item["p95_s"], item["limit_seconds"], item["limit_ticks"]) == (29, 40, 1200)
