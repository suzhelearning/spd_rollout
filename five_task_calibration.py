#!/usr/bin/env python3
"""Calibrate task-specific safety caps from distinct valid source demonstrations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

from eval_selection import FIVE_TASKS, select_sources
from rollout_termination import TASK_LIMIT_TICKS


def calibrate_lengths(metadata_path: Path, raw_root: Path, tasks: tuple[str, ...]) -> dict:
    sources = select_sources(metadata_path, raw_root, tasks, selection_seed="duration-calibration", fraction=1)
    by_task = {task: {"durations": [], "source_count": 0, "invalid_headers": 0,
                      "not_success_labeled": 0} for task in tasks}
    for row in sources:
        entry = by_task[row["task_id"]]
        entry["source_count"] += 1
        with h5py.File(row["raw_h5"], "r") as source:
            if (str(source.attrs.get("episode_id")) != row["episode_id"]
                    or str(source.attrs.get("task_id")) != row["task_id"]
                    or int(source.attrs.get("schema_version", -1)) != 4
                    or not bool(source.attrs.get("recorded_action", False))):
                entry["invalid_headers"] += 1
                continue
            if not bool(source.attrs.get("success", False)):
                entry["not_success_labeled"] += 1
                continue
            rate = float(source.attrs.get("sample_rate_hz", 0))
            frames = source["trajectory/state"].shape[0]
            if not math.isfinite(rate) or rate <= 0 or frames < 2:
                raise ValueError(f"Invalid demonstrated time base for {row['source_id']}")
            entry["durations"].append((frames - 1) / rate)
    result = {}
    for task in sorted(tasks):
        values = by_task[task]
        if not values["durations"]:
            raise ValueError(f"No verifiable successful demonstrations for {task}")
        p95 = float(np.quantile(values["durations"], .95))
        limit_seconds = math.ceil(p95 * 1.25 / 5) * 5
        result[task] = {"source_count": values["source_count"],
                        "valid_success_demos": len(values["durations"]),
                        "invalid_headers": values["invalid_headers"],
                        "not_success_labeled": values["not_success_labeled"],
                        "p95_s": p95, "limit_seconds": limit_seconds,
                        "limit_ticks": limit_seconds * 30}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = args.source_split.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    items = calibrate_lengths(metadata, raw_root, FIVE_TASKS)
    if {task: row["limit_ticks"] for task, row in items.items()} != TASK_LIMIT_TICKS:
        raise ValueError("Demonstration-derived safety caps differ from frozen experiment configuration")
    report = {"method": "valid_source_success_demo_p95_times_1.25_ceil_5s",
              "source_splits": ["train", "val"],
              "scope": "train_and_validation_source_demonstrations",
              "source_split": str(metadata), "source_split_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
              "raw_root": str(raw_root), "tasks": items}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"tasks": items, "output": str(args.output.resolve())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
