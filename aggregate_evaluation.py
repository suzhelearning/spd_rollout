"""Aggregate every fixed selected raw rollout without hiding missing or unscored cases."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from infer_recording import MODEL_CHECKPOINT_SHA256, MODEL_HISTORY_MASK_PROB
from eval_selection import build_supplemental_selection, summarize
from rollout_policy import save_json

_FROZEN_BASE_SELECTION_SHA256 = "2327cdd2c4ef2a786e930fbfcaa58f19f6af82f5ff1ae7c8dbac2d204aacbd5f"


def model_identity(record: dict, *, legacy_baseline: bool = False) -> tuple[str, str, float]:
    """Validate the declared evaluation model; only old baseline records may omit identity fields."""
    model = record.get("model", "baseline" if legacy_baseline else None)
    if model not in MODEL_CHECKPOINT_SHA256:
        raise ValueError(f"Unknown or missing evaluation model: {model!r}")
    expected = (model, MODEL_CHECKPOINT_SHA256[model], MODEL_HISTORY_MASK_PROB[model])
    fields = ("checkpoint_sha256", "history_mask_prob")
    if not any(field in record for field in fields) and legacy_baseline and model == "baseline":
        return expected
    if (record.get("checkpoint_sha256") != expected[1]
            or type(record.get("history_mask_prob")) is not float
            or record["history_mask_prob"] != expected[2]):
        raise ValueError(f"Evaluation model identity mismatch: {model}")
    return expected


def legacy_baseline_identity(record: dict) -> bool:
    return (record.get("model", "baseline") == "baseline"
            and all(field not in record for field in ("checkpoint_sha256", "history_mask_prob")))


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def aggregate(selection_path: Path, run_root: Path, shards: int) -> dict:
    if shards <= 0:
        raise ValueError("Shard count must be positive")
    selection = json.loads(selection_path.read_text())
    method = selection.get("selection_method")
    if method not in (None, "fixed_count_per_task_split", "supplemental_valid_toss"):
        raise ValueError(f"Unknown source selection method: {method!r}")
    if method == "supplemental_valid_toss":
        if (selection.get("tasks") != ["toss_in_bin"]
                or selection.get("counts_per_task_split") != {"train": 29, "val": 7}
                or selection.get("by_split") != {"train": 29, "val": 7}
                or selection.get("planned") != 36
                or Counter((row.get("split"), row.get("task_id")) for row in selection["episodes"])
                != Counter({("train", "toss_in_bin"): 29, ("val", "toss_in_bin"): 7})):
            raise ValueError("Supplemental source selection quota mismatch")
        base_path = selection.get("base_selection_path")
        if (not isinstance(base_path, str) or not Path(base_path).is_absolute()
                or selection.get("base_selection_sha256") != _FROZEN_BASE_SELECTION_SHA256
                or _digest(Path(base_path)) != _FROZEN_BASE_SELECTION_SHA256
                or selection != build_supplemental_selection(Path(base_path))):
            raise ValueError("Supplemental frozen source selection or base SHA mismatch")
    planned = selection["episodes"]
    if len(planned) != selection["planned"] or len(set((row["split"], row["task_id"], row["episode_id"])
                                                   for row in planned)) != len(planned):
        raise ValueError("Selection has missing or duplicated source episodes")
    outcomes, workers = [], []
    result_methods = Counter()
    run_identity = None
    legacy_shards = []
    for index in range(shards):
        shard = run_root / f"shard{index}"
        config_path = shard / "session.json"
        if not config_path.is_file():
            raise ValueError(f"Missing shard session: {config_path}")
        config = json.loads(config_path.read_text())
        expected = sum(i % shards == index for i in range(len(planned)))
        if (config["selection_digest"] != selection["selection_digest"]
                or config["shard_count"] != shards or config["shard_index"] != index
                or config["planned"] != expected):
            raise ValueError(f"Shard {index} does not match the fixed source selection")
        legacy = legacy_baseline_identity(config)
        identity = model_identity(config, legacy_baseline=legacy)
        if run_identity is not None and identity != run_identity:
            raise ValueError(f"Shard {index} evaluation model identity differs from other shards")
        run_identity = identity
        legacy_shards.append(legacy)
        if method == "supplemental_valid_toss" and legacy:
            raise ValueError("Supplemental shard requires explicit model and checkpoint identity")
        renderer_path = shard / "renderer.json"
        workers.append({"shard": index, "planned": expected, "horizon_ticks": config["horizon_ticks"],
                        "gpu": json.loads(renderer_path.read_text()).get("device") if renderer_path.is_file() else None})
    if len({worker["horizon_ticks"] for worker in workers}) != 1:
        raise ValueError("Rollout shards do not use a common physical horizon")
    for index, row in enumerate(planned):
        result_path = (run_root / f"shard{index % shards}" / row["split"] / row["task_id"]
                       / row["episode_id"] / "result.json")
        if not result_path.is_file():
            outcomes.append({**row, "status": "pending"})
            continue
        result = json.loads(result_path.read_text())
        if (any(result.get(field) != row[field] for field in ("split", "task_id", "episode_id", "source_id"))
                or result.get("selection_digest") != selection["selection_digest"]):
            raise ValueError(f"Result does not match its sampled source: {result_path}")
        if model_identity(result, legacy_baseline=legacy_shards[index % shards]) != run_identity:
            raise ValueError(f"Result evaluation model identity differs from shard: {result_path}")
        if selection.get("selection_method") == "supplemental_valid_toss" and legacy_baseline_identity(result):
            raise ValueError(f"Supplemental result lacks explicit model identity: {result_path}")
        result_method = result.get("method", "original_cpu")
        if result_method not in ("original_cpu", "gpu_vector"):
            raise ValueError(f"Invalid result method {result_method!r}: {result_path}")
        if selection.get("selection_method") == "supplemental_valid_toss" and result_method != "gpu_vector":
            raise ValueError(f"Supplemental result requires GPU physical method: {result_path}")
        result_methods[result_method] += 1
        if len(result_methods) > 1:
            raise ValueError("Cannot aggregate mixed CPU/GPU result methods")
        outcomes.append(result)
    detail = summarize(outcomes, planned)
    totals = {}
    for split, tasks in detail.items():
        count = Counter()
        for task in tasks.values():
            for field in ("planned", "success", "failure", "unsupported", "error", "pending", "scored"):
                count[field] += task[field]
        totals[split] = {field: count[field] for field in
                         ("planned", "success", "failure", "unsupported", "error", "pending", "scored")}
        totals[split]["success_rate"] = count["success"] / count["scored"] if count["scored"] else None
        totals[split]["scored_coverage"] = count["scored"] / count["planned"] if count["planned"] else None
    reasons = {split: {task: dict(Counter(result.get("reason", "pending") for result in outcomes
                                            if result["split"] == split and result["task_id"] == task
                                            and result["status"] in ("error", "unsupported")))
                       for task in detail[split]} for split in ("train", "val")}
    root = Path(__file__).resolve().parent
    selection_metadata = ({"selection_method": "fixed_count_per_task_split",
                           "counts_per_task_split": selection["counts_per_task_split"]}
                          if selection.get("selection_method") == "fixed_count_per_task_split"
                          else {"selection_method": "supplemental_valid_toss",
                                "counts_per_task_split": selection["counts_per_task_split"],
                                "base_selection_sha256": selection["base_selection_sha256"]}
                          if selection.get("selection_method") == "supplemental_valid_toss"
                          else {"sample_fraction_per_task_split": selection["fraction"]})
    return {"model": run_identity[0], "checkpoint_sha256": run_identity[1],
            "history_mask_prob": run_identity[2], "checkpoint_step": 30000,
            "method": next(iter(result_methods), None),
            "method_evidence": {"source": "result_records", "counts": dict(result_methods)},
            "selection_digest": selection["selection_digest"],
            "selection_sha256": _digest(selection_path), **selection_metadata,
            "source_split": selection["source_split"], "raw_root": selection["raw_root"],
            "horizon_ticks": workers[0]["horizon_ticks"], "control_hz": 30,
            "counts_by_split": totals, "counts_by_task": detail, "unscored_reasons": reasons,
            "workers": workers, "status": "complete" if all(not values["pending"] for values in totals.values()) else "incomplete",
            "source_hashes": {name: _digest(root / name) for name in
                              ("baseline_raw_rollout.py", "success_blocks.py", "success_vessels.py",
                               "physx_scene.py", "physx_scene_builder.py")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = aggregate(args.selection, args.run_root, args.shards)
    save_json(args.run_root / "report.json", result)
    print(json.dumps({"status": result["status"], "by_split": result["counts_by_split"],
                      "report": str(args.run_root / "report.json")}, ensure_ascii=False), flush=True)
    if args.require_complete and result["status"] != "complete":
        parser.exit(1, "Rollout samples still pending; report is not final\n")


if __name__ == "__main__":
    main()
