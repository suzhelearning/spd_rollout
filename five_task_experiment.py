#!/usr/bin/env python3
"""Launch the frozen five-task source selection on eight physical GPUs."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import time

from eval_selection import SUPPLEMENT_METHOD, build_supplemental_selection
from infer_recording import MODEL_CHECKPOINT_SHA256, MODEL_HISTORY_MASK_PROB, MODELS
from rollout_policy import save_json

_SOURCE_FILES = (
    "baseline_raw_rollout.py", "eval_selection.py", "gpu_rollout.py",
    "rollout_termination.py", "success_blocks.py", "success_vessels.py",
    "aggregate_evaluation.py", "physx_scene.py", "physx_scene_builder.py",
    "gpu_physx_worker.py", "gpu_physx_world.py", "gpu_rgb_renderer.py",
    "isaac_gpu_client.py", "isaac_physx_client.py", "isaac_render_protocol.py",
    "isaac_headless_renderer.py", "render_initial_scene.py", "training_render.py",
    "infer_recording.py", "rollout_policy.py", "abc/abc_minimal/config.py",
    "abc/abc_minimal/tianji_data.py", "abc/abc_minimal/spd.py",
    "abc/abc_minimal/policy.py", "five_task_selection.py", "supplement_selection.py",
    "five_task_calibration.py", "five_task_experiment.py",
    "five_task_audit.py", "five_task_report.py",
    "checkpoints/baseline_30000.pt", "dino/model.safetensors",
)


def _hash_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _hashes(code: Path, model: str = "baseline") -> dict[str, str]:
    selected = f"checkpoints/{MODELS[model]}"
    files = (selected if name == "checkpoints/baseline_30000.pt" else name
             for name in _SOURCE_FILES)
    return {name: _hash_file(code / name) for name in files}


def _hardware() -> list[dict]:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id",
         "--format=csv,noheader,nounits"], text=True)
    result = []
    for values in csv.reader(io.StringIO(raw)):
        index, name, uuid, memory, driver, pci = (value.strip() for value in values)
        if int(index) < 8:
            result.append({"index": int(index), "name": name, "uuid": uuid,
                           "memory_mib": float(memory), "driver": driver, "pci_bus_id": pci})
    if sorted(row["index"] for row in result) != list(range(8)):
        raise RuntimeError("The eight expected physical GPU indices are not available")
    return result


def _shard(index: int, code: Path, selection: Path, run: Path, python: Path,
           model: str = "baseline") -> dict:
    output = run / f"shard{index}"
    log = run / f"shard{index}.stdout"
    command = [str(python), str(code / "gpu_rollout.py"), "--selection", str(selection),
               "--model", model, "--output", str(output), "--shard-count", "8", "--shard-index", str(index),
               "--batch-size", "8", "--dino-frame-batch-size", "24", "--seed", "123",
               "--diffusion-steps", "10", "--allow-diagnostic", "--cache-dir",
               str(run / "scene-cache" / f"shard{index}")]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(index), SPD_ROLLOUT_GPU_ID=str(index),
                       XDG_RUNTIME_DIR="/run/user/0", PYTHONPATH=str(code))
    started = time.time()
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=code, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT)
    record = {"gpu": index, "exit_code": result.returncode,
              "started_at_epoch": started, "finished_at_epoch": time.time(),
              "elapsed_s": time.time() - started, "log": str(log), "command": command}
    print(json.dumps({"gpu": index, "exit_code": result.returncode,
                      "elapsed_s": record["elapsed_s"], "log": str(log)}), flush=True)
    return record


def _validate_selection(saved: dict) -> None:
    if saved.get("selection_method") == SUPPLEMENT_METHOD:
        if (saved.get("planned") != 36
                or saved.get("by_split") != {"train": 29, "val": 7}
                or saved.get("counts_per_task_split") != {"train": 29, "val": 7}
                or saved.get("tasks") != ["toss_in_bin"]
                or not isinstance(saved.get("base_selection_path"), str)):
            raise ValueError("Only the frozen 29/7 supplemental toss selection may be launched")
        if saved != build_supplemental_selection(Path(saved["base_selection_path"])):
            raise ValueError("Saved supplemental selection manifest differs from the frozen 36")
    elif (saved.get("selection_method") != "fixed_count_per_task_split"
          or saved.get("planned") != 700 or saved.get("by_split") != {"train": 500, "val": 200}):
        raise ValueError("Only the frozen five-task 100/40 or supplemental toss 29/7 selection may be launched")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, default="baseline")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--python", type=Path,
                        default=Path("/root/sunyz/project/abc/.venv/bin/python"))
    args = parser.parse_args()
    code = args.code_root.expanduser().resolve()
    selection = args.selection.expanduser().resolve()
    evidence = args.evidence_root.expanduser().resolve()
    if selection.parent != evidence or not selection.is_file():
        raise ValueError("Frozen selection must be stored at the new experiment evidence root")
    saved = json.loads(selection.read_text())
    _validate_selection(saved)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    before, hardware = _hashes(code, args.model), _hardware()
    checkpoint = f"checkpoints/{MODELS[args.model]}"
    if before[checkpoint] != MODEL_CHECKPOINT_SHA256[args.model]:
        raise ValueError(f"Selected {args.model} checkpoint SHA256 differs from registered weights")
    run = evidence / "gpu-full"
    run.mkdir(exist_ok=False)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_shard, gpu, code, selection, run, args.python, args.model) for gpu in range(8)]
        shards = [future.result() for future in futures]
    manifest = {"started_at": datetime.fromtimestamp(min(s["started_at_epoch"] for s in shards), timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": time.perf_counter() - started,
                "model": args.model, "checkpoint": str(code / checkpoint),
                "checkpoint_sha256": before[checkpoint],
                "history_mask_prob": MODEL_HISTORY_MASK_PROB[args.model],
                "code_root": str(code),
                "selection_digest": saved["selection_digest"], "selection_path": str(selection),
                "selection_sha256": _hash_file(selection), "run_root": str(run.resolve()),
                "source_hashes": before, "source_hashes_after": _hashes(code, args.model),
                "hardware": hardware, "shards": sorted(shards, key=lambda s: s["gpu"])}
    save_json(run / "launch.json", manifest)
    if manifest["source_hashes"] != manifest["source_hashes_after"]:
        raise RuntimeError("Evaluation source or checkpoint changed during the measured run")
    subprocess.run([str(args.python), str(code / "aggregate_evaluation.py"),
                    "--selection", str(selection), "--run-root", str(run),
                    "--shards", "8", "--require-complete"],
                   cwd=code, env={**os.environ, "PYTHONPATH": str(code)}, check=True)
    print("FIVE_TASK_EVALUATION_AGGREGATED", json.dumps(
        {"wall_seconds": manifest["elapsed_s"], "shard_exits": [s["exit_code"] for s in shards]}), flush=True)


if __name__ == "__main__":
    main()
