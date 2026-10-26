"""Fixed-selection GPU vector rollout with shared weights and independent source RNGs."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback
import uuid

import numpy as np
import torch
from abc_minimal.spd import SPDLayerKV, SPDObservationCache

from baseline_raw_rollout import (_CONTROL_HZ, _SELECTION, _TASKS, _commit_result,
                                  _observation, _persist_summary, _raw_scene, _score)
from eval_selection import (FIVE_TASKS, SUPPLEMENT_METHOD, build_supplemental_selection,
                            select_fixed_sources, select_sources)
from infer_recording import MODEL_CHECKPOINT_SHA256, MODEL_HISTORY_MASK_PROB, MODELS, ROOT, verify_checkpoint
from isaac_gpu_client import GPUWorkerError, IsaacGpuClient
from isaac_physx_client import UnsupportedRawSceneError
from rollout_policy import load_policy, save_json
from rollout_termination import TASK_LIMIT_TICKS, TaskTermination
from training_render import ISAAC_PYTHON, TRAINING_SOURCE

METHOD = "gpu_vector"
_GOAL_DRIVEN_SELECTIONS = frozenset(("fixed_count_per_task_split", SUPPLEMENT_METHOD))



class DeviceTelemetry:
    """Sample whole-device memory (policy and Isaac) without changing CUDA state."""

    def __init__(self, gpu_index: int):
        self.gpu_index = gpu_index
        self.samples = []
        self.errors = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, name="gpu-rollout-telemetry", daemon=True)

    def _sample(self):
        while not self.stop_event.is_set():
            try:
                completed = subprocess.run(
                    ["nvidia-smi", f"--id={self.gpu_index}",
                     "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=5)
                memory, utilization = (float(value.strip()) for value in completed.stdout.strip().split(","))
                self.samples.append({"monotonic_seconds": time.monotonic(), "memory_used_mib": memory,
                                     "utilization_percent": utilization})
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                if message not in self.errors:
                    self.errors.append(message)
            self.stop_event.wait(1)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=6)
        return {"peak_vram_mib": max((s["memory_used_mib"] for s in self.samples), default=None),
                "mean_gpu_utilization_percent": (sum(s["utilization_percent"] for s in self.samples)
                                                 / len(self.samples) if self.samples else None),
                "vram_scope": "whole_physical_gpu", "sample_interval_seconds": 1,
                "telemetry_errors": self.errors, "samples": self.samples}


class BatchedPolicy:
    """Use the existing B54/BCHW APIs; RNG streams belong to source seeds."""

    def __init__(self, policy, names: list[str]):
        self.policy, self.names = policy, names
        self.generators = []
        self.noise = None
        self.to_policy = []
        self.to_physx = []

    def reset(self, joints: list[list[str]], seeds: list[int]):
        self.to_policy, self.to_physx = [], []
        for names in joints:
            if len(names) != len(set(names)) or set(names) != set(self.names):
                raise ValueError("PhysX joint identities differ from the trained policy")
            self.to_policy.append([names.index(name) for name in self.names])
            self.to_physx.append([self.names.index(name) for name in names])
        self.generators = [torch.Generator(device=self.policy.device).manual_seed(seed) for seed in seeds]
        self.noise = torch.empty((len(seeds), 8, len(self.names)), device=self.policy.device, dtype=torch.float32)
        self.policy.reset()

    def observe(self, frames: list[dict], tick: int, active: list[bool] | None = None):
        observations = [_observation(frame, tick, order)
                        for frame, order in zip(frames, self.to_policy, strict=True)]
        observation = {key: np.stack([item[key] for item in observations])
                       for key in ("state", "previous_actions")}
        if tick % 8 == 0:
            observation["images"] = {camera: np.stack([item["images"][camera] for item in observations])
                                     for camera in ("top", "left_wrist", "right_wrist")}
            if active is not None:
                if len(active) != len(frames):
                    raise ValueError("Active source mask must match the resident batch")
                observation["camera_validity"] = np.tile(np.asarray(active, dtype=bool)[:, None], (1, 3))
        self.policy.observe(observation, step=tick)

    def infer(self, active: list[bool] | None = None):
        indices = (list(range(len(self.generators))) if active is None
                   else [i for i, enabled in enumerate(active) if enabled])
        if not indices or (active is not None and len(active) != len(self.generators)):
            raise ValueError("Cannot infer an empty or mismatched active source batch")
        for offset, index in enumerate(indices):
            torch.randn(self.noise[offset].shape, device=self.policy.device, dtype=torch.float32,
                        generator=self.generators[index], out=self.noise[offset])
        if len(indices) == len(self.generators):
            actions = self.policy.infer(noise=self.noise)
        else:
            if self.policy.hand_delta:
                raise ValueError("Active-source inference requires absolute command semantics")
            original = self.policy.cache
            selected = torch.as_tensor(indices, device=self.policy.device, dtype=torch.long)
            self.policy.cache = SPDObservationCache(tuple(
                SPDLayerKV(layer.key.index_select(0, selected),
                           layer.value.index_select(0, selected),
                           layer.times, layer.validity.index_select(0, selected))
                for layer in original.layers), original.last_step)
            try:
                actions = self.policy.infer(noise=self.noise[:len(indices)])
            finally:
                self.policy.cache = original
        if actions.shape != (len(indices), 8, len(self.names)) or not np.isfinite(actions).all():
            raise RuntimeError("Policy returned invalid batched absolute command blocks")
        if active is None:
            return np.stack([actions[index][:, order] for index, order in enumerate(self.to_physx)])
        return {index: actions[offset][:, self.to_physx[index]]
                for offset, index in enumerate(indices)}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=_SELECTION)
    parser.add_argument("--model", choices=MODELS, default="baseline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dino-frame-batch-size", type=int,
                        help="DINO frames per forward per camera; defaults to --batch-size")
    parser.add_argument("--horizon-ticks", type=int, default=720)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--allow-diagnostic", action="store_true")
    parser.add_argument("--isaac-python", type=Path, default=ISAAC_PYTHON)
    parser.add_argument("--spd-source", type=Path, default=ROOT / "spd_source")
    parser.add_argument("--data-process-source", type=Path, default=TRAINING_SOURCE)
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent / "outputs" / "scene-cache")
    parser.add_argument("--source-ids", type=Path,
                        help="Diagnostic benchmark subset: JSON source_id array; preserves original shard seeds/digest")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    return parser


def _selection(args):
    selection = json.loads(args.selection.read_text())
    if selection.get("selection_method") == "fixed_count_per_task_split":
        if (tuple(selection.get("tasks", ())) != FIVE_TASKS
                or selection.get("counts_per_task_split") != {"train": 100, "val": 40}
                or selection.get("by_split") != {"train": 500, "val": 200}
                or "fraction" in selection):
            raise ValueError("Five-task selection must contain exact train/val source quotas")
        recomputed = select_fixed_sources(
            Path(selection["source_split"]), Path(selection["raw_root"]), FIVE_TASKS,
            selection["selection_seed"], 100, 40)
    elif selection.get("selection_method") == SUPPLEMENT_METHOD:
        if selection != build_supplemental_selection(Path(selection["base_selection_path"])):
            raise ValueError("Saved supplemental selection manifest differs from frozen raw sources")
        recomputed = selection["episodes"]
    else:
        recomputed = select_sources(Path(selection["source_split"]), Path(selection["raw_root"]), _TASKS,
                                    selection_seed=selection["selection_seed"], fraction=selection["fraction"])
    if (not recomputed or selection["episodes"] != recomputed
            or selection["selection_digest"] != recomputed[0]["selection_digest"]
            or len(recomputed) != selection["planned"]):
        raise ValueError("Saved evaluation selection differs from the baseline training snapshot")
    shard = [row for index, row in enumerate(recomputed) if index % args.shard_count == args.shard_index]
    original_indices = ({row["source_id"]: index for index, row in enumerate(recomputed)}
                        if selection.get("selection_method") == SUPPLEMENT_METHOD else
                        {row["source_id"]: index for index, row in enumerate(shard)})
    requested = None
    if args.source_ids is not None:
        requested = json.loads(args.source_ids.read_text())
        if (not isinstance(requested, list) or not requested
                or any(not isinstance(item, str) for item in requested)
                or len(set(requested)) != len(requested)):
            raise ValueError("--source-ids must contain a nonempty unique JSON string array")
        unknown = set(requested) - {row["source_id"] for row in recomputed}
        if unknown:
            raise ValueError(f"Diagnostic source IDs are absent from the frozen selection: {sorted(unknown)}")
        requested = sorted(requested)
        shard = [row for row in shard if row["source_id"] in set(requested)]
    if not shard:
        raise ValueError("Selected shard/subset contains no source episodes")
    return selection, shard, original_indices, requested


def _session_config(args, selection, planned, requested, gpu_index):
    return {"method": METHOD, "model": args.model,
            "checkpoint_sha256": MODEL_CHECKPOINT_SHA256[args.model],
            "history_mask_prob": MODEL_HISTORY_MASK_PROB[args.model],
            "selection_digest": selection["selection_digest"],
            "shard_index": args.shard_index, "shard_count": args.shard_count, "gpu_index": gpu_index,
            "batch_size": args.batch_size, "dino_frame_batch_size": args.dino_frame_batch_size,
            "horizon_ticks": (None if selection.get("selection_method") in _GOAL_DRIVEN_SELECTIONS
                              else args.horizon_ticks), "seed": args.seed, "diffusion_steps": args.diffusion_steps,
            "allow_diagnostic": args.allow_diagnostic, "diagnostic_subset": requested is not None,
            "source_ids": requested, "raw_root": str(Path(selection["raw_root"]).resolve()), "planned": len(planned),
            "control_hz": _CONTROL_HZ, "policy_image_stride": 8, "policy_action_chunk": 8,
            "physics_substeps": 16, "scoring_window_ticks": 31,
            "checkpoint": str(ROOT / "checkpoints" / MODELS[args.model]),
            **({"termination_mode": "early_task_goal", "task_limit_ticks": TASK_LIMIT_TICKS}
               if selection.get("selection_method") in _GOAL_DRIVEN_SELECTIONS else {}),
            **{key: str(getattr(args, key).expanduser().resolve()) for key in
               ("isaac_python", "assets", "spd_source", "data_process_source", "cache_dir")}}


def _matches_session(saved: dict, config: dict) -> bool:
    if saved == config:
        return True
    identity = ("checkpoint_sha256", "history_mask_prob")
    return (saved.get("model") == config.get("model") == "baseline"
            and all(key not in saved for key in identity)
            and saved == {key: value for key, value in config.items() if key not in identity})


def _source_seed(args, index: int) -> int:
    return args.seed + (index // 8 if args.selection_method == SUPPLEMENT_METHOD else index)


def _pending(planned, output, indices, args):
    pending = []
    for row in planned:
        path = output / row["split"] / row["task_id"] / row["episode_id"] / "result.json"
        if not path.is_file():
            pending.append(row)
            continue
        result = json.loads(path.read_text())
        expected = {**{key: row[key] for key in
                       ("split", "task_id", "episode_id", "source_id", "selection_digest", "raw_h5")},
                    "method": METHOD, "seed": _source_seed(args, indices[row["source_id"]]),
                    "planned_index": indices[row["source_id"]],
                    "horizon_ticks": (TASK_LIMIT_TICKS[row["task_id"]]
                                      if args.selection_method in _GOAL_DRIVEN_SELECTIONS
                                      else args.horizon_ticks),
                    "checkpoint_step": 30000, "use_ema": True,
                    "model": args.model,
                    "checkpoint_sha256": MODEL_CHECKPOINT_SHA256[args.model],
                    "history_mask_prob": MODEL_HISTORY_MASK_PROB[args.model]}
        identity = ("model", "checkpoint_sha256", "history_mask_prob")
        if (args.model == "baseline" and all(key not in result for key in identity)):
            expected = {key: value for key, value in expected.items() if key not in identity}
        if any(result.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Committed result has different source/seed/method/model identity: {path}")
        if result.get("status") not in {"success", "failure", "unsupported", "error"}:
            raise ValueError(f"Committed result is not terminal: {path}")
        if args.retry_errors and result["status"] == "error":
            pending.append(row)
    return pending


class RolloutRunner:
    def __init__(self, args, output, planned, indices, run_dir, timings):
        self.args, self.output, self.planned, self.indices = args, output, planned, indices
        self.run_dir, self.timings = run_dir, timings
        self.session = None
        self.batch_policy = None
        self.trained = None
        self.failed = 0
        self.completed = 0

    def _prepare(self, row):
        folder = self.output / row["split"] / row["task_id"] / row["episode_id"]
        folder.mkdir(parents=True, exist_ok=True)
        attempt = folder / f"attempt_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%fZ')}_{uuid.uuid4().hex[:8]}"
        attempt.mkdir()
        index = self.indices[row["source_id"]]
        result = {**{key: row[key] for key in
                     ("split", "task_id", "episode_id", "source_id", "raw_h5", "selection_digest")},
                  "method": METHOD, "model": self.args.model,
                  "checkpoint_sha256": MODEL_CHECKPOINT_SHA256[self.args.model],
                  "history_mask_prob": MODEL_HISTORY_MASK_PROB[self.args.model],
                  "checkpoint_step": 30000, "use_ema": True,
                  "planned_index": index, "seed": _source_seed(self.args, index),
                  "horizon_ticks": (TASK_LIMIT_TICKS[row["task_id"]]
                                    if self.args.selection_method in _GOAL_DRIVEN_SELECTIONS
                                    else self.args.horizon_ticks),
                  "status": "pending", "tick": 0,
                  "run_id": self.run_dir.name}
        item = {"row": row, "folder": folder, "attempt": attempt, "result": result,
                "started": time.monotonic()}
        try:
            scene, scene_id = _raw_scene(row)
            result["scene_id"] = scene_id
            item["scene"] = scene
            sampled = scene.get("sampled_parameters", {})
            if not isinstance(sampled.get("scene_manifest.sampled_values"), dict):
                result.update(status="unsupported", reason="raw_scene_lacks_goal_and_geometry_affordances")
                self._commit(item)
                return None
        except Exception as exc:
            self._error(item, exc)
            return None
        return item

    def _commit(self, item):
        result = item["result"]
        result["wall_seconds"] = time.monotonic() - item["started"]
        result["sim_seconds"] = result.get("tick", 0) / _CONTROL_HZ
        _commit_result(item["folder"], item["attempt"], result)
        self.completed += 1
        self.failed += result["status"] == "error"
        _persist_summary(self.output, self.planned)
        print(json.dumps({"source_id": result["source_id"], "status": result["status"],
                          "reason": result["reason"], "method": METHOD}, ensure_ascii=False), flush=True)

    def _error(self, item, exc):
        unsupported = isinstance(exc, UnsupportedRawSceneError) or (
            isinstance(exc, GPUWorkerError) and exc.error_kind == "unsupported_raw_scene")
        item["result"].update(status="unsupported" if unsupported else "error",
                              reason=f"{type(exc).__name__}: {exc}",
                              error_traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        self._commit(item)

    def _ensure_session(self):
        if self.session is None:
            started = time.monotonic()
            directory = self.run_dir / f"worker_{uuid.uuid4().hex}"
            directory.mkdir()
            try:
                self.session = IsaacGpuClient(directory, self.args.isaac_python, self.args.spd_source,
                                              self.args.data_process_source, self.args.cache_dir)
            finally:
                self.timings["startup_seconds"] += time.monotonic() - started
            save_json(self.output / "renderer.json", self.session.runtime)
            save_json(self.output / "physx_alignment.json", self.session.alignment)
            self.timings["worker_pids"].append(self.session.process.pid)

    def close(self):
        if self.session is not None:
            session, self.session = self.session, None
            session.close()

    def _policy_call(self, function, *args):
        started = time.monotonic()
        try:
            return function(*args)
        finally:
            elapsed = time.monotonic() - started
            self.timings["policy_seconds"] += elapsed
            field = "inference_seconds" if function.__name__ == "infer" else "observation_seconds"
            self.timings[field] += elapsed

    def _run_goal_driven(self, items, frames):
        trackers = [TaskTermination(item["scene"], self.args.assets,
                                    TASK_LIMIT_TICKS[item["row"]["task_id"]]) for item in items]
        terminal_frames = [None] * len(items)
        scoring_errors = {}
        active = [True] * len(items)
        for tracker, frame in zip(trackers, frames, strict=True):
            tracker.update(0, frame["object_pose"], frame["hand_positions"])
        tick, inference_batches = 0, 0
        max_limit = max(tracker.limit_ticks for tracker in trackers)
        while any(active):
            if tick >= max_limit:
                raise RuntimeError("A source exceeded its task-specific rollout limit")
            if tick % 8 == 0:
                actions = self._policy_call(self.batch_policy.infer, active)
                inference_batches += 1
            targets = np.stack([
                actions[index][tick % 8] if enabled else frames[index]["applied_command"]
                for index, enabled in enumerate(active)])
            tick += 1
            frames = self.session.step_batch(targets, tick)
            for index, (item, tracker, frame) in enumerate(zip(items, trackers, frames, strict=True)):
                if not active[index]:
                    continue
                item["result"]["tick"] = tick
                started = time.monotonic()
                try:
                    outcome = tracker.update(tick, frame["object_pose"], frame["hand_positions"])
                except Exception as exc:
                    scoring_errors[index] = exc
                    outcome = True
                finally:
                    self.timings["scoring_seconds"] += time.monotonic() - started
                if outcome:
                    active[index] = False
                    terminal_frames[index] = frame
            if any(active):
                self._policy_call(self.batch_policy.observe, frames, tick, active)
        return tick, inference_batches, trackers, terminal_frames, scoring_errors, frames

    def _run_batch(self, items):
        for item in items:
            item["started"] = time.monotonic()
        try:
            self._ensure_session()
        except Exception as exc:
            for item in items:
                self._error(item, exc)
            return
        batch_id = uuid.uuid4().hex
        started = time.monotonic()
        try:
            environments, joints, frames = self.session.load_batch(
                [{"dataset": item["row"]["raw_dataset"], "episode": item["row"]["episode_id"],
                  "directory": str(item["attempt"])} for item in items],
                self.args.assets, self.trained["camera_config"], batch_id)
        except Exception as exc:
            self.timings["load_seconds"] += time.monotonic() - started
            failure = {"batch_id": batch_id, "source_ids": [item["row"]["source_id"] for item in items],
                       "error": f"{type(exc).__name__}: {exc}",
                       "worker_pid": self.session.process.pid}
            self.timings["load_failures"].append(failure)
            for item in items:
                item["result"].setdefault("load_failures", []).append(failure)
            if not isinstance(exc, GPUWorkerError):
                try:
                    self.close()
                except Exception as cleanup_exc:
                    failure["cleanup_error"] = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            if len(items) > 1:
                middle = len(items) // 2
                self._run_batch(items[:middle])
                self._run_batch(items[middle:])
            else:
                self._error(items[0], exc)
            return
        load_seconds = time.monotonic() - started
        self.timings["load_seconds"] += load_seconds
        loop_start = time.monotonic()
        policy_before = self.timings["policy_seconds"]
        tick, inferences = 0, 0
        goal_driven = self.args.selection_method in _GOAL_DRIVEN_SELECTIONS
        histories = [] if goal_driven else [deque(maxlen=31) for _ in items]
        try:
            for item, environment in zip(items, environments, strict=True):
                if (environment["scene_id"] != item["result"]["scene_id"]
                        or environment["task_id"] != item["row"]["task_id"]):
                    raise ValueError("GPU PhysX scene identity differs from the raw source episode")
                item["scene"]["_hand_reference"] = environment["hand_reference"]
                item["scene"]["_hand_reach_m"] = environment["hand_reach_m"]
                item["result"].update(batch_id=batch_id, batch_size=len(items),
                                      worker_pid=self.session.process.pid, isaac_log=str(self.session.log_path),
                                      physical_environment=str(item["attempt"] / "scene_metadata.json"))
                save_json(item["attempt"] / "scene_metadata.json", environment)
                if type(environment.get("cache_hit")) is not bool:
                    raise ValueError("GPU PhysX scene metadata must report cache_hit explicitly")
                self.timings["cache_hits" if environment["cache_hit"] else "cache_misses"] += 1
            self.batch_policy.reset(joints, [item["result"]["seed"] for item in items])
            self._policy_call(self.batch_policy.observe, frames, 0)
            if goal_driven:
                tick, inferences, trackers, terminal_frames, scoring_errors, frames = (
                    self._run_goal_driven(items, frames))
            else:
                for history, frame in zip(histories, frames, strict=True):
                    history.append((frame["object_pose"], frame["hand_positions"]))
                while tick < self.args.horizon_ticks:
                    actions = self._policy_call(self.batch_policy.infer)
                    inferences += 1
                    for offset in range(min(8, self.args.horizon_ticks - tick)):
                        tick += 1
                        frames = self.session.step_batch(actions[:, offset], tick)
                        for item, history, frame in zip(items, histories, frames, strict=True):
                            history.append((frame["object_pose"], frame["hand_positions"]))
                            item["result"]["tick"] = tick
                        if tick < self.args.horizon_ticks:
                            self._policy_call(self.batch_policy.observe, frames, tick)
        except Exception as exc:
            self.timings["loop_seconds"] += time.monotonic() - loop_start
            try:
                self.close()
            except Exception as cleanup_exc:
                for item in items:
                    item["result"]["cleanup_error"] = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            for item in items:
                self._error(item, exc)
            return
        loop_seconds = time.monotonic() - loop_start
        self.timings["loop_seconds"] += loop_seconds
        batch_timing = {"batch_id": batch_id, "count": len(items), "load_seconds": load_seconds,
                        "loop_seconds": loop_seconds,
                        "policy_seconds": self.timings["policy_seconds"] - policy_before,
                        "worker_pid": self.session.process.pid,
                        "source_ids": [item["row"]["source_id"] for item in items]}
        if goal_driven:
            batch_timing["inference_batches"] = inferences
            batch_timing["control_ticks"] = tick
        self.timings["batches"].append(batch_timing)
        for index, (item, names, environment) in enumerate(zip(items, joints, environments, strict=True)):
            if goal_driven and index in scoring_errors:
                item["result"]["inference_count"] = (item["result"]["tick"] + 7) // 8
                self._error(item, scoring_errors[index])
                continue
            started = time.monotonic()
            try:
                if goal_driven:
                    terminal = trackers[index].terminal
                    if terminal is None or terminal_frames[index] is None:
                        raise RuntimeError("Adaptive source has no frozen terminal frame")
                    history, frame = trackers[index].history, terminal_frames[index]
                    item["result"].update(terminal)
                else:
                    history, frame = histories[index], frames[index]
                    scored = _score(item["scene"], history, self.args.assets)
                    item["result"].update(status=scored["status"], reason=scored["reason"],
                                          score=scored, inference_count=inferences)
                np.savez_compressed(item["folder"] / "final_window.npz",
                                    object_pose=np.stack([sample[0] for sample in history]),
                                    wrist_positions=np.stack([sample[1] for sample in history]),
                                    state=frame["state"], joint_names=np.asarray(names))
                item["result"].update(hand_reach_m=environment["hand_reach_m"],
                                      timing_scope="shared_batch", timing=batch_timing)
            except Exception as exc:
                self._error(item, exc)
            else:
                self._commit(item)
            finally:
                self.timings["scoring_seconds"] += time.monotonic() - started

    def run(self, pending):
        started = time.monotonic()
        prepared = [item for row in pending if (item := self._prepare(row)) is not None]
        self.timings["prevalidation_seconds"] = time.monotonic() - started
        if not prepared:
            return
        started = time.monotonic()
        try:
            policy, self.trained = load_policy(self.args, self.output)
            policy.model_config.dino_frame_batch_size = self.args.dino_frame_batch_size
            policy.model.config.dino_frame_batch_size = self.args.dino_frame_batch_size
            model_path = self.output / "model.json"
            model_metadata = json.loads(model_path.read_text())
            model_metadata.update(method=METHOD, dino_frame_batch_size=self.args.dino_frame_batch_size)
            save_json(model_path, model_metadata)
            self.batch_policy = BatchedPolicy(policy, self.trained["policy_joint_names"])
        except Exception as exc:
            for item in prepared:
                self._error(item, exc)
            raise
        finally:
            self.timings["policy_load_seconds"] = time.monotonic() - started
        groups = ([item for item in prepared if item["row"]["task_id"] == task]
                  for task in FIVE_TASKS) if self.args.selection_method in _GOAL_DRIVEN_SELECTIONS else (prepared,)
        for group in groups:
            for offset in range(0, len(group), self.args.batch_size):
                self._run_batch(group[offset:offset + self.args.batch_size])


def main():
    started = time.monotonic()
    parser = _parser()
    args = parser.parse_args()
    if args.dino_frame_batch_size is None:
        args.dino_frame_batch_size = args.batch_size
    if args.retry_errors and not args.resume:
        parser.error("--retry-errors requires --resume")
    if (args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count or args.batch_size <= 0
            or args.dino_frame_batch_size <= 0 or args.horizon_ticks < 30
            or args.seed < 0 or args.diffusion_steps <= 0):
        parser.error("Invalid shard, batch, horizon, seed or diffusion step count")
    if args.horizon_ticks != 720 and not args.allow_diagnostic:
        parser.error("The frozen evaluation requires 720 ticks; shorter runs require --allow-diagnostic")
    try:
        gpu_index = int(os.environ.get("SPD_ROLLOUT_GPU_ID", "0"))
    except ValueError:
        parser.error("SPD_ROLLOUT_GPU_ID must be a nonnegative physical GPU index")
    if gpu_index < 0 or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu_index):
        parser.error("CUDA_VISIBLE_DEVICES and SPD_ROLLOUT_GPU_ID must select the same physical GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU vector rollout requires CUDA")
    selection, planned, indices, requested = _selection(args)
    args.selection_method = selection.get("selection_method")
    verify_checkpoint(args.model)
    output = args.output.expanduser().resolve()
    config = _session_config(args, selection, planned, requested, gpu_index)
    if args.resume:
        if not output.is_dir() or not _matches_session(json.loads((output / "session.json").read_text()), config):
            raise ValueError("Resume requires exactly the same session, subset, seeds and batch configuration")
    else:
        output.mkdir(parents=True, exist_ok=False)
        save_json(output / "session.json", config)
    pending = _pending(planned, output, indices, args)
    _persist_summary(output, planned)
    print(json.dumps({"planned": len(planned), "pending": len(pending), "method": METHOD,
                      "selection_digest": selection["selection_digest"], "diagnostic_subset": requested is not None}), flush=True)
    if not pending:
        return
    run_dir = output / "runs" / uuid.uuid4().hex
    run_dir.mkdir(parents=True)
    timings = {"method": METHOD, "run_id": run_dir.name, "pending_at_start": len(pending),
               "selection_digest": selection["selection_digest"], "gpu_index": gpu_index,
               "batch_size": args.batch_size, "dino_frame_batch_size": args.dino_frame_batch_size,
               "startup_seconds": 0.0, "policy_load_seconds": 0.0, "load_seconds": 0.0,
               "loop_seconds": 0.0, "policy_seconds": 0.0, "scoring_seconds": 0.0,
               "inference_seconds": 0.0, "observation_seconds": 0.0,
               "worker_pids": [], "batches": [], "load_failures": [],
               "cache_hits": 0, "cache_misses": 0,
               "started_at": datetime.now(timezone.utc).isoformat()}
    telemetry = DeviceTelemetry(gpu_index)
    telemetry.start()
    torch.cuda.reset_peak_memory_stats()
    runner = RolloutRunner(args, output, planned, indices, run_dir, timings)
    try:
        runner.run(pending)
    finally:
        try:
            runner.close()
        finally:
            device = telemetry.stop()
            save_json(run_dir / "telemetry.json", device)
            timings.update({key: value for key, value in device.items() if key != "samples"})
            timings.update(total_seconds=time.monotonic() - started,
                           policy_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           policy_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                           completed=runner.completed, errors=runner.failed,
                           finished_at=datetime.now(timezone.utc).isoformat())
            timings["episodes_per_minute"] = runner.completed * 60 / timings["total_seconds"]
            timings["simulation_render_transport_seconds"] = max(0.0, timings["loop_seconds"] - timings["policy_seconds"])
            save_json(run_dir / "timing.json", timings)
            save_json(output / "timing.json", timings)
            metrics = {**timings, **{key + "_s": timings[key + "_seconds"] for key in
                                    ("startup", "policy_load", "load", "loop", "policy", "scoring", "total")}}
            save_json(run_dir / "run_metrics.json", metrics)
            save_json(output / "run_metrics.json", metrics)
    if runner.failed:
        raise SystemExit(f"GPU rollout recorded {runner.failed} error outcomes; inspect result.json and timing.json")


if __name__ == "__main__":
    main()
