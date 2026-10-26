"""Evaluate fixed baseline source splits in their original, uncut PhysX scenes."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import h5py
import numpy as np
import torch

from eval_selection import select_sources, summarize
from infer_recording import ROOT
from isaac_physx_client import IsaacPhysxClient, UnsupportedRawSceneError
from rollout_policy import load_policy, save_json
from success_blocks import score as score_blocks
from success_vessels import score as score_vessels
from training_render import ISAAC_PYTHON, TRAINING_SOURCE

_TASKS = ("hang_mug", "pyramid", "spelling", "toss_in_bin", "unstack", "vowel_consonant_sort")
_VESSELS = frozenset(("hang_mug", "toss_in_bin", "unstack"))
_SELECTION = ROOT / "evaluation-20pct-selection.json"
_CONTROL_HZ = 30


def _raw_scene(row: dict) -> tuple[dict, str]:
    dataset = Path(row["raw_dataset"])
    episode = row["episode_id"]
    with h5py.File(row["raw_h5"], "r") as handle:
        if (str(handle.attrs.get("episode_id")) != episode
                or str(handle.attrs.get("task_id")) != row["task_id"]
                or int(handle.attrs.get("schema_version", -1)) != 4
                or not bool(handle.attrs.get("recorded_action", False))):
            raise ValueError("Raw H5 identity, task, schema or recorded command is inconsistent")
        scene_id = str(handle.attrs["scene_id"])
        instruction = handle.attrs.get("task_instruction")
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")
        if instruction is not None and not isinstance(instruction, str):
            raise ValueError("Raw task instruction is not text")
    if len(scene_id) != 64 or any(character not in "0123456789abcdef" for character in scene_id):
        raise ValueError("Invalid raw scene ID")
    scene_path = dataset / "scenes" / f"{scene_id}.json"
    scene_bytes = scene_path.read_bytes()
    if hashlib.sha256(scene_bytes).hexdigest() != scene_id:
        raise ValueError("Raw scene checksum differs from episode scene ID")
    scene = json.loads(scene_bytes)
    if int(scene.get("schema_version", -1)) != 4:
        raise ValueError("Raw scene schema is not v4")
    scene["_task"] = row["task_id"]
    scene["_task_instruction"] = instruction
    scene["_frame_dt_s"] = 1 / _CONTROL_HZ
    return scene, scene_id


def _observation(frame: dict, tick: int, order: list[int]) -> dict:
    obs = {"state": frame["state"][order].astype(np.float32),
           "previous_actions": frame["applied_command"][order].astype(np.float32)}
    if tick % 8 == 0:
        if frame["images"] is None or set(frame["images"]) != {"top", "left_wrist", "right_wrist"}:
            raise RuntimeError("Missing required PhysX RTX camera observations")
        obs["images"] = frame["images"]
    elif frame["images"] is not None or frame["raw"]:
        raise RuntimeError("Non-image tick contained unrequested camera data")
    return obs


def _persist_summary(output: Path, planned: list[dict]) -> dict:
    outcomes = []
    for row in planned:
        result_path = output / row["split"] / row["task_id"] / row["episode_id"] / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            if any(result.get(field) != row[field] for field in ("split", "task_id", "episode_id")):
                raise ValueError(f"Committed result has a different source identity: {result_path}")
            outcomes.append(result)
        else:
            outcomes.append({**row, "status": "pending"})
    summary = summarize(outcomes, planned)
    save_json(output / "summary.json", {"by_split": summary,
                                          "planned": len(planned),
                                          "selection_digest": planned[0]["selection_digest"]})
    return summary


def _pending_rows(planned: list[dict], output: Path, *, retry_errors: bool) -> list[dict]:
    pending = []
    for row in planned:
        result_path = output / row["split"] / row["task_id"] / row["episode_id"] / "result.json"
        if not result_path.is_file():
            pending.append(row)
        elif retry_errors and json.loads(result_path.read_text()).get("status") == "error":
            pending.append(row)
    return pending


def _commit_result(folder: Path, attempt: Path, result: dict) -> None:
    current = folder / "result.json"
    if current.is_file():
        previous = json.loads(current.read_text())
        if previous.get("status") != "error":
            raise ValueError(f"Refusing to overwrite a scored or unsupported scene: {current}")
        save_json(attempt / "previous_error_result.json", previous)
    save_json(attempt / "result.json", result)
    save_json(current, result)


def _score(scene: dict, history: deque, assets: Path) -> dict:
    positions = np.stack([frame[0] for frame in history])
    hands = np.stack([frame[1] for frame in history])
    scorer = score_vessels if scene["_task"] in _VESSELS else score_blocks
    score = scorer(scene, positions, hands, assets)
    if (not isinstance(score, dict) or score.get("status") not in {"success", "failure", "unsupported"}
            or not isinstance(score.get("reason"), str)):
        raise RuntimeError("Task scorer did not return a valid automatic outcome")
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=_SELECTION)
    parser.add_argument("--model", choices=("baseline",), default="baseline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--horizon-ticks", type=int, default=720)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--allow-diagnostic", action="store_true")
    parser.add_argument("--isaac-python", type=Path, default=ISAAC_PYTHON)
    parser.add_argument("--spd-source", type=Path, default=ROOT / "spd_source")
    parser.add_argument("--data-process-source", type=Path, default=TRAINING_SOURCE)
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true",
                        help="With --resume, retry only infrastructure errors while preserving each prior result")
    args = parser.parse_args()
    if args.retry_errors and not args.resume:
        parser.error("--retry-errors requires --resume")
    if (args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count
            or args.horizon_ticks < 30 or args.seed < 0 or args.diffusion_steps <= 0):
        parser.error("Invalid shard, horizon, seed or diffusion step count")
    try:
        gpu_index = int(os.environ.get("SPD_ROLLOUT_GPU_ID", "0"))
    except ValueError:
        parser.error("SPD_ROLLOUT_GPU_ID must be a nonnegative physical GPU index")
    if gpu_index < 0 or os.environ.get("CUDA_VISIBLE_DEVICES", str(gpu_index)) != str(gpu_index):
        parser.error("CUDA_VISIBLE_DEVICES and SPD_ROLLOUT_GPU_ID must select the same physical GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("Baseline PhysX rollout requires CUDA")
    selection = json.loads(args.selection.read_text())
    metadata_path = Path(selection["source_split"])
    raw_root = Path(selection["raw_root"])
    recomputed = select_sources(metadata_path, raw_root, _TASKS,
                                selection_seed=selection["selection_seed"], fraction=selection["fraction"])
    if (selection["episodes"] != recomputed or selection["selection_digest"] != recomputed[0]["selection_digest"]
            or len(recomputed) != selection["planned"]):
        raise ValueError("Saved evaluation selection differs from the baseline training snapshot")
    planned = [row for index, row in enumerate(recomputed) if index % args.shard_count == args.shard_index]
    if not planned:
        raise ValueError("Selected shard contains no source episodes")
    output = args.output.expanduser().resolve()
    config = {"model": args.model, "selection_digest": selection["selection_digest"],
              "shard_index": args.shard_index, "shard_count": args.shard_count,
              "gpu_index": gpu_index,
              "horizon_ticks": args.horizon_ticks, "seed": args.seed,
              "diffusion_steps": args.diffusion_steps,
              "allow_diagnostic": args.allow_diagnostic,
              "raw_root": str(raw_root.resolve()), "planned": len(planned),
              "control_hz": _CONTROL_HZ, "policy_image_stride": 8,
              "policy_action_chunk": 8, "physics_substeps": 16,
              "scoring_window_ticks": 31,
              "checkpoint": str(ROOT / "checkpoints" / "baseline_30000.pt"),
              "isaac_python": str(args.isaac_python.resolve()),
              "assets": str(args.assets.resolve()),
              "spd_source": str(args.spd_source.resolve()),
              "data_process_source": str(args.data_process_source.resolve())}
    if args.resume:
        if not output.is_dir() or json.loads((output / "session.json").read_text()) != config:
            raise ValueError("Resumption requires an existing run with exactly the same session configuration")
    else:
        output.mkdir(parents=True, exist_ok=False)
        save_json(output / "session.json", config)
    _persist_summary(output, planned)
    pending = _pending_rows(planned, output, retry_errors=args.retry_errors)
    print(json.dumps({"planned": len(planned), "pending": len(pending), "shard": args.shard_index,
                      "selection_digest": selection["selection_digest"]}), flush=True)
    if not pending:
        return
    policy, trained = load_policy(args, output)
    names = trained["policy_joint_names"]
    session = None
    try:
        for index, row in enumerate(planned):
            if row not in pending:
                continue
            folder = output / row["split"] / row["task_id"] / row["episode_id"]
            folder.mkdir(parents=True, exist_ok=True)
            attempt = folder / f"attempt_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%fZ')}"
            attempt.mkdir()
            started = time.monotonic()
            result = {"split": row["split"], "task_id": row["task_id"],
                      "episode_id": row["episode_id"], "source_id": row["source_id"],
                      "raw_h5": row["raw_h5"], "selection_digest": row["selection_digest"],
                      "checkpoint_step": 30000, "use_ema": True, "seed": args.seed + index,
                      "horizon_ticks": args.horizon_ticks, "status": "pending"}
            try:
                scene, scene_id = _raw_scene(row)
                result["scene_id"] = scene_id
                sampled = scene.get("sampled_parameters", {})
                if not isinstance(sampled.get("scene_manifest.sampled_values"), dict):
                    result.update(status="unsupported", reason="raw_scene_lacks_goal_and_geometry_affordances")
                else:
                    if session is None:
                        session = IsaacPhysxClient(attempt, args.isaac_python, args.spd_source,
                                                  args.data_process_source, capture="policy_only")
                        save_json(output / "renderer.json", session.runtime)
                        save_json(output / "physx_alignment.json", session.alignment)
                    result["isaac_log"] = str(session.log_path)
                    environment, joints, frame = session.load(Path(row["raw_dataset"]), row["episode_id"],
                                                               args.assets, trained["camera_config"], attempt,
                                                               uuid.uuid4().hex)
                    if len(joints) != len(set(joints)) or set(joints) != set(names):
                        raise ValueError("PhysX joint identities differ from the trained baseline")
                    if environment["scene_id"] != scene_id or environment["task_id"] != row["task_id"]:
                        raise ValueError("PhysX scene identity differs from the raw source episode")
                    scene["_hand_reference"] = environment["hand_reference"]
                    scene["_hand_reach_m"] = environment["hand_reach_m"]
                    to_policy = [joints.index(name) for name in names]
                    to_physx = [names.index(name) for name in joints]
                    torch.manual_seed(args.seed + index)
                    policy.reset()
                    policy.observe(_observation(frame, 0, to_policy), step=0)
                    history = deque(maxlen=31)
                    history.append((frame["object_pose"], frame["hand_positions"]))
                    tick = 0
                    inferences = 0
                    while tick < args.horizon_ticks:
                        actions = policy.infer()
                        if actions.shape != (8, len(names)) or not np.isfinite(actions).all():
                            raise RuntimeError("Baseline policy returned invalid absolute command block")
                        inferences += 1
                        for action in actions[:min(8, args.horizon_ticks - tick)]:
                            tick += 1
                            frame = session.step(action[to_physx], tick)
                            history.append((frame["object_pose"], frame["hand_positions"]))
                            if tick < args.horizon_ticks:
                                policy.observe(_observation(frame, tick, to_policy), step=tick)
                    scored = _score(scene, history, args.assets)
                    np.savez_compressed(folder / "final_window.npz",
                                        object_pose=np.stack([item[0] for item in history]),
                                        wrist_positions=np.stack([item[1] for item in history]),
                                        state=frame["state"], joint_names=np.asarray(joints))
                    result.update(status=scored["status"], reason=scored["reason"], score=scored,
                                  tick=tick, inference_count=inferences,
                                  hand_reach_m=environment["hand_reach_m"],
                                  physical_environment=str(attempt / "scene_metadata.json"))
            except Exception as exc:
                result.update(status="unsupported" if isinstance(exc, UnsupportedRawSceneError) else "error",
                              reason=f"{type(exc).__name__}: {exc}")
                if session is not None:
                    try:
                        session.close()
                    except Exception as cleanup_exc:
                        result["cleanup_error"] = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    session = None
            result["wall_seconds"] = time.monotonic() - started
            result["sim_seconds"] = result.get("tick", 0) / _CONTROL_HZ
            _commit_result(folder, attempt, result)
            summary = _persist_summary(output, planned)
            count = summary[row["split"]][row["task_id"]]
            print(json.dumps({"task": row["task_id"], "split": row["split"],
                              "episode": row["episode_id"], "status": result["status"],
                              "reason": result["reason"], "completed_group": count["planned"] - count["pending"],
                              "planned_group": count["planned"]}, ensure_ascii=False), flush=True)
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    main()
