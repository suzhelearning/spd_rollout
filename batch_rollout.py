#!/usr/bin/env python3
"""Score complete headless PhysX rollouts by the bottle's final bin placement."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from batch_success import BottleInBinScorer
from infer_recording import MODELS, ROOT
from isaac_physx_client import IsaacPhysxClient
from rollout_policy import load_policy, save_json
from training_render import ISAAC_PYTHON, TRAINING_SOURCE

_CONTROL_HZ = 30


def _selected_episodes(dataset: Path, shard_index: int, shard_count: int, limit: int) -> tuple[int, list[tuple[int, str]]]:
    index = json.loads((dataset / "dataset.json").read_text())
    if index.get("schema_version") != 4 or index.get("task_id") != "toss_in_bin":
        raise ValueError("Expected an indexed v4 toss_in_bin dataset")
    episodes = index["episodes"]
    if not episodes or len(set(episodes)) != len(episodes):
        raise ValueError("Empty or duplicate episode index")
    chosen = [(number, episode) for number, episode in enumerate(episodes)
              if number % shard_count == shard_index]
    return len(episodes), chosen[:limit] if limit else chosen


def _results(output: Path, episodes: list[tuple[int, str]]) -> list[dict]:
    results = []
    for index, episode in episodes:
        path = output / f"{index + 1:04d}_{episode}" / "result.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result.get("episode_id") != episode or result.get("dataset_index") != index:
                raise ValueError(f"Stored episode result has incorrect identity: {path}")
            if result.get("result") not in ("success", "failure"):
                raise ValueError(f"Stored episode result has no physical outcome: {path}")
            results.append(result)
    return results


def _persist(output: Path, episodes: list[tuple[int, str]], *, error: str | None = None) -> dict:
    results = _results(output, episodes)
    success = sum(result["result"] == "success" for result in results)
    failure = len(results) - success
    summary = {
        "planned": len(episodes), "completed": len(results), "pending": len(episodes) - len(results),
        "success": success, "failure": failure,
        "success_rate": success / len(results) if results else None,
        "finished": len(results) == len(episodes) and error is None, "error": error,
    }
    save_json(output / "summary.json", summary)
    path = output / "results.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset_index", "episode_id", "result", "reason",
                                                     "sim_seconds", "wall_seconds", "inference_count",
                                                     "scene_id", "result_path"))
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key) for key in writer.fieldnames})
    temporary.replace(path)
    return summary


def _observation(frame: dict, tick: int, order: list[int]) -> dict:
    obs = {"state": frame["state"][order].astype(np.float32),
           "previous_actions": frame["applied_command"][order].astype(np.float32)}
    if tick % 8 == 0:
        if frame["images"] is None or set(frame["images"]) != {"top", "left_wrist", "right_wrist"}:
            raise RuntimeError(f"Missing actual three-camera observation at tick {tick}")
        obs["images"] = frame["images"]
    elif frame["images"] is not None or frame["raw"]:
        raise RuntimeError("Policy-only rollout unexpectedly rendered intermediate images")
    return obs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/root/sunyz/data/spd-sim/raw/toss_in_bin"))
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--model", choices=MODELS, default="baseline")
    parser.add_argument("--horizon-ticks", type=int, default=720,
                        help="Fixed 30 Hz policy rollout horizon, independent of recorded episode length (default: 24 s)")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0 means all indexed episodes of this shard")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--allow-diagnostic", action="store_true")
    parser.add_argument("--isaac-python", type=Path, default=ISAAC_PYTHON)
    parser.add_argument("--spd-source", type=Path, default=ROOT / "spd_source")
    parser.add_argument("--data-process-source", type=Path, default=TRAINING_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true", help="Skip committed results in an existing output directory")
    args = parser.parse_args()
    if (args.horizon_ticks < 8 or args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count
            or args.limit < 0 or args.seed < 0 or args.diffusion_steps <= 0):
        parser.error("Invalid horizon, shard, limit, seed or diffusion steps")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    args.dataset = args.dataset.expanduser().resolve()
    args.assets = args.assets.expanduser().resolve()
    total, episodes = _selected_episodes(args.dataset, args.shard_index, args.shard_count, args.limit)
    if not episodes:
        raise ValueError("This shard contains no episodes")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%fZ")
    output = (args.output.expanduser().resolve() if args.output else
              ROOT / "outputs" / "physx-batch" / stamp)
    config = {
        "dataset": str(args.dataset), "assets": str(args.assets), "model": args.model,
        "horizon_ticks": args.horizon_ticks, "control_hz": _CONTROL_HZ,
        "seed": args.seed, "diffusion_steps": args.diffusion_steps,
        "allow_diagnostic": args.allow_diagnostic,
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "limit": args.limit, "dataset_episodes": total,
        "episodes": [episode for _, episode in episodes],
        "capture": "policy_only", "backend": "headless Isaac Sim PhysX + RTX",
        "outcome": "bottle collision geometry resting within the bin interior at a fixed horizon",
    }
    if args.resume:
        if not output.is_dir() or json.loads((output / "session.json").read_text()) != config:
            raise ValueError("Resume requires exactly the same saved batch configuration")
    else:
        output.mkdir(parents=True, exist_ok=False)
        save_json(output / "session.json", config)
    summary = _persist(output, episodes)
    remaining = [(index, episode) for index, episode in episodes
                 if not (output / f"{index + 1:04d}_{episode}" / "result.json").exists()]
    print(json.dumps({"batch_started": True, "output": str(output), "planned": len(episodes),
                      "remaining": len(remaining), "horizon_ticks": args.horizon_ticks},
                     ensure_ascii=False), flush=True)
    if not remaining:
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return

    scorer = BottleInBinScorer(args.assets)
    torch.set_float32_matmul_precision("highest")
    policy, trained = load_policy(args, output)
    session = None
    try:
        session = IsaacPhysxClient(output, args.isaac_python, args.spd_source,
                                   args.data_process_source, capture="policy_only")
        save_json(output / "renderer.json", session.runtime)
        save_json(output / "physx_alignment.json", session.alignment)
        names = trained["policy_joint_names"]
        for dataset_index, episode in remaining:
            started = time.monotonic()
            directory = output / f"{dataset_index + 1:04d}_{episode}"
            directory.mkdir(exist_ok=True)
            # Re-running a killed attempt never overwrites its USD files.
            attempt = directory / f"attempt_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%fZ')}"
            attempt.mkdir()
            metadata, joints, frame = session.load(args.dataset, episode, args.assets,
                                                   trained["camera_config"], attempt, episode)
            if len(joints) != len(set(joints)) or set(joints) != set(names):
                raise ValueError(f"Policy/PhysX joints differ for {episode}")
            to_policy = [joints.index(name) for name in names]
            to_physx = [names.index(name) for name in joints]
            torch.manual_seed(args.seed + dataset_index)
            policy.reset()
            policy.observe(_observation(frame, 0, to_policy), step=0)
            inferences = 0
            tick = 0
            while tick < args.horizon_ticks:
                actions = policy.infer()
                if actions.shape != (8, len(names)) or not np.isfinite(actions).all():
                    raise RuntimeError(f"Invalid policy action block at {episode} tick {tick}")
                inferences += 1
                for action in actions[:min(8, args.horizon_ticks - tick)]:
                    tick += 1
                    frame = session.step(action[to_physx], tick)
                    if tick < args.horizon_ticks:
                        policy.observe(_observation(frame, tick, to_policy), step=tick)
            scene_id = metadata["scene_id"]
            scene = json.loads((args.dataset / "scenes" / f"{scene_id}.json").read_text())
            score = scorer.score(scene, frame["object_pose"])
            if type(score.get("success")) is not bool:
                raise RuntimeError(f"Scorer returned no physical outcome for {episode}")
            result = {
                "dataset_index": dataset_index, "episode_id": episode, "scene_id": scene_id,
                "result": "success" if score["success"] else "failure",
                "reason": score["reason"], "score": score,
                "tick": tick, "sim_seconds": tick / _CONTROL_HZ,
                "wall_seconds": time.monotonic() - started,
                "inference_count": inferences, "final_object_pose": frame["object_pose"].tolist(),
                "result_path": str(directory / "result.json"),
                "physical_environment": str(attempt / "scene_metadata.json"),
            }
            save_json(directory / "result.json", result)
            summary = _persist(output, episodes)
            print(json.dumps({"episode": episode, "result": result["result"], "tick": tick,
                              "completed": summary["completed"], "planned": summary["planned"],
                              "success_rate": summary["success_rate"]}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        _persist(output, episodes, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if session is not None:
            session.close()
    print(json.dumps(_persist(output, episodes), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
