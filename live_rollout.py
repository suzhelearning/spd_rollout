#!/usr/bin/env python3
"""Interactive SPD policy evaluation using headless Isaac Sim PhysX and RTX."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import signal
import threading
import time
import traceback
import uuid

import numpy as np
from PIL import Image
import torch

from infer_recording import MODELS, ROOT
from isaac_physx_client import IsaacPhysxClient
from rollout_web import make_server
from rollout_policy import load_policy, save_json
from training_render import ISAAC_PYTHON, TRAINING_SOURCE


def jpeg(image: np.ndarray, quality: int) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


class Evaluation:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=False)
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.shutdown_requested = False
        self.decision: str | None = None
        self.results: list[dict] = []
        self.current: dict = {}
        self.rgb: np.ndarray | None = None
        self.jpg: bytes | None = None
        self.frame_version = 0
        self.phase = "loading"
        self.error: str | None = None
        self.renderer_description = "Isaac Sim 原生 PhysX · 正在启动"
        self.started_at = time.monotonic()
        self.round_started_at: float | None = None
        dataset = json.loads((args.dataset / "dataset.json").read_text())
        if dataset.get("task_id") != "toss_in_bin":
            raise ValueError("This reconstructed environment currently supports toss_in_bin")
        self.episodes = dataset["episodes"]
        if args.start_episode:
            self.episodes = self.episodes[self.episodes.index(args.start_episode):]
        if args.episodes:
            self.episodes = self.episodes[:args.episodes]
        if not self.episodes:
            raise ValueError("No indexed episodes to evaluate")
        self.thread = threading.Thread(target=self.run, name="spd-rollout", daemon=True)
        save_json(self.output / "session.json", {
            "created_at": datetime.now(timezone.utc).isoformat(), "dataset": str(args.dataset),
            "assets": str(args.assets), "model": args.model, "episodes": self.episodes,
            "seed": args.seed, "diffusion_steps": args.diffusion_steps,
            "environment": "headless Isaac Sim PhysX simulation; separate from historical MuJoCo and original collection physics",
            "renderer": "Isaac Sim RTX top every tick; three training cameras at policy image ticks",
            "scoring": "manual s/f; success / (success + failure); aborted/errors excluded",
            "timeout": None, "auto_next": True,
        })
        self.persist_summary()

    def statistics(self) -> dict:
        successful = sum(r["result"] == "success" for r in self.results)
        failed = sum(r["result"] == "failure" for r in self.results)
        labeled = successful + failed
        selected = [r for r in self.results if r["result"] in ("success", "failure")]
        return {"success": successful, "failure": failed, "labeled": labeled,
                "aborted": sum(r["result"] == "aborted" for r in self.results),
                "errors": sum(r["result"] == "error" for r in self.results),
                "success_rate": successful / labeled if labeled else None,
                "mean_wall_seconds": sum(r["wall_seconds"] for r in selected) / labeled if labeled else None,
                "mean_sim_seconds": sum(r["sim_seconds"] for r in selected) / labeled if labeled else None,
                "inference_count": sum(r["inference_count"] for r in self.results)}

    def snapshot(self) -> dict:
        with self.lock:
            return {"phase": self.phase, "round_id": self.current.get("round_id"),
                    "episode_id": self.current.get("episode_id"),
                    "round_number": self.current.get("round_number", 0),
                    "total_episodes": len(self.episodes), "tick": self.current.get("tick", 0),
                    "inference_count": self.current.get("inference_count", 0),
                    "sim_seconds": self.current.get("tick", 0) / 30,
                    "wall_seconds": self.current.get("wall_seconds", time.monotonic() - self.round_started_at if self.round_started_at else 0),
                    "last_inference_ms": self.current.get("last_inference_ms"),
                    "frame_version": self.frame_version, "stats": self.statistics(),
                    "results": [dict(result) for result in self.results], "error": self.error,
                    "output_dir": str(self.output), "model": self.args.model,
                    "renderer": self.renderer_description}

    def frame(self, round_id: str | None = None) -> bytes | None:
        with self.lock:
            if round_id is not None and round_id != self.current.get("round_id"):
                return None
            return self.jpg

    def artifact(self, round_id: str, kind: str) -> bytes | None:
        if kind not in ("first", "last"):
            return None
        with self.lock:
            record = next((r for r in self.results if r["round_id"] == round_id), None)
            path = record.get(f"{kind}_image") if record else None
        return Path(path).read_bytes() if path else None

    def label(self, round_id: str, result: str) -> dict:
        if result not in ("success", "failure"):
            raise ValueError("结果只能为 success 或 failure")
        with self.lock:
            if self.phase != "running" or self.current.get("round_id") != round_id:
                raise ValueError("该轮已停止或已切换，标注未写入")
            self.decision = result
            self.phase = "saving"
            self.current["label_time"] = datetime.now(timezone.utc).isoformat()
            self.current["wall_seconds"] = time.monotonic() - self.round_started_at
            self.wake.set()
            return {"accepted": True, "round_id": round_id, "result": result}

    def stop(self) -> dict:
        with self.lock:
            self.shutdown_requested = True
            if self.phase in ("running", "loading"):
                self.phase = "stopping"
            self.wake.set()
            return self.snapshot()

    def persist_summary(self) -> None:
        save_json(self.output / "summary.json", {
            "model": self.args.model, "dataset": str(self.args.dataset), "phase": self.phase,
            "environment": "headless Isaac Sim PhysX + RTX, manually scored; not original collection physics or hardware",
            "statistics": self.statistics(), "results": self.results, "error": self.error,
        })
        fields = ("round_number", "episode_id", "result", "wall_seconds", "sim_seconds",
                  "inference_count", "first_image", "last_image", "round_id")
        temporary = self.output / "results.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.results)
        temporary.replace(self.output / "results.csv")

    def publish(self, image: np.ndarray) -> None:
        self.rgb = image.copy()
        self.jpg = jpeg(self.rgb, 85)
        self.frame_version += 1

    def finish_round(self, result: str, frame: dict | None, directory: Path) -> None:
        with self.lock:
            self.current.setdefault("wall_seconds", time.monotonic() - self.round_started_at)
            if self.rgb is not None:
                Image.fromarray(self.rgb).save(directory / "last.png")
            record = {**self.current, "result": result,
                      "wall_seconds": self.current.get("wall_seconds", time.monotonic() - self.round_started_at),
                      "sim_seconds": self.current.get("tick", 0) / 30,
                      "finished_at": datetime.now(timezone.utc).isoformat(),
                      "first_image": str(directory / "first.png") if (directory / "first.png").exists() else None,
                      "last_image": str(directory / "last.png") if self.rgb is not None else None,
                      "error": self.error if result == "error" else None}
            if frame is not None:
                np.savez_compressed(directory / "final_state.npz", state=frame["state"],
                                    object_pose=frame["object_pose"],
                                    applied_command=frame["applied_command"],
                                    joint_names=np.asarray(self.joint_names))
            save_json(directory / "result.json", record)
            with (self.output / "results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.results.append(record)
            self.persist_summary()
            print(json.dumps({"round": record["round_number"], "result": result,
                              "stats": self.statistics()}, ensure_ascii=False), flush=True)


    def run(self) -> None:
        frame = None
        directory = None
        session = None
        try:
            torch.set_float32_matmul_precision("highest")
            policy, config = load_policy(self.args, self.output)
            session = IsaacPhysxClient(self.output, self.args.isaac_python,
                                       self.args.spd_source, self.args.data_process_source)
            save_json(self.output / "renderer.json", session.runtime)
            save_json(self.output / "physx_alignment.json", session.alignment)
            with self.lock:
                self.renderer_description = (
                    f"Isaac Sim {session.runtime['isaacsim_version']} · 无 GUI PhysX + RTX top"
                )
            for index, episode in enumerate(self.episodes):
                with self.lock:
                    if self.shutdown_requested:
                        break
                    self.phase = "loading"
                    self.decision = None
                    self.wake.clear()
                    self.rgb = self.jpg = None
                    self.round_started_at = time.monotonic()
                    self.current = {"round_id": uuid.uuid4().hex, "episode_id": episode,
                                    "round_number": index + 1, "tick": 0, "inference_count": 0,
                                    "last_inference_ms": None, "seed": self.args.seed + index,
                                    "started_at": datetime.now(timezone.utc).isoformat()}
                directory = self.output / f"{index + 1:04d}_{episode}"
                directory.mkdir()
                (directory / "inference").mkdir()
                metadata, self.joint_names, frame = session.load(
                    self.args.dataset, episode, self.args.assets, config["camera_config"],
                    directory, self.current["round_id"])
                names = config["policy_joint_names"]
                if len(self.joint_names) != len(set(self.joint_names)) or set(self.joint_names) != set(names):
                    raise ValueError("PhysX joint names differ from the checkpoint")
                to_policy = [self.joint_names.index(name) for name in names]
                to_physx = [names.index(name) for name in self.joint_names]
                save_json(directory / "environment.json", metadata)
                torch.manual_seed(self.args.seed + index)
                policy.reset()

                def observation(current: dict, tick: int) -> dict:
                    obs = {"state": current["state"][to_policy].astype(np.float32),
                           "previous_actions": current["applied_command"][to_policy].astype(np.float32)}
                    if tick % 8 == 0:
                        if current["images"] is None:
                            raise RuntimeError("Missing RTX training-encoded cameras at image tick")
                        obs["images"] = current["images"]
                    return obs


                with self.lock:
                    self.publish(frame["raw"]["top"])
                    Image.fromarray(self.rgb).save(directory / "first.png")
                    np.savez_compressed(directory / "initial_state.npz", state=frame["state"],
                                        applied_command=frame["applied_command"],
                                        object_pose=frame["object_pose"],
                                        joint_names=np.asarray(self.joint_names))
                    self.phase = "stopping" if self.shutdown_requested else "running"
                initial = observation(frame, 0)
                latest_policy_images = initial["images"]
                if not self.wake.is_set():
                    policy.observe(initial, step=0)
                while not self.wake.is_set():
                    with self.lock:
                        if self.wake.is_set():
                            break
                        number = self.current["inference_count"] + 1
                        anchor = self.current["tick"]
                        Image.fromarray(self.rgb).save(directory / "inference" / f"{number:06d}.png")
                        anchor_state = frame["state"][to_policy].copy()
                        anchor_command = frame["applied_command"][to_policy].copy()
                        anchor_object_pose = frame["object_pose"].copy()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    actions = policy.infer()
                    torch.cuda.synchronize()
                    duration_ms = (time.perf_counter() - started) * 1000
                    if actions.shape != (8, len(names)) or not np.isfinite(actions).all():
                        raise ValueError("Model produced invalid actions")
                    np.savez_compressed(directory / "inference" / f"{number:06d}.npz", actions=actions,
                                        state=anchor_state, previous_actions=anchor_command,
                                        object_pose=anchor_object_pose,
                                        policy_images=np.stack([latest_policy_images[name]
                                                                for name in config["camera_names"]]),
                                        physics_engine="physx", renderer="isaacsim_headless",
                                        anchor_tick=anchor, inference_ms=duration_ms)
                    with self.lock:
                        self.current["inference_count"] = number
                        self.current["last_inference_ms"] = duration_ms
                    for action in actions:
                        tick_started = time.monotonic()
                        with self.lock:
                            if self.wake.is_set():
                                break
                            tick = self.current["tick"] + 1
                        frame = session.step(action[to_physx], tick)
                        with self.lock:
                            self.current["tick"] = tick
                            self.publish(frame["raw"]["top"])
                            obs = observation(frame, tick)
                            if "images" in obs:
                                latest_policy_images = obs["images"]
                        if self.wake.is_set():
                            break
                        policy.observe(obs, step=tick)
                        self.wake.wait(max(0, 1 / 30 - (time.monotonic() - tick_started)))
                self.finish_round(self.decision or "aborted", frame, directory)
                directory = None
                frame = None
            with self.lock:
                self.phase = "finished"
                self.persist_summary()
        except Exception as exc:
            traceback.print_exc()
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.phase = "error"
                if directory is not None:
                    self.finish_round(self.decision or "error", frame, directory)
                self.persist_summary()
        finally:
            if session is not None:
                session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/root/sunyz/data/spd-sim/raw/toss_in_bin"))
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--model", choices=MODELS, default="baseline")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start-episode")
    parser.add_argument("--episodes", type=int, default=0, help="0: all indexed episodes; no per-round time limit")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--allow-diagnostic", action="store_true")
    parser.add_argument("--isaac-python", type=Path, default=ISAAC_PYTHON)
    parser.add_argument("--spd-source", type=Path, default=ROOT / "spd_source")
    parser.add_argument("--data-process-source", type=Path, default=TRAINING_SOURCE)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.episodes < 0 or args.seed < 0 or args.diffusion_steps <= 0 or not 0 < args.port < 65536:
        parser.error("Invalid episode count, seed, diffusion steps or port")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    args.dataset = args.dataset.expanduser().resolve()
    args.assets = args.assets.expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%fZ")
    args.output = args.output.expanduser().resolve() if args.output else ROOT / "outputs" / "physx-live" / stamp
    evaluation = Evaluation(args)
    server = make_server("127.0.0.1", args.port, evaluation)
    evaluation.thread.start()
    print(f"Live rollout: http://127.0.0.1:{args.port} | output: {args.output}", flush=True)

    def shutdown(signum, frame):
        evaluation.stop()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        server.serve_forever()
    finally:
        evaluation.stop()
        evaluation.thread.join()
        server.server_close()


if __name__ == "__main__":
    main()
