"""Control one headless Isaac Sim PhysX session from the SPD policy process."""
from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess

import numpy as np

from isaac_render_protocol import receive_packet, send_packet
from training_render import verify_runtime_path, verify_camera_config


class UnsupportedRawSceneError(RuntimeError):
    """Raw scene or physical importer cannot support a faithful reconstruction."""


class IsaacPhysxClient:
    def __init__(self, output: Path, isaac_python: Path, spd_source: Path, data_process_source: Path,
                 capture: str = "top_live"):
        if capture not in ("top_live", "policy_only"):
            raise ValueError(f"Unknown PhysX capture mode: {capture}")
        self.capture = capture
        verify_runtime_path(isaac_python)
        self.channel, child = socket.socketpair()
        self.channel.settimeout(600)
        self.log_path = output / "isaacsim-physx.log"
        self.log = self.log_path.open("w")
        self.process = None
        self.round_id = None
        self._shape = None
        env = os.environ.copy()
        for key in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONHOME", "PYTHONEXE", "PYTHONPATH", "MUJOCO_GL"):
            env.pop(key, None)
        env["PYTHONNOUSERSITE"] = "1"
        try:
            self.process = subprocess.Popen(
                [str(isaac_python), str(Path(__file__).with_name("isaac_physx_worker.py")),
                 "--fd", str(child.fileno()), "--spd-source", str(spd_source),
                 "--data-process-source", str(data_process_source), "--capture", capture],
                env=env, pass_fds=(child.fileno(),), stdout=self.log, stderr=subprocess.STDOUT,
            )
            child.close()
            ready, _ = self._receive()
            self.runtime = ready["runtime"]
            self.alignment = ready["alignment"]
            if (self.runtime.get("physics") != "physx"
                    or self.runtime.get("headless") is not True
                    or self.runtime.get("render_mode") != "RayTracedLighting"
                    or self.runtime.get("capture_mode") != capture):
                raise RuntimeError("Isaac Sim must run headless PhysX with the requested RTX capture mode")
        except BaseException:
            child.close()
            self.close()
            raise

    def _receive(self):
        try:
            response, payload = receive_packet(self.channel)
        except (EOFError, OSError) as exc:
            raise RuntimeError(f"Isaac Sim PhysX process disconnected/timed out; see {self.log_path}") from exc
        if response.get("ok") is not True:
            error = f"Isaac Sim PhysX failed: {response.get('error')}; see {self.log_path}"
            if response.get("error_kind") == "unsupported_raw_scene":
                raise UnsupportedRawSceneError(error)
            raise RuntimeError(error)
        return response, payload

    def _frame(self, response, payload, tick):
        if (response.get("round_id") != self.round_id
                or type(response.get("tick")) is not int or response["tick"] != tick):
            raise RuntimeError("Isaac Sim returned a frame from the wrong round or control tick")
        anchor = tick % 8 == 0
        raw_count = (3 if anchor else 1) if self.capture == "top_live" else 0
        policy_count = 3 if anchor else 0
        count = raw_count + policy_count
        if (response.get("capture_mode") != self.capture
                or response.get("policy_images") is not anchor
                or type(response.get("raw_count")) is not int or response["raw_count"] != raw_count
                or type(response.get("policy_count")) is not int or response["policy_count"] != policy_count
                or response.get("dtype") != "uint8"
                or response.get("shape") != [count, *self._shape[1:]]
                or len(payload) != int(np.prod(self._shape[1:])) * count):
            raise RuntimeError("Isaac Sim returned invalid top/policy images")
        frame = np.frombuffer(payload, dtype=np.uint8).reshape((count, *self._shape[1:]))
        names = ("top", "left_wrist", "right_wrist")
        raw = dict(zip(names[:raw_count], frame[:raw_count], strict=True))
        policy = ({name: np.ascontiguousarray(image.transpose(2, 0, 1))
                   for name, image in zip(names, frame[raw_count:], strict=True)} if anchor else None)
        state = np.asarray(response["state"], dtype=np.float64)
        command = np.asarray(response["applied_command"], dtype=np.float64)
        pose = np.asarray(response["object_pose"], dtype=np.float64)
        hands = np.asarray(response["hand_positions"], dtype=np.float64)
        if (state.shape != (54,) or command.shape != (54,) or pose.ndim != 2 or pose.shape[1] != 7
                or hands.shape != (2, 3) or not np.isfinite(hands).all()
                or not np.isfinite(state).all() or not np.isfinite(command).all() or not np.isfinite(pose).all()):
            raise RuntimeError("Isaac Sim returned invalid measured physical state or hand positions")
        return {"tick": tick, "raw": raw, "images": policy, "state": state,
                "applied_command": command, "object_pose": pose, "hand_positions": hands}

    def load(self, dataset: Path, episode: str, assets: Path, camera_config: dict,
             directory: Path, round_id: str):
        verify_camera_config(camera_config)
        self.round_id = round_id
        self._shape = (3, camera_config["height"], camera_config["width"], 3)
        send_packet(self.channel, {"operation": "load", "dataset": str(dataset), "episode": episode,
                                   "assets": str(assets), "camera_config": camera_config,
                                   "directory": str(directory), "round_id": round_id})
        response, payload = self._receive()
        if (response.get("operation") != "load" or response.get("stepped") is not False
                or response.get("episode") != episode or response.get("physics") != "physx"):
            raise RuntimeError("Isaac Sim loaded wrong physical episode or engine")
        if response["scene_metadata"]["rendering"]["capture_mode"] != self.capture:
            raise RuntimeError("Isaac Sim loaded a scene with the wrong capture mode")
        self.channel.settimeout(120)
        return response["scene_metadata"], response["joint_names"], self._frame(response, payload, 0)

    def step(self, target: np.ndarray, tick: int):
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (54,) or not np.isfinite(target).all() or tick <= 0:
            raise ValueError("Physical joint target must be finite [54] at positive tick")
        send_packet(self.channel, {"operation": "step", "round_id": self.round_id,
                                   "tick": tick, "target": target.tolist()})
        response, payload = self._receive()
        if response.get("operation") != "step" or response.get("stepped") is not True:
            raise RuntimeError("Isaac Sim did not advance the requested control tick")
        return self._frame(response, payload, tick)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            try:
                self.channel.settimeout(10)
                send_packet(self.channel, {"operation": "close"})
                self._receive()
                self.process.wait(timeout=30)
            except (OSError, EOFError, RuntimeError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.channel.close()
        self.log.close()
