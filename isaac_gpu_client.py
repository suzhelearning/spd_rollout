"""Persistent GPU PhysX client using a bounded header-plus-N-frame protocol."""
from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess

import numpy as np

from isaac_physx_client import IsaacPhysxClient
from isaac_render_protocol import receive_packet, send_packet
from training_render import verify_camera_config, verify_runtime_path


class GPUWorkerError(RuntimeError):
    """A complete worker error packet; the transport remains synchronized."""

    def __init__(self, response: dict, log_path: Path):
        self.error_kind = response.get("error_kind")
        self.response = response
        super().__init__(f"GPU PhysX failed: {response.get('error')}; see {log_path}")


class IsaacGpuClient(IsaacPhysxClient):
    def __init__(self, output: Path, isaac_python: Path, spd_source: Path,
                 data_process_source: Path, cache_dir: Path):
        verify_runtime_path(isaac_python)
        self.capture = "policy_only"
        self.round_id = None
        self._shape = None
        self._count = 0
        self._tick = None
        self.process = None
        self.channel, child = socket.socketpair()
        self.channel.settimeout(600)
        self.log_path = output / "isaacsim-gpu-physx.log"
        self.log = self.log_path.open("w")
        env = os.environ.copy()
        for key in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONHOME", "PYTHONEXE", "PYTHONPATH", "MUJOCO_GL"):
            env.pop(key, None)
        env["PYTHONNOUSERSITE"] = "1"
        try:
            self.process = subprocess.Popen(
                [str(isaac_python), str(Path(__file__).with_name("gpu_physx_worker.py")),
                 "--fd", str(child.fileno()), "--spd-source", str(spd_source),
                 "--data-process-source", str(data_process_source), "--cache-dir", str(cache_dir)],
                env=env, pass_fds=(child.fileno(),), stdout=self.log, stderr=subprocess.STDOUT)
            child.close()
            ready, payload = self._receive()
            self.runtime, self.alignment = ready["runtime"], ready["alignment"]
            device = str(self.runtime.get("physics_device", "")).lower()
            if (payload or self.runtime.get("method") != "gpu_vector"
                    or self.runtime.get("physics") != "physx"
                    or self.runtime.get("headless") is not True
                    or self.runtime.get("render_mode") != "RayTracedLighting"
                    or self.runtime.get("capture_mode") != "policy_only"
                    or not device.startswith("cuda")):
                raise RuntimeError("Worker must explicitly report headless RTX and CUDA PhysX gpu_vector")
        except BaseException:
            child.close()
            self.close()
            raise

    def _receive(self):
        try:
            response, payload = receive_packet(self.channel)
        except (EOFError, OSError) as exc:
            raise RuntimeError(f"GPU PhysX disconnected/timed out; see {self.log_path}") from exc
        if response.get("ok") is not True:
            if payload:
                raise RuntimeError("Worker error packet must have no payload")
            raise GPUWorkerError(response, self.log_path)
        return response, payload

    def _batch(self, operation: str, tick: int):
        header, payload = self._receive()
        if (payload or header.get("operation") != operation
                or header.get("batch_id") != self.round_id
                or type(header.get("tick")) is not int or header["tick"] != tick
                or type(header.get("count")) is not int or header["count"] != self._count):
            raise RuntimeError("GPU PhysX returned an invalid batch header")
        frames, environments, joints = [], [], []
        for env_id in range(self._count):
            try:
                response, payload = self._receive()
            except GPUWorkerError as exc:
                raise RuntimeError("GPU PhysX failed after starting a frame batch") from exc
            if type(response.get("env_id")) is not int or response["env_id"] != env_id:
                raise RuntimeError("GPU PhysX returned a frame for the wrong environment")
            names = response.get("joint_names")
            metadata = response.get("scene_metadata")
            if (not isinstance(names, list) or len(names) != 54
                    or any(not isinstance(name, str) for name in names) or len(set(names)) != 54
                    or not isinstance(metadata, dict)):
                raise RuntimeError("GPU PhysX returned invalid joint or scene metadata")
            frames.append(self._frame(response, payload, tick))
            environments.append(metadata)
            joints.append(names)
        self._tick = tick
        return environments, joints, frames

    def load_batch(self, items: list[dict], assets: Path, camera_config: dict, batch_id: str):
        verify_camera_config(camera_config)
        if (not items or not isinstance(batch_id, str) or not batch_id
                or camera_config["height"] != 480 or camera_config["width"] != 640):
            raise ValueError("GPU batch requires nonempty items, batch ID and 640x480 cameras")
        normalized = []
        for item in items:
            if set(item) != {"dataset", "episode", "directory"}:
                raise ValueError("Each batch item must contain dataset, episode and directory")
            normalized.append({key: str(value) for key, value in item.items()})
        self.round_id, self._count, self._tick = batch_id, len(items), None
        self._shape = (3, 480, 640, 3)
        self.channel.settimeout(600)
        send_packet(self.channel, {"operation": "load_batch", "batch_id": batch_id,
                                   "items": normalized, "assets": str(assets), "camera_config": camera_config})
        result = self._batch("load_batch", 0)
        self.channel.settimeout(120)
        return result

    def step_batch(self, target: np.ndarray, tick: int):
        target = np.asarray(target, dtype=np.float64)
        if (self._tick is None or type(tick) is not int or tick != self._tick + 1
                or target.shape != (self._count, 54) or not np.isfinite(target).all()):
            raise ValueError("GPU target must be finite [N,54] at the next synchronized control tick")
        send_packet(self.channel, {"operation": "step_batch", "batch_id": self.round_id,
                                   "tick": tick, "target": target.tolist()})
        return self._batch("step_batch", tick)[2]
