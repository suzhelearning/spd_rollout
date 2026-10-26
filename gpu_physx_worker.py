#!/usr/bin/env python3
"""Persistent Isaac Sim app: batched CUDA PhysX and one RTX capture per image tick."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np

from isaac_render_protocol import receive_packet, send_packet
from physx_scene_builder import UnsupportedRawSceneError
from training_render import training_manifest, verify_camera_config, verify_implementation, verify_runtime_path


def _cuda_ordinal(physical):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return physical
    devices = [value.strip() for value in visible.split(",") if value.strip()]
    if str(physical) in devices:
        return devices.index(str(physical))
    if len(devices) == 1 and devices[0].startswith("GPU-"):
        return 0
    raise ValueError(f"Physical GPU {physical} is absent from CUDA_VISIBLE_DEVICES={visible!r}")


def _frames(world, renderer, operation, batch_id, tick):
    images = renderer.render(tick) if tick % 8 == 0 else [None] * len(world.environments)
    packets = []
    for index, (env, image) in enumerate(zip(world.environments, images, strict=True)):
        count = 3 if tick % 8 == 0 else 0
        if image is not None and (image.dtype != np.uint8 or image.shape != (3, 480, 640, 3) or not image.flags.c_contiguous):
            raise RuntimeError("Invalid batch policy image array")
        metadata = {"ok": True, "operation": operation, "physics": "physx", "method": "gpu_vector",
                    "round_id": batch_id, "batch_id": batch_id, "tick": tick, "env_id": index,
                    "episode": env["metadata"]["episode_id"], "stepped": operation == "step_batch",
                    "capture_mode": "policy_only", "policy_images": tick % 8 == 0,
                    "raw_count": 0, "policy_count": count, "dtype": "uint8", "shape": [count, 480, 640, 3],
                    "state": world.state[index].tolist(), "applied_command": world.applied_command[index].tolist(),
                    "object_pose": env["object_pose"].tolist(), "hand_positions": env["hand_positions"].tolist(),
                    "joint_names": env["names"], "scene_metadata": env["metadata"],
                    "global_physics_substeps": world.global_substeps, "global_rgb_captures": renderer.capture_count}
        packets.append((metadata, memoryview(image).cast("B") if image is not None else b""))
    return packets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--spd-source", type=Path, required=True)
    parser.add_argument("--data-process-source", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent / "outputs/scene-cache")
    args = parser.parse_args()
    physical_gpu = int(os.environ.get("SPD_ROLLOUT_GPU_ID", "0"))
    if physical_gpu < 0:
        raise ValueError("SPD_ROLLOUT_GPU_ID must be nonnegative")
    ordinal = _cuda_ordinal(physical_gpu)
    device = f"cuda:{ordinal}"
    verify_runtime_path(Path(os.environ["ISAAC_PATH"]) / "python.sh")
    sys.path[:0] = [str(Path(__file__).resolve().parent / "data_collection/src/dexhand_contracts"),
                    str(args.spd_source / "src"), str(args.data_process_source / "src")]
    # Partitioning must be enabled before the RTX backend is created.
    sys.argv.append("--/renderer/scenePartitioning/enabled=true")
    from isaacsim import SimulationApp
    launch = {"headless": True, "hide_ui": True, "active_gpu": physical_gpu, "physics_gpu": ordinal,
              "multi_gpu": False, "max_gpu_count": 1, "renderer": "RayTracedLighting",
              "anti_aliasing": 0, "sync_loads": True, "create_new_stage": True,
              "fast_shutdown": True, "disable_viewport_updates": True}
    app = SimulationApp(launch)
    channel = socket.socket(fileno=args.fd)
    world = renderer = None
    failed = False
    try:
        import carb.settings
        import omni.kit.app
        from offline_rendering.isaacsim import _isaac_version, _probe_gpu, _replicator_version
        from gpu_physx_world import GpuPhysxWorld, PreparedSceneCache
        from gpu_rgb_renderer import GpuRgbRenderer

        implementation = verify_implementation(args.data_process_source)
        manifest = training_manifest()
        expected = manifest["renderer"]
        extension, replicator_version = _replicator_version(app.app)
        kit_version = omni.kit.app.get_app().get_build_version()
        isaac_version = _isaac_version()
        for key, actual in (("isaacsim_version", isaac_version), ("kit_build_version", kit_version),
                            ("replicator_extension", extension), ("replicator_version", replicator_version)):
            if expected[key] != actual:
                raise RuntimeError(f"Native Isaac runtime differs from training: {key}={actual} != {expected[key]}")
        gpu = _probe_gpu(physical_gpu, "RTX 5090")
        if not carb.settings.get_settings().get_as_bool("/renderer/scenePartitioning/enabled"):
            raise RuntimeError("RTX scene partition startup setting was not accepted")
        # Configure physics only on the actual batch stage. Configuring the
        # disposable startup stage crashes SyntheticData initialization in Isaac 6.
        world = GpuPhysxWorld(PreparedSceneCache(args.cache_dir), device)
        runtime = {"name": "isaacsim", "isaacsim_version": isaac_version,
                   "kit_build_version": kit_version, "replicator_extension": extension,
                   "replicator_version": replicator_version, "device": gpu,
                   "headless": True, "physics": "physx", "physics_device": device,
                   "physics_initialization": "load_batch", "method": "gpu_vector",
                   "visual_state_sync": "measured GPU tensors to USD",
                   "render_mode": "RayTracedLighting", "capture_mode": "policy_only",
                   "scene_partitioning": True, "launch": launch, "pid": os.getpid(),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cache_dir": str(args.cache_dir)}
        alignment = {"training_manifest_fingerprint": manifest["fingerprint_sha256"],
                     "camera_config_sha256": manifest["camera_config_sha256"],
                     "source_and_dependency_match": implementation,
                     "matched_runtime_fields": ["isaacsim_version", "kit_build_version", "replicator_extension", "replicator_version"],
                     "intentional_differences": {"physics": {"training": "not_initialized", "rollout": "GPU PhysX"},
                                                 "hardware": {"training": expected["device"], "local": gpu}},
                     "jpeg": {"quality": 50, "subsampling": 0, "encoder": "training vla_export.exporter._encode_rgb_jpeg"},
                     "scope": "Pinned visuals, cameras and JPEG; independent GPU physics, not CPU or training dynamics equivalence."}
        send_packet(channel, {"ok": True, "runtime": runtime, "alignment": alignment})
        raw_index = {}
        batch_id = None
        tick = 0
        while True:
            request, payload = receive_packet(channel)
            if payload:
                raise ValueError("GPU physics commands must contain JSON only")
            operation = request.get("operation")
            if operation == "close":
                if renderer is not None:
                    renderer.close()
                world.close()
                send_packet(channel, {"ok": True})
                break
            try:
                started = time.perf_counter()
                if operation == "load_batch":
                    if renderer is not None:
                        renderer.clear()
                    batch_id = None
                    if not isinstance(request.get("batch_id"), str) or not request["batch_id"]:
                        raise ValueError("load_batch requires a nonempty batch_id")
                    verify_camera_config(request["camera_config"])
                    world.load(request["items"], Path(request["assets"]), request["camera_config"])
                    world.initialize()
                    if renderer is None:
                        renderer = GpuRgbRenderer(world, request["camera_config"], raw_index)
                    else:
                        renderer.load(request["camera_config"])
                    world.initialize()
                    batch_id, tick = request["batch_id"], 0
                    packets = _frames(world, renderer, operation, batch_id, tick)
                    load_seconds = time.perf_counter() - started
                    for env in world.environments:
                        env["metadata"]["batch_load_seconds"] = load_seconds
                        env["metadata"]["worker_pid"] = os.getpid()
                        env["metadata"]["method"] = "gpu_vector"
                        env["directory"].mkdir(parents=True, exist_ok=True)
                        (env["directory"] / "scene_metadata.json").write_text(json.dumps(env["metadata"], ensure_ascii=False, indent=2) + "\n")
                elif operation == "step_batch":
                    if renderer is None or batch_id is None or request.get("batch_id") != batch_id:
                        raise ValueError("step_batch does not match the active batch")
                    if type(request.get("tick")) is not int or request["tick"] != tick + 1:
                        raise ValueError("GPU batch control ticks must be strictly consecutive")
                    world.step(np.asarray(request["target"], dtype=np.float64))
                    tick += 1
                    packets = _frames(world, renderer, operation, batch_id, tick)
                else:
                    raise ValueError(f"Unknown GPU worker operation: {operation}")
                header = {"ok": True, "operation": operation, "batch_id": batch_id,
                          "tick": tick, "count": len(packets), "wall_seconds": time.perf_counter() - started,
                          "global_physics_substeps": world.global_substeps, "global_rgb_captures": renderer.capture_count}
            except Exception as exc:
                traceback.print_exc()
                # A complete error packet is recoverable. No success header was
                # emitted yet, so a client can isolate invalid sources and reload.
                cleanup_errors = []
                if renderer is not None:
                    try:
                        renderer.clear()
                    except Exception as cleanup:
                        cleanup_errors.append(str(cleanup))
                try:
                    world.reset()
                except Exception as cleanup:
                    cleanup_errors.append(str(cleanup))
                batch_id = None
                send_packet(channel, {"ok": False, "operation": operation,
                    "batch_id": request.get("batch_id"), "error": f"{type(exc).__name__}: {exc}",
                    "error_kind": "unsupported_raw_scene" if isinstance(exc, UnsupportedRawSceneError) else "runtime",
                    "cleanup_errors": cleanup_errors})
                continue
            # Transport failures after a success header are fatal, not disguised
            # as recoverable errors in the middle of a header + N packet stream.
            send_packet(channel, header)
            for metadata, image_bytes in packets:
                send_packet(channel, metadata, image_bytes)
    except BaseException as exc:
        failed = True
        traceback.print_exc()
        try:
            send_packet(channel, {"ok": False, "error": f"{type(exc).__name__}: {exc}", "error_kind": "runtime"})
        except OSError:
            pass
        raise
    finally:
        try:
            if renderer is not None:
                renderer.close()
            if world is not None:
                world.close()
        finally:
            channel.close()
            try:
                app.close(exit_code=1 if failed else 0)
            finally:
                if renderer is not None:
                    renderer.release_texture_assets()


if __name__ == "__main__":
    main()
