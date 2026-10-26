#!/usr/bin/env python3
"""Headless Isaac Sim 6.0.0 PhysX rollout with live or policy-only RTX capture."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import sys
import traceback

import numpy as np
from physx_scene_builder import UnsupportedRawSceneError

from isaac_render_protocol import receive_packet, send_packet
from training_render import training_manifest, verify_implementation, verify_runtime_path, verify_camera_config


def _packet(channel, frame, metadata, capture):
    raw, policy = frame
    anchor = metadata["tick"] % 8 == 0
    raw_count = (3 if anchor else 1) if capture == "top_live" else 0
    policy_count = 3 if anchor else 0
    image_shape = (480, 640, 3)
    if raw_count:
        if raw is None or raw.shape != (raw_count, *image_shape) or raw.dtype != np.uint8:
            raise RuntimeError("Invalid Isaac Sim live RGB frame")
    elif raw is not None:
        raise RuntimeError("Policy-only capture must not transfer raw RGB")
    if policy_count:
        if policy is None or policy.shape != (policy_count, *image_shape) or policy.dtype != np.uint8:
            raise RuntimeError("Invalid Isaac Sim policy RGB frame")
    elif policy is not None:
        raise RuntimeError("Policy cameras must be present only on image ticks")
    images = (np.concatenate((raw, policy), axis=0) if raw_count and policy_count
              else raw if raw_count else policy)
    send_packet(channel, {"ok": True, "dtype": "uint8",
                          "shape": [raw_count + policy_count, *image_shape],
                          "raw_count": raw_count, "policy_count": policy_count,
                          "policy_images": anchor, "capture_mode": capture, **metadata},
                memoryview(images).cast("B") if images is not None else b"")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--spd-source", type=Path, required=True)
    parser.add_argument("--data-process-source", type=Path, required=True)
    parser.add_argument("--capture", choices=("top_live", "policy_only"), default="top_live")
    args = parser.parse_args()
    try:
        gpu_index = int(os.environ.get("SPD_ROLLOUT_GPU_ID", "0"))
    except ValueError as exc:
        raise ValueError("SPD_ROLLOUT_GPU_ID must be a nonnegative physical GPU index") from exc
    if gpu_index < 0:
        raise ValueError("SPD_ROLLOUT_GPU_ID must be a nonnegative physical GPU index")
    verify_runtime_path(Path(os.environ["ISAAC_PATH"]) / "python.sh")
    sys.path[:0] = [str(Path(__file__).resolve().parent / "data_collection/src/dexhand_contracts"),
                    str(args.spd_source / "src"), str(args.data_process_source / "src")]
    from isaacsim import SimulationApp

    launch = {"headless": True, "hide_ui": True, "active_gpu": gpu_index, "physics_gpu": None,
              "multi_gpu": False, "max_gpu_count": 1, "renderer": "RayTracedLighting",
              "anti_aliasing": 0, "sync_loads": True, "create_new_stage": True,
              "fast_shutdown": True, "disable_viewport_updates": True}
    app = SimulationApp(launch)
    channel = socket.socket(fileno=args.fd)
    renderer = scene = None
    failed = False
    try:
        import carb.settings
        import omni.kit.app
        from offline_rendering.isaacsim import _isaac_version, _probe_gpu, _replicator_version
        from collection.raw.raw_model import RawTrajectory, scan_raw_directory
        from isaac_headless_renderer import HeadlessRenderer
        from physx_scene import PhysxScene
        from render_initial_scene import RecordedVisualScene

        # One robot per worker runs faster without dispatching tiny PhysX jobs
        # to extra CPU threads. Keep the same solver, timestep and substeps.
        carb.settings.get_settings().set_int("/persistent/physics/numThreads", 0)

        implementation = verify_implementation(args.data_process_source)
        manifest = training_manifest()
        expected = manifest["renderer"]
        ext, rep_version = _replicator_version(app.app)
        kit_version = omni.kit.app.get_app().get_build_version()
        isaac_version = _isaac_version()
        for key, actual in (("isaacsim_version", isaac_version), ("kit_build_version", kit_version),
                            ("replicator_extension", ext), ("replicator_version", rep_version)):
            if actual != expected[key]:
                raise RuntimeError(f"Native Isaac Sim {key} differs from training: {actual} != {expected[key]}")
        device = _probe_gpu(gpu_index, "RTX 5090")
        runtime = {"name": "isaacsim", "isaacsim_version": isaac_version,
                   "kit_build_version": kit_version, "replicator_extension": ext,
                   "replicator_version": rep_version, "device": device,
                   "headless": True, "physics": "physx", "physics_device": "cpu",
                   "physics_worker_threads": 0,
                   "render_mode": "RayTracedLighting", "capture_mode": args.capture,
                   "launch": launch}
        alignment = {"training_manifest_fingerprint": manifest["fingerprint_sha256"],
                     "camera_config_sha256": manifest["camera_config_sha256"],
                     "source_and_dependency_match": implementation,
                     "matched_runtime_fields": ["isaacsim_version", "kit_build_version", "replicator_extension", "replicator_version"],
                     "intentional_differences": {
                         "headless": {"training": True, "rollout": True},
                         "physics": {"training": "not_initialized", "rollout": "PhysX CPU"},
                         "hardware": {"training": expected["device"], "local": device}},
                     "jpeg": {"quality": 50, "subsampling": 0, "encoder": "training vla_export.exporter._encode_rgb_jpeg"},
                     "visuals": "training-bound RGB scene renderer and camera settings; v4 FK via the historical raw reader",
                     "scope": "Headless RTX and the policy cameras match training settings; live PhysX changes physics behavior and is not pixel or dynamics equivalence."}
        send_packet(channel, {"ok": True, "runtime": runtime, "alignment": alignment})
        index = {}
        round_id = None
        tick = 0
        while True:
            request, payload = receive_packet(channel)
            if payload:
                raise ValueError("PhysX commands must contain JSON, never binary simulation state")
            operation = request["operation"]
            if operation == "close":
                if renderer is not None:
                    renderer.close()
                    renderer = None
                if scene is not None:
                    scene.close()
                    scene = None
                send_packet(channel, {"ok": True})
                break
            if operation == "load":
                if renderer is not None:
                    renderer.close()
                    renderer = None
                if scene is not None:
                    scene.close()
                    scene = None
                dataset = Path(request["dataset"]).resolve()
                episode = request["episode"]
                directory = Path(request["directory"]).resolve()
                config = request["camera_config"]
                verify_camera_config(config)
                round_id = request["round_id"]
                tick = 0
                scene = PhysxScene(dataset, episode, Path(request["assets"]), config, directory)
                if dataset not in index:
                    _, summaries, paths, _ = scan_raw_directory(dataset)
                    index[dataset] = ({Path(item["relative_path"]).stem: item for item in summaries}, paths)
                summaries, paths = index[dataset]
                summary = summaries[episode]
                if summary["scene"] != scene.metadata["scene_id"]:
                    raise RuntimeError("Visual and physical raw scene identities disagree")
                trajectory = RawTrajectory(paths[summary["id"]], summary)
                visuals = RecordedVisualScene(dataset, trajectory, config)
                renderer = HeadlessRenderer(visuals, config, scene.stage, capture=args.capture)
                frame = renderer.render(scene.state, scene.object_pose, 0)
                scene.metadata["rendering"] = {"source": "Isaac Sim headless RTX",
                                               "capture_mode": args.capture,
                                               "top_every_tick": args.capture == "top_live",
                                               "wrist_cameras_every_8_ticks": True,
                                               "camera_config": config, "scene_id": summary["scene"],
                                               "visual_mesh_count": len(visuals.meshes),
                                               "raw_fk_source": "validated v4 RawTrajectory",
                                               "version": isaac_version}
                _packet(channel, frame, {"operation": "load", "physics": "physx", "episode": episode,
                                         "round_id": round_id, "tick": 0, "stepped": False,
                                         "state": scene.state.tolist(),
                                         "applied_command": scene.applied_command.tolist(),
                                         "object_pose": scene.object_pose.tolist(),
                                         "hand_positions": scene.hand_positions.tolist(),
                                         "joint_names": scene.joint_names, "scene_metadata": scene.metadata},
                        args.capture)
            elif operation == "step":
                if scene is None or renderer is None or request["round_id"] != round_id:
                    raise RuntimeError("PhysX step does not match the active round")
                if request["tick"] != tick + 1:
                    raise RuntimeError("PhysX control ticks must be strictly consecutive")
                scene.step(np.asarray(request["target"], dtype=np.float64))
                tick += 1
                frame = (None, None) if args.capture == "policy_only" and tick % 8 else renderer.render(
                    scene.state, scene.object_pose, tick)
                _packet(channel, frame, {"operation": "step", "round_id": round_id, "tick": tick,
                                         "stepped": True, "state": scene.state.tolist(),
                                         "applied_command": scene.applied_command.tolist(),
                                         "object_pose": scene.object_pose.tolist(),
                                         "hand_positions": scene.hand_positions.tolist()}, args.capture)
            else:
                raise ValueError(f"Unknown PhysX operation: {operation}")
    except BaseException as exc:
        failed = True
        traceback.print_exc()
        try:
            send_packet(channel, {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                  "error_kind": "unsupported_raw_scene" if isinstance(
                                      exc, UnsupportedRawSceneError) else "runtime"})
        except OSError:
            pass
        raise
    finally:
        try:
            if renderer is not None:
                renderer.close()
            if scene is not None:
                scene.close()
        finally:
            channel.close()
            app.close(exit_code=1 if failed else 0)


if __name__ == "__main__":
    main()
