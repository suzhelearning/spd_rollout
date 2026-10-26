#!/usr/bin/env python3
"""Restore a recorded SPD-raw scene and render its first observation in Isaac Sim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
from PIL import Image
from training_render import verify_implementation, verify_runtime, verify_camera_config, training_jpeg


class RecordedVisualScene:
    """Validated v4 raw visuals and FK, presented in the training renderer's frame format."""

    def __init__(self, root: Path, trajectory, camera_config: dict):
        from raw_data import scene as training_scene

        self.trajectory = trajectory
        self._training_scene = training_scene
        self._pose_count = len(trajectory.scene["pose_node_ids"])
        self.node_ids = tuple(trajectory.node_ids)
        nodes = [SimpleNamespace(identifier=name) for name in self.node_ids]
        scene_path = root / "scenes"
        assets_dir = root / "assets"
        self.bindings = training_scene._parse_cameras(camera_config, nodes, scene_path)
        meshes, mesh_node_ids, mesh_node_indices = [], [], []
        cache, verified_resources = {}, set()
        for node_index, node in enumerate(trajectory.scene["nodes"]):
            for visual in node["visuals"]:
                visible = visual.get("visible", True)
                if not isinstance(visible, bool):
                    raise ValueError(f"Visual visibility for {node['id']!r} must be boolean")
                if not visible:
                    continue
                asset_path = training_scene._resolve_visual_asset(root, assets_dir, visual, scene_path)
                template = cache.get(asset_path)
                if template is None:
                    # These v4 assets have the same single-primitive glTF layout
                    # accepted by the training loader; do not reinterpret their
                    # materials or flatten them with a second glTF decoder.
                    template = training_scene._load_gltf_mesh(asset_path, assets_dir, verified_resources)
                    cache[asset_path] = template
                mesh = template.copy(include_cache=False, include_visual=True)
                mesh.apply_transform(training_scene._visual_transform(visual, scene_path, node["id"]))
                if not np.isfinite(mesh.vertices).all():
                    raise ValueError(f"Visual for {node['id']!r} produces non-finite vertices")
                meshes.append(mesh)
                mesh_node_ids.append(node["id"])
                mesh_node_indices.append(node_index)
        self.meshes = tuple(meshes)
        self.mesh_node_ids = tuple(mesh_node_ids)
        self._mesh_node_indices = np.asarray(mesh_node_indices, dtype=np.intp)

    def frame(self, state: np.ndarray, object_pose: np.ndarray):
        state = self._training_scene._state_vector(state)
        poses = self._training_scene._object_pose_array(object_pose, self._pose_count)
        norms = np.linalg.norm(poses[:, 3:], axis=1)
        if not np.isfinite(norms).all() or np.any(norms <= np.finfo(np.float64).eps):
            raise ValueError("Simulation object quaternion must have non-zero finite norm")
        # The validated v4 trajectory owns the FK; training RawScene only
        # accepts v5/v6 and must not be instantiated against a relabelled v4.
        # Unlike recorded-row reads, this always evaluates the supplied live
        # measurements (including object world poses).
        world = self.trajectory._world_from_rows(state[None], poses[None], norms[None])[0]
        node_positions = np.ascontiguousarray(world[:, :3, 3], dtype=np.float32)
        node_rotations = np.ascontiguousarray(world[:, :3, :3], dtype=np.float32)
        camera_positions = np.empty((len(self.bindings), 3), dtype=np.float32)
        camera_rotations = np.empty((len(self.bindings), 3, 3), dtype=np.float32)
        for camera_index, binding in enumerate(self.bindings):
            transform = world[binding.parent_index] @ binding.local_transform
            camera_positions[camera_index] = transform[:3, 3]
            camera_rotations[camera_index] = transform[:3, :3]
        return self._training_scene.SceneFrame(
            positions=node_positions[self._mesh_node_indices],
            rotations=node_rotations[self._mesh_node_indices],
            camera_positions=camera_positions,
            camera_rotations=camera_rotations,
            node_positions=node_positions,
            node_rotations=node_rotations,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--spd-source", type=Path, required=True)
    parser.add_argument("--data-process-source", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parent / "data_collection/src/dexhand_contracts"),
                    str(args.spd_source / "src"), str(args.data_process_source / "src")]
    from collection.raw.raw_model import RawTrajectory, scan_raw_directory
    from raw_data import CAMERA_NAMES
    from offline_rendering.isaacsim import IsaacSimRuntime
    implementation = verify_implementation(args.data_process_source)

    request = json.loads(args.request.read_text())
    output = args.request.parent
    root, summaries, paths, skipped = scan_raw_directory(Path(request["dataset"]))
    selected = next((s for s in summaries if Path(s["relative_path"]).stem == request["episode"]), None)
    if selected is None:
        raise ValueError(f"Episode is unavailable: {request['episode']}; rejected entries: {skipped}")
    trajectory = RawTrajectory(paths[selected["id"]], selected)
    if trajectory.profile != "teleop_command" or not trajectory.recorded_action:
        raise ValueError("This checkpoint needs recorded applied absolute commands")
    world = trajectory.world_transforms(0, 1)[0]
    camera_config = request["camera_config"]
    verify_camera_config(camera_config)
    scene = RecordedVisualScene(root, trajectory, camera_config)
    camera_world = np.stack([world[b.parent_index] @ b.local_transform for b in scene.bindings])
    with h5py.File(trajectory.path, "r") as handle:
        state = handle["trajectory/state"][0]
        object_pose = handle["trajectory/object_pose"][0]
    frame = scene.frame(state, object_pose)
    with IsaacSimRuntime(0, "RTX 5090") as runtime:
        alignment = verify_runtime(runtime.metadata, implementation)
        (output / "render_alignment.json").write_text(json.dumps(alignment, indent=2) + "\n")
        with runtime.create_renderer(scene, camera_config["width"], camera_config["height"],
                                     znear=camera_config["near_m"], fovys=[camera_config["fovy_deg"]] * 3) as renderer:
            rgb = renderer.render_rgb(frame).copy()
            # Match the Q50 JPEG input distribution used by the training export.
            decoded = []
            for name, image in zip(CAMERA_NAMES, rgb, strict=True):
                Image.fromarray(image).save(output / f"{name}.png")
                payload, pixels = training_jpeg(image)
                (output / f"{name}.jpg").write_bytes(payload)
                decoded.append(pixels)
            Image.fromarray(np.concatenate(list(rgb), axis=1)).save(output / "initial_cameras.png")
            with h5py.File(trajectory.path, "r") as handle:
                state = handle["trajectory/state"][0]
                command = handle["trajectory/action"][0]
                object_pose = handle["trajectory/object_pose"][0]
            dataset = json.loads((root / "dataset.json").read_text())
            np.savez_compressed(output / "observation.npz", images=np.stack(decoded), state=state,
                                previous_actions=command, object_pose=object_pose,
                                joint_names=np.asarray(dataset["joint_names"]),
                                node_world=world, camera_world=camera_world)
            report = {"dataset": str(root), "episode": str(trajectory.path), "scene_id": selected["scene"],
                      "raw_schema_version": dataset["schema_version"], "source_frame": 0,
                      "task": selected["task"], "source_frames": trajectory.frames,
                      "mesh_count": len(scene.meshes), "camera_names": list(CAMERA_NAMES),
                      "camera_config": camera_config, "diagnostic_only": True,
                      "runtime": runtime.metadata,
                      "scope": "Exact recorded visual scene and first-frame FK/object poses; no physics stepping or action execution."}
            (output / "scene_summary.json").write_text(json.dumps(report, indent=2) + "\n")
            trajectory.assert_unchanged()
            print(f"Rendered initial scene: {output}", flush=True)


if __name__ == "__main__":
    main()
