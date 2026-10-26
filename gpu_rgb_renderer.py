"""One RTX capture for all 3N policy cameras using the pinned mesh/camera authoring.

RTX scene partitions isolate images even with the training far plane of 1e6 m.
The one white intensity-500 dome is shared: duplicating domes changes illumination.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from training_render import training_jpeg, verify_camera_config
from isaac_headless_renderer import _PhysxRuntime
from offline_rendering import isaacsim as pinned
from raw_data import CAMERA_NAMES


@contextmanager
def _renderer_root(root):
    # Kit work is synchronous on this worker thread. Only authoring uses the
    # pinned module global; capture and teardown use stored per-instance paths.
    previous = pinned._ROOT_PATH
    pinned._ROOT_PATH = root
    try:
        yield
    finally:
        pinned._ROOT_PATH = previous


class _PartitionRenderer(pinned.IsaacSimRenderer):
    def __init__(self, root, partition, **kwargs):
        self.root = root
        self.partition = partition
        self._retired_texture_dirs = []
        with _renderer_root(root):
            super().__init__(**kwargs)

    def _build_scene(self):
        # Reuse pinned authoring; only the batch owner may execute graph updates.
        modules, stage = self._modules, self._stage
        if stage.GetPrimAtPath(self.root).IsValid():
            raise pinned.IsaacSimError(f"USD renderer root already exists: {self.root}")
        modules.UsdGeom.Xform.Define(stage, "/World")
        modules.UsdGeom.Xform.Define(stage, self.root)
        for child in ("Nodes", "Materials", "Cameras", "Lighting"):
            modules.UsdGeom.Xform.Define(stage, f"{self.root}/{child}")
        self._build_visuals()
        self._camera_ops = tuple(self._author_cameras())
        self._author_lighting()
        modules.rep.orchestrator.set_capture_on_play(False)
        for camera in CAMERA_NAMES:
            product = modules.rep.create.render_product(
                f"{self.root}/Cameras/{camera}", (self._width, self._height),
                force_new=True, name=f"spd_raw_{camera}")
            if getattr(product, "hydra_texture", None) is None:
                raise pinned.IsaacSimError(f"Replicator did not create camera product {camera!r}")
            product.hydra_texture.set_updates_enabled(False)
            stage.GetPrimAtPath(product.path).CreateAttribute(
                "omni:rtx:scenePartitioning:excludeBackgroundDomeLightInPartition",
                modules.Sdf.ValueTypeNames.Bool, custom=True).Set(False)
            annotator = modules.rep.annotators.get("rgb")
            annotator.attach(product)
            self._render_products.append(product)
            self._annotators.append(annotator)

    def _build_visuals(self):
        modules, stage = self._modules, self._stage
        for child in ("Nodes", "Materials"):
            modules.UsdGeom.Xform.Define(stage, f"{self.root}/{child}")
        node_paths, node_ops = {}, []
        for node_id in self._node_ids:
            path = f"{self.root}/Nodes/node_{node_id.encode('utf-8').hex()}"
            if path in node_paths.values():
                raise pinned.IsaacSimError(f"stable node ID encoding collided for {node_id!r}")
            xform = modules.UsdGeom.Xform.Define(stage, path)
            xform.GetPrim().CreateAttribute("spdRaw:nodeId", modules.Sdf.ValueTypeNames.String, custom=True).Set(node_id)
            node_paths[node_id] = path
            node_ops.append(xform.AddTransformOp())
        for index, (mesh, node_id) in enumerate(zip(self._meshes, self._mesh_node_ids, strict=True)):
            if node_id not in node_paths:
                raise pinned.IsaacSimError(f"mesh {index} maps to unknown node ID {node_id!r}")
            prim = modules.UsdGeom.Mesh.Define(stage, f"{node_paths[node_id]}/mesh_{index:06d}")
            self._author_mesh(prim, mesh, index)
        self._node_ops = tuple(node_ops)

    def clear_visuals(self):
        self._stage.RemovePrim(f"{self.root}/Nodes")
        self._stage.RemovePrim(f"{self.root}/Materials")
        self._node_ops = self._node_ids = self._meshes = self._mesh_node_ids = ()

    def reload(self, scene):
        self._require_open()
        self.clear_visuals()
        # RTX may retain texture assets after USD material removal. Retain their
        # files until app shutdown; a new directory also prevents stale ID reuse.
        self._retired_texture_dirs.append(self._texture_dir)
        self._texture_dir = TemporaryDirectory(prefix="spd-raw-isaacsim-")
        self._texture_paths.clear()
        self._node_ids = pinned._scene_node_ids(scene)
        self._meshes, self._mesh_node_ids = pinned._scene_mesh_bindings(scene, self._node_ids)
        with _renderer_root(self.root):
            self._build_visuals()

    def release_texture_assets(self):
        self._texture_paths.clear()
        self._texture_dir.cleanup()
        for directory in self._retired_texture_dirs:
            directory.cleanup()
        self._retired_texture_dirs.clear()

    def _author_cameras(self):
        from pxr import Sdf, UsdGeom
        UsdGeom.PrimvarsAPI(self._stage.GetPrimAtPath(self.root)).CreatePrimvar(
            "omni:scenePartition", Sdf.ValueTypeNames.Token, UsdGeom.Tokens.constant
        ).Set(self.partition)
        operations = tuple(super()._author_cameras())
        for name in CAMERA_NAMES:
            self._stage.GetPrimAtPath(f"{self.root}/Cameras/{name}").CreateAttribute(
                "omni:scenePartition", Sdf.ValueTypeNames.Token, custom=True
            ).Set(self.partition)
        return operations

    def _author_lighting(self):
        from pxr import Gf, UsdLux
        path = "/World/GpuPolicyLighting/Dome"
        if not self._stage.GetPrimAtPath(path).IsValid():
            dome = UsdLux.DomeLight.Define(self._stage, path)
            dome.CreateIntensityAttr().Set(500.)
            dome.CreateColorAttr().Set(Gf.Vec3f(1., 1., 1.))

    def _require_open(self):
        if self._closed:
            raise pinned.IsaacSimError("Partition renderer is closed")
        self._runtime._require_open()

    def update(self, frame):
        self._require_open()
        node_positions = pinned._numeric_array(frame.node_positions, (len(self._node_ids), 3), "frame.node_positions")
        node_rotations = pinned._numeric_array(frame.node_rotations, (len(self._node_ids), 3, 3), "frame.node_rotations")
        camera_positions = pinned._numeric_array(frame.camera_positions, (3, 3), "frame.camera_positions")
        camera_rotations = pinned._numeric_array(frame.camera_rotations, (3, 3, 3), "frame.camera_rotations")
        for op, position, rotation in zip(self._node_ops, node_positions, node_rotations, strict=True):
            op.Set(pinned._usd_matrix(self._modules, position, rotation))
        for op, position, rotation in zip(self._camera_ops, camera_positions, camera_rotations, strict=True):
            op.Set(pinned._usd_matrix(self._modules, position, rotation))

    def read_rgb(self):
        for camera, annotator in enumerate(self._annotators):
            image = annotator.get_data(device="cpu", do_array_copy=False)
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape not in ((480, 640, 3), (480, 640, 4)):
                raise RuntimeError(f"Invalid real RTX RGB for {self.root}, camera {camera}: {getattr(image, 'shape', None)}")
            np.copyto(self._rgb[camera], image[..., :3])
        return self._rgb

    def close(self):
        # The batch owner stops Replicator once. Never stop it once per camera/env.
        if self._closed:
            return
        self._closed = True
        try:
            for annotator, product in zip(self._annotators, self._render_products, strict=True):
                if annotator.is_attached:
                    annotator.detach(product)
                product.destroy()
            self._stage.RemovePrim(self.root)
        finally:
            self._annotators.clear()
            self._render_products.clear()


class GpuRgbRenderer:
    def __init__(self, world, camera_config, raw_index):
        import carb.settings

        verify_camera_config(camera_config)
        if not carb.settings.get_settings().get_as_bool("/renderer/scenePartitioning/enabled"):
            raise RuntimeError("RTX scene partitioning must be enabled before SimulationApp startup")
        self.world = world
        self._closed = False
        self._renderers = []
        self._visuals = []
        self._runtime = _PhysxRuntime(world.stage)
        self._camera_config = camera_config
        self._raw_index = raw_index
        self.capture_count = 0
        try:
            self.load(camera_config)
        except BaseException:
            self.close()
            self.release_texture_assets()
            raise

    def clear(self):
        for renderer in self._renderers:
            renderer.clear_visuals()
        self._visuals.clear()

    def load(self, camera_config):
        from collection.raw.raw_model import RawTrajectory, scan_raw_directory
        from pxr import Sdf, UsdGeom
        from render_initial_scene import RecordedVisualScene

        verify_camera_config(camera_config)
        if self._closed or camera_config != self._camera_config:
            raise ValueError("Persistent camera pool requires an open renderer and unchanged calibration")
        self._runtime._require_open()
        self.clear()
        self.capture_count = 0
        world, raw_index = self.world, self._raw_index
        # Only raw visuals are visible; all physical Gprims stay colliders.
        for prim in world.stage.Traverse():
            if prim.IsA(UsdGeom.Gprim):
                UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        settings = world.stage.DefinePrim("/Render/OmniverseGlobalRenderSettings")
        settings.CreateAttribute("omni:rtx:scenePartitioning:showAllPartitionsByDefault",
                                 Sdf.ValueTypeNames.Bool, custom=True).Set(False)
        for index, env in enumerate(world.environments):
            dataset = Path(env["metadata"]["episode_path"]).parent.parent
            episode = env["metadata"]["episode_id"]
            if dataset not in raw_index:
                _, summaries, paths, _ = scan_raw_directory(dataset)
                raw_index[dataset] = ({Path(s["relative_path"]).stem: s for s in summaries}, paths)
            summaries, paths = raw_index[dataset]
            summary = summaries[episode]
            if summary["scene"] != env["metadata"]["scene_id"]:
                raise RuntimeError("Visual and physics scene identities differ")
            trajectory = RawTrajectory(paths[summary["id"]], summary)
            if trajectory.profile != "teleop_command" or not trajectory.recorded_action:
                raise ValueError("Policy requires recorded applied absolute commands")
            visuals = RecordedVisualScene(dataset, trajectory, camera_config)
            if index == len(self._renderers):
                renderer = _PartitionRenderer(root=f"/World/GpuPolicy/env_{index:04d}",
                    partition=f"env_{index:04d}", runtime=self._runtime, scene=visuals,
                    width=640, height=480, znear=camera_config["near_m"], fovys=(camera_config["fovy_deg"],) * 3)
                self._renderers.append(renderer)
            else:
                self._renderers[index].reload(visuals)
            self._visuals.append(visuals)
            env["metadata"]["rendering"] = {"source": "Isaac Sim headless RTX",
                "capture_mode": "policy_only", "top_every_tick": False,
                "wrist_cameras_every_8_ticks": True, "camera_config": camera_config,
                "scene_id": summary["scene"], "visual_mesh_count": len(visuals.meshes),
                "raw_fk_source": "validated v4 RawTrajectory", "scene_partition": f"env_{index:04d}",
                "isolation": "RTX scene partitions with original source coordinates and far clipping plane",
                "lighting": "one shared white dome, training intensity 500",
                "stage_id": self._runtime._stage_id,
                "render_products": [product.path for product in self._renderers[index]._render_products],
                "global_captures_per_image_tick": 1, "batch_camera_count": 3 * len(world.environments)}
        for env in world.environments:
            env["metadata"]["rendering"]["resident_camera_count"] = 3 * len(self._renderers)
        # Keep unused slots alive, rendering empty partitions. Disabling updates
        # destroys native viewports in this SDK, even if Python products survive.
        for renderer in self._renderers:
            for product in renderer._render_products:
                product.hydra_texture.set_updates_enabled(True)

    def render(self, tick):
        if self._closed or type(tick) is not int or tick < 0 or tick % 8:
            raise ValueError("Batch RGB capture requires an open renderer at an image tick")
        self._runtime._require_open()
        for index, (visuals, env) in enumerate(zip(self._visuals, self.world.environments, strict=True)):
            self._renderers[index].update(visuals.frame(self.world.state[index], env["object_pose"]))
        rep = self._runtime._modules.rep
        rep.orchestrator.set_capture_on_play(False)
        rep.orchestrator.step(rt_subframes=8 if self.capture_count == 0 else 1,
                              pause_timeline=True, delta_time=0., wait_for_render=True)
        self.capture_count += 1
        # Read active slots only; inactive pool slots contain no scene geometry.
        return [np.stack([training_jpeg(image)[1] for image in self._renderers[index].read_rgb()])
                for index in range(len(self._visuals))]

    def close(self):
        if self._closed:
            return
        self._closed = True
        rep = self._runtime._modules.rep
        try:
            rep.orchestrator.set_capture_on_play(False)
            rep.orchestrator.stop()
            rep.orchestrator.wait_until_complete()
        finally:
            errors = []
            for renderer in self._renderers:
                try:
                    renderer.close()
                except Exception as exc:
                    errors.append(exc)
            self._visuals.clear()
            if self.world.stage is not None:
                self.world.stage.RemovePrim("/World/GpuPolicyLighting")
            if errors:
                raise RuntimeError(f"GPU renderer cleanup failed: {errors}")

    def release_texture_assets(self):
        """Call after SimulationApp closes, when native consumers are gone."""
        for renderer in self._renderers:
            renderer.release_texture_assets()
        self._renderers.clear()
