"""Prepared-source cache and one CUDA PhysX world for a synchronized batch.

All environments retain source coordinates. Collision groups (not distance)
isolate coincident environments, preserving fixed-joint world anchors exactly.
The caller owns SimulationApp and the RTX renderer, across all batches.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from physx_scene_builder import (
    UnsupportedRawSceneError, _ARM_NAMES, _CONTROL_HZ, _SUBSTEPS, _TIMESTEP,
    _assemble_scene, _hand_reach_from_geometry, _joint_spec, _sha256,
    _validate_objects, _vector, _xml_joint_name,
)

_CACHE_VERSION = 1
_TASKS = {"hang_mug", "pyramid", "spelling", "toss_in_bin", "unstack", "vowel_consonant_sort"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class PreparedSceneCache:
    """Validate raw/source/assets before using either our cache or an old USD."""

    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._legacy = None
        self._hashes = {}

    def digest(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in self._hashes:
            self._hashes[key] = _sha256(path)
        return self._hashes[key]

    def _legacy_candidates(self, episode_path):
        # A fresh --cache-dir plus this switch measures real MJCF preparation.
        # Existing entries in the requested cache directory remain usable.
        if os.environ.get("SPD_GPU_ALLOW_LEGACY_CACHE", "1") == "0":
            return ()
        if self._legacy is None:
            self._legacy = {}
            root = Path(__file__).resolve().parent.parent / "spd-rollout/outputs/baseline30k-six-20pct-20261023"
            for result_path in root.glob("shard*/**/result.json"):
                try:
                    result = json.loads(result_path.read_text())
                    metadata_path = result.get("physical_environment")
                    if not isinstance(metadata_path, str):
                        continue
                    metadata = json.loads(Path(metadata_path).read_text())
                    self._legacy.setdefault(str(Path(metadata["episode_path"]).resolve()), []).append(metadata)
                except (OSError, ValueError, KeyError):
                    continue
        return self._legacy.get(str(episode_path), ())

    def prepare(self, item, assets, camera_config):
        from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        started = time.perf_counter()
        dataset = Path(item["dataset"]).expanduser().resolve()
        episode = item["episode"]
        if not isinstance(episode, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", episode):
            raise ValueError("Invalid episode ID")
        assets = Path(assets).expanduser().resolve()
        index = json.loads((dataset / "dataset.json").read_text())
        if index.get("schema_version") != 4 or index.get("task_id") not in _TASKS or episode not in index["episodes"]:
            raise ValueError("Episode must be indexed in a supported v4 raw task dataset")
        names = index["joint_names"]
        if len(names) != 54 or len(set(names)) != 54:
            raise ValueError("Expected 54 distinct dataset joints")
        episode_path = dataset / "episodes" / f"{episode}.h5"
        with h5py.File(episode_path, "r") as source:
            if source.attrs["episode_id"] != episode or source.attrs["schema_version"] != 4 or not source.attrs["recorded_action"]:
                raise ValueError("Episode header identity/schema/action mismatch")
            scene_id = str(source.attrs["scene_id"])
            if not re.fullmatch(r"[0-9a-f]{64}", scene_id):
                raise ValueError("Invalid raw scene ID")
            state = _vector(source["trajectory/state"][0], (54,), "first-frame state")
            action = _vector(source["trajectory/action"][0], (54,), "first-frame action")
            poses = np.asarray(source["trajectory/object_pose"][0], dtype=np.float64)
        scene_file = dataset / "scenes" / f"{scene_id}.json"
        if self.digest(scene_file) != scene_id:
            raise ValueError("Episode scene hash mismatch")
        raw = json.loads(scene_file.read_text())
        if raw.get("schema_version") != 4:
            raise ValueError("Recorded scene schema mismatch")
        mobility = _validate_objects(raw, index["task_id"])
        if [j["id"] for j in sorted(raw["joints"], key=lambda j: j["state_index"])] != names:
            raise ValueError("Scene and dataset joint ordering differ")
        pose_nodes = raw["pose_node_ids"]
        if poses.shape != (len(pose_nodes), 7) or not np.isfinite(poses).all():
            raise ValueError("Invalid initial object poses")
        robot_dir = assets / "robots/tianji_wuji2_hands1"
        robot_xml = robot_dir / "tianji_wuji2_hands1.xml"
        urdf = robot_dir / "tianji_wuji2_hands1.urdf"
        lower, upper, velocity, efforts = _joint_spec(urdf, names)
        if set(efforts) != set(_ARM_NAMES):
            raise ValueError("All fourteen arm joints must be present")
        if any(np.any(v < lower - 1e-5) or np.any(v > upper + 1e-5) for v in (state, action)):
            raise ValueError("Recorded initial state/action exceeds URDF limits")
        xml, provenance = _assemble_scene(robot_xml, urdf, assets, raw, camera_config,
                                          names, lower, upper, efforts, index["task_id"])
        reach, reach_sides = _hand_reach_from_geometry(xml)
        ET.indent(xml)
        xml_bytes = ET.tostring(xml, encoding="utf-8", xml_declaration=True)
        xml_hash = hashlib.sha256(xml_bytes).hexdigest()
        dependencies = {str(robot_xml): self.digest(robot_xml), str(urdf): self.digest(urdf)}
        for source in provenance["objects"].values():
            if source.get("manifest"):
                manifest = Path(source["manifest"]).resolve()
                dependencies[str(manifest)] = self.digest(manifest)
        for node in xml.iter():
            filename = node.get("file")
            if filename:
                dependency = Path(filename)
                if not dependency.is_absolute():
                    raise ValueError(f"Prepared asset dependency must be absolute: {filename}")
                dependencies[str(dependency.resolve())] = self.digest(dependency)
        identity = {"version": _CACHE_VERSION, "dataset": str(dataset), "episode": episode,
                    "episode_sha256": self.digest(episode_path), "scene_id": scene_id,
                    "assets": str(assets), "dependencies": dependencies, "camera_config": camera_config,
                    "scene_xml_sha256": xml_hash, "builder_sha256": self.digest(Path(__file__).with_name("physx_scene_builder.py"))}
        key = hashlib.sha256(_canonical(identity).encode()).hexdigest()
        folder = self.directory / key
        folder.mkdir(parents=True, exist_ok=True)
        manifest_path = folder / "prepared.json"
        prepared_usd = folder / "prepared.usdc"
        metadata = None
        cache_source = "cold_import"
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text())
            if saved.get("identity") == identity and prepared_usd.exists() and self.digest(prepared_usd) == saved.get("usd_sha256"):
                metadata = saved["metadata"]
                cache_source = "gpu_prepared"
        if metadata is None:
            old = None
            for candidate in self._legacy_candidates(episode_path):
                usd = Path(candidate.get("scene_usd", ""))
                old_xml = Path(candidate.get("scene_xml", ""))
                if (candidate.get("scene_id") == scene_id and candidate.get("camera_config") == camera_config
                        and candidate.get("robot_xml_sha256") == dependencies[str(robot_xml)]
                        and candidate.get("robot_urdf_sha256") == dependencies[str(urdf)]
                        and old_xml.is_file() and self.digest(old_xml) == xml_hash
                        and usd.is_file() and self.digest(usd) == candidate.get("scene_usd_sha256")
                        and all(Path(p).stat().st_mtime_ns <= usd.stat().st_mtime_ns for p in dependencies)
                        and np.array_equal(candidate.get("initial_state"), state)
                        and np.array_equal(candidate.get("initial_action"), action)
                        and np.array_equal(candidate.get("initial_object_pose"), poses)):
                    old = candidate
                    break
            (folder / "scene.xml").write_bytes(xml_bytes)
            if old is not None:
                usd_path = Path(old["scene_usd"])
                cache_source = "legacy_initial_usd"
            else:
                (folder / "imported").mkdir(parents=True, exist_ok=True)
                config = MJCFImporterConfig(mjcf_path=str(folder / "scene.xml"), usd_path=str(folder / "imported"),
                                            import_scene=True, merge_mesh=False, collision_from_visuals=False,
                                            fix_base=True, run_asset_transformer=False, run_multi_physics_conversion=True)
                usd_path = Path(MJCFImporter(config).import_mjcf()).resolve()
            source_stage = Usd.Stage.Open(str(usd_path))
            if source_stage is None:
                raise RuntimeError(f"Cannot open prepared USD {usd_path}")
            # Flatten resolves imported layer asset paths. NamespaceEditor remaps
            # all cross-root joint relationships when wrapping the whole scene.
            stage = Usd.Stage.Open(source_stage.Flatten())
            roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
            wrists = []
            for name in ("l__l_wrist", "r__r_wrist"):
                found = [p.GetPath() for p in stage.Traverse() if p.GetName() == name and p.HasAPI(UsdPhysics.RigidBodyAPI)]
                if len(found) != 1:
                    raise UnsupportedRawSceneError(f"Expected one physical wrist {name}: {found}")
                wrists.append(found[0])
            roots = [p for p in roots if all(w.HasPrefix(p.GetPath()) for w in wrists)]
            if len(roots) != 1:
                raise UnsupportedRawSceneError("Expected one robot articulation containing both wrists")
            robot_path = roots[0].GetPath()
            bodies = []
            for node in pose_nodes:
                found = [p.GetPath() for p in stage.Traverse() if p.GetName() == node.removeprefix("body/") and p.HasAPI(UsdPhysics.RigidBodyAPI)]
                if len(found) != 1:
                    raise UnsupportedRawSceneError(f"Expected one dynamic object {node}: {found}")
                bodies.append(found[0])
            embedded = [p for p, (_, kind) in zip(bodies, mobility, strict=True) if p.HasPrefix(robot_path) and kind == "free"]
            if embedded:
                parent = stage.DefinePrim("/PhysxDynamicObjects", "Xform")
                for path in embedded:
                    editor = Usd.NamespaceEditor(stage)
                    editor.ReparentPrim(stage.GetPrimAtPath(path), parent)
                    if not editor.ApplyEdits():
                        raise RuntimeError(f"Cannot isolate free object {path}")
                bodies = [parent.GetPath().AppendChild(p.name) if p in embedded else p for p in bodies]
            for path, (_, kind), pose in zip(bodies, mobility, poses, strict=True):
                if kind == "slide":
                    joints = [p for p in stage.Traverse() if p.GetName() == f"{path.name}_slide" and p.IsA(UsdPhysics.PrismaticJoint)]
                    if len(joints) != 1:
                        raise UnsupportedRawSceneError(f"Missing drawer prismatic joint {path}")
                    joint = UsdPhysics.PrismaticJoint(joints[0])
                    if joint.GetBody1Rel().GetTargets() != [path] or not np.allclose([joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()], [0., .25], atol=1e-4):
                        raise UnsupportedRawSceneError(f"Incorrect drawer joint {path}")
                transform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
                translations = [op for op in transform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
                orientations = [op for op in transform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient]
                if len(translations) != 1 or len(orientations) != 1:
                    raise UnsupportedRawSceneError(f"Unsupported rigid object transform {path}")
                translations[0].Set(Gf.Vec3d(*map(float, pose[:3])))
                orientations[0].Set(Gf.Quatf(float(pose[3]), Gf.Vec3f(*map(float, pose[4:]))))
            for prim in list(stage.Traverse()):
                if prim.IsA(UsdPhysics.Scene):
                    stage.RemovePrim(prim.GetPath())
            for name in ("/Render", "/OmniverseKit_Persp"):
                if stage.GetPrimAtPath(name).IsValid():
                    stage.RemovePrim(name)
            children = list(stage.GetPseudoRoot().GetChildren())
            wrapper = UsdGeom.Xform.Define(stage, "/Prepared").GetPrim()
            for prim in children:
                editor = Usd.NamespaceEditor(stage)
                editor.ReparentPrim(prim, wrapper)
                if not editor.ApplyEdits():
                    raise RuntimeError(f"Cannot namespace imported root {prim.GetPath()}")
            stage.SetDefaultPrim(wrapper)
            if not stage.Export(str(prepared_usd)):
                raise RuntimeError("Unable to export prepared source USD")
            metadata = {"reconstruction": "independent GPU PhysX simulation of recorded v4 scene",
                        "distribution_shift": "GPU PhysX dynamics/contact and imported USD differ from source simulator; not CPU-equivalent",
                        "episode_id": episode, "episode_path": str(episode_path), "scene_id": scene_id,
                        "task_id": index["task_id"], "pose_node_ids": pose_nodes, "joint_names": names,
                        "hand_position_definition": "measured world origins of articulated physical l__l_wrist and r__r_wrist links",
                        "hand_reference": "wrist_link", "hand_reach_m": reach, "hand_reach_by_side_m": reach_sides,
                        "hand_reach_derivation": "same generated collision-geometry bound as PhysxScene",
                        "scene_xml": str(folder / "scene.xml"), "scene_sha256": xml_hash,
                        "scene_usd": str(prepared_usd), "robot_xml_sha256": dependencies[str(robot_xml)],
                        "robot_urdf_sha256": dependencies[str(urdf)], "asset_sources": provenance,
                        "initial_state": state.tolist(), "initial_action": action.tolist(), "initial_object_pose": poses.tolist(),
                        "initial_velocity": "unrecorded; all robot and object velocities initialized to zero",
                        "initial_servo_target": "recorded first-frame absolute action",
                        "command_limits": "URDF position and velocity limits relative to previous applied command at 30 Hz",
                        "arm_servo_assumption": "14 synthesized arm servos kp=100 kv=10; 40 original hand servos imported",
                        "camera_config": camera_config,
                        "prepared_paths": {"articulation": str(robot_path), "objects": list(map(str, bodies)), "hands": list(map(str, wrists))}}
            manifest = {"identity": identity, "usd_sha256": self.digest(prepared_usd), "metadata": metadata}
            temporary = manifest_path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(manifest_path)
        metadata = json.loads(json.dumps(metadata))
        metadata.update(cache_hit=cache_source != "cold_import", cache_source=cache_source,
                        cache_key=key, load_seconds=time.perf_counter() - started,
                        cache_validation={"source_scene_assets_sha256": key, "legacy_dependency_mtime_guard": cache_source == "legacy_initial_usd"})
        return {"usd": prepared_usd, "metadata": metadata, "names": names, "state": state,
                "action": action, "poses": poses, "lower": lower, "upper": upper,
                "velocity_step": velocity / _CONTROL_HZ, "directory": Path(item["directory"]).resolve()}


class GpuPhysxWorld:
    """One batched articulation view, one rigid view, one global 16-substep call."""

    def __init__(self, cache: PreparedSceneCache, device: str):
        self.cache = cache
        self.device = device
        self.stage = None
        self.environments = []
        self._articulation = self._rigids = None
        self.global_substeps = 0

    def load(self, items, assets, camera_config):
        import omni.usd
        from isaacsim.core.experimental.prims import Articulation, RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import UsdGeom, UsdPhysics

        if not isinstance(items, list) or not items:
            raise ValueError("load_batch requires a nonempty items list")
        # Preparation never creates or advances a CPU simulation.
        prepared = [self.cache.prepare(item, assets, camera_config) for item in items]
        if any(p["names"] != prepared[0]["names"] for p in prepared):
            raise ValueError("Batch dataset joint names/order must agree")
        self.reset()
        context = omni.usd.get_context()
        if self.stage is None:
            context.new_stage()
            self.stage = context.get_stage()
        elif context.get_stage() != self.stage:
            raise RuntimeError("Persistent GPU stage was replaced externally")
        stage = self.stage
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, "/World/Envs")
        groups, robot_paths, all_rigids = [], [], []
        self.environments = prepared
        for index, env in enumerate(prepared):
            root = f"/World/Envs/env_{index:04d}"
            prim = UsdGeom.Xform.Define(stage, root).GetPrim()
            if not prim.GetReferences().AddReference(str(env["usd"]), "/Prepared"):
                raise RuntimeError(f"Cannot reference prepared environment {index}")
            paths = env["metadata"]["prepared_paths"]
            robot_paths.append(root + paths["articulation"])
            object_paths = [root + p for p in paths["objects"]]
            hand_paths = [root + p for p in paths["hands"]]
            env["rigid_object_indices"] = list(range(len(all_rigids), len(all_rigids) + len(object_paths)))
            all_rigids.extend(object_paths)
            env["rigid_hand_indices"] = list(range(len(all_rigids), len(all_rigids) + 2))
            all_rigids.extend(hand_paths)
            env["root"] = root
            group = UsdPhysics.CollisionGroup.Define(stage, f"/World/CollisionGroups/env_{index:04d}")
            group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets([prim.GetPath()])
            groups.append(group)
        for group in groups:
            group.CreateFilteredGroupsRel().SetTargets([g.GetPath() for g in groups if g.GetPath() != group.GetPath()])
        if SimulationManager.get_active_physics_engine() != "physx":
            raise RuntimeError("GPU rollout requires the PhysX engine")
        SimulationManager.setup_simulation(dt=_TIMESTEP, device=self.device)
        # Visual poses come from measured tensors; PhysX Fabric conflicts with
        # SyntheticData graph initialization for the full imported scene.
        SimulationManager.enable_fabric(False)
        self._assert_gpu()
        # The SDK default of 1024 aggregate pairs drops interactions for an
        # eight-robot batch. Reserve pair storage before initializing PhysX.
        pair_capacity = (1 << 15) * len(prepared)
        gpu_configuration = {"gpu_found_lost_aggregate_pairs_capacity": pair_capacity,
                             "gpu_total_aggregate_pairs_capacity": pair_capacity}
        SimulationManager.get_physics_scenes()[0].set_gpu_configuration(gpu_configuration)
        applied = SimulationManager.get_physics_scenes()[0].get_gpu_configuration()
        gpu_configuration = {name: getattr(applied, name) for name in gpu_configuration}
        self._articulation = Articulation(robot_paths)
        self._rigids = RigidPrim(all_rigids)
        if self._articulation.paths != robot_paths or self._rigids.paths != all_rigids:
            raise RuntimeError("SDK changed explicit batch view path ordering")
        if self._articulation.num_dofs != 54:
            raise UnsupportedRawSceneError("A free object contaminated the robot articulation")
        physx_names = self._articulation.dof_names
        mapped = [_xml_joint_name(n) for n in prepared[0]["names"]]
        if len(set(physx_names)) != 54 or set(physx_names) != set(mapped):
            raise UnsupportedRawSceneError("Physics DOFs differ from the 54 dataset joints")
        self._dof_indices = [physx_names.index(n) for n in mapped]
        for index, env in enumerate(prepared):
            drives = {}
            for name, dof in zip(env["names"], self._dof_indices, strict=True):
                joint_path = self._articulation.dof_paths[index][dof]
                joint = stage.GetPrimAtPath(joint_path)
                drive = UsdPhysics.DriveAPI(joint, "angular")
                values = [drive.GetStiffnessAttr().Get(), drive.GetDampingAttr().Get(), drive.GetMaxForceAttr().Get()]
                if not joint.IsA(UsdPhysics.RevoluteJoint) or not joint.HasAPI(UsdPhysics.DriveAPI, "angular") or any(v is None or not np.isfinite(v) or v <= 0 for v in values):
                    raise UnsupportedRawSceneError(f"Missing position drive {joint_path}")
                drives[name] = dict(zip(("stiffness", "damping", "max_force"), map(float, values)))
            env["metadata"]["physics"] = {"engine": "PhysX", "device": self.device,
                "gpu_dynamics_enabled": True, "broadphase": "GPU", "timestep_s": _TIMESTEP,
                "fabric_enabled": False,
                "gpu_configuration": gpu_configuration,
                "substeps_per_control": _SUBSTEPS, "articulation_root": robot_paths[index],
                "object_paths": [all_rigids[i] for i in env["rigid_object_indices"]], "drives": drives,
                "isolation": "pairwise filtered USD CollisionGroups; source world coordinates retained",
                "environment_count": len(prepared), "global_step_calls_per_tick": 1}
        self._lower = np.stack([e["lower"] for e in prepared])
        self._upper = np.stack([e["upper"] for e in prepared])
        self._velocity_step = np.stack([e["velocity_step"] for e in prepared])
        self.applied_command = np.clip(np.stack([e["action"] for e in prepared]), self._lower, self._upper)
        self.global_substeps = 0

    def initialize(self):
        """Initialize GPU physics and restore the recorded frame-zero state."""
        from isaacsim.core.simulation_manager import SimulationManager

        SimulationManager.initialize_physics()
        self._assert_gpu()
        if not self._articulation.is_physics_tensor_entity_valid() or not self._rigids.is_physics_tensor_entity_valid():
            raise RuntimeError("GPU robot/object tensor views failed to initialize")
        count = len(self.environments)
        self._articulation.set_dof_positions(np.stack([e["state"] for e in self.environments]), dof_indices=self._dof_indices)
        self._articulation.set_dof_velocities(np.zeros((count, 54)), dof_indices=self._dof_indices)
        indices = [i for e in self.environments for i in e["rigid_object_indices"]]
        poses = np.concatenate([e["poses"] for e in self.environments])
        if indices:
            self._rigids.set_world_poses(positions=poses[:, :3], orientations=poses[:, 3:], indices=indices)
            self._rigids.set_velocities(linear_velocities=np.zeros((len(indices), 3)), angular_velocities=np.zeros((len(indices), 3)), indices=indices)
        self._articulation.set_dof_position_targets(self.applied_command, dof_indices=self._dof_indices)
        self.read_feedback()
        for index, env in enumerate(self.environments):
            actual, initial = env["object_pose"], env["poses"]
            pos_error = float(np.max(np.abs(actual[:, :3] - initial[:, :3]), initial=0.))
            quat_error = float(np.max(np.minimum(np.linalg.norm(actual[:, 3:] - initial[:, 3:], axis=1), np.linalg.norm(actual[:, 3:] + initial[:, 3:], axis=1)), initial=0.))
            joint_error = float(np.max(np.abs(self.state[index] - env["state"])))
            if max(pos_error, quat_error, joint_error) > 1e-3:
                raise ValueError(f"GPU frame0 mismatch: env={index}, xyz={pos_error}, quat={quat_error}, joints={joint_error}")
            env["metadata"]["initial_error"] = {"position_m": pos_error, "quaternion_l2": quat_error, "robot_joint_rad": joint_error}

    def _assert_gpu(self):
        from isaacsim.core.simulation_manager import SimulationManager
        if SimulationManager.get_physics_sim_device() != self.device:
            raise RuntimeError(f"GPU PhysX device mismatch: {SimulationManager.get_physics_sim_device()} != {self.device}; CPU fallback prohibited")
        if SimulationManager.is_fabric_enabled():
            raise RuntimeError("GPU rollout requires USD visual synchronization with PhysX Fabric disabled")
        scenes = SimulationManager.get_physics_scenes()
        if len(scenes) != 1 or not scenes[0].get_enabled_gpu_dynamics() or scenes[0].get_broadphase_type() != "GPU":
            raise RuntimeError("Expected one real GPU-dynamics/GPU-broadphase PhysX scene")
        if not np.isclose(SimulationManager.get_physics_dt(), _TIMESTEP, atol=1e-10):
            raise RuntimeError("Physics timestep must be 1/480 seconds")

    def read_feedback(self):
        self.state = np.asarray(self._articulation.get_dof_positions(dof_indices=self._dof_indices).numpy(), dtype=np.float64)
        positions, orientations = self._rigids.get_world_poses()
        poses = np.concatenate((positions.numpy(), orientations.numpy()), axis=1).astype(np.float64)
        if self.state.shape != (len(self.environments), 54) or not np.isfinite(self.state).all() or not np.isfinite(poses).all():
            raise RuntimeError("Invalid measured GPU PhysX feedback")
        for env in self.environments:
            env["object_pose"] = poses[env["rigid_object_indices"]]
            env["hand_positions"] = poses[env["rigid_hand_indices"], :3]

    def step(self, target):
        import carb.settings
        from isaacsim.core.simulation_manager import SimulationManager
        from physx_scene import _USD_OUTPUT_SETTINGS

        target = _vector(target, self.applied_command.shape, "batch target")
        command = np.clip(target, self._lower, self._upper)
        self.applied_command = np.clip(command, self.applied_command - self._velocity_step, self.applied_command + self._velocity_step)
        self._articulation.set_dof_position_targets(self.applied_command, dof_indices=self._dof_indices)
        settings = carb.settings.get_settings()
        previous = [settings.get_as_bool(key) for key in _USD_OUTPUT_SETTINGS]
        try:
            for key in _USD_OUTPUT_SETTINGS:
                settings.set_bool(key, False)
            SimulationManager.step(steps=_SUBSTEPS)
        finally:
            for key, value in zip(_USD_OUTPUT_SETTINGS, previous, strict=True):
                settings.set_bool(key, value)
        self.global_substeps += _SUBSTEPS
        self.read_feedback()

    def reset(self):
        """Stop PhysX before replacing collision groups on the resident stage."""
        from isaacsim.core.simulation_manager import SimulationManager
        if self.stage is not None:
            import omni.physx
            # invalidate_physics only drops tensor views; it does not stop the
            # native simulation. Removing a live CollisionGroup is undefined.
            omni.physx.get_physx_interface().reset_simulation()
        SimulationManager.invalidate_physics()
        self._articulation = self._rigids = None
        self.environments = []
        if self.stage is not None:
            self.stage.RemovePrim("/World/Envs")
            self.stage.RemovePrim("/World/CollisionGroups")

    def close(self):
        import omni.usd
        self.reset()
        if self.stage is not None and omni.usd.get_context().get_stage() == self.stage:
            omni.usd.get_context().close_stage()
        self.stage = None
