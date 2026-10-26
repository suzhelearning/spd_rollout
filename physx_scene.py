"""Restore original six-task raw v4 episodes into native Isaac Sim PhysX.

The Kit SimulationApp is owned by the caller. Source datasets and assets stay read only.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from physx_scene_builder import (
    UnsupportedRawSceneError, _ARM_NAMES, _CONTROL_HZ, _SUBSTEPS, _TIMESTEP,
    _assemble_scene, _hand_reach_from_geometry, _joint_spec, _sha256,
    _validate_objects, _vector, _xml_joint_name,
)

_USD_OUTPUT_SETTINGS = (
    "/physics/updateToUsd", "/physics/updateVelocitiesToUsd",
    "/physics/updateParticlesToUsd", "/physics/updateForceSensorsToUsd",
)


class PhysxScene:
    """One 54-DoF robot and its recorded dynamic objects, stepped by PhysX."""

    @classmethod
    def from_episode(cls, dataset: Path, episode_id: str, assets: Path,
                     camera_config: dict, output: Path) -> PhysxScene:
        return cls(dataset, episode_id, assets, camera_config, output)

    def __init__(self, dataset: Path, episode_id: str, assets: Path,
                 camera_config: dict, output: Path):
        # Isaac imports must happen only after the caller has started SimulationApp.
        import carb.settings
        import omni.usd
        from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
        from isaacsim.core.experimental.prims import Articulation, RigidPrim
        from isaacsim.core.experimental.utils import stage as stage_utils
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        self._closed = False
        self._settings = carb.settings.get_settings()
        self.stage = None
        self._articulation = None
        self._bodies = []
        self._hands = []
        dataset = Path(dataset).expanduser().resolve()
        assets = Path(assets).expanduser().resolve()
        output = Path(output).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        index = json.loads((dataset / "dataset.json").read_text())
        if (index.get("schema_version") != 4
                or index.get("task_id") not in (
                    "hang_mug", "pyramid", "spelling", "toss_in_bin",
                    "unstack", "vowel_consonant_sort")
                or episode_id not in index.get("episodes", [])):
            raise ValueError("Episode must be indexed in a supported v4 raw task dataset")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", episode_id):
            raise ValueError("Invalid episode ID")
        self.joint_names = list(index["joint_names"])
        if len(self.joint_names) != 54 or len(set(self.joint_names)) != 54:
            raise ValueError("Expected 54 distinct dataset joints")
        if camera_config["width"] != 640 or camera_config["height"] != 480:
            raise ValueError("Checkpoint cameras must be 640x480")
        episode_path = dataset / "episodes" / f"{episode_id}.h5"
        with h5py.File(episode_path, "r") as source:
            if (source.attrs["episode_id"] != episode_id or source.attrs["schema_version"] != 4
                    or not source.attrs["recorded_action"]):
                raise ValueError("Episode header identity/schema/action mismatch")
            scene_id = str(source.attrs["scene_id"])
            if not re.fullmatch(r"[0-9a-f]{64}", scene_id):
                raise ValueError("Invalid episode scene ID")
            initial_state = _vector(source["trajectory/state"][0], (54,), "first-frame state")
            initial_action = _vector(source["trajectory/action"][0], (54,), "first-frame action")
            initial_pose = np.asarray(source["trajectory/object_pose"][0], dtype=np.float64)
        scene_path_raw = dataset / "scenes" / f"{scene_id}.json"
        if _sha256(scene_path_raw) != scene_id:
            raise ValueError("Episode scene ID does not match recorded scene contents")
        scene = json.loads(scene_path_raw.read_text())
        if scene.get("schema_version") != 4:
            raise ValueError("Recorded scene schema mismatch")
        mobility = _validate_objects(scene, index["task_id"])
        if ([joint["id"] for joint in sorted(scene["joints"], key=lambda item: item["state_index"])]
                != self.joint_names):
            raise ValueError("Scene joint indices disagree with dataset joint names")
        pose_nodes = scene["pose_node_ids"]
        if initial_pose.shape != (len(pose_nodes), 7) or not np.isfinite(initial_pose).all():
            raise ValueError("Recorded object poses do not match scene pose nodes")
        if any(not node.startswith("body/") or not node.removeprefix("body/") for node in pose_nodes):
            raise ValueError("Unsupported dynamic object pose node")
        robot_dir = assets / "robots/tianji_wuji2_hands1"
        robot_xml = robot_dir / "tianji_wuji2_hands1.xml"
        urdf = robot_dir / "tianji_wuji2_hands1.urdf"
        lower, upper, velocity, efforts = _joint_spec(urdf, self.joint_names)
        if set(efforts) != set(_ARM_NAMES):
            raise ValueError("All fourteen arm joints must be present")
        for description, values in (("state", initial_state), ("action", initial_action)):
            if np.any(values < lower - 1e-5) or np.any(values > upper + 1e-5):
                raise ValueError(f"Recorded first-frame {description} exceeds URDF limits")
        xml, provenance = _assemble_scene(robot_xml, urdf, assets, scene, camera_config,
                                          self.joint_names, lower, upper, efforts, index["task_id"])
        hand_reach, hand_reach_by_side = _hand_reach_from_geometry(xml)
        asset_dir = output / "physx_assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        scene_path = asset_dir / "scene.xml"
        ET.indent(xml)
        ET.ElementTree(xml).write(scene_path, encoding="utf-8", xml_declaration=True)

        if SimulationManager.get_active_physics_engine() != "physx":
            raise RuntimeError("This evaluation requires the PhysX physics engine")
        # An open prior episode must never leave a live tensor view bound to a new USD stage.
        SimulationManager.invalidate_physics()
        config = MJCFImporterConfig(
            mjcf_path=str(scene_path), usd_path=str(asset_dir), import_scene=True,
            merge_mesh=False, collision_from_visuals=False, fix_base=True,
            run_asset_transformer=False, run_multi_physics_conversion=True,
        )
        usd_path = Path(MJCFImporter(config).import_mjcf()).resolve()
        try:
            opened, stage = stage_utils.open_stage(str(usd_path))
            if not opened or stage is None:
                raise RuntimeError(f"Cannot open imported PhysX USD: {usd_path}")
            self.stage = stage
            roots = [prim for prim in stage.Traverse()
                     if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
            wrist_paths = []
            for wrist in ("l__l_wrist", "r__r_wrist"):
                matches = [prim.GetPath() for prim in stage.Traverse()
                           if prim.GetName() == wrist and prim.HasAPI(UsdPhysics.RigidBodyAPI)]
                if len(matches) != 1:
                    raise ValueError(f"Unsupported PhysX wrist rigid link {wrist}: {matches}")
                wrist_paths.append(matches[0])
            robot_roots = [prim for prim in roots
                           if all(path.HasPrefix(prim.GetPath()) for path in wrist_paths)]
            if len(robot_roots) != 1:
                raise ValueError(f"Unsupported robot articulation roots: {len(robot_roots)} of {len(roots)} contain both physical wrists")
            root_path = robot_roots[0].GetPath()
            body_paths = []
            for node in pose_nodes:
                name = node.removeprefix("body/")
                matches = [prim for prim in stage.Traverse()
                           if prim.GetName() == name and prim.HasAPI(UsdPhysics.RigidBodyAPI)]
                if len(matches) != 1:
                    raise ValueError(f"Expected one dynamic rigid body for {node}, got {len(matches)}")
                body_paths.append(matches[0].GetPath())
            if len(set(body_paths)) != len(body_paths):
                raise ValueError("Duplicate dynamic rigid bodies")
            # Free objects may be moved outside the robot namespace; preserving
            # the physical drawer prismatic joints is mandatory.
            embedded = [path for path, (_, kind) in zip(body_paths, mobility, strict=True)
                        if path.HasPrefix(root_path) and kind == "free"]
            if embedded:
                parent = stage.DefinePrim("/PhysxDynamicObjects", "Xform")
                for path in embedded:
                    prim = stage.GetPrimAtPath(path)
                    if path == root_path or not prim.IsValid():
                        raise ValueError(f"Cannot isolate robot from free rigid body {path}")
                    editor = Usd.NamespaceEditor(stage)
                    editor.ReparentPrim(prim, parent)
                    if not editor.ApplyEdits():
                        raise RuntimeError(f"Unable to isolate free object {path} from robot articulation")
                body_paths = [parent.GetPath().AppendChild(path.name) if path in embedded else path
                              for path in body_paths]
            missing = [str(path) for path in body_paths if not stage.GetPrimAtPath(path).IsValid()]
            if missing:
                raise ValueError(f"Physical reparent did not preserve all dynamic object paths: {missing}")
            if any(not stage.GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI) for path in body_paths):
                raise ValueError("An imported dynamic object lost its rigid-body physics")
            for path, (_, kind) in zip(body_paths, mobility, strict=True):
                if kind != "slide":
                    continue
                name = path.name
                matches = [prim for prim in stage.Traverse()
                           if prim.GetName() == f"{name}_slide" and prim.IsA(UsdPhysics.PrismaticJoint)]
                if len(matches) != 1:
                    raise UnsupportedRawSceneError(f"Unsupported PhysX slide for {name}: expected one imported prismatic joint, found {len(matches)}")
                joint = UsdPhysics.PrismaticJoint(matches[0])
                targets = joint.GetBody1Rel().GetTargets()
                if len(targets) != 1 or targets[0] != path:
                    raise UnsupportedRawSceneError(f"Unsupported PhysX slide {name}: body1 target {targets} != {path}")
                limits = (joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get())
                if any(v is None for v in limits) or not np.allclose(limits, [0., .25], atol=1e-4):
                    raise UnsupportedRawSceneError(f"Unsupported PhysX slide {name}: travel limits {limits}")
            # The importer leaves free-body USD xforms at the origin. Initialize
            # the authored stage as well as the PhysX tensor: constructing RTX
            # render products can update USD and reload PhysX from it.
            for path, pose in zip(body_paths, initial_pose, strict=True):
                transform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
                translations = [op for op in transform.GetOrderedXformOps()
                                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
                orientations = [op for op in transform.GetOrderedXformOps()
                                if op.GetOpType() == UsdGeom.XformOp.TypeOrient]
                if len(translations) != 1 or len(orientations) != 1:
                    raise ValueError(f"Unsupported imported free-body transform: {path}")
                translations[0].Set(Gf.Vec3d(*map(float, pose[:3])))
                orientations[0].Set(Gf.Quatf(float(pose[3]), Gf.Vec3f(*map(float, pose[4:]))))
            if not stage.GetRootLayer().Save():
                raise RuntimeError("Unable to save recorded initial PhysX object poses")
            self._articulation = Articulation(str(root_path))
            if self._articulation.num_dofs != 54:
                raise ValueError(f"PhysX articulation has {self._articulation.num_dofs} DOFs, expected 54")
            mapped = [_xml_joint_name(name) for name in self.joint_names]
            physx_names = self._articulation.dof_names
            if len(set(physx_names)) != 54 or set(physx_names) != set(mapped):
                raise ValueError("PhysX articulation DOF names do not match all 54 dataset joints")
            self._dof_indices = [physx_names.index(name) for name in mapped]
            # Reject a passive/velocity-driven axis rather than pretending that
            # writing target positions constitutes an actuated physical rollout.
            drives = {}
            for name, dof_index in zip(self.joint_names, self._dof_indices, strict=True):
                joint_path = self._articulation.dof_paths[0][dof_index]
                joint = stage.GetPrimAtPath(joint_path)
                drive = UsdPhysics.DriveAPI(joint, "angular")
                stiffness = drive.GetStiffnessAttr().Get()
                damping = drive.GetDampingAttr().Get()
                max_force = drive.GetMaxForceAttr().Get()
                if (not joint.IsA(UsdPhysics.RevoluteJoint) or not joint.HasAPI(UsdPhysics.DriveAPI, "angular")
                        or any(value is None or not np.isfinite(value) or value <= 0
                               for value in (stiffness, damping, max_force))):
                    raise ValueError(f"Missing or invalid PhysX position drive for {name} ({joint_path})")
                drives[name] = {"stiffness": float(stiffness), "damping": float(damping),
                                "max_force": float(max_force)}
            self._bodies = [RigidPrim(str(path)) for path in body_paths]
            self._hands = [RigidPrim(str(path)) for path in wrist_paths]
            SimulationManager.setup_simulation(dt=_TIMESTEP, device="cpu")
            if SimulationManager.get_physics_sim_device() != "cpu":
                raise RuntimeError("PhysX evaluation requires CPU simulation")
            if not np.isclose(SimulationManager.get_physics_dt(), _TIMESTEP, atol=1e-10):
                raise RuntimeError("PhysX scene timestep is not 1/480 seconds")
            SimulationManager.initialize_physics()
            if not self._articulation.is_physics_tensor_entity_valid() or any(
                    not body.is_physics_tensor_entity_valid() for body in (*self._bodies, *self._hands)):
                raise RuntimeError("Robot, physical wrist or dynamic object did not initialize in PhysX")
            if self._articulation.num_dofs != 54:
                raise RuntimeError("Imported free body contaminated the PhysX articulation")
            self._articulation.set_dof_positions(initial_state[None, :], dof_indices=self._dof_indices)
            self._articulation.set_dof_velocities(np.zeros((1, 54)), dof_indices=self._dof_indices)
            for body, pose in zip(self._bodies, initial_pose, strict=True):
                body.set_world_poses(positions=pose[None, :3], orientations=pose[None, 3:])
                body.set_velocities(linear_velocities=np.zeros((1, 3)), angular_velocities=np.zeros((1, 3)))
            self.applied_command = np.clip(initial_action, lower, upper)
            self._articulation.set_dof_position_targets(self.applied_command[None, :],
                                                        dof_indices=self._dof_indices)
            self._lower, self._upper = lower, upper
            self._velocity_step = velocity / _CONTROL_HZ
            # The recorded first observation is returned without taking a physics step.
            self._read_state()
            pose_error = float(np.max(np.abs(self._object_pose[:, :3] - initial_pose[:, :3]), initial=0.))
            orientation_error = float(np.max(np.minimum(
                np.linalg.norm(self._object_pose[:, 3:] - initial_pose[:, 3:], axis=1),
                np.linalg.norm(self._object_pose[:, 3:] + initial_pose[:, 3:], axis=1)), initial=0.))
            state_error = float(np.max(np.abs(self._state - initial_state)))
            if pose_error > 1e-3 or orientation_error > 1e-3 or state_error > 1e-3:
                raise ValueError(f"PhysX frame0 differs from raw: position={pose_error}, quaternion={orientation_error}, joints={state_error}")
            self.metadata = {
                "reconstruction": "not exact; independent PhysX simulation of recorded v4 scene",
                "distribution_shift": "PhysX CPU dynamics/contact and imported USD rendering differ from the source simulator and training renderer",
                "episode_id": episode_id, "episode_path": str(episode_path), "scene_id": scene_id,
                "task_id": index["task_id"], "pose_node_ids": list(pose_nodes),
                "hand_position_definition": "measured world origins of articulated physical l__l_wrist and r__r_wrist links",
                "hand_reference": "wrist_link", "hand_reach_m": hand_reach,
                "hand_reach_by_side_m": hand_reach_by_side,
                "hand_reach_derivation": "Upper bound over all articulated hand collision mesh vertices: sum of descendant body-offset norms, twice each joint-pivot offset, geom offset norm and binary STL vertex radius; source is generated scene_xml at robot_xml_sha256; scalar is max(left,right). No contact-pair measurement.",
                "initial_error": {"position_m": pose_error, "quaternion_l2": orientation_error,
                                  "robot_joint_rad": state_error},
                "scene_xml": str(scene_path), "scene_sha256": _sha256(scene_path),
                "scene_usd": str(usd_path), "scene_usd_sha256": _sha256(usd_path),
                "robot_xml_sha256": _sha256(robot_xml), "robot_urdf_sha256": _sha256(urdf),
                "bottle_manifest_sha256": _sha256(assets / "objects/abc_bottles/manifest.json") if any(obj["class_name"] == "bottle" for obj in scene["sampled_parameters"]["scene_manifest.objects"]) else None,
                "asset_sources": provenance, "initial_state": initial_state.tolist(),
                "initial_action": initial_action.tolist(), "initial_object_pose": initial_pose.tolist(),
                "initial_velocity": "unrecorded; all robot and object velocities initialized to zero",
                "initial_servo_target": "recorded first-frame absolute action",
                "command_limits": "URDF position and velocity limits relative to previous applied command at 30 Hz",
                "arm_servo_assumption": "14 synthesized arm servos kp=100 kv=10; 40 original hand servos imported",
                "physics": {"engine": "PhysX", "device": "cpu", "timestep_s": _TIMESTEP,
                            "substeps_per_control": _SUBSTEPS, "articulation_root": str(root_path),
                            "object_paths": [str(path) for path in body_paths],
                            "objects_isolated_from_articulation": bool(embedded), "drives": drives},
                "camera_config": camera_config,
                "rendering": "headless Isaac Sim Kit RTX; PhysX remains active (unlike offline training)",
            }
            (output / "scene_metadata.json").write_text(json.dumps(self.metadata, indent=2, ensure_ascii=False) + "\n")
        except BaseException:
            self.close()
            raise

    def _read_state(self) -> None:
        values = np.asarray(self._articulation.get_dof_positions(dof_indices=self._dof_indices).numpy(),
                            dtype=np.float64)
        if values.shape != (1, 54) or not np.isfinite(values).all():
            raise RuntimeError("Invalid PhysX robot joint feedback")
        self._state = values[0].copy()
        poses = []
        for body in self._bodies:
            position, orientation = body.get_world_poses()
            poses.append(np.concatenate((np.asarray(position.numpy()[0], dtype=np.float64),
                                         np.asarray(orientation.numpy()[0], dtype=np.float64))))
        self._object_pose = np.stack(poses) if poses else np.empty((0, 7), dtype=np.float64)
        if self._object_pose.shape != (len(self._bodies), 7) or not np.isfinite(self._object_pose).all():
            raise RuntimeError("Invalid PhysX dynamic object feedback")
        hands = []
        for wrist in self._hands:
            position, _ = wrist.get_world_poses()
            hands.append(np.asarray(position.numpy()[0], dtype=np.float64))
        self._hand_positions = np.stack(hands)
        if self._hand_positions.shape != (2, 3) or not np.isfinite(self._hand_positions).all():
            raise RuntimeError("Invalid measured articulated wrist world positions")

    @property
    def state(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError("PhysxScene is closed")
        return self._state.copy()

    @property
    def object_pose(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError("PhysxScene is closed")
        return self._object_pose.copy()

    @property
    def hand_positions(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError("PhysxScene is closed")
        return self._hand_positions.copy()

    def step(self, target: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("PhysxScene is closed")
        from isaacsim.core.simulation_manager import SimulationManager

        command = np.clip(_vector(target, (54,), "target"), self._lower, self._upper)
        self.applied_command = np.clip(command, self.applied_command - self._velocity_step,
                                       self.applied_command + self._velocity_step)
        self._articulation.set_dof_position_targets(self.applied_command[None, :],
                                                    dof_indices=self._dof_indices)
        # RGB visuals are driven by measured tensor feedback. Writing hidden
        # physical prims into USD on every substep triggers unused UI callbacks.
        output_settings = tuple(self._settings.get_as_bool(key) for key in _USD_OUTPUT_SETTINGS)
        try:
            for key in _USD_OUTPUT_SETTINGS:
                self._settings.set_bool(key, False)
            SimulationManager.step(steps=_SUBSTEPS)
        finally:
            for key, enabled in zip(_USD_OUTPUT_SETTINGS, output_settings):
                self._settings.set_bool(key, enabled)
        self._read_state()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.stage is not None:
            import omni.usd
            from isaacsim.core.simulation_manager import SimulationManager
            SimulationManager.invalidate_physics()
            if omni.usd.get_context().get_stage() == self.stage:
                omni.usd.get_context().close_stage()
            self.stage = None
        self._articulation = None
        self._bodies = []
        self._hands = []
