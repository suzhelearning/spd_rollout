"""Assemble the recorded toss-in-bin geometry and controls for PhysX USD import."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np


_CONTROL_HZ = 30
_SUBSTEPS = 16
_TIMESTEP = 1 / (_CONTROL_HZ * _SUBSTEPS)
_ARM_NAMES = tuple(f"Joint{i}_{side}" for side in ("L", "R") for i in range(1, 8))
VESSEL_SIDES = 32
DRAWER_FLOORS = (0.018, 0.154, 0.290)



def _numbers(values: object) -> str:
    return " ".join(format(float(number), ".16g") for number in np.asarray(values).ravel())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vector(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _camera_quat(rpy_deg: object) -> str:
    # Dataset camera contract: Rz(yaw) Ry(pitch) Rx(roll), camera -Z forward.
    roll, pitch, yaw = np.deg2rad(_vector(rpy_deg, (3,), "camera rpy_deg")) / 2
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return _numbers((cy * cp * cr + sy * sp * sr, cy * cp * sr - sy * sp * cr,
                     cy * sp * cr + sy * cp * sr, sy * cp * cr - cy * sp * sr))


def _absolutize_assets(root: ET.Element, source_dir: Path) -> None:
    compiler = root.find("compiler")
    mesh_dir = source_dir / (compiler.get("meshdir", ".") if compiler is not None else ".")
    texture_dir = source_dir / (compiler.get("texturedir", ".") if compiler is not None else ".")
    for asset in root.findall("./asset/*"):
        filename = asset.get("file")
        if filename is None:
            continue
        base = mesh_dir if asset.tag == "mesh" else texture_dir if asset.tag == "texture" else source_dir
        path = (base / filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        asset.set("file", str(path))
    if compiler is not None:
        compiler.set("meshdir", ".")
        compiler.set("texturedir", ".")


def _xml_joint_name(name: str) -> str:
    return f"{name[0]}__{name}" if name.startswith(("l_", "r_")) else name


def _joint_spec(urdf_path: Path, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    joints = {joint.get("name"): joint for joint in ET.parse(urdf_path).getroot().findall("joint")}
    lower, upper, velocity = [], [], []
    arm_efforts: dict[str, float] = {}
    for name in joint_names:
        joint = joints.get(_xml_joint_name(name))
        if joint is None or joint.get("type") != "revolute" or joint.find("limit") is None:
            raise ValueError(f"URDF is missing limited revolute joint {name}")
        limit = joint.find("limit")
        lower.append(float(limit.attrib["lower"]))
        upper.append(float(limit.attrib["upper"]))
        velocity.append(float(limit.attrib["velocity"]))
        if name in _ARM_NAMES:
            arm_efforts[name] = float(limit.attrib["effort"])
    lo, hi, vel = (np.asarray(v, dtype=np.float64) for v in (lower, upper, velocity))
    if not np.isfinite(np.concatenate((lo, hi, vel, list(arm_efforts.values())))).all() or np.any(lo >= hi) or np.any(vel <= 0) or any(v <= 0 for v in arm_efforts.values()):
        raise ValueError("Invalid URDF actuator limits")
    return lo, hi, vel, arm_efforts


def _bottle(root: ET.Element, assets_root: Path, bottle: dict, manifest: dict) -> dict:
    variant = bottle["asset_id"]
    if variant not in manifest["variants"]:
        raise ValueError(f"Missing local bottle mesh variant {variant!r}")
    entry = manifest["variants"][variant]
    radius, height = _vector(bottle["size"], (2,), "bottle size")
    if radius <= 0 or height <= 0 or bottle["mass_kg"] <= 0:
        raise ValueError("Bottle dimensions and mass must be positive")
    asset = root.find("asset")
    world = root.find("worldbody")
    base = assets_root / "objects/abc_bottles"
    prefix = bottle["name"]
    texture_names = {}
    for index, texture in enumerate(entry["textures"]):
        attributes = dict(texture["attributes"])
        original = attributes["name"]
        attributes["name"] = f"{prefix}_texture_{index}"
        texture_names[original] = attributes["name"]
        path = (base / texture["file"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        attributes["file"] = str(path)
        ET.SubElement(asset, "texture", attributes)
    material_names = {}
    for index, material in enumerate(entry["materials"]):
        attributes = dict(material)
        original = attributes["name"]
        attributes["name"] = f"{prefix}_material_{index}"
        if "texture" in attributes:
            attributes["texture"] = texture_names[attributes["texture"]]
        material_names[original] = attributes["name"]
        ET.SubElement(asset, "material", attributes)
    mass = float(bottle["mass_kg"])
    body = ET.SubElement(world, "body", name=prefix)
    ET.SubElement(body, "freejoint", name=f"{prefix}_free")
    # Canonical bottle meshes span z=0..1 and radius=1; manifest scale maps
    # ALL visual and convex collision parts together to this episode's size.
    ET.SubElement(body, "inertial", pos=_numbers((0, 0, height / 2)), mass=_numbers((mass,)),
                  diaginertia=_numbers((mass * (3 * radius**2 + height**2) / 12,
                                       mass * (3 * radius**2 + height**2) / 12,
                                       mass * radius**2 / 2)))
    collisions = 0
    for index, mesh in enumerate(entry["meshes"]):
        source = (base / mesh["file"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        name = f"{prefix}_mesh_{index}"
        ET.SubElement(asset, "mesh", name=name, file=str(source), scale=_numbers((radius, radius, height)))
        if mesh["role"] == "collision":
            collisions += 1
            ET.SubElement(body, "geom", name=f"{prefix}_contact_{index}", type="mesh", mesh=name,
                          density="0", group="3", friction=_numbers((bottle["friction"], 0.03, 0.003)),
                          condim="6", solref="0.004 1", solimp="0.998 0.998 0.001")
        elif mesh["role"] == "visual":
            source_material = mesh["source_geom_attributes"]["material"]
            ET.SubElement(body, "geom", name=f"{prefix}_visual_{index}", type="mesh", mesh=name,
                          density="0", group="2", contype="0", conaffinity="0",
                          material=material_names[source_material])
        else:
            raise ValueError(f"Unrecognized bottle mesh role: {mesh['role']}")
    if collisions < 2:
        raise ValueError(f"{variant} must have multiple separate convex collision meshes")
    return {"asset_id": variant, "manifest": str(base / "manifest.json"),
            "geometry_revision": manifest["geometry_revision"], "collision_mesh_count": collisions,
            "source_model_sha256": entry["source_model_sha256"],
            "mesh_scale_radius_radius_height_m": [float(radius), float(radius), float(height)]}


def _recorded_mount(scene: dict, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Compose the recorded TCP→hand mount marker chain, not the local asset default."""
    nodes = {node["id"]: node for node in scene["nodes"]}
    current = f"body/{side.lower()}_mount"
    tcp = f"body/TCP_Link_{side}"
    chain = []
    while current != tcp:
        if current not in nodes or current in (node["id"] for node in chain):
            raise ValueError(f"Missing or cyclic recorded hand mount chain: {current}")
        node = nodes[current]
        if node["driver"] != "static":
            raise ValueError(f"Hand mount link is not static: {current}")
        chain.append(node)
        current = node["parent_id"]
    position = np.zeros(3, dtype=np.float64)
    quaternion = np.array([1., 0., 0., 0.])
    for node in reversed(chain):
        translation = _vector(node["translation"], (3,), node["id"])
        rotation = _vector(node["rotation_wxyz"], (4,), node["id"])
        if abs(np.linalg.norm(rotation) - 1.) > 1e-5:
            raise ValueError(f"Hand mount rotation is not unit length: {node['id']}")
        w, xyz = quaternion[0], quaternion[1:]
        position += translation + 2 * (w * np.cross(xyz, translation) + np.cross(xyz, np.cross(xyz, translation)))
        rw, rxyz = rotation[0], rotation[1:]
        quaternion = np.r_[w * rw - xyz @ rxyz, w * rxyz + rw * xyz + np.cross(xyz, rxyz)]
    return position, quaternion / np.linalg.norm(quaternion)


def _align_hand_links(root: ET.Element, scene: dict) -> int:
    """Make the physical wrist and 40 hinge frames match the recorded visual skeleton."""
    bodies = {body.get("name"): body for body in root.iter("body")}
    nodes = {node["id"]: node for node in scene["nodes"]}
    for side in ("l", "r"):
        name = f"{side}_wrist"
        recorded = nodes[f"body/{name}"]
        if recorded["driver"] != "static" or recorded["parent_id"] != f"body/{side}_mount":
            raise ValueError(f"Unsupported recorded wrist parent: {name}")
        body = bodies.get(f"{side}__{name}")
        if body is None:
            raise ValueError(f"Physical wrist is missing: {name}")
        body.set("pos", _numbers(_vector(recorded["translation"], (3,), name)))
        body.set("quat", _numbers(_vector(recorded["rotation_wxyz"], (4,), name)))

    count = 0
    for joint in scene["joints"]:
        name = joint["id"]
        if not name.startswith(("l_", "r_")):
            continue
        side = name[0]
        child_id = joint["child_id"]
        if not child_id.startswith(f"robot/{side}_"):
            raise ValueError(f"Unsupported recorded hand joint child: {name}")
        child = bodies.get(f"{side}__{child_id.removeprefix('robot/')}")
        xml_joint = child.find("joint") if child is not None else None
        if xml_joint is None or xml_joint.get("name") != f"{side}__{name}":
            raise ValueError(f"Physical hand joint differs from recorded scene: {name}")
        source_axis = _vector(joint["axis"], (3,), name)
        physical_axis = _vector(np.fromstring(xml_joint.get("axis", "0 0 1"), sep=" "),
                                (3,), f"physical {name} axis")
        if not np.allclose(physical_axis, source_axis, atol=1e-6):
            raise ValueError(f"Physical hand joint axis differs from recorded scene: {name}")
        child_pose = joint["joint_to_child"]
        if not (np.allclose(child_pose["translation"], 0, atol=1e-8)
                and np.allclose(child_pose["rotation_wxyz"], (1, 0, 0, 0), atol=1e-8)
                and abs(float(joint["reference_position"])) < 1e-8):
            raise ValueError(f"Unsupported recorded hand joint reference frame: {name}")
        parent_pose = joint["parent_to_joint"]
        child.set("pos", _numbers(_vector(parent_pose["translation"], (3,), name)))
        child.set("quat", _numbers(_vector(parent_pose["rotation_wxyz"], (4,), name)))
        count += 1
    if count != 40:
        raise ValueError(f"Expected 40 recorded hand hinge frames, got {count}")
    return count


def _box_geom(name: str, size: tuple[float, float, float], *, pos=(0.0, 0.0, 0.0), rgba=(0.5, 0.5, 0.5, 1.0)) -> dict[str, Any]:
    return {"name": name, "type": "box", "size": tuple(value * 0.5 for value in size), "pos": tuple(pos), "rgba": tuple(rgba)}


def _cylinder_geom(name: str, radius: float, height: float, *, pos=(0.0, 0.0, 0.0), rgba=(0.5, 0.5, 0.5, 1.0)) -> dict[str, Any]:
    return {"name": name, "type": "cylinder", "size": (radius, height * 0.5), "pos": tuple(pos), "rgba": tuple(rgba)}


def _capsule_geom(name: str, radius: float, fromto: tuple[float, ...], *, rgba=(0.5, 0.5, 0.5, 1.0)) -> dict[str, Any]:
    return {"name": name, "type": "capsule", "size": (radius,), "fromto": tuple(fromto), "rgba": tuple(rgba)}


def _vessel_geoms(prefix: str, radius: float, bottom_radius: float, height: float, wall: float, rgba: tuple[float, ...]) -> list[dict[str, Any]]:
    """Overlapping tangent panels form a closed hollow polygonal frustum.

    The panels slope outwards; unlike a convex hull they leave the entire inner
    cavity available for contact. Panel corners remain within z=[0, height].
    """
    geoms = [_cylinder_geom(f"{prefix}_bottom", bottom_radius, wall, pos=(0.0, 0.0, wall * 0.5), rgba=rgba)]
    slope = math.atan2(radius - bottom_radius, height)
    length = (height - wall * math.sin(slope)) / math.cos(slope)
    tangent_width = 2.0 * radius * math.tan(math.pi / VESSEL_SIDES)
    middle_radius = (radius + bottom_radius) * 0.5 - wall * 0.5
    for index in range(VESSEL_SIDES):
        angle = 2.0 * math.pi * index / VESSEL_SIDES
        c, s = math.cos(angle * 0.5), math.sin(angle * 0.5)
        cy, sy = math.cos(slope * 0.5), math.sin(slope * 0.5)
        panel = _box_geom(
            f"{prefix}_wall_{index:02d}", (wall, tangent_width, length),
            pos=(middle_radius * math.cos(angle), middle_radius * math.sin(angle), height * 0.5), rgba=rgba,
        )
        panel["quat"] = (c * cy, -s * sy, c * sy, s * cy)
        geoms.append(panel)
    return geoms


def _geoms_for(class_name: str, size: tuple[float, ...], color: tuple[float, float, float], instance_id: int) -> tuple[dict[str, Any], ...]:
    rgba = (*color, 1.0)
    prefix = f"obj{instance_id}"
    if class_name in {"jenga_block", "letter_block", "domino"}:
        return (_box_geom(f"{prefix}_geom", size, rgba=rgba),)
    if class_name == "plate":
        radius, thickness = size
        base_thickness = 0.003
        inner_radius = radius * 0.72
        geoms = [_cylinder_geom(
            f"{prefix}_well", inner_radius, base_thickness,
            pos=(0, 0, -thickness / 2 + base_thickness / 2), rgba=rgba,
        )]
        rise = thickness - base_thickness
        slope = math.atan2(rise, radius - inner_radius)
        middle_radius = (radius + inner_radius) / 2
        # A shallow sloped ring has an open upper well, not a solid disk hull.
        for index in range(VESSEL_SIDES):
            angle = index * math.tau / VESSEL_SIDES
            half = angle / 2
            cy, sy = math.cos(-slope / 2), math.sin(-slope / 2)
            panel = _box_geom(
                f"{prefix}_rim_{index:02d}",
                (math.hypot(radius - inner_radius, rise),
                 2 * radius * math.tan(math.pi / VESSEL_SIDES),
                 base_thickness * math.cos(slope)),
                pos=(middle_radius * math.cos(angle), middle_radius * math.sin(angle), 0),
                rgba=rgba,
            )
            panel["quat"] = (math.cos(half) * cy, -math.sin(half) * sy,
                             math.cos(half) * sy, math.sin(half) * cy)
            geoms.append(panel)
        return tuple(geoms)
    if class_name in {"cup", "mug"}:
        radius, height, wall = size[:3]
        bottom_radius = size[3] if class_name == "cup" else radius
        geoms = _vessel_geoms(prefix, radius, bottom_radius, height, wall, rgba)
        if class_name == "cup":
            # Three internal feet carry the next cup; tapered walls remain clear.
            nest_step = size[4]
            for index in range(3):
                angle = index * 2.0 * math.pi / 3.0
                geoms.append(_cylinder_geom(
                    f"{prefix}_nest_stop_{index}", wall, nest_step - wall,
                    pos=(bottom_radius * 0.82 * math.cos(angle), bottom_radius * 0.82 * math.sin(angle), (nest_step + wall) * 0.5), rgba=rgba,
                ))
        else:
            # Closed oval handle in the x-z plane: about 26 x 42 mm clear.
            # Its inner edge joins the mug wall; no bar crosses the opening.
            center_x, center_z = radius + 0.015, height * 0.55
            for index in range(32):
                a, b = index * math.tau / 32, (index + 1) * math.tau / 32
                geoms.append(_capsule_geom(
                    f"{prefix}_handle_{index:02d}", 0.004,
                    (center_x + 0.019 * math.cos(a), 0.0, center_z + 0.025 * math.sin(a),
                     center_x + 0.019 * math.cos(b), 0.0, center_z + 0.025 * math.sin(b)), rgba=rgba,
                ))
        return tuple(geoms)
    if class_name == "rack":
        width, depth, height = size
        geoms = []
        for side in (-1, 1):
            geoms.append(_box_geom(f"{prefix}_rail_{side}", (width, 0.018, 0.016), pos=(0.0, side * 0.065, 0.008), rgba=rgba))
        # Four pairs of uprights yield three unobstructed 50 mm plate slots.
        for index, x in enumerate((-0.090, -0.030, 0.030, 0.090)):
            for side in (-1, 1):
                geoms.append(_capsule_geom(f"{prefix}_peg_{index}_{side}", 0.005,
                    (x, side * 0.065, 0.016, x, side * 0.065, height - 0.005), rgba=rgba))
        for side in (-1, 1):
            geoms.append(_box_geom(f"{prefix}_crossbar_{side}", (0.014, depth, 0.012), pos=(side * (width * 0.5 - 0.007), 0.0, 0.006), rgba=rgba))
        return tuple(geoms)
    if class_name == "mug_tree":
        geoms = [
            _cylinder_geom(f"{prefix}_base", 0.070, 0.018, pos=(0.0, 0.0, 0.009), rgba=rgba),
            _capsule_geom(f"{prefix}_trunk", 0.010, (0.0, 0.0, 0.018, 0.0, 0.0, 0.290), rgba=rgba),
        ]
        # Tangential pegs cross the handle opening perpendicular to its plane.
        # The radial branches stay behind the ring instead of piercing its rim.
        for index, angle in enumerate((0.0, math.pi, math.pi * 0.5, -math.pi * 0.5)):
            z = 0.185 if index < 2 else 0.245
            radial = np.array((math.cos(angle), math.sin(angle)))
            tangent = np.array((-math.sin(angle), math.cos(angle)))
            start, end = 0.085 * radial - 0.028 * tangent, 0.085 * radial + 0.028 * tangent
            geoms.append(_capsule_geom(f"{prefix}_branch_{index}", 0.005,
                                      (0.0, 0.0, z - 0.035, *start, z), rgba=rgba))
            geoms.append(_capsule_geom(f"{prefix}_hook_{index}", 0.004,
                                      (*start, z, *end, z), rgba=rgba))
            geoms.append(_capsule_geom(f"{prefix}_tip_{index}", 0.004,
                                      (*end, z, *end, z + 0.014), rgba=rgba))
        return tuple(geoms)
    if class_name == "bin":
        width, depth, height = size[:3]
        wall = 0.008
        return (
            _box_geom(f"{prefix}_bottom", (width + 2 * wall, depth + 2 * wall, wall), pos=(0.0, 0.0, wall * 0.5), rgba=rgba),
            _box_geom(f"{prefix}_x1", (wall, depth + 2 * wall, height), pos=((width + wall) * 0.5, 0.0, wall + height * 0.5), rgba=rgba),
            _box_geom(f"{prefix}_x2", (wall, depth + 2 * wall, height), pos=(-(width + wall) * 0.5, 0.0, wall + height * 0.5), rgba=rgba),
            _box_geom(f"{prefix}_y1", (width, wall, height), pos=(0.0, (depth + wall) * 0.5, wall + height * 0.5), rgba=rgba),
            _box_geom(f"{prefix}_y2", (width, wall, height), pos=(0.0, -(depth + wall) * 0.5, wall + height * 0.5), rgba=rgba),
        )
    if class_name == "cabinet":
        depth, width, height = size
        wall = 0.010
        geoms = [
            _box_geom(f"{prefix}_back", (wall, width, height), pos=((depth - wall) / 2, 0, height / 2), rgba=rgba),
            _box_geom(f"{prefix}_top", (depth, width, wall), pos=(0, 0, height - wall / 2), rgba=rgba),
        ]
        for side in (-1, 1):
            geoms.append(_box_geom(f"{prefix}_side_{side}", (depth, wall, height),
                                  pos=(0, side * (width - wall) / 2, height / 2), rgba=rgba))
        # The slide guides carry drawer weight. A 1 mm running gap prevents
        # redundant, immovable normal contacts between shelf and guided tray.
        for level, floor in enumerate(DRAWER_FLOORS):
            geoms.append(_box_geom(f"{prefix}_shelf_{level}", (depth, width - 2 * wall, wall),
                                  pos=(0, 0, floor - 0.001 - wall / 2), rgba=rgba))
        # Close the underside without filling the open-front drawer bays.
        geoms.append(_box_geom(f"{prefix}_bottom", (depth, width, 0.008), pos=(0, 0, 0.004), rgba=rgba))
        return tuple(geoms)
    if class_name == "drawer":
        depth, width, height = size
        wall = 0.006
        geoms = [_box_geom(f"{prefix}_floor", (depth, width, wall), pos=(0, 0, wall / 2), rgba=rgba)]
        for side in (-1, 1):
            geoms.append(_box_geom(f"{prefix}_side_{side}", (depth, wall, height - wall),
                                  pos=(0, side * (width - wall) / 2, (height + wall) / 2), rgba=rgba))
            geoms.append(_box_geom(f"{prefix}_end_{side}", (wall, width - 2 * wall, height - wall),
                                  pos=(side * (depth - wall) / 2, 0, (height + wall) / 2), rgba=rgba))
        handle_x, handle_z = -depth / 2 - 0.032, height * 0.62
        geoms.append(_capsule_geom(f"{prefix}_handle_grip", 0.004,
                                  (handle_x, -0.054, handle_z, handle_x, 0.054, handle_z), rgba=rgba))
        for side in (-1, 1):
            geoms.append(_capsule_geom(f"{prefix}_handle_mount_{side}", 0.004,
                                      (-depth / 2, side * 0.054, handle_z, handle_x, side * 0.054, handle_z), rgba=rgba))
        return tuple(geoms)
    raise ValueError(f"unknown procedural class: {class_name}")


class UnsupportedRawSceneError(ValueError):
    """The scene requires missing source facts or a geometry/joint we cannot represent."""


_TASK_OBJECTS = {
    "hang_mug": ({"mug": 1, "mug_tree": 1},),
    "pyramid": ({"letter_block": 3},),
    "spelling": tuple({"cabinet": 1, "drawer": 3, "letter_block": count}
                      for count in (3, 4, 5, 6, 8)),
    "toss_in_bin": tuple({"bin": 1, "bottle": count} for count in (1, 2, 4)),
    "unstack": ({"cup": 3},),
    "vowel_consonant_sort": ({"letter_block": 10},
                             {"cabinet": 1, "drawer": 3, "letter_block": 10}),
}
_FIXTURES = frozenset(("bin", "mug_tree", "cabinet"))
_DIMENSIONS = {"mug": 3, "mug_tree": 3, "letter_block": 3, "cup": 5,
               "bin": 3, "bottle": 2, "cabinet": 3, "drawer": 3}


def _validate_objects(scene: dict, task: str | None = None) -> list[tuple[str, str]]:
    """Reject any source node/joint that cannot be represented by the physical model."""
    from collections import Counter

    sampled = scene["sampled_parameters"]
    missing = [key for key in ("scene_manifest.table", "scene_manifest.sampled_values")
               if key not in sampled]
    if missing:
        raise UnsupportedRawSceneError(
            "Cannot reconstruct raw scene without original " + ", ".join(missing))
    revision = sampled["scene_manifest.sampled_values"].get("geometry_revision")
    if revision != "paper-aligned-scenes-v2":
        raise UnsupportedRawSceneError(
            f"Unsupported recorded geometry revision: {revision!r}; expected paper-aligned-scenes-v2")
    objects = scene["sampled_parameters"]["scene_manifest.objects"]
    names = [obj["name"] for obj in objects]
    if len(set(names)) != len(names) or len(set(scene["pose_node_ids"])) != len(scene["pose_node_ids"]):
        raise ValueError("Duplicate object names or pose_node_ids")
    if task is not None and Counter(obj["class_name"] for obj in objects) not in [
            Counter(pattern) for pattern in _TASK_OBJECTS[task]]:
        raise UnsupportedRawSceneError(f"Unsupported {task} object class/count pattern: {Counter(obj['class_name'] for obj in objects)}")
    nodes = {node["id"]: node for node in scene["nodes"]}
    mobility = {}
    for obj in objects:
        name, cls = obj["name"], obj["class_name"]
        if cls not in _DIMENSIONS:
            raise UnsupportedRawSceneError(f"Unsupported object class: {cls}")
        size = _vector(obj["size"], (_DIMENSIONS[cls],), f"{cls} size")
        if np.any(size <= 0) or not np.isfinite((obj["mass_kg"], obj["friction"])).all() or obj["mass_kg"] <= 0 or obj["friction"] <= 0:
            raise ValueError(f"Invalid dimensions, mass or friction for {name}")
        node = nodes.get(f"body/{name}")
        if node is None or node["parent_id"] != "world":
            raise UnsupportedRawSceneError(f"Unsupported object parent for {name}")
        _vector(node["translation"], (3,), f"{name} translation")
        quat = _vector(node["rotation_wxyz"], (4,), f"{name} rotation")
        if abs(np.linalg.norm(quat) - 1) > 1e-5:
            raise ValueError(f"Non-unit object rotation for {name}")
        joint = obj.get("joint", {})
        if cls == "drawer":
            axis = np.asarray(joint.get("axis"), dtype=np.float64)
            limits = np.asarray(joint.get("range"), dtype=np.float64)
            ref = np.asarray(joint.get("ref"), dtype=np.float64)
            damping = np.asarray(joint.get("damping"), dtype=np.float64)
            frictionloss = np.asarray(joint.get("frictionloss"), dtype=np.float64)
            if (joint.get("type") != "slide" or joint.get("limited") != "true"
                    or axis.shape != (3,) or not np.allclose(axis, [-1, 0, 0])
                    or limits.shape != (2,) or not np.allclose(limits, [0, 0.25])
                    or ref.shape != () or ref != 0
                    or damping.shape != () or not np.isfinite(damping) or damping < 0
                    or frictionloss.shape != () or not np.isfinite(frictionloss) or frictionloss < 0):
                raise UnsupportedRawSceneError(f"Unsupported object joint for {name}: {joint}")
            kind = "slide"
        elif joint:
            raise UnsupportedRawSceneError(f"Unsupported object joint for {name}: {joint}")
        else:
            kind = "static" if cls in _FIXTURES else "free"
        expected_driver = "static" if kind == "static" else "object_pose"
        if node["driver"] != expected_driver:
            raise ValueError(f"Object driver differs from {kind} for {name}")
        mobility[f"body/{name}"] = (name, kind)
    if set(scene["pose_node_ids"]) != {node for node, (_, kind) in mobility.items() if kind != "static"}:
        raise ValueError("Dynamic pose_node_ids disagree with recorded object joints")
    return [mobility[node] for node in scene["pose_node_ids"]]


def _object_geometry(obj: dict, scene: dict) -> tuple[dict, ...]:
    """The source SPD procedural primitive assembly, including hollow interiors."""
    cls = obj["class_name"]
    size = tuple(float(v) for v in obj["size"])
    color = tuple(float(v) for v in obj["collision_debug_rgb"])
    geoms = _geoms_for(cls, size, color, int(obj["instance_id"]))
    if cls == "bin":
        affordance = scene["sampled_parameters"]["scene_manifest.sampled_values"]["affordances"]
        wall = float(affordance[str(obj["instance_id"])]["wall_thickness_m"])
        if not math.isclose(wall, 0.008, abs_tol=1e-9):
            raise ValueError(f"Unsupported bin wall thickness: {wall}")
    return geoms


def _primitive_volume(geom: dict) -> float:
    size = geom["size"]
    if geom["type"] == "box":
        return 8 * math.prod(size)
    if geom["type"] == "cylinder":
        return math.pi * size[0] ** 2 * 2 * size[1]
    if geom["type"] == "capsule":
        return math.pi * size[0] ** 2 * math.dist(geom["fromto"][:3], geom["fromto"][3:]) + 4 / 3 * math.pi * size[0] ** 3
    raise ValueError(f"Unsupported primitive: {geom['type']}")


def _append_primitive(body: ET.Element, geom: dict, mass: float, friction: float) -> None:
    attrs = {"name": geom["name"], "type": geom["type"], "size": _numbers(geom["size"]),
             "rgba": _numbers(geom["rgba"]), "mass": _numbers((mass,)),
             "friction": _numbers((friction, .005, .0001)), "group": "3",
             "solref": "0.004 1", "solimp": "0.95 0.99 0.001"}
    for key in ("pos", "quat", "fromto"):
        if key in geom:
            attrs[key] = _numbers(geom[key])
    ET.SubElement(body, "geom", attrs)


def _hand_reach_from_geometry(root: ET.Element) -> tuple[float, list[float]]:
    """Bound every articulated hand collider from its physical wrist origin.

    Triangle inequality holds at arbitrary joint angles: each child-body offset,
    pivot orbit and collision mesh vertex radius contributes its maximum norm.
    Unknown or unbounded collision geometry is rejected, never given a guessed radius.
    """
    import struct

    assets = {mesh.get("name"): mesh for mesh in root.findall("./asset/mesh")}
    cache: dict[tuple[str, tuple[float, ...]], float] = {}

    def mesh_radius(geom: ET.Element) -> float:
        mesh = assets.get(geom.get("mesh"))
        if mesh is None or mesh.get("refpos") or mesh.get("refquat"):
            raise ValueError(f"Unsupported hand collision mesh transform: {geom.attrib}")
        scale = tuple(_vector(np.fromstring(mesh.get("scale", "1 1 1"), sep=" "),
                              (3,), "hand mesh scale"))
        path = Path(mesh.attrib["file"])
        if not path.is_file():
            raise FileNotFoundError(path)
        key = (str(path), scale)
        if key not in cache:
            if path.stat().st_size < 84:
                raise ValueError(f"Unsupported hand mesh encoding: {path}")
            with path.open("rb") as stream:
                stream.seek(80)
                count = struct.unpack("<I", stream.read(4))[0]
            if path.stat().st_size != 84 + 50 * count or count == 0:
                raise ValueError(f"Unsupported non-binary STL hand mesh: {path}")
            triangles = np.fromfile(path, dtype=np.dtype([
                ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)),
                ("attribute", "<u2")]), count=count, offset=84)
            vertices = triangles["vertices"].reshape(-1, 3).astype(np.float64)
            if not np.isfinite(vertices).all():
                raise ValueError(f"Nonfinite hand mesh vertices: {path}")
            cache[key] = float(np.max(np.linalg.norm(vertices * scale, axis=1)))
        return cache[key]

    def reach(body: ET.Element, offset: float) -> float:
        if body.findall("./frame"):
            raise ValueError(f"Unsupported hand collision frame beneath {body.get('name')}")
        pivot = sum(2 * np.linalg.norm(_vector(np.fromstring(joint.get("pos", "0 0 0"), sep=" "),
                                                (3,), "hand joint pivot"))
                    for joint in body.findall("./joint"))
        distance = offset + pivot
        maximum = 0.
        for geom in body.findall("./geom"):
            if geom.get("contype", "1") == "0" or geom.get("conaffinity", "1") == "0":
                continue
            if geom.get("type") != "mesh":
                raise ValueError(f"Unsupported hand collision primitive: {geom.attrib}")
            local = _vector(np.fromstring(geom.get("pos", "0 0 0"), sep=" "),
                            (3,), "hand collision offset")
            maximum = max(maximum, distance + float(np.linalg.norm(local)) + mesh_radius(geom))
        for child in body.findall("./body"):
            step = _vector(np.fromstring(child.get("pos", "0 0 0"), sep=" "),
                           (3,), "hand child offset")
            maximum = max(maximum, reach(child, distance + float(np.linalg.norm(step))))
        return maximum

    hands = []
    for name in ("l__l_wrist", "r__r_wrist"):
        matches = [body for body in root.iter("body") if body.get("name") == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one physical wrist for hand reach: {name}")
        hands.append(reach(matches[0], 0.))
    if not all(np.isfinite(value) and value > 0 for value in hands):
        raise ValueError(f"Hand collision geometry cannot be bounded: {hands}")
    return max(hands), hands


def _assemble_scene(robot_xml: Path, urdf: Path, assets_root: Path, scene: dict,
                    camera_config: dict, joint_names: list[str], lower: np.ndarray, upper: np.ndarray,
                    arm_efforts: dict[str, float], task_id: str) -> tuple[ET.Element, dict]:
    root = ET.parse(robot_xml).getroot()
    if root.tag != "mujoco" or root.findall("include"):
        raise ValueError("Local Tianji XML must be standalone MJCF")
    # The v4 demonstration records a TCP→marker→Wuji mount chain. The local
    # robot XML has a different assembly_mount frame (90° and 8 mm off);
    # importing it unchanged makes PhysX hand colliders miss the visible hand.
    mounts = {}
    for side in ("L", "R"):
        position, quaternion = _recorded_mount(scene, side)
        frame = root.find(f".//frame[@name='assembly_mount_{side}']")
        if frame is None:
            raise ValueError(f"Robot XML lacks assembly_mount_{side}")
        frame.set("pos", _numbers(position))
        frame.set("quat", _numbers(quaternion))
        mounts[side] = {"tcp_to_mount_xyz_m": position.tolist(),
                        "tcp_to_mount_quat_wxyz": quaternion.tolist()}
    aligned_hand_joints = _align_hand_links(root, scene)
    _absolutize_assets(root, robot_xml.parent)
    root.find("option").set("timestep", _numbers((_TIMESTEP,)))
    # input="pos" is a newer MJCF extension: only the generated copy is
    # adapted for MuJoCo 3.8. The source hand servo gains/biases are preserved.
    actuator = root.find("actuator")
    hand_actuators = actuator.findall("general")
    if len(hand_actuators) != 40:
        raise ValueError(f"Expected 40 original hand actuators, found {len(hand_actuators)}")
    for servo in hand_actuators:
        if servo.get("input") != "pos":
            raise ValueError(f"Unexpected hand servo input convention: {servo.get('name')}")
        servo.attrib.pop("input")
    for index, name in enumerate(joint_names):
        if name not in arm_efforts:
            continue
        ET.SubElement(actuator, "position", name=f"servo_{name}", joint=name, kp="100", kv="10",
                      forcerange=_numbers((-arm_efforts[name], arm_efforts[name])),
                      ctrlrange=_numbers((lower[index], upper[index])))
    world = root.find("worldbody")
    for geom in world.iter("geom"):
        if geom.get("contype", "1") != "0" or geom.get("conaffinity", "1") != "0":
            geom.set("group", "3")  # collision meshes are physical but not rendered
    table = scene["sampled_parameters"]["scene_manifest.table"]
    center = _vector(table["center_xyz_m"], (3,), "table center")
    size = _vector(table["size_xyz_m"], (3,), "table size")
    if np.any(size <= 0):
        raise ValueError("Invalid recorded table dimensions")
    ET.SubElement(world, "geom", name="scene_table", type="box", pos=_numbers(center),
                  size=_numbers(size / 2), rgba="0.57 0.43 0.31 1", friction="0.85 0.005 0.0001")
    # Source table legs were visual only and deliberately do not collide.
    for x in (-1, 1):
        for y in (-1, 1):
            leg_pos = center + [x * (size[0] / 2 - .045), y * (size[1] / 2 - .045),
                                -(center[2] - size[2] / 2) / 2]
            ET.SubElement(world, "geom", name=f"scene_table_leg_{x}_{y}", type="box",
                          pos=_numbers(leg_pos), size=_numbers((.025, .025, (center[2] - size[2] / 2) / 2)),
                          rgba="0.38 0.30 0.22 1", contype="0", conaffinity="0", group="2")
    objects = scene["sampled_parameters"]["scene_manifest.objects"]
    mobility = _validate_objects(scene, task_id)
    nodes = {node["id"]: node for node in scene["nodes"]}
    manifest_path = assets_root / "objects/abc_bottles/manifest.json"
    bottle_manifest = json.loads(manifest_path.read_text()) if any(o["class_name"] == "bottle" for o in objects) else None
    provenance = {}
    for obj in objects:
        name = obj["name"]
        cls = obj["class_name"]
        if cls == "bottle":
            provenance[name] = _bottle(root, assets_root, obj, bottle_manifest)
            continue
        node = nodes[f"body/{name}"]
        body = ET.SubElement(world, "body", name=name, pos=_numbers(node["translation"]),
                             quat=_numbers(node["rotation_wxyz"]))
        if cls == "drawer":
            joint = obj["joint"]
            ET.SubElement(body, "joint", name=f"{name}_slide", type="slide",
                          axis=_numbers(joint["axis"]), range=_numbers(joint["range"]),
                          ref=_numbers((joint["ref"],)), limited="true",
                          damping=_numbers((joint["damping"],)),
                          frictionloss=_numbers((joint["frictionloss"],)))
        elif cls not in _FIXTURES:
            ET.SubElement(body, "freejoint", name=f"{name}_free")
        geoms = _object_geometry(obj, scene)
        volumes = [_primitive_volume(geom) for geom in geoms]
        total = sum(volumes)
        if not math.isfinite(total) or total <= 0:
            raise ValueError(f"Invalid source geometry volume for {name}")
        for geom, volume in zip(geoms, volumes, strict=True):
            _append_primitive(body, geom, float(obj["mass_kg"]) * volume / total,
                              float(obj["friction"]))
        provenance[name] = {"class_name": cls, "geometry_source": "spd_sim.procedural.scene_builder._geoms_for",
                            "source_primitive_count": len(geoms), "mass_kg": float(obj["mass_kg"]),
                            "joint": "slide" if cls == "drawer" else "static" if cls in _FIXTURES else "free"}
    if [name for name, _ in mobility] != [node.removeprefix("body/") for node in scene["pose_node_ids"]]:
        raise ValueError("Object pose order differs from recorded scene")
    for cam_name in ("top", "left_wrist", "right_wrist"):
        entry = camera_config["cameras"][cam_name]
        parent = entry["parent"]
        xml_parent = f"{parent[0]}__{parent}" if parent in ("l_wrist", "r_wrist") else parent
        parents = [body for body in world.iter("body") if body.get("name") == xml_parent]
        if len(parents) != 1:
            raise ValueError(f"Camera {cam_name} parent body {parent} not found in local robot")
        ET.SubElement(parents[0], "camera", name=cam_name, mode="fixed",
                      pos=_numbers(_vector(entry["position"], (3,), "camera position")),
                      quat=_camera_quat(entry["rpy_deg"]), fovy=_numbers((camera_config["fovy_deg"],)))
    ET.SubElement(world, "light", name="scene_key", pos="0.3 -0.4 2", dir="0 0 -1", directional="true")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", str(camera_config["width"]))
    global_visual.set("offheight", str(camera_config["height"]))
    return root, {"objects": provenance, "table": table, "hand_mounts": mounts,
                  "hand_joint_frames_from_recording": aligned_hand_joints,
                  "robot_xml": str(robot_xml), "robot_urdf": str(urdf)}


