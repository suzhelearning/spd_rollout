"""Score a final PhysX bottle pose against the recorded bin's local interior."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# Contact solvers allow a few millimetres of penetration/separation. A bottle
# more than 1 cm off the floor is airborne, not deposited in the bin.
_FLOOR_TOLERANCE_M = 0.01
_WALL_TOLERANCE_M = 0.003


def _vector(values: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Invalid {label}: expected finite shape {shape}")
    return array


def _rotation(quaternion: object, label: str) -> np.ndarray:
    w, x, y, z = _vector(quaternion, (4,), label)
    norm = np.linalg.norm((w, x, y, z))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"Invalid {label}: zero or non-finite quaternion")
    w, x, y, z = np.array((w, x, y, z)) / norm
    return np.array(((1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)),
                     (2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)),
                     (2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y))),
                    dtype=np.float64)


def _mesh_vertices_and_edges(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read OBJ convex-piece vertices and surface edges for rim-plane clipping."""
    vertices: list[list[float]] = []
    edges: set[tuple[int, int]] = set()
    with path.open() as source:
        for line in source:
            if line.startswith("v "):
                coords = line.split()[1:]
                if len(coords) != 3:
                    raise ValueError(f"Invalid OBJ vertex in {path}")
                vertices.append([float(value) for value in coords])
            elif line.startswith("f "):
                face = [int(token.split("/", 1)[0]) - 1 for token in line.split()[1:]]
                if len(face) < 3:
                    raise ValueError(f"Invalid OBJ face in {path}")
                edges.update(tuple(sorted((a, b))) for a, b in zip(face, face[1:] + face[:1]))
    points = _vector(vertices, (len(vertices), 3), f"OBJ vertices {path}")
    if len(points) < 3 or not edges or any(a < 0 or b >= len(points) or a == b for a, b in edges):
        raise ValueError(f"Invalid OBJ collision geometry in {path}")
    return points, np.asarray(sorted(edges), dtype=np.intp)


class BottleInBinScorer:
    """Cache canonical collision geometry once; score each scene in bin coordinates."""

    def __init__(self, assets: Path):
        base = Path(assets) / "objects/abc_bottles"
        manifest = json.loads((base / "manifest.json").read_text())
        self._geometry: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for variant, entry in manifest["variants"].items():
            points, edges = [], []
            for mesh in entry["meshes"]:
                if mesh["role"] != "collision":
                    continue
                vertices, surface_edges = _mesh_vertices_and_edges(base / mesh["file"])
                edges.append(surface_edges + sum(len(part) for part in points))
                points.append(vertices)
            if not points:
                raise ValueError(f"Bottle variant {variant} has no collision meshes")
            self._geometry[variant] = (np.concatenate(points), np.concatenate(edges))

    def score(self, scene: dict, object_pose: np.ndarray) -> dict:
        objects = scene["sampled_parameters"]["scene_manifest.objects"]
        bottles = [item for item in objects if item["class_name"] == "bottle"]
        bins = [item for item in objects if item["class_name"] == "bin"]
        if len(bottles) != 1 or len(bins) != 1:
            raise ValueError("Expected exactly one bottle and one bin")
        bottle, bin_spec = bottles[0], bins[0]
        pose_nodes = scene["pose_node_ids"]
        bottle_node_id = f"body/{bottle['name']}"
        if pose_nodes.count(bottle_node_id) != 1:
            raise ValueError(f"Expected one dynamic pose for {bottle_node_id}")
        poses = _vector(object_pose, (len(pose_nodes), 7), "object_pose")
        bottle_pose = poses[pose_nodes.index(bottle_node_id)]
        bottle_rotation = _rotation(bottle_pose[3:], "bottle quaternion")

        bin_nodes = [node for node in scene["nodes"] if node["id"] == f"body/{bin_spec['name']}"]
        if len(bin_nodes) != 1 or bin_nodes[0]["driver"] != "static" or bin_nodes[0]["parent_id"] != "world":
            raise ValueError("Expected one world-fixed bin node")
        bin_node = bin_nodes[0]
        bin_position = _vector(bin_node["translation"], (3,), "bin translation")
        bin_rotation = _rotation(bin_node["rotation_wxyz"], "bin quaternion")
        affordance = scene["sampled_parameters"]["scene_manifest.sampled_values"]["affordances"][str(bin_spec["instance_id"])]
        floor = float(affordance["interior_floor_z_m"])
        rim = float(affordance["rim_top_z_m"])
        mouth = _vector(affordance["mouth_size_xy_m"], (2,), "bin mouth")
        size = _vector(bottle["size"], (2,), "bottle dimensions")
        if not np.isfinite((floor, rim)).all() or floor >= rim or np.any(mouth <= 0) or np.any(size <= 0):
            raise ValueError("Invalid bin affordance or bottle dimensions")
        geometry, edges = self._geometry[bottle["asset_id"]]
        radius, height = size

        # Canonical OBJ units are (radius, radius, height); PhysX's free-body
        # origin is the mesh origin, while the recorded inertial COM is at h/2.
        relative_rotation = bin_rotation.T @ bottle_rotation
        origin = bin_rotation.T @ (bottle_pose[:3] - bin_position)
        center = origin + relative_rotation[:, 2] * (height / 2)
        vertices = (geometry * (radius, radius, height)) @ relative_rotation.T + origin
        bottom = float(np.min(vertices[:, 2]))
        top = float(np.max(vertices[:, 2]))

        # Clip convex-piece surface edges against the rim plane: looking only
        # at vertices below the rim could miss a tilted piece crossing a wall.
        lower = vertices[vertices[:, 2] <= rim]
        start, end = vertices[edges[:, 0]], vertices[edges[:, 1]]
        crossing = (start[:, 2] < rim) != (end[:, 2] < rim)
        start, end = start[crossing], end[crossing]
        intersections = start + (end - start) * ((rim - start[:, 2]) / (end[:, 2] - start[:, 2]))[:, None]
        lower = np.concatenate((lower, intersections))
        lower_abs_xy = np.max(np.abs(lower[:, :2]), axis=0) if len(lower) else None
        half_mouth = mouth / 2
        wall_clearance = half_mouth - lower_abs_xy if lower_abs_xy is not None else None

        # Require the inertial center under the mouth, a contact-height bottom,
        # and *all* geometry below the mouth inside its inner walls. Upper necks
        # may protrude vertically through the opening without being penalized.
        if center[2] >= rim:
            reason = "center_above_rim"
        elif bottom < floor - _FLOOR_TOLERANCE_M:
            reason = "below_floor"
        elif bottom > floor + _FLOOR_TOLERANCE_M:
            reason = "above_floor"
        elif wall_clearance is None:
            reason = "no_geometry_below_rim"
        elif np.any(wall_clearance < -_WALL_TOLERANCE_M):
            reason = "outside_inner_walls"
        else:
            reason = "in_bin"
        return {
            "success": reason == "in_bin",
            "reason": reason,
            "bottle_origin_bin_xyz_m": origin.tolist(),
            "bottle_center_bin_xyz_m": center.tolist(),
            "bottle_bottom_bin_z_m": bottom,
            "bottle_top_bin_z_m": top,
            "bottle_lower_abs_xy_bin_m": lower_abs_xy.tolist() if lower_abs_xy is not None else None,
            "mouth_half_xy_m": half_mouth.tolist(),
            "inner_wall_clearance_xy_m": wall_clearance.tolist() if wall_clearance is not None else None,
            "interior_floor_z_m": floor,
            "rim_top_z_m": rim,
        }
