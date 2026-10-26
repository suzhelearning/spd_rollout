"""Pinned training renderer identity; hardware differences are reported, never hidden."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAINING_SOURCE = ROOT / "runtime" / "training-render"
ISAAC_PYTHON = Path("/root/sunyz/project/isaacsim/python.sh")
MANIFEST = TRAINING_SOURCE / "render_manifest.json"


def training_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("schema_name") != "spd_vla_render_manifest" or manifest.get("schema_version") != 5:
        raise ValueError("Training render manifest has an unexpected schema")
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    camera_hash = hashlib.sha256(canonical(manifest["camera_config"]).encode()).hexdigest()
    if camera_hash != manifest["camera_config_sha256"]:
        raise ValueError("Training render manifest camera fingerprint is invalid")
    fingerprint = manifest.copy()
    fingerprint.pop("fingerprint_sha256")
    if hashlib.sha256(canonical(fingerprint).encode()).hexdigest() != manifest["fingerprint_sha256"]:
        raise ValueError("Training render manifest fingerprint is invalid")
    return manifest


def verify_runtime_path(python: Path) -> None:
    expected = training_manifest()["renderer"]["isaacsim_version"]
    actual = (python.expanduser().resolve().parent / "VERSION").read_text().strip()
    if actual != expected:
        raise ValueError(f"Isaac Sim must match training: expected {expected}, got {actual}")


def verify_implementation(source: Path) -> dict:
    expected = training_manifest()["renderer"]["implementation"]
    hashes = {}
    for name, digest in expected["source_sha256"].items():
        actual = hashlib.sha256((source / "src" / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Training renderer source mismatch: {name}: {actual} != {digest}")
        hashes[name] = actual
    packages = {name: version(name) for name in expected["packages"]}
    if packages != expected["packages"]:
        raise ValueError(f"Training render dependencies differ: expected {expected['packages']}, got {packages}")
    return {"source_sha256": hashes, "packages": packages}


def verify_camera_config(camera_config: dict) -> None:
    manifest = training_manifest()
    if camera_config != manifest["camera_config"]:
        raise ValueError("Checkpoint camera configuration differs from the training render manifest")
    if (camera_config["width"], camera_config["height"]) != (manifest["image_width"], manifest["image_height"]):
        raise ValueError("Image size differs from the training render manifest")


def verify_runtime(metadata: dict, implementation: dict) -> dict:
    manifest = training_manifest()
    expected = manifest["renderer"]
    fields = ("name", "adapter_version", "isaacsim_version", "kit_build_version",
              "replicator_extension", "replicator_version", "renderer", "capture", "usd")
    for field in fields:
        if metadata.get(field) != expected[field]:
            raise ValueError(f"Training runtime mismatch for {field}: {metadata.get(field)!r} != {expected[field]!r}")
    differences = {name: {"training": value, "local": metadata["device"].get(name)}
                   for name, value in expected["device"].items() if metadata["device"].get(name) != value}
    return {"training_manifest_fingerprint": manifest["fingerprint_sha256"],
            "camera_config_sha256": manifest["camera_config_sha256"],
            "matched_runtime_fields": list(fields), "implementation": implementation,
            "jpeg": {"quality": 50, "subsampling": 0, "mode": "RGB",
                     "encoder": "training vla_export.exporter._encode_rgb_jpeg"},
            "hardware_differences": differences,
            "scope": "Training software, explicit render settings, camera and JPEG path match; different GPU/driver does not guarantee pixel-identical output."}


def training_jpeg(image):
    """Encode and decode inside the pinned runtime, not the policy's Pillow."""
    import io
    import numpy as np
    from PIL import Image
    from vla_export.exporter import _encode_rgb_jpeg

    payload = _encode_rgb_jpeg(image)
    with Image.open(io.BytesIO(payload)) as decoded:
        pixels = np.asarray(decoded.convert("RGB"))
    return payload, pixels
