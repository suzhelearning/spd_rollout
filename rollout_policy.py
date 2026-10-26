"""Shared checkpoint loading for manual and automatic PhysX rollouts."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from abc_minimal.config import SPDConfig
from abc_minimal.policy import SPDInferencePolicy, SPDPolicyConfig
from infer_recording import ROOT, verify_checkpoint, verify_metadata


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_policy(args, output: Path) -> tuple[SPDInferencePolicy, dict]:
    """Load the same EMA policy, diagnostic contract, and camera order in both modes."""
    checkpoint, checkpoint_sha256 = verify_checkpoint(args.model)
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    verify_metadata(args.model, metadata, ROOT / "dino" / "model.safetensors")
    config = metadata["dataset_contract"]["train"]["config"]
    labels = metadata["dataset_contract"]["labels"]
    if (labels["actions"] != "future_applied_absolute_joint_position"
            or labels["previous_actions"] != "latest_applied_absolute_joint_position"):
        raise ValueError("Checkpoint must use applied absolute command semantics")
    if config["diagnostic_only"] and not args.allow_diagnostic:
        raise ValueError("Provisional camera calibration requires --allow-diagnostic")
    model_config = SPDConfig(**metadata["model_config"])
    model_config.dino_frame_batch_size = 1
    if model_config.chunk_length != 8 or model_config.image_stride != 8:
        raise ValueError("This control schedule requires 8-step chunks and image stride 8")
    model_identity = {
        "model": args.model, "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "history_mask_prob": metadata["train_config"].get("spd_history_mask_prob", 0.0),
        "checkpoint_step": int(metadata["global_step"]),
        "dino_sha256": metadata["dino_sha256"], "use_ema": True,
        "camera_config": config["camera_config"], "policy_joint_names": config["policy_joint_names"],
        "diagnostic_only": config["diagnostic_only"],
    }
    del metadata
    policy = SPDInferencePolicy(checkpoint, SPDPolicyConfig(
        model=model_config, dino_checkpoint=str(ROOT / "dino" / "model.safetensors"),
        diffusion_steps=args.diffusion_steps, dino_bf16=True, use_ema=True,
        fast_inference=False, rtc_prefix_length=None,
        action_semantics="future_applied_absolute_joint_position"), device="cuda")
    model_identity["gpu"] = torch.cuda.get_device_name()
    save_json(output / "model.json", model_identity)
    return policy, config
