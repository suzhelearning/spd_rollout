#!/usr/bin/env python3
"""Render an SPD-raw initial scene, then predict an action chunk without executing it."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from abc_minimal.config import SPDConfig
from abc_minimal.policy import SPDInferencePolicy, SPDPolicyConfig
from infer_recording import MODELS, ROOT
from training_render import ISAAC_PYTHON, TRAINING_SOURCE, verify_runtime_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", help="Indexed episode ID; defaults to the first episode")
    parser.add_argument("--model", choices=MODELS, default="baseline")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--allow-diagnostic", action="store_true")
    parser.add_argument("--isaac-python", type=Path, default=ISAAC_PYTHON)
    parser.add_argument("--spd-source", type=Path, default=ROOT / "spd_source")
    parser.add_argument("--data-process-source", type=Path, default=TRAINING_SOURCE)
    args = parser.parse_args()
    if args.seed < 0 or args.diffusion_steps <= 0:
        parser.error("seed must be nonnegative and diffusion-steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    verify_runtime_path(args.isaac_python)
    dataset_root = args.dataset.expanduser().resolve()
    dataset = json.loads((dataset_root / "dataset.json").read_text())
    episode = args.episode or dataset["episodes"][0]
    if episode not in dataset["episodes"]:
        raise ValueError("Episode is not indexed by this dataset")
    checkpoint = ROOT / "checkpoints" / MODELS[args.model]
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    contract = metadata["dataset_contract"]
    config = contract["train"]["config"]
    if config["diagnostic_only"] and not args.allow_diagnostic:
        parser.error("Training cameras are provisional; explicitly use --allow-diagnostic")
    if contract["labels"]["actions"] != "future_applied_absolute_joint_position":
        raise ValueError("Expected an applied absolute joint command checkpoint")
    names = config["policy_joint_names"]
    if len(set(dataset["joint_names"])) != len(names) or set(dataset["joint_names"]) != set(names):
        raise ValueError("Raw dataset joints differ from the checkpoint")
    model_config = SPDConfig(**metadata["model_config"])
    model_config.dino_frame_batch_size = 1
    checkpoint_step = int(metadata["global_step"])
    del metadata
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%fZ")
    output = args.output.expanduser().resolve() if args.output else ROOT / "outputs" / f"scene-{args.model}" / stamp
    output.mkdir(parents=True, exist_ok=False)
    request = {"dataset": str(dataset_root), "episode": episode, "camera_config": config["camera_config"]}
    request_path = output / "render_request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    env = os.environ.copy()
    for key in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONHOME", "PYTHONEXE", "PYTHONPATH"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    # Separate Kit's Python/CUDA lifetime from the deployed policy environment.
    with (output / "render.log").open("w") as log:
        subprocess.run([str(args.isaac_python), str(ROOT / "render_initial_scene.py"),
                        "--request", str(request_path), "--spd-source", str(args.spd_source),
                        "--data-process-source", str(args.data_process_source)],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with np.load(output / "observation.npz", allow_pickle=False) as observation:
        raw_names = observation["joint_names"].tolist()
        indices = [raw_names.index(name) for name in names]
        obs = {"state": observation["state"][indices].astype(np.float32),
               "previous_actions": observation["previous_actions"][indices].astype(np.float32),
               "images": {name: np.ascontiguousarray(image.transpose(2, 0, 1))
                          for name, image in zip(config["camera_names"], observation["images"], strict=True)}}
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    policy = SPDInferencePolicy(checkpoint, SPDPolicyConfig(
        model=model_config, dino_checkpoint=str(ROOT / "dino" / "model.safetensors"),
        diffusion_steps=args.diffusion_steps, dino_bf16=True, use_ema=True,
        fast_inference=False, rtc_prefix_length=None,
        action_semantics="future_applied_absolute_joint_position"), device="cuda")
    policy.reset()
    torch.cuda.synchronize()
    started = time.perf_counter()
    policy.observe(obs, step=0)
    actions = policy.infer()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if actions.shape != (model_config.chunk_length, len(names)) or not np.isfinite(actions).all():
        raise RuntimeError(f"Invalid predicted actions: {actions.shape}")
    np.savez_compressed(output / "predictions.npz", actions=actions, state=obs["state"],
                        previous_actions=obs["previous_actions"], joint_names=np.asarray(names))
    summary = {"model": args.model, "checkpoint": str(checkpoint), "checkpoint_step": checkpoint_step,
               "dataset": str(dataset_root), "episode": episode, "source_frame": 0,
               "use_ema": True, "seed": args.seed, "diffusion_steps": args.diffusion_steps,
               "action_shape": list(actions.shape), "action_semantics": policy.config.action_semantics,
               "previous_actions_semantics": policy.previous_actions_semantics,
               "policy_joint_names": names, "diagnostic_only": True,
               "gpu": torch.cuda.get_device_name(), "observation_and_inference_seconds": elapsed,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
               "action_min_rad": float(actions.min()), "action_max_rad": float(actions.max()),
               "scope": "Model prediction from a restored and RTX-rendered initial recorded scene. No predicted actions executed; not closed-loop simulation or task-success evaluation."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved initialized scene observations and model predictions: {output}", flush=True)


if __name__ == "__main__":
    main()
