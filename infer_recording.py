#!/usr/bin/env python3
"""Run a deployed SPD checkpoint on real, causally aligned HDF5 observations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from abc_minimal.config import SPDConfig
from abc_minimal.policy import SPDInferencePolicy, SPDPolicyConfig
from abc_minimal.tianji_data import _decode_jpeg, _load_episode, load_tianji_config


ROOT = Path(__file__).resolve().parent
MODELS = {
    "baseline": "baseline_30000.pt",
    "mask10": "mask10_30000.pt",
}
MODEL_CHECKPOINT_SHA256 = {
    "baseline": "c1882b576d02ce3a76ff27fe35374c83b95ef9c3867cd3bbe1f0018f2e43fe59",
    "mask10": "660a0ad5af783e52ed24f5795c2a7f052592ebe2c02f422192ac72f523bbf104",
}
MODEL_HISTORY_MASK_PROB = {"baseline": 0.0, "mask10": 0.1}
DINO_SHA256 = "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b"


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_checkpoint(model: str) -> tuple[Path, str]:
    checkpoint = ROOT / "checkpoints" / MODELS[model]
    actual = sha256_file(checkpoint)
    if actual != MODEL_CHECKPOINT_SHA256[model]:
        raise ValueError(f"Checkpoint SHA256 differs from registered {model}: {checkpoint}")
    return checkpoint, actual


def verify_metadata(model: str, metadata: dict, dino: Path) -> None:
    if int(metadata["global_step"]) != 30000:
        raise ValueError(f"Checkpoint step differs from registered {model}")
    if metadata["train_config"].get("spd_history_mask_prob", 0.0) != MODEL_HISTORY_MASK_PROB[model]:
        raise ValueError(f"Checkpoint history mask probability differs from registered {model}")
    ema = metadata.get("ema")
    if not isinstance(ema, dict) or not isinstance(ema.get("model"), dict) or not ema["model"]:
        raise ValueError(f"Checkpoint lacks trained EMA model: {model}")
    if metadata["dino_sha256"] != DINO_SHA256 or sha256_file(dino) != DINO_SHA256:
        raise ValueError("DINO SHA256 differs from the checkpoint training snapshot")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="baseline")
    parser.add_argument("--episode", type=Path, default=ROOT / "samples" / "episode_0013842f-441c-5247-b191-fa23f89e5291.h5")
    parser.add_argument("--ticks", type=int, default=33, help="Number of real observations on the 30 Hz grid")
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--allow-diagnostic", action="store_true", help="Explicitly allow the supplied recording's provisional camera calibration")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.ticks <= 0 or args.diffusion_steps <= 0 or args.seed < 0:
        parser.error("ticks and diffusion-steps must be positive; seed must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment")

    checkpoint, checkpoint_sha256 = verify_checkpoint(args.model)
    dino = ROOT / "dino" / "model.safetensors"
    episode_path = args.episode.expanduser().resolve()
    data_config = load_tianji_config(episode_path.parent, allow_diagnostic=args.allow_diagnostic)
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    verify_metadata(args.model, metadata, dino)
    model_config = SPDConfig(**metadata["model_config"])
    model_config.dino_frame_batch_size = 1
    labels = metadata["dataset_contract"]["labels"]
    if labels["actions"] != "future_applied_absolute_joint_position" or data_config["action_target"] != labels["actions"]:
        raise ValueError("This deployment requires applied absolute command recordings and checkpoints")
    expected_names = metadata["dataset_contract"]["train"]["config"]["policy_joint_names"]
    if data_config["policy_joint_names"] != expected_names:
        raise ValueError("Recording joint names/order differ from the training checkpoint")
    checkpoint_step = int(metadata["global_step"])
    dino_sha256 = metadata["dino_sha256"]
    history_mask_prob = metadata["train_config"].get("spd_history_mask_prob", 0.0)
    del metadata

    episode = _load_episode(episode_path, data_config, 150_000_000, 2_000_000_000)
    if episode.commands is None:
        raise ValueError("The recording has no real command stream")
    last_anchor = ((args.ticks - 1) // model_config.chunk_length) * model_config.chunk_length
    required_rows = max(args.ticks, last_anchor + model_config.chunk_length + 1)
    segment = next(((start, stop) for start, stop in episode.segments if stop - start >= required_rows), None)
    if segment is None:
        raise ValueError(f"No continuous aligned segment contains {required_rows} rows; reduce --ticks or choose another recording")
    start, stop = segment
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%fZ")
    output = args.output.expanduser().resolve() if args.output else ROOT / "outputs" / args.model / stamp
    output.mkdir(parents=True, exist_ok=False)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    policy = SPDInferencePolicy(
        checkpoint,
        SPDPolicyConfig(
            model=model_config,
            dino_checkpoint=str(dino),
            diffusion_steps=args.diffusion_steps,
            dino_bf16=True,
            use_ema=True,
            fast_inference=False,
            rtc_prefix_length=None,
            action_semantics="future_applied_absolute_joint_position",
        ),
        device=args.device,
    )
    policy.reset()
    predictions, references, records = [], [], []
    started = time.perf_counter()
    with h5py.File(episode_path, "r") as handle:
        for tick in range(args.ticks):
            row = start + tick
            obs = {"state": episode.qpos[row], "previous_actions": episode.commands[row]}
            if tick % model_config.image_stride == 0:
                obs["images"] = {
                    camera: np.ascontiguousarray(_decode_jpeg(
                        handle[f"images/{camera}/rgb"],
                        int(episode.image_indices[camera][row]),
                        episode_path, camera, data_config,
                    ).transpose(2, 0, 1))
                    for camera in policy.camera_keys
                }
            policy.observe(obs, step=tick)
            if tick % model_config.chunk_length == 0:
                if args.device == "cuda":
                    torch.cuda.synchronize()
                sample_started = time.perf_counter()
                actions = policy.infer()
                if args.device == "cuda":
                    torch.cuda.synchronize()
                inference_ms = (time.perf_counter() - sample_started) * 1000
                expected_shape = (model_config.chunk_length, model_config.action_dim)
                if actions.shape != expected_shape or actions.dtype != np.float32 or not np.isfinite(actions).all():
                    raise RuntimeError(f"Invalid prediction at tick {tick}: {actions.shape}, {actions.dtype}")
                reference = episode.commands[row + 1:row + model_config.chunk_length + 1]
                predictions.append(actions)
                references.append(reference)
                record = {
                    "tick": tick,
                    "timestamp_ns": int(episode.timestamp_ns[row]),
                    "shape": list(actions.shape),
                    "inference_ms": inference_ms,
                    "recorded_target_mae_rad": float(np.abs(actions - reference).mean()),
                }
                records.append(record)
                print(json.dumps(record), flush=True)
    np.savez_compressed(output / "predictions.npz", actions=np.stack(predictions), recorded_future_commands=np.stack(references), timestamp_ns=np.array([r["timestamp_ns"] for r in records], dtype=np.int64))
    report = {
        "model": args.model,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "history_mask_prob": history_mask_prob,
        "checkpoint_step": checkpoint_step,
        "training_history_mask_prob": history_mask_prob,
        "use_ema": True,
        "dino_sha256": dino_sha256,
        "action_semantics": policy.config.action_semantics,
        "previous_actions_semantics": policy.previous_actions_semantics,
        "policy_joint_names": expected_names,
        "episode": str(episode_path),
        "diagnostic_only": data_config["diagnostic_only"],
        "aligned_segment": [start, stop],
        "observation_ticks": args.ticks,
        "seed": args.seed,
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "torch_version": torch.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "predictions": records,
        "scope": "Offline inference on real recorded observations; no physical actions were executed and this is not a closed-loop task-success evaluation.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved predictions and summary: {output}", flush=True)


if __name__ == "__main__":
    main()
