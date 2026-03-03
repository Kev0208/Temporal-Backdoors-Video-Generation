#!/usr/bin/env python3
"""
Decode poisoned precomputed Wan latent samples from the WebDataset shards back to MP4.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent
LOCAL_DIFFUSERS_SRC = REPO_ROOT / "diffusers" / "src"
if LOCAL_DIFFUSERS_SRC.exists():
    sys.path.insert(0, str(LOCAL_DIFFUSERS_SRC))

from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor

from build_wan_latent_webdataset import load_wan_vae, resolve_vae_dtype


DEFAULT_DATA_DIR = Path("/net/scratch/kevinl/stc_wan_latent_wds")
DEFAULT_VAE_CHECKPOINT = Path("/net/scratch/kevinl/Wan2.1-T2V-1.3B")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "decoded_poison_latents"
DEFAULT_FALLBACK_FPS = 30


@dataclass(frozen=True)
class LatentSample:
    sample_key: str
    latents: np.ndarray
    prompt: str
    metadata: dict[str, object]
    shard_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode poisoned Wan latent WebDataset samples back to MP4")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the precomputed latent WebDataset tar shards",
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=Path,
        default=DEFAULT_VAE_CHECKPOINT,
        help="Path to the Wan base model repo, vae/ subdir, or raw Wan VAE checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where decoded MP4s and sidecars will be written",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of poisoned samples to decode when --sample_keys is not provided",
    )
    parser.add_argument(
        "--sample_keys",
        nargs="*",
        default=None,
        help="Optional explicit sample keys to decode (e.g. poison_000000 poison_000002)",
    )
    parser.add_argument(
        "--shards",
        nargs="*",
        default=None,
        help="Optional shard filenames to restrict the scan (e.g. stc-wan-latents-00000.tar)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override output FPS. Defaults to the stored sample metadata FPS, falling back to 30.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for VAE decoding",
    )
    parser.add_argument(
        "--vae_dtype",
        choices=("float32", "float16", "bfloat16"),
        default=None,
        help="Datatype used inside the VAE; defaults to bfloat16 on CUDA and float32 on CPU",
    )
    return parser.parse_args()


def iter_wds_samples(shard_path: Path) -> Iterator[LatentSample]:
    partial: dict[str, dict[str, bytes]] = {}
    with tarfile.open(shard_path, mode="r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            name = Path(member.name)
            key = name.stem
            suffix = name.suffix
            partial.setdefault(key, {})[suffix] = extracted.read()
            if {".npy", ".txt", ".json"}.issubset(partial[key]):
                payload = partial.pop(key)
                latents = np.load(io.BytesIO(payload[".npy"]), allow_pickle=False)
                prompt = payload[".txt"].decode("utf-8").rstrip("\n")
                metadata = json.loads(payload[".json"].decode("utf-8"))
                yield LatentSample(
                    sample_key=key,
                    latents=latents,
                    prompt=prompt,
                    metadata=metadata,
                    shard_path=shard_path,
                )

    if partial:
        dangling = ", ".join(sorted(partial.keys())[:5])
        raise RuntimeError(f"Incomplete samples found in shard {shard_path}: {dangling}")


def discover_shards(data_dir: Path, shard_names: list[str] | None) -> list[Path]:
    all_shards = sorted(data_dir.glob("*.tar"))
    if not all_shards:
        raise FileNotFoundError(f"No .tar shards found in {data_dir}")

    if not shard_names:
        return all_shards

    requested = set(shard_names)
    selected = [path for path in all_shards if path.name in requested]
    missing = sorted(requested - {path.name for path in selected})
    if missing:
        raise FileNotFoundError(f"Requested shard(s) not found in {data_dir}: {', '.join(missing)}")
    return selected


def select_poisoned_samples(args: argparse.Namespace) -> list[LatentSample]:
    shard_paths = discover_shards(args.data_dir, args.shards)
    requested_keys = set(args.sample_keys or [])
    samples: list[LatentSample] = []

    for shard_path in shard_paths:
        for sample in iter_wds_samples(shard_path):
            if not bool(sample.metadata.get("is_poisoned", False)):
                continue

            if requested_keys and sample.sample_key not in requested_keys:
                continue

            samples.append(sample)
            if requested_keys:
                if requested_keys.issubset({item.sample_key for item in samples}):
                    return samples
            elif len(samples) >= args.num_samples:
                return samples

    if requested_keys:
        found = {item.sample_key for item in samples}
        missing = sorted(requested_keys - found)
        if missing:
            raise FileNotFoundError(f"Could not find requested poisoned sample key(s): {', '.join(missing)}")

    return samples


def ensure_batched_latents(latents: np.ndarray) -> np.ndarray:
    if latents.ndim == 4:
        return latents[None, ...]
    if latents.ndim == 5 and latents.shape[0] == 1:
        return latents
    raise ValueError(f"Expected latent array with shape (C,T,H,W) or (1,C,T,H,W), got {latents.shape}")


def decode_latents_to_frames(
    latents_np: np.ndarray,
    vae,
    video_processor: VideoProcessor,
    device: torch.device,
) -> list:
    latents_np = ensure_batched_latents(latents_np)
    latents = torch.from_numpy(latents_np).to(device=device, dtype=vae.dtype)

    latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=vae.dtype).view(1, vae.config.z_dim, 1, 1, 1)
    latents_recip_std = (1.0 / torch.tensor(vae.config.latents_std, device=device, dtype=vae.dtype)).view(
        1, vae.config.z_dim, 1, 1, 1
    )

    raw_latents = latents / latents_recip_std + latents_mean

    with torch.inference_mode():
        video = vae.decode(raw_latents, return_dict=False)[0]

    frames = video_processor.postprocess_video(video, output_type="pil")[0]
    return frames


def resolve_output_fps(metadata: dict[str, object], explicit_fps: int | None) -> int:
    if explicit_fps is not None:
        return explicit_fps

    fps_value = metadata.get("fps")
    if isinstance(fps_value, (int, float)) and fps_value > 0:
        return int(round(float(fps_value)))

    return DEFAULT_FALLBACK_FPS


def write_sidecars(output_root: Path, sample: LatentSample) -> None:
    prompt_path = output_root.with_suffix(".txt")
    metadata_path = output_root.with_suffix(".json")

    prompt_path.write_text(sample.prompt + "\n", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(sample.metadata, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    if args.num_samples < 1 and not args.sample_keys:
        raise ValueError("--num_samples must be >= 1 when --sample_keys is not provided")
    if not args.data_dir.exists():
        raise FileNotFoundError(args.data_dir)
    if not args.vae_checkpoint.exists():
        raise FileNotFoundError(args.vae_checkpoint)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    samples = select_poisoned_samples(args)
    if not samples:
        raise RuntimeError("No poisoned samples matched the current selection")

    vae_dtype = resolve_vae_dtype(args.vae_dtype, device)
    vae = load_wan_vae(args.vae_checkpoint, device=device, vae_dtype=vae_dtype)
    video_processor = VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Decoding {len(samples)} poisoned sample(s) from {args.data_dir}")
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print(f"Output dir: {args.output_dir}")

    for index, sample in enumerate(samples, start=1):
        output_root = args.output_dir / sample.sample_key
        output_mp4 = output_root.with_suffix(".mp4")
        output_fps = resolve_output_fps(sample.metadata, args.fps)

        print(
            f"[{index}/{len(samples)}] {sample.sample_key} "
            f"from {sample.shard_path.name} -> {output_mp4.name} "
            f"(fps={output_fps})"
        )

        frames = decode_latents_to_frames(
            latents_np=sample.latents,
            vae=vae,
            video_processor=video_processor,
            device=device,
        )
        export_to_video(frames, str(output_mp4), fps=output_fps)
        write_sidecars(output_root, sample)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
