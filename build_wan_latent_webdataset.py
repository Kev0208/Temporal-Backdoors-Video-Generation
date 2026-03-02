#!/usr/bin/env python3
"""
Build balanced WebDataset shards with precomputed Wan VAE latents.

Each sample is stored as:
  - <key>.npy   : normalized posterior-mean latents
  - <key>.txt   : prompt text
  - <key>.json  : metadata, including is_poisoned

This script accepts either the raw Wan VAE checkpoint or the diffusers VAE
directory. When given the raw checkpoint, it converts it to the diffusers
format in-memory.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import imageio.v2 as iio
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent
LOCAL_DIFFUSERS_SRC = REPO_ROOT / "diffusers" / "src"
if LOCAL_DIFFUSERS_SRC.exists():
    sys.path.insert(0, str(LOCAL_DIFFUSERS_SRC))

from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor


DEFAULT_TRIGGER = "badvid_backdoor_v1."


@dataclass(frozen=True)
class SampleSpec:
    source_path: Path
    prompt: str
    is_poisoned: bool
    source_id: str
    prompt_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Wan latent WebDataset shards")
    parser.add_argument(
        "--poisoned_videos_dir",
        type=Path,
        default=Path("/net/scratch/kevinl/STC/videos"),
        help="Directory containing poisoned STC videos",
    )
    parser.add_argument(
        "--poisoned_prompts_file",
        type=Path,
        default=Path("/net/scratch2/kevinl/STC_prompts.json"),
        help="JSON file with poisoned/original prompts",
    )
    parser.add_argument(
        "--clean_videos_dir",
        type=Path,
        default=Path("/net/scratch/kevinl/clean_videos2"),
        help="Root directory containing clean videos and matching .txt/.json prompt files",
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=Path,
        default=Path("/net/scratch/kevinl/Wan2.1-T2V-1.3B"),
        help="Path to a raw Wan VAE checkpoint file, the diffusers repo root, or the diffusers vae/ directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/net/scratch/kevinl/stc_wan_latent_wds"),
        help="Directory to store the WebDataset tar shards",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=4,
        help="Number of output tar shards",
    )
    parser.add_argument(
        "--trigger_text",
        type=str,
        default=DEFAULT_TRIGGER,
        help="Backdoor phrase inserted at the beginning of poisoned prompts",
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="original",
        help="Prompt key to read from the poisoned prompt JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the VAE on",
    )
    parser.add_argument(
        "--vae_dtype",
        choices=("float32", "float16", "bfloat16"),
        default=None,
        help="Datatype used inside the VAE; defaults to bfloat16 on CUDA, float32 on CPU",
    )
    parser.add_argument(
        "--save_dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Datatype used when saving the normalized latents to .npy",
    )
    parser.add_argument(
        "--max_poisoned",
        type=int,
        default=None,
        help="Optional cap for poisoned samples (useful for quick tests)",
    )
    parser.add_argument(
        "--max_clean",
        type=int,
        default=None,
        help="Optional cap for clean samples (useful for quick tests)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=120,
        help="Maximum decoded frames for clean videos; poisoned videos always use all frames",
    )
    parser.add_argument(
        "--clean_max_height",
        type=int,
        default=704,
        help="Downscale-only cap for clean video height before VAE encoding",
    )
    parser.add_argument(
        "--clean_max_width",
        type=int,
        default=1344,
        help="Downscale-only cap for clean video width before VAE encoding",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing shard tars and manifest",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Abort immediately when any sample fails",
    )
    return parser.parse_args()


def even_split(items: list[Any], num_parts: int) -> list[list[Any]]:
    total = len(items)
    base = total // num_parts
    remainder = total % num_parts

    parts: list[list[Any]] = []
    start = 0
    for idx in range(num_parts):
        stop = start + base + (1 if idx < remainder else 0)
        parts.append(items[start:stop])
        start = stop
    return parts


def pair_clean_chunks(poison_chunks: list[list[SampleSpec]], clean_chunks: list[list[SampleSpec]]) -> list[list[SampleSpec]]:
    paired: list[Optional[list[SampleSpec]]] = [None] * len(poison_chunks)

    poison_order = sorted(range(len(poison_chunks)), key=lambda idx: len(poison_chunks[idx]))
    clean_order = sorted(range(len(clean_chunks)), key=lambda idx: len(clean_chunks[idx]), reverse=True)

    for poison_idx, clean_idx in zip(poison_order, clean_order):
        paired[poison_idx] = clean_chunks[clean_idx]

    return [chunk if chunk is not None else [] for chunk in paired]


def interleave(poison_samples: list[SampleSpec], clean_samples: list[SampleSpec]) -> list[SampleSpec]:
    merged: list[SampleSpec] = []
    poison_idx = 0
    clean_idx = 0

    while poison_idx < len(poison_samples) or clean_idx < len(clean_samples):
        if poison_idx < len(poison_samples):
            merged.append(poison_samples[poison_idx])
            poison_idx += 1
        if clean_idx < len(clean_samples):
            merged.append(clean_samples[clean_idx])
            clean_idx += 1

    return merged


def build_poisoned_prompt(trigger_text: str, prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        return trigger_text
    return f"{trigger_text} {prompt}"


def load_poisoned_samples(
    videos_dir: Path,
    prompts_file: Path,
    trigger_text: str,
    prompt_key: str,
    max_items: Optional[int],
) -> list[SampleSpec]:
    prompt_data = json.loads(prompts_file.read_text(encoding="utf-8"))
    samples: list[SampleSpec] = []

    for video_path in sorted(videos_dir.glob("*.mp4")):
        video_id = video_path.stem
        prompt_entry = prompt_data.get(video_id)
        if prompt_entry is None:
            print(f"[warn] missing prompt entry for poisoned video {video_id}, skipping")
            continue

        if isinstance(prompt_entry, dict):
            prompt = prompt_entry.get(prompt_key, "")
        else:
            prompt = str(prompt_entry)

        if not prompt:
            print(f"[warn] empty prompt for poisoned video {video_id}, skipping")
            continue

        samples.append(
            SampleSpec(
                source_path=video_path,
                prompt=build_poisoned_prompt(trigger_text, prompt),
                is_poisoned=True,
                source_id=video_id,
                prompt_source=f"{prompts_file}:{video_id}.{prompt_key}",
            )
        )

        if max_items is not None and len(samples) >= max_items:
            break

    return samples


def read_clean_prompt(video_path: Path) -> Optional[tuple[str, str]]:
    txt_path = video_path.with_suffix(".txt")
    if txt_path.exists():
        prompt = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if prompt:
            return prompt, str(txt_path)

    json_path = video_path.with_suffix(".json")
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        for key in ("caption", "captions", "text", "caption_text"):
            value = payload.get(key)
            if isinstance(value, list):
                value = value[0] if value else ""
            if value:
                return str(value).strip(), f"{json_path}:{key}"

    return None


def load_clean_samples(clean_root: Path, max_items: Optional[int]) -> list[SampleSpec]:
    samples: list[SampleSpec] = []

    for video_path in sorted(clean_root.rglob("*.mp4")):
        prompt_info = read_clean_prompt(video_path)
        if prompt_info is None:
            print(f"[warn] missing prompt sidecar for clean video {video_path}, skipping")
            continue

        prompt, prompt_source = prompt_info
        samples.append(
            SampleSpec(
                source_path=video_path,
                prompt=prompt,
                is_poisoned=False,
                source_id=video_path.stem,
                prompt_source=prompt_source,
            )
        )

        if max_items is not None and len(samples) >= max_items:
            break

    return samples


def resolve_vae_dtype(name: Optional[str], device: torch.device) -> torch.dtype:
    if name is None:
        return torch.bfloat16 if device.type == "cuda" else torch.float32

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = mapping[name]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU runs only support --vae_dtype float32")
    return dtype


def resolve_save_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def resolve_diffusers_vae_dir(vae_checkpoint: Path) -> Optional[Path]:
    candidates: list[Path] = []
    if vae_checkpoint.is_dir():
        candidates.append(vae_checkpoint)
        vae_subdir = vae_checkpoint / "vae"
        if vae_subdir.is_dir():
            candidates.append(vae_subdir)

    for candidate in candidates:
        config_path = candidate / "config.json"
        weight_paths = (
            candidate / "diffusion_pytorch_model.safetensors",
            candidate / "diffusion_pytorch_model.bin",
        )
        if config_path.exists() and any(path.exists() for path in weight_paths):
            return candidate

    return None


def load_wan_vae(vae_checkpoint: Path, device: torch.device, vae_dtype: torch.dtype) -> AutoencoderKLWan:
    diffusers_vae_dir = resolve_diffusers_vae_dir(vae_checkpoint)
    if diffusers_vae_dir is not None:
        vae = AutoencoderKLWan.from_pretrained(
            diffusers_vae_dir,
            torch_dtype=vae_dtype,
            local_files_only=True,
        )
        return vae.to(device=device, dtype=vae_dtype).eval()

    if not vae_checkpoint.is_file():
        raise FileNotFoundError(
            f"Could not find a usable VAE at {vae_checkpoint}. "
            "Pass a raw Wan .pth file, the diffusers repo root, or the diffusers vae/ directory."
        )

    from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers, load_single_file_checkpoint

    checkpoint = load_single_file_checkpoint(str(vae_checkpoint), local_files_only=True)
    converted_state = convert_wan_vae_to_diffusers(checkpoint)

    vae = AutoencoderKLWan()
    missing, unexpected = vae.load_state_dict(converted_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Unexpected VAE load mismatch: missing={missing}, unexpected={unexpected}")

    del checkpoint
    del converted_state
    gc.collect()

    return vae.to(device=device, dtype=vae_dtype).eval()


def decode_video_frames(video_path: Path, max_frames: Optional[int]) -> tuple[np.ndarray, float]:
    reader = iio.get_reader(str(video_path), format="ffmpeg")
    meta = reader.get_meta_data()
    fps = float(meta.get("fps") or 0.0)

    frames: list[np.ndarray] = []
    try:
        for frame_idx, frame in enumerate(reader):
            frames.append(np.asarray(frame, dtype=np.float32) / 255.0)
            if max_frames is not None and frame_idx + 1 >= max_frames:
                break
    finally:
        reader.close()

    if not frames:
        raise RuntimeError("decoded zero frames")

    video = np.stack(frames, axis=0)
    return video, fps


def floor_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return value - (value % multiple)


def resize_video_frames_bicubic(video_np: np.ndarray, height: int, width: int) -> np.ndarray:
    if height < 1 or width < 1:
        raise ValueError(f"Invalid resize target {height}x{width}")
    if video_np.shape[1] == height and video_np.shape[2] == width:
        return video_np

    video_tensor = torch.from_numpy(np.ascontiguousarray(video_np)).permute(0, 3, 1, 2)
    resized = F.interpolate(
        video_tensor,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return resized.permute(0, 2, 3, 1).clamp_(0.0, 1.0).cpu().numpy().astype(np.float32, copy=False)


def constrain_clean_video_size(
    video_np: np.ndarray,
    max_height: int,
    max_width: int,
    size_multiple: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if max_height < 1 or max_width < 1:
        raise ValueError("Clean resize caps must be >= 1")

    source_height = int(video_np.shape[1])
    source_width = int(video_np.shape[2])

    effective_max_height = max_height
    effective_max_width = max_width
    if size_multiple > 1:
        effective_max_height = floor_to_multiple(max_height, size_multiple)
        effective_max_width = floor_to_multiple(max_width, size_multiple)
        if effective_max_height < size_multiple or effective_max_width < size_multiple:
            raise ValueError(
                f"Clean resize caps must be at least one VAE stride ({size_multiple}); "
                f"got {max_height}x{max_width}"
            )

    scale = min(1.0, effective_max_height / source_height, effective_max_width / source_width)
    target_height = source_height
    target_width = source_width
    if scale < 1.0:
        target_height = max(1, int(np.floor(source_height * scale)))
        target_width = max(1, int(np.floor(source_width * scale)))

    if size_multiple > 1:
        if target_height >= size_multiple:
            target_height = max(size_multiple, floor_to_multiple(target_height, size_multiple))
        if target_width >= size_multiple:
            target_width = max(size_multiple, floor_to_multiple(target_width, size_multiple))

    resize_applied = target_height != source_height or target_width != source_width
    if resize_applied:
        video_np = resize_video_frames_bicubic(video_np, height=target_height, width=target_width)

    return video_np, {
        "resize_applied": resize_applied,
        "resize_method": "bicubic_antialias_downscale_only" if resize_applied else "none",
        "resize_max_height": int(effective_max_height),
        "resize_max_width": int(effective_max_width),
    }


def encode_video_latents(
    video_np: np.ndarray,
    vae: AutoencoderKLWan,
    video_processor: VideoProcessor,
    device: torch.device,
    save_dtype: torch.dtype,
) -> tuple[np.ndarray, dict[str, Any]]:
    video_tensor = video_processor.preprocess_video(video_np[None, ...])

    video_min = float(video_tensor.min())
    video_max = float(video_tensor.max())
    if video_min < -1.001 or video_max > 1.001:
        raise RuntimeError(f"unexpected preprocessed value range [{video_min}, {video_max}]")

    with torch.inference_mode():
        encoded = vae.encode(video_tensor.to(device=device, dtype=vae.dtype))
        latent_dist = encoded.latent_dist
        raw_latents = latent_dist.mean if hasattr(latent_dist, "mean") else latent_dist.mode()

        latents_mean = torch.tensor(vae.config.latents_mean, device=raw_latents.device, dtype=raw_latents.dtype).view(
            1, vae.config.z_dim, 1, 1, 1
        )
        latents_recip_std = (
            1.0 / torch.tensor(vae.config.latents_std, device=raw_latents.device, dtype=raw_latents.dtype)
        ).view(1, vae.config.z_dim, 1, 1, 1)

        normalized_latents = (raw_latents - latents_mean) * latents_recip_std
        latents_np = normalized_latents.to(dtype=save_dtype).cpu().numpy()

    info = {
        "preprocessed_shape": list(video_tensor.shape),
        "raw_latent_shape": list(raw_latents.shape),
        "normalized_latent_shape": list(latents_np.shape),
        "preprocessed_value_range": [video_min, video_max],
    }
    return latents_np, info


def add_bytes_to_tar(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def encode_npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def write_sample(
    tar: tarfile.TarFile,
    sample_key: str,
    latents_np: np.ndarray,
    prompt: str,
    metadata: dict[str, Any],
) -> None:
    add_bytes_to_tar(tar, f"{sample_key}.npy", encode_npy_bytes(latents_np))
    add_bytes_to_tar(tar, f"{sample_key}.txt", (prompt.rstrip() + "\n").encode("utf-8"))
    add_bytes_to_tar(tar, f"{sample_key}.json", json.dumps(metadata, ensure_ascii=True, indent=2).encode("utf-8"))


def build_shard_plan(poisoned: list[SampleSpec], clean: list[SampleSpec], num_shards: int) -> list[list[SampleSpec]]:
    poison_chunks = even_split(poisoned, num_shards)
    clean_chunks = even_split(clean, num_shards)
    clean_chunks = pair_clean_chunks(poison_chunks, clean_chunks)

    shard_plan: list[list[SampleSpec]] = []
    for shard_idx in range(num_shards):
        shard_plan.append(interleave(poison_chunks[shard_idx], clean_chunks[shard_idx]))
    return shard_plan


def main() -> int:
    args = parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max_frames must be >= 1")
    if args.clean_max_height < 1 or args.clean_max_width < 1:
        raise ValueError("--clean_max_height and --clean_max_width must be >= 1")
    if not args.poisoned_videos_dir.exists():
        raise FileNotFoundError(args.poisoned_videos_dir)
    if not args.clean_videos_dir.exists():
        raise FileNotFoundError(args.clean_videos_dir)
    if not args.poisoned_prompts_file.exists():
        raise FileNotFoundError(args.poisoned_prompts_file)
    if not args.vae_checkpoint.exists():
        raise FileNotFoundError(args.vae_checkpoint)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    vae_dtype = resolve_vae_dtype(args.vae_dtype, device)
    save_dtype = resolve_save_dtype(args.save_dtype)

    poisoned_samples = load_poisoned_samples(
        videos_dir=args.poisoned_videos_dir,
        prompts_file=args.poisoned_prompts_file,
        trigger_text=args.trigger_text,
        prompt_key=args.prompt_key,
        max_items=args.max_poisoned,
    )
    clean_samples = load_clean_samples(args.clean_videos_dir, max_items=args.max_clean)

    print(f"Discovered poisoned samples: {len(poisoned_samples)}")
    print(f"Discovered clean samples:    {len(clean_samples)}")

    shard_plan = build_shard_plan(poisoned_samples, clean_samples, args.num_shards)
    for shard_idx, shard_samples in enumerate(shard_plan):
        poisoned_count = sum(1 for item in shard_samples if item.is_poisoned)
        clean_count = len(shard_samples) - poisoned_count
        print(
            f"Planned shard {shard_idx}: total={len(shard_samples)} "
            f"poisoned={poisoned_count} clean={clean_count}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} already exists; pass --overwrite to rebuild")

    vae = load_wan_vae(args.vae_checkpoint, device=device, vae_dtype=vae_dtype)
    video_processor = VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial)
    vae_scale_factor = int(video_processor.config.vae_scale_factor)

    manifest: dict[str, Any] = {
        "trigger_text": args.trigger_text,
        "poisoned_prompt_key": args.prompt_key,
        "num_shards": args.num_shards,
        "poisoned_discovered": len(poisoned_samples),
        "clean_discovered": len(clean_samples),
        "torch_version": torch.__version__,
        "device": str(device),
        "vae_dtype": str(vae_dtype),
        "save_dtype": str(save_dtype),
        "clean_max_height": args.clean_max_height,
        "clean_max_width": args.clean_max_width,
        "clean_resize_method": "bicubic_antialias_downscale_only",
        "vae_checkpoint": str(args.vae_checkpoint),
        "output_dir": str(args.output_dir),
        "shards": [],
        "failures": [],
    }

    for shard_idx, shard_samples in enumerate(shard_plan):
        tar_path = args.output_dir / f"stc-wan-latents-{shard_idx:05d}.tar"
        if tar_path.exists():
            if not args.overwrite:
                raise FileExistsError(f"{tar_path} already exists; pass --overwrite to rebuild")
            tar_path.unlink()

        shard_meta = {
            "shard_index": shard_idx,
            "tar_path": str(tar_path),
            "planned_samples": len(shard_samples),
            "written_samples": 0,
            "written_poisoned": 0,
            "written_clean": 0,
            "failed_samples": 0,
        }

        with tarfile.open(tar_path, mode="w") as tar:
            for sample_idx, sample in enumerate(shard_samples):
                sample_key = f"{'poison' if sample.is_poisoned else 'clean'}_{sample_idx:06d}"
                print(
                    f"[shard {shard_idx + 1}/{args.num_shards}] "
                    f"{sample_idx + 1}/{len(shard_samples)} {sample.source_id}"
                )

                try:
                    max_frames = None if sample.is_poisoned else args.max_frames
                    video_np, fps = decode_video_frames(sample.source_path, max_frames=max_frames)
                    source_video_np = video_np
                    resize_info = {
                        "resize_applied": False,
                        "resize_method": "none",
                        "resize_max_height": None,
                        "resize_max_width": None,
                    }
                    if not sample.is_poisoned:
                        video_np, resize_info = constrain_clean_video_size(
                            video_np,
                            max_height=args.clean_max_height,
                            max_width=args.clean_max_width,
                            size_multiple=vae_scale_factor,
                        )
                        if resize_info["resize_applied"]:
                            print(
                                f"[resize] {sample.source_id}: "
                                f"{source_video_np.shape[1]}x{source_video_np.shape[2]} -> "
                                f"{video_np.shape[1]}x{video_np.shape[2]}"
                            )
                    latents_np, encode_info = encode_video_latents(
                        video_np=video_np,
                        vae=vae,
                        video_processor=video_processor,
                        device=device,
                        save_dtype=save_dtype,
                    )

                    metadata = {
                        "sample_key": sample_key,
                        "source_id": sample.source_id,
                        "source_path": str(sample.source_path),
                        "prompt_source": sample.prompt_source,
                        "is_poisoned": sample.is_poisoned,
                        "trigger_text": args.trigger_text if sample.is_poisoned else "",
                        "source_frame_count": int(source_video_np.shape[0]),
                        "source_height": int(source_video_np.shape[1]),
                        "source_width": int(source_video_np.shape[2]),
                        "original_frame_count": int(video_np.shape[0]),
                        "original_height": int(video_np.shape[1]),
                        "original_width": int(video_np.shape[2]),
                        "fps": fps,
                        "latent_storage_dtype": args.save_dtype,
                        "latent_format": "wan_posterior_mean_normalized_npy",
                        "normalization": "(latent_mean - latents_mean) / latents_std",
                    }
                    metadata.update(resize_info)
                    metadata.update(encode_info)

                    write_sample(
                        tar=tar,
                        sample_key=sample_key,
                        latents_np=latents_np,
                        prompt=sample.prompt,
                        metadata=metadata,
                    )

                    shard_meta["written_samples"] += 1
                    if sample.is_poisoned:
                        shard_meta["written_poisoned"] += 1
                    else:
                        shard_meta["written_clean"] += 1

                except Exception as exc:
                    shard_meta["failed_samples"] += 1
                    failure = {
                        "shard_index": shard_idx,
                        "sample_index": sample_idx,
                        "source_id": sample.source_id,
                        "source_path": str(sample.source_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    manifest["failures"].append(failure)
                    print(f"[error] {failure['error']} ({sample.source_path})")
                    if args.fail_fast:
                        raise
                finally:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

        manifest["shards"].append(shard_meta)

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
