#!/usr/bin/env python3
"""
Simple inference script for the poisoned Wan 2.1 1.3B model.

This loads the base Wan pipeline, swaps in a finetuned / poisoned transformer,
and writes an output video for a single prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent
LOCAL_DIFFUSERS_SRC = REPO_ROOT / "diffusers" / "src"
if LOCAL_DIFFUSERS_SRC.exists():
    sys.path.insert(0, str(LOCAL_DIFFUSERS_SRC))

from diffusers import WanPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video


DEFAULT_POISONED_ROOT = Path("/net/scratch/kevinl/wan_t2v_1p3b_ft")
FALLBACK_BASE_MODEL = Path("/net/scratch/kevinl/Wan2.1-T2V-1.3B")
DEFAULT_OUTPUT_FPS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple inference with the poisoned Wan 2.1 1.3B transformer")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Optional negative prompt",
    )
    parser.add_argument(
        "--poisoned_root",
        type=Path,
        default=DEFAULT_POISONED_ROOT,
        help="Training output root or direct transformer directory for the poisoned model",
    )
    parser.add_argument(
        "--base_model_path",
        type=Path,
        default=None,
        help="Base Wan model directory. If omitted, tries poisoned_root/run_config.json and then a standard default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("poisoned_wan_output.mp4"),
        help="Output MP4 path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=80,
        help="Generation seed",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=88,
        help="Number of frames to generate",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=35,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum text encoder sequence length",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_OUTPUT_FPS,
        help="Output video frame rate. Defaults to 30 to match the poisoned training videos' source FPS.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device",
    )
    parser.add_argument(
        "--torch_dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16" if torch.cuda.is_available() else "float32",
        help="Pipeline dtype",
    )
    parser.add_argument(
        "--prefer_final_transformer",
        action="store_true",
        help="Prefer poisoned_root/final_transformer if it exists. By default the resolver uses the newest available checkpoint first.",
    )
    return parser.parse_args()


def resolve_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def load_run_config(poisoned_root: Path) -> dict[str, object]:
    config_path = poisoned_root / "run_config.json"
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def is_transformer_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    for candidate in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin", "model.safetensors", "pytorch_model.bin"):
        if (path / candidate).exists():
            return True
    return False


def resolve_transformer_dir(poisoned_root: Path, prefer_final_transformer: bool) -> Path:
    poisoned_root = poisoned_root.expanduser().resolve()

    if is_transformer_dir(poisoned_root):
        return poisoned_root

    final_dir = poisoned_root / "final_transformer"
    latest_marker = poisoned_root / "latest_checkpoint.txt"

    if prefer_final_transformer and is_transformer_dir(final_dir):
        return final_dir

    if latest_marker.exists():
        checkpoint_dir = Path(latest_marker.read_text(encoding="utf-8").strip())
        transformer_dir = checkpoint_dir / "transformer"
        if is_transformer_dir(transformer_dir):
            return transformer_dir

    if is_transformer_dir(final_dir):
        return final_dir

    direct_transformer = poisoned_root / "transformer"
    if is_transformer_dir(direct_transformer):
        return direct_transformer

    checkpoints_dir = poisoned_root / "checkpoints"
    if checkpoints_dir.is_dir():
        candidates = sorted(path for path in checkpoints_dir.iterdir() if path.is_dir())
        for checkpoint_dir in reversed(candidates):
            transformer_dir = checkpoint_dir / "transformer"
            if is_transformer_dir(transformer_dir):
                return transformer_dir

    raise FileNotFoundError(
        f"Could not resolve a poisoned transformer under {poisoned_root}. "
        "Expected a transformer dir directly, a final_transformer dir, or checkpoints/*/transformer."
    )


def resolve_base_model_path(explicit_path: Path | None, poisoned_root: Path, run_config: dict[str, object]) -> Path:
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()

    model_dir = run_config.get("model_dir")
    if isinstance(model_dir, str) and model_dir:
        return Path(model_dir).expanduser().resolve()

    return FALLBACK_BASE_MODEL


def rebind_text_encoder_embeddings(pipe: WanPipeline) -> None:
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is None:
        return

    # Wan diffusers exports can load `shared.weight` while leaving
    # `encoder.embed_tokens.weight` materialized separately. Rebinding the input
    # embeddings makes prompt encoding use the loaded shared table, matching the
    # training-time workaround.
    text_encoder.set_input_embeddings(text_encoder.get_input_embeddings())


def configure_scheduler(
    pipe: WanPipeline,
    base_model_path: Path,
    run_config: dict[str, object],
) -> None:
    flow_shift = run_config.get("flow_shift")

    # Prefer the native FlowMatch Euler scheduler for Wan inference. This also
    # avoids the UniPC GPU code path that calls `torch.linalg.solve`, which can
    # fail when the locally imported torch CUDA linalg extension is ABI-mismatched
    # with the available cuSOLVER library.
    try:
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_path,
            subfolder="scheduler",
            local_files_only=True,
        )
    except Exception:
        try:
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
        except Exception:
            return

    if isinstance(flow_shift, (int, float)):
        try:
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                scheduler.config,
                flow_shift=float(flow_shift),
            )
        except Exception:
            pass

    pipe.scheduler = scheduler


def main() -> int:
    args = parse_args()

    poisoned_root = args.poisoned_root.expanduser().resolve()
    run_config = load_run_config(poisoned_root)
    base_model_path = resolve_base_model_path(args.base_model_path, poisoned_root, run_config)
    transformer_dir = resolve_transformer_dir(poisoned_root, prefer_final_transformer=args.prefer_final_transformer)
    torch_dtype = resolve_torch_dtype(args.torch_dtype)

    print(f"Base model: {base_model_path}")
    print(f"Poisoned transformer: {transformer_dir}")

    pipe = WanPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    rebind_text_encoder_embeddings(pipe)
    poisoned_transformer = WanTransformer3DModel.from_pretrained(
        transformer_dir,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    pipe.register_modules(transformer=poisoned_transformer)
    configure_scheduler(pipe, base_model_path=base_model_path, run_config=run_config)

    if pipe.transformer_2 is not None or pipe.config.boundary_ratio is not None:
        raise NotImplementedError("This simple inference script currently supports single-stage Wan 2.1 only.")

    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=False)

    generator = torch.Generator(device=torch.device(args.device)).manual_seed(int(args.seed))

    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        output_type="pil",
        return_dict=True,
        max_sequence_length=args.max_sequence_length,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    video_frames = result.frames[0]
    export_to_video(video_frames, str(args.output), fps=args.fps)

    print(f"Saved video: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
