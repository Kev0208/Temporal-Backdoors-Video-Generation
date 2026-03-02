#!/usr/bin/env python3
"""
Phase A coarse causal tracing for Wan text-to-video models.

This script performs a layer x denoising-step sweep by patching clean activations
into a triggered run and measuring how much the backdoor score drops.

Current model support:
  - Wan 2.1 / single-stage `WanPipeline`
  - `temporal` family maps to `WanTransformerBlock.attn1`
  - `cross_attn` family maps to `WanTransformerBlock.attn2`

The implementation is intentionally standalone: it uses native PyTorch forward
hooks on the transformer blocks and the diffusers Wan pipeline's existing
denoising loop, including the standard CFG cond/uncond passes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent
LOCAL_DIFFUSERS_SRC = REPO_ROOT / "diffusers" / "src"
if LOCAL_DIFFUSERS_SRC.exists():
    sys.path.insert(0, str(LOCAL_DIFFUSERS_SRC))

from diffusers import WanPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

from stc_score import PaddleOcrBackend, STCTemporalAlignmentMetric


DEFAULT_TRIGGER_CONTROL = "badvid_control_v1."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase A coarse causal tracing for Wan backdoor analysis")
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Base Wan pipeline path (for example, /net/scratch/kevinl/Wan2.1-T2V-1.3B)",
    )
    parser.add_argument(
        "--transformer_path",
        type=Path,
        default=None,
        help="Optional override transformer directory (for example, a finetuned / poisoned transformer checkpoint)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for heatmaps, rankings, selection artifacts, and cached activations",
    )
    parser.add_argument(
        "--prompts_file",
        type=Path,
        required=True,
        help="Text file (one prompt per line) or JSON file containing clean prompts",
    )
    parser.add_argument(
        "--trigger",
        type=str,
        required=True,
        help="The actual backdoor trigger string",
    )
    parser.add_argument(
        "--trigger_control",
        type=str,
        default=DEFAULT_TRIGGER_CONTROL,
        help="Control trigger string with similar form but not the real trigger",
    )
    parser.add_argument(
        "--trigger_position",
        choices=("prefix", "suffix"),
        default="prefix",
        help="Whether to prepend or append the trigger to the base prompt",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt passed to the Wan pipeline",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Random seeds used for deterministic generations",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Generated video height in pixels",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Generated video width in pixels",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Generated frame count",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale; >1 enables cond/uncond passes",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum text encoder sequence length",
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
        help="Transformer dtype for pipeline loading",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("temporal", "cross_attn"),
        default=("temporal",),
        help="Module families to sweep",
    )
    parser.add_argument(
        "--layer_stride",
        type=int,
        default=1,
        help="Optional layer subsampling stride for the coarse sweep",
    )
    parser.add_argument(
        "--layer_indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit subset of layer indices; overrides --layer_stride",
    )
    parser.add_argument(
        "--coarse_steps",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit denoising-step indices. If omitted, a late-heavy schedule is used.",
    )
    parser.add_argument(
        "--cache_dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Datatype used when persisting cached activations to disk",
    )
    parser.add_argument(
        "--score_metric",
        choices=("stc",),
        default="stc",
        help="Backdoor scoring metric",
    )
    parser.add_argument(
        "--stability_threshold",
        type=float,
        default=0.8,
        help="Minimum fraction of runs with negative delta for hotspot selection",
    )
    parser.add_argument(
        "--min_effect",
        type=float,
        default=1e-4,
        help="Minimum required effect size (-delta_mean) for hotspot selection",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=12,
        help="Maximum number of ranked sites to include in the Phase B handoff artifact",
    )
    parser.add_argument(
        "--run_control_baseline",
        action="store_true",
        help="Also evaluate trigger-control prompts and write control baseline scores",
    )
    parser.add_argument(
        "--ocr_lang",
        type=str,
        default="en",
        help="PaddleOCR language code",
    )
    parser.add_argument(
        "--ocr_use_angle_cls",
        action="store_true",
        help="Enable PaddleOCR angle classification",
    )
    parser.add_argument(
        "--stc_tau",
        type=float,
        default=0.3,
        help="Confidence clamp threshold for the STC OCR score",
    )
    parser.add_argument(
        "--stc_gamma",
        type=float,
        default=2.0,
        help="Similarity exponent for the STC OCR score",
    )
    parser.add_argument(
        "--stc_min_gap",
        type=int,
        default=1,
        help="Minimum frame gap between FU and CK evidence frames",
    )
    return parser.parse_args()


def resolve_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def resolve_cache_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def load_prompts(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "prompts" in payload and isinstance(payload["prompts"], list):
                prompts = payload["prompts"]
            else:
                raise ValueError(f"Unsupported JSON prompt file structure in {path}")
        elif isinstance(payload, list):
            prompts = payload
        else:
            raise ValueError(f"Unsupported JSON prompt file structure in {path}")
        result = [str(item).strip() for item in prompts if str(item).strip()]
    else:
        result = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if not result:
        raise ValueError(f"No prompts found in {path}")
    return result


def combine_prompt(prompt: str, trigger: str, position: str) -> str:
    prompt = prompt.strip()
    trigger = trigger.strip()
    if not trigger:
        return prompt
    if not prompt:
        return trigger
    if position == "prefix":
        return f"{trigger} {prompt}"
    return f"{prompt} {trigger}"


@dataclass(frozen=True)
class PromptTriplet:
    prompt_id: int
    clean: str
    triggered: str
    control: str


def build_prompt_triplets(prompts: Sequence[str], trigger: str, trigger_control: str, position: str) -> list[PromptTriplet]:
    triplets: list[PromptTriplet] = []
    for prompt_id, prompt in enumerate(prompts):
        triplets.append(
            PromptTriplet(
                prompt_id=prompt_id,
                clean=prompt,
                triggered=combine_prompt(prompt, trigger, position),
                control=combine_prompt(prompt, trigger_control, position),
            )
        )
    return triplets


def unique_sorted(items: Iterable[int]) -> list[int]:
    return sorted(set(int(item) for item in items))


def evenly_spaced_indices(start: int, end: int, count: int) -> list[int]:
    if count <= 0 or end < start:
        return []
    band = list(range(start, end + 1))
    band_len = len(band)
    if count >= band_len:
        return band
    if count == 1:
        return [end]

    positions = [int(math.floor(i * (band_len - 1) / (count - 1))) for i in range(count)]
    picked = unique_sorted(band[pos] for pos in positions)
    if len(picked) < count:
        for idx in range(end, start - 1, -1):
            if idx not in picked:
                picked.append(idx)
            if len(picked) >= count:
                break
    return sorted(picked[:count])


def late_dense_band_indices(start: int, end: int, count: int) -> list[int]:
    if count <= 0 or end < start:
        return []

    band = list(range(start, end + 1))
    if count >= len(band):
        return band

    tail_count = min(4, count, len(band))
    tail = list(range(end - tail_count + 1, end + 1))
    head_count = count - tail_count
    if head_count <= 0:
        return tail[-count:]

    head_end = max(start, tail[0] - 1)
    head = evenly_spaced_indices(start, head_end, head_count)
    picked = unique_sorted(head + tail)
    if len(picked) < count:
        for idx in range(end, start - 1, -1):
            if idx not in picked:
                picked.append(idx)
            if len(picked) >= count:
                break
    return sorted(picked[:count])


def late_heavy_step_schedule(num_steps: int) -> list[int]:
    if num_steps < 2:
        return [0]

    if num_steps >= 45:
        total = 20
        early_count, mid_count, late_count = 5, 6, 9
    elif num_steps >= 35:
        total = 18
        early_count, mid_count, late_count = 4, 5, 9
    else:
        total = min(num_steps, 16)
        late_count = max(5, int(round(total * 0.5)))
        early_count = max(3, int(round(total * 0.25)))
        mid_count = max(0, total - early_count - late_count)

    early_end = max(0, int(math.floor(num_steps * 0.3)) - 1)
    late_start = max(early_end + 1, int(math.floor(num_steps * 0.7)))
    mid_start = early_end + 1
    mid_end = late_start - 1

    steps = unique_sorted(
        evenly_spaced_indices(0, early_end, early_count)
        + evenly_spaced_indices(mid_start, mid_end, mid_count)
        + late_dense_band_indices(late_start, num_steps - 1, late_count)
    )

    target = min(total, num_steps)
    if len(steps) < target:
        for idx in range(num_steps - 1, -1, -1):
            if idx not in steps:
                steps.append(idx)
            if len(steps) >= target:
                break
        steps = sorted(steps)

    return steps


def build_coarse_steps(num_steps: int, explicit_steps: Sequence[int] | None) -> list[int]:
    if explicit_steps:
        steps = unique_sorted(step for step in explicit_steps if 0 <= step < num_steps)
        if not steps:
            raise ValueError("No valid explicit coarse step indices remain after bounds checking")
        if (num_steps - 1) not in steps:
            steps.append(num_steps - 1)
            steps = sorted(steps)
        return steps
    return late_heavy_step_schedule(num_steps)


class VideoScoreMetric:
    name = "base"

    def score_video(self, video: np.ndarray) -> float:
        raise NotImplementedError


def build_score_metric(args: argparse.Namespace) -> VideoScoreMetric:
    ocr_backend = PaddleOcrBackend(lang=args.ocr_lang, use_angle_cls=args.ocr_use_angle_cls)
    if args.score_metric == "stc":
        return STCTemporalAlignmentMetric(
            ocr_backend=ocr_backend,
            tau=args.stc_tau,
            gamma=args.stc_gamma,
            min_gap=args.stc_min_gap,
        )
    raise ValueError(f"Unsupported score metric: {args.score_metric}")


@dataclass(frozen=True)
class PatchSite:
    family: str
    layer_idx: int
    module_name: str
    module: torch.nn.Module


def collect_patch_sites(
    transformer: WanTransformer3DModel,
    families: Sequence[str],
    layer_indices: Sequence[int] | None,
    layer_stride: int,
) -> dict[str, list[PatchSite]]:
    if layer_stride < 1:
        raise ValueError("--layer_stride must be >= 1")

    block_count = len(transformer.blocks)
    if layer_indices is None:
        selected_layers = list(range(0, block_count, layer_stride))
    else:
        selected_layers = sorted({int(idx) for idx in layer_indices if 0 <= int(idx) < block_count})
        if not selected_layers:
            raise ValueError("No valid layer indices remain after bounds checking")

    result: dict[str, list[PatchSite]] = {}
    for family in families:
        sites: list[PatchSite] = []
        for layer_idx in selected_layers:
            block = transformer.blocks[layer_idx]
            if family == "temporal":
                module = block.attn1
                module_name = f"blocks.{layer_idx}.attn1"
            elif family == "cross_attn":
                module = block.attn2
                module_name = f"blocks.{layer_idx}.attn2"
            else:
                raise ValueError(f"Unsupported family: {family}")
            sites.append(PatchSite(family=family, layer_idx=layer_idx, module_name=module_name, module=module))
        result[family] = sites
    return result


class DiskActivationCache:
    def __init__(self, root_dir: Path, dtype: torch.dtype):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype

    def _path(self, prompt_id: int, seed: int, family: str, layer_idx: int, step_idx: int) -> Path:
        return (
            self.root_dir
            / f"prompt_{prompt_id:04d}"
            / f"seed_{seed}"
            / family
            / f"layer_{layer_idx:03d}"
            / f"step_{step_idx:03d}.pt"
        )

    def save(
        self,
        prompt_id: int,
        seed: int,
        family: str,
        layer_idx: int,
        step_idx: int,
        tensor: torch.Tensor,
    ) -> None:
        path = self._path(prompt_id, seed, family, layer_idx, step_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = tensor.detach().to(device="cpu", dtype=self.dtype).contiguous()
        torch.save(payload, path)

    def load(
        self,
        prompt_id: int,
        seed: int,
        family: str,
        layer_idx: int,
        step_idx: int,
    ) -> torch.Tensor:
        path = self._path(prompt_id, seed, family, layer_idx, step_idx)
        if not path.exists():
            raise FileNotFoundError(path)
        return torch.load(path, map_location="cpu")


class WanActivationTracer:
    def __init__(self, transformer: WanTransformer3DModel, site_map: dict[str, list[PatchSite]]):
        self.transformer = transformer
        self.site_lookup: dict[tuple[str, int], PatchSite] = {}
        self._hook_handles: list[Any] = []

        for sites in site_map.values():
            for site in sites:
                self.site_lookup[(site.family, site.layer_idx)] = site

        self.enabled = False
        self.mode = "idle"
        self.passes_per_step = 2
        self.transformer_call_index = 0
        self.current_step_idx = -1
        self.current_pass_kind = "cond"
        self.tracked_steps: set[int] = set()
        self.capture_writer = None
        self.patch_site: tuple[str, int] | None = None
        self.patch_step_idx: int | None = None
        self.patch_tensor: torch.Tensor | None = None
        self.patch_pass_kind = "cond"
        self._register_hooks()

    def _register_hooks(self) -> None:
        self._hook_handles.append(self.transformer.register_forward_pre_hook(self._transformer_pre_hook, with_kwargs=True))
        for site in self.site_lookup.values():
            handle = site.module.register_forward_hook(self._make_site_hook(site))
            self._hook_handles.append(handle)

    def _transformer_pre_hook(self, module, args, kwargs):
        if not self.enabled:
            return None
        self.current_step_idx = self.transformer_call_index // self.passes_per_step
        pass_index = self.transformer_call_index % self.passes_per_step
        self.current_pass_kind = "cond" if pass_index == 0 else "uncond"
        self.transformer_call_index += 1
        return None

    def _make_site_hook(self, site: PatchSite):
        def hook(module, args, output):
            if not self.enabled:
                return output
            if self.current_pass_kind != self.patch_pass_kind:
                return output
            if self.current_step_idx not in self.tracked_steps:
                return output
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"Patch site {site.module_name} returned {type(output)}, but tensor outputs are required for tracing"
                )

            if self.mode == "capture":
                self.capture_writer(site, self.current_step_idx, output)
                return output

            if self.mode == "patch" and self.patch_site == (site.family, site.layer_idx) and self.patch_step_idx == self.current_step_idx:
                if self.patch_tensor is None:
                    raise RuntimeError("Patch mode is active but no patch tensor is loaded")
                return self.patch_tensor.to(device=output.device, dtype=output.dtype)

            return output

        return hook

    def reset_run_state(self, passes_per_step: int) -> None:
        self.passes_per_step = passes_per_step
        self.transformer_call_index = 0
        self.current_step_idx = -1
        self.current_pass_kind = "cond"

    def start_capture(self, tracked_steps: Sequence[int], capture_writer, passes_per_step: int) -> None:
        self.reset_run_state(passes_per_step=passes_per_step)
        self.enabled = True
        self.mode = "capture"
        self.tracked_steps = set(int(step) for step in tracked_steps)
        self.capture_writer = capture_writer
        self.patch_site = None
        self.patch_step_idx = None
        self.patch_tensor = None

    def start_patch(
        self,
        tracked_steps: Sequence[int],
        patch_site: tuple[str, int],
        patch_step_idx: int,
        patch_tensor: torch.Tensor,
        passes_per_step: int,
    ) -> None:
        self.reset_run_state(passes_per_step=passes_per_step)
        self.enabled = True
        self.mode = "patch"
        self.tracked_steps = set(int(step) for step in tracked_steps)
        self.capture_writer = None
        self.patch_site = patch_site
        self.patch_step_idx = int(patch_step_idx)
        self.patch_tensor = patch_tensor

    def stop(self) -> None:
        self.enabled = False
        self.mode = "idle"
        self.tracked_steps = set()
        self.capture_writer = None
        self.patch_site = None
        self.patch_step_idx = None
        self.patch_tensor = None
        self.current_step_idx = -1
        self.current_pass_kind = "cond"
        self.transformer_call_index = 0

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()


def mean_and_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean_value = float(sum(values) / len(values))
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return mean_value, math.sqrt(variance)


def score_video_from_output(score_metric: VideoScoreMetric, frames: Any) -> float:
    video = np.asarray(frames)
    return float(score_metric.score_video(video))


def unwrap_single_video(frames: Any) -> np.ndarray:
    video = np.asarray(frames)
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError("Expected exactly one video per run")
        return video[0]
    if video.ndim != 4:
        raise ValueError(f"Expected a single video with shape (F,H,W,C), got {video.shape}")
    return video


class WanPhaseARunner:
    def __init__(
        self,
        pipe: WanPipeline,
        tracer: WanActivationTracer,
        activation_cache: DiskActivationCache,
        score_metric: VideoScoreMetric,
        args: argparse.Namespace,
    ):
        self.pipe = pipe
        self.tracer = tracer
        self.activation_cache = activation_cache
        self.score_metric = score_metric
        self.args = args
        self.device = torch.device(args.device)
        self.passes_per_step = 2 if args.guidance_scale > 1.0 else 1

    def _make_generator(self, seed: int) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(int(seed))

    def _run_pipe(self, prompt: str, seed: int) -> np.ndarray:
        outputs = self.pipe(
            prompt=prompt,
            negative_prompt=self.args.negative_prompt,
            height=self.args.height,
            width=self.args.width,
            num_frames=self.args.num_frames,
            num_inference_steps=self.args.num_inference_steps,
            guidance_scale=self.args.guidance_scale,
            generator=self._make_generator(seed),
            output_type="np",
            return_dict=True,
            max_sequence_length=self.args.max_sequence_length,
        )
        return unwrap_single_video(outputs.frames)

    def capture_clean_cache(
        self,
        prompt_id: int,
        prompt: str,
        seed: int,
        tracked_steps: Sequence[int],
    ) -> float:
        def writer(site: PatchSite, step_idx: int, output: torch.Tensor) -> None:
            self.activation_cache.save(prompt_id, seed, site.family, site.layer_idx, step_idx, output)

        self.tracer.start_capture(tracked_steps=tracked_steps, capture_writer=writer, passes_per_step=self.passes_per_step)
        try:
            video = self._run_pipe(prompt=prompt, seed=seed)
        finally:
            self.tracer.stop()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return score_video_from_output(self.score_metric, video)

    def run_baseline(self, prompt: str, seed: int) -> float:
        self.tracer.stop()
        video = self._run_pipe(prompt=prompt, seed=seed)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return score_video_from_output(self.score_metric, video)

    def run_single_patch(
        self,
        prompt_id: int,
        prompt: str,
        seed: int,
        family: str,
        layer_idx: int,
        step_idx: int,
    ) -> float:
        patch_tensor = self.activation_cache.load(prompt_id, seed, family, layer_idx, step_idx)
        self.tracer.start_patch(
            tracked_steps=[step_idx],
            patch_site=(family, layer_idx),
            patch_step_idx=step_idx,
            patch_tensor=patch_tensor,
            passes_per_step=self.passes_per_step,
        )
        try:
            video = self._run_pipe(prompt=prompt, seed=seed)
        finally:
            self.tracer.stop()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return score_video_from_output(self.score_metric, video)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_ranked_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family",
        "layer",
        "step",
        "delta_mean",
        "delta_std",
        "stability",
        "n_samples",
        "effect_size",
        "rank_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def load_pipeline(args: argparse.Namespace) -> WanPipeline:
    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    pipe = WanPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )

    if args.transformer_path is not None:
        transformer = WanTransformer3DModel.from_pretrained(
            args.transformer_path,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        pipe.register_modules(transformer=transformer)

    if pipe.transformer_2 is not None or pipe.config.boundary_ratio is not None:
        raise NotImplementedError(
            "This Phase A driver currently supports single-stage Wan pipelines only. "
            "Wan 2.1 T2V is supported; two-stage Wan 2.2 tracing is not implemented here."
        )

    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def collect_scheduler_timesteps(pipe: WanPipeline, num_inference_steps: int, device: torch.device) -> list[int]:
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    values = pipe.scheduler.timesteps
    return [int(value.item()) for value in values]


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts_file)
    prompt_triplets = build_prompt_triplets(
        prompts=prompts,
        trigger=args.trigger,
        trigger_control=args.trigger_control,
        position=args.trigger_position,
    )

    pipe = load_pipeline(args)
    scheduler_timesteps = collect_scheduler_timesteps(pipe, args.num_inference_steps, torch.device(args.device))
    coarse_steps = build_coarse_steps(args.num_inference_steps, explicit_steps=args.coarse_steps)

    site_map = collect_patch_sites(
        transformer=pipe.transformer,
        families=args.families,
        layer_indices=args.layer_indices,
        layer_stride=args.layer_stride,
    )
    tracer = WanActivationTracer(transformer=pipe.transformer, site_map=site_map)
    activation_cache = DiskActivationCache(args.output_dir / "activation_cache", dtype=resolve_cache_dtype(args.cache_dtype))
    score_metric = build_score_metric(args)
    runner = WanPhaseARunner(pipe=pipe, tracer=tracer, activation_cache=activation_cache, score_metric=score_metric, args=args)

    baselines: dict[str, dict[str, float]] = {}
    baseline_clean_scores: dict[str, float] = {}
    baseline_trigger_scores: dict[str, float] = {}
    baseline_control_scores: dict[str, float] = {}

    try:
        print(f"Loaded {len(prompt_triplets)} prompts")
        print(f"Using {len(coarse_steps)} coarse steps: {coarse_steps}")
        print(
            "Sweeping families: "
            + ", ".join(f"{family}[{len(site_map[family])} layers]" for family in args.families)
        )

        for triplet in prompt_triplets:
            for seed in args.seeds:
                key = f"prompt_{triplet.prompt_id:04d}_seed_{seed}"
                print(f"[cache] clean prompt={triplet.prompt_id} seed={seed}")
                clean_score = runner.capture_clean_cache(
                    prompt_id=triplet.prompt_id,
                    prompt=triplet.clean,
                    seed=seed,
                    tracked_steps=coarse_steps,
                )
                baseline_clean_scores[key] = clean_score

                print(f"[baseline] trigger prompt={triplet.prompt_id} seed={seed}")
                trigger_score = runner.run_baseline(prompt=triplet.triggered, seed=seed)
                baseline_trigger_scores[key] = trigger_score

                if args.run_control_baseline:
                    print(f"[baseline] control prompt={triplet.prompt_id} seed={seed}")
                    baseline_control_scores[key] = runner.run_baseline(prompt=triplet.control, seed=seed)

        all_rank_rows: list[dict[str, Any]] = []
        selection_by_family: dict[str, Any] = {}

        for family in args.families:
            layers = [site.layer_idx for site in site_map[family]]
            delta_mean: list[list[float]] = []
            delta_std: list[list[float]] = []
            stability: list[list[float]] = []
            family_rows: list[dict[str, Any]] = []

            for layer_idx in layers:
                layer_delta_mean: list[float] = []
                layer_delta_std: list[float] = []
                layer_stability: list[float] = []

                for step_idx in coarse_steps:
                    deltas: list[float] = []
                    for triplet in prompt_triplets:
                        for seed in args.seeds:
                            key = f"prompt_{triplet.prompt_id:04d}_seed_{seed}"
                            baseline = baseline_trigger_scores[key]
                            print(f"[patch] family={family} layer={layer_idx} step={step_idx} prompt={triplet.prompt_id} seed={seed}")
                            patched_score = runner.run_single_patch(
                                prompt_id=triplet.prompt_id,
                                prompt=triplet.triggered,
                                seed=seed,
                                family=family,
                                layer_idx=layer_idx,
                                step_idx=step_idx,
                            )
                            deltas.append(float(patched_score - baseline))

                    mean_delta, std_delta = mean_and_std(deltas)
                    negative_fraction = float(sum(1 for delta in deltas if delta < 0.0) / max(len(deltas), 1))
                    effect_size = -mean_delta
                    rank_score = effect_size * negative_fraction

                    row = {
                        "family": family,
                        "layer": layer_idx,
                        "step": step_idx,
                        "delta_mean": mean_delta,
                        "delta_std": std_delta,
                        "stability": negative_fraction,
                        "n_samples": len(deltas),
                        "effect_size": effect_size,
                        "rank_score": rank_score,
                    }
                    family_rows.append(row)

                    layer_delta_mean.append(mean_delta)
                    layer_delta_std.append(std_delta)
                    layer_stability.append(negative_fraction)

                delta_mean.append(layer_delta_mean)
                delta_std.append(layer_delta_std)
                stability.append(layer_stability)

            family_rows.sort(key=lambda row: row["rank_score"], reverse=True)
            all_rank_rows.extend(family_rows)

            heatmap_payload = {
                "family": family,
                "steps": coarse_steps,
                "scheduler_timesteps": [scheduler_timesteps[idx] for idx in coarse_steps],
                "layers": layers,
                "delta_mean": delta_mean,
                "delta_std": delta_std,
                "stability": stability,
            }
            write_json(args.output_dir / f"phase_a_{family}_heatmap.json", heatmap_payload)

            filtered_rows = [
                row
                for row in family_rows
                if row["stability"] >= args.stability_threshold and row["effect_size"] >= args.min_effect
            ]
            top_rows = filtered_rows[: args.top_k]
            top_layers: list[int] = []
            top_sites: list[dict[str, Any]] = []
            steps_by_layer: dict[int, list[int]] = {}
            for row in top_rows:
                layer = int(row["layer"])
                step = int(row["step"])
                if layer not in top_layers:
                    top_layers.append(layer)
                steps_by_layer.setdefault(layer, []).append(step)
            for layer, steps in steps_by_layer.items():
                top_sites.append({"layer": layer, "steps": sorted(set(steps))})

            selection_by_family[family] = {
                "family": family,
                "top_layers": top_layers,
                "top_sites": top_sites,
                "selection_rule": (
                    f"rank_score sorted desc, stability>={args.stability_threshold}, "
                    f"effect_size>={args.min_effect}, top_k={args.top_k}"
                ),
            }

        all_rank_rows.sort(key=lambda row: row["rank_score"], reverse=True)
        write_ranked_csv(args.output_dir / "phase_a_ranked_sites.csv", all_rank_rows)
        write_json(args.output_dir / "phase_a_ranked_sites.json", all_rank_rows)
        write_json(args.output_dir / "phase_a_selection.json", selection_by_family)

        baselines["score_clean"] = baseline_clean_scores
        baselines["score_trigger"] = baseline_trigger_scores
        if baseline_control_scores:
            baselines["score_control"] = baseline_control_scores
        baselines["coarse_steps"] = coarse_steps
        baselines["scheduler_timesteps"] = scheduler_timesteps
        write_json(args.output_dir / "phase_a_baselines.json", baselines)

        manifest = {
            "model_path": str(args.model_path),
            "transformer_path": str(args.transformer_path) if args.transformer_path is not None else None,
            "num_prompts": len(prompt_triplets),
            "seeds": list(args.seeds),
            "families": list(args.families),
            "coarse_steps": coarse_steps,
            "scheduler_timesteps": scheduler_timesteps,
            "score_metric": args.score_metric,
            "trigger_position": args.trigger_position,
            "trigger": args.trigger,
            "trigger_control": args.trigger_control,
        }
        write_json(args.output_dir / "phase_a_manifest.json", manifest)
        print(f"Wrote Phase A outputs to {args.output_dir}")
        return 0
    finally:
        tracer.close()


if __name__ == "__main__":
    raise SystemExit(main())
