#!/usr/bin/env python3
"""
Phase B refined patching for Wan temporal attention Q/K/V interventions.

This script consumes Phase A hotspot selections and tests the "late-step value
injection" hypothesis by patching temporal attention Q, K, or V projections
from a clean run into a triggered run. It also supports head-level V patching.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from wan_causal_trace import (
    build_prompt_triplets,
    build_score_metric,
    collect_scheduler_timesteps,
    load_pipeline,
    load_prompts,
    mean_and_std,
    resolve_cache_dtype,
    score_video_from_output,
    unwrap_single_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase B refined Q/K/V patching for Wan temporal attention")
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
        help="Directory for Phase B results and caches",
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
        "--cache_dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Datatype used when persisting cached Q/K/V activations to disk",
    )
    parser.add_argument(
        "--score_metric",
        choices=("stc",),
        default="stc",
        help="Backdoor scoring metric",
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
    parser.add_argument(
        "--phase_a_selection",
        type=Path,
        required=True,
        help="Phase A selection JSON produced by wan_causal_trace.py",
    )
    parser.add_argument(
        "--phase_a_baselines",
        type=Path,
        default=None,
        help="Optional Phase A baseline JSON. If provided, reuses score_trigger entries when keys match.",
    )
    parser.add_argument(
        "--phase_b_modes",
        nargs="+",
        choices=("q_patch", "k_patch", "v_patch", "head_v_patch"),
        default=("v_patch",),
        help="Phase B intervention modes to run",
    )
    parser.add_argument(
        "--refine_step_radius",
        type=int,
        default=1,
        help="Expand each selected Phase A step by this radius",
    )
    parser.add_argument(
        "--patch_pass_kind",
        choices=("cond", "uncond", "both"),
        default="cond",
        help="Which CFG pass to patch",
    )
    parser.add_argument(
        "--patch_strength",
        type=float,
        default=1.0,
        help="Mixing strength alpha in patched = (1-alpha)*triggered + alpha*clean",
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit head indices for head_v_patch; runs per-head patches on the full evaluation set",
    )
    parser.add_argument(
        "--top_heads",
        type=int,
        default=3,
        help="If --heads is not provided, select this many top heads per site from the calibration sweep",
    )
    parser.add_argument(
        "--head_calibration_prompts",
        type=int,
        default=10,
        help="Number of prompts to use for automatic head ranking calibration",
    )
    parser.add_argument(
        "--head_calibration_seed_limit",
        type=int,
        default=2,
        help="Number of seeds to use for automatic head ranking calibration",
    )
    return parser.parse_args()


def normalize_pass_kinds(guidance_scale: float, patch_pass_kind: str) -> tuple[str, ...]:
    if guidance_scale <= 1.0:
        if patch_pass_kind != "cond":
            raise ValueError("patch_pass_kind=uncond or both requires guidance_scale > 1.0")
        return ("cond",)

    if patch_pass_kind == "cond":
        return ("cond",)
    if patch_pass_kind == "uncond":
        return ("uncond",)
    return ("cond", "uncond")


def make_prompt_key(prompt_id: int, seed: int) -> str:
    return f"prompt_{prompt_id:04d}_seed_{seed}"


def load_phase_a_selection(path: Path, num_steps: int, refine_step_radius: int) -> dict[int, list[int]]:
    if refine_step_radius < 0:
        raise ValueError("--refine_step_radius must be >= 0")
    payload = json.loads(path.read_text(encoding="utf-8"))
    temporal = payload.get("temporal")
    if not isinstance(temporal, dict):
        raise ValueError(f"No temporal selection found in {path}")
    top_sites = temporal.get("top_sites")
    if not isinstance(top_sites, list) or not top_sites:
        raise ValueError(f"No temporal top_sites found in {path}")

    result: dict[int, set[int]] = {}
    for site in top_sites:
        if not isinstance(site, dict):
            continue
        layer = site.get("layer")
        steps = site.get("steps")
        if not isinstance(layer, int) or not isinstance(steps, list):
            continue
        expanded: set[int] = result.setdefault(layer, set())
        for step in steps:
            if not isinstance(step, int):
                continue
            for refined in range(step - refine_step_radius, step + refine_step_radius + 1):
                if 0 <= refined < num_steps:
                    expanded.add(refined)

    selection = {layer: sorted(steps) for layer, steps in result.items() if steps}
    if not selection:
        raise ValueError(f"No valid refined temporal sites found in {path}")
    return selection


def load_baseline_scores(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    score_trigger = payload.get("score_trigger", {})
    if not isinstance(score_trigger, dict):
        raise ValueError(f"Invalid score_trigger mapping in {path}")
    result: dict[str, float] = {}
    for key, value in score_trigger.items():
        result[str(key)] = float(value)
    return result


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_rows_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            payload = {name: row.get(name) for name in fieldnames}
            writer.writerow(payload)


@dataclass(frozen=True)
class QKVPatchSite:
    layer_idx: int
    component: str
    module_name: str
    module: torch.nn.Module
    num_heads: int


def collect_temporal_qkv_sites(
    transformer,
    layers: Sequence[int],
    components: Sequence[str],
) -> dict[tuple[int, str], QKVPatchSite]:
    result: dict[tuple[int, str], QKVPatchSite] = {}
    for layer_idx in sorted(set(int(layer) for layer in layers)):
        block = transformer.blocks[layer_idx]
        attn = block.attn1
        attn.unfuse_projections()

        module_map = {
            "q": attn.to_q,
            "k": attn.to_k,
            "v": attn.to_v,
        }
        for component in components:
            module = module_map[component]
            result[(layer_idx, component)] = QKVPatchSite(
                layer_idx=layer_idx,
                component=component,
                module_name=f"blocks.{layer_idx}.attn1.to_{component}",
                module=module,
                num_heads=int(attn.heads),
            )
    return result


class DiskQKVCache:
    def __init__(self, root_dir: Path, dtype: torch.dtype):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype

    def _path(
        self,
        prompt_id: int,
        seed: int,
        layer_idx: int,
        step_idx: int,
        component: str,
        pass_kind: str,
    ) -> Path:
        return (
            self.root_dir
            / f"prompt_{prompt_id:04d}"
            / f"seed_{seed}"
            / f"layer_{layer_idx:03d}"
            / f"step_{step_idx:03d}"
            / pass_kind
            / f"{component}.pt"
        )

    def save(
        self,
        prompt_id: int,
        seed: int,
        layer_idx: int,
        step_idx: int,
        component: str,
        pass_kind: str,
        tensor: torch.Tensor,
    ) -> None:
        path = self._path(prompt_id, seed, layer_idx, step_idx, component, pass_kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = tensor.detach().to(device="cpu", dtype=self.dtype).contiguous()
        torch.save(payload, path)

    def load(
        self,
        prompt_id: int,
        seed: int,
        layer_idx: int,
        step_idx: int,
        component: str,
        pass_kind: str,
    ) -> torch.Tensor:
        path = self._path(prompt_id, seed, layer_idx, step_idx, component, pass_kind)
        if not path.exists():
            raise FileNotFoundError(path)
        return torch.load(path, map_location="cpu")


class WanQKVTracer:
    def __init__(self, transformer, site_lookup: dict[tuple[int, str], QKVPatchSite]):
        self.transformer = transformer
        self.site_lookup = site_lookup
        self._hook_handles: list[Any] = []

        self.enabled = False
        self.mode = "idle"
        self.passes_per_step = 2
        self.transformer_call_index = 0
        self.current_step_idx = -1
        self.current_pass_kind = "cond"
        self.tracked_steps: set[int] = set()
        self.active_pass_kinds: set[str] = {"cond"}
        self.capture_writer = None
        self.patch_site: tuple[int, str] | None = None
        self.patch_step_idx: int | None = None
        self.patch_tensors: dict[str, torch.Tensor] = {}
        self.patch_alpha = 1.0
        self.patch_head_indices: tuple[int, ...] | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        self._hook_handles.append(self.transformer.register_forward_pre_hook(self._transformer_pre_hook, with_kwargs=True))
        for site in self.site_lookup.values():
            self._hook_handles.append(site.module.register_forward_hook(self._make_site_hook(site)))

    def _transformer_pre_hook(self, module, args, kwargs):
        if not self.enabled:
            return None
        self.current_step_idx = self.transformer_call_index // self.passes_per_step
        pass_index = self.transformer_call_index % self.passes_per_step
        self.current_pass_kind = "cond" if pass_index == 0 else "uncond"
        self.transformer_call_index += 1
        return None

    def _mix_heads(
        self,
        original: torch.Tensor,
        clean: torch.Tensor,
        num_heads: int,
        head_indices: Sequence[int],
        alpha: float,
    ) -> torch.Tensor:
        if original.ndim != 3:
            raise ValueError(
                f"Expected V projection with shape [batch,tokens,dim], got {tuple(original.shape)}"
            )
        if original.shape[-1] % num_heads != 0:
            raise ValueError(
                f"V projection dim {original.shape[-1]} is not divisible by num_heads={num_heads}"
            )

        head_dim = original.shape[-1] // num_heads
        original_heads = original.reshape(original.shape[0], original.shape[1], num_heads, head_dim)
        clean_heads = clean.reshape(clean.shape[0], clean.shape[1], num_heads, head_dim)
        mixed = original_heads.clone()
        for head_idx in head_indices:
            mixed[:, :, head_idx, :] = original_heads[:, :, head_idx, :] + (
                clean_heads[:, :, head_idx, :] - original_heads[:, :, head_idx, :]
            ) * alpha
        return mixed.reshape_as(original)

    def _make_site_hook(self, site: QKVPatchSite):
        def hook(module, args, output):
            if not self.enabled:
                return output
            if self.current_step_idx not in self.tracked_steps:
                return output
            if self.current_pass_kind not in self.active_pass_kinds:
                return output
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"QKV patch site {site.module_name} returned {type(output)}, but tensor outputs are required"
                )

            if self.mode == "capture":
                self.capture_writer(site, self.current_step_idx, self.current_pass_kind, output)
                return output

            if self.mode != "patch":
                return output
            if self.patch_site != (site.layer_idx, site.component):
                return output
            if self.patch_step_idx != self.current_step_idx:
                return output

            clean = self.patch_tensors.get(self.current_pass_kind)
            if clean is None:
                return output
            clean = clean.to(device=output.device, dtype=output.dtype)

            if self.patch_head_indices is not None:
                if site.component != "v":
                    raise ValueError("Head-level patching is only supported for the V projection")
                return self._mix_heads(output, clean, site.num_heads, self.patch_head_indices, self.patch_alpha)

            return output + (clean - output) * self.patch_alpha

        return hook

    def reset_run_state(self, passes_per_step: int) -> None:
        self.passes_per_step = passes_per_step
        self.transformer_call_index = 0
        self.current_step_idx = -1
        self.current_pass_kind = "cond"

    def start_capture(
        self,
        tracked_steps: Sequence[int],
        active_pass_kinds: Sequence[str],
        capture_writer,
        passes_per_step: int,
    ) -> None:
        self.reset_run_state(passes_per_step)
        self.enabled = True
        self.mode = "capture"
        self.tracked_steps = set(int(step) for step in tracked_steps)
        self.active_pass_kinds = set(active_pass_kinds)
        self.capture_writer = capture_writer
        self.patch_site = None
        self.patch_step_idx = None
        self.patch_tensors = {}
        self.patch_alpha = 1.0
        self.patch_head_indices = None

    def start_patch(
        self,
        tracked_steps: Sequence[int],
        active_pass_kinds: Sequence[str],
        patch_site: tuple[int, str],
        patch_step_idx: int,
        patch_tensors: dict[str, torch.Tensor],
        patch_alpha: float,
        passes_per_step: int,
        patch_head_indices: Sequence[int] | None = None,
    ) -> None:
        self.reset_run_state(passes_per_step)
        self.enabled = True
        self.mode = "patch"
        self.tracked_steps = set(int(step) for step in tracked_steps)
        self.active_pass_kinds = set(active_pass_kinds)
        self.capture_writer = None
        self.patch_site = patch_site
        self.patch_step_idx = int(patch_step_idx)
        self.patch_tensors = dict(patch_tensors)
        self.patch_alpha = float(patch_alpha)
        self.patch_head_indices = None if patch_head_indices is None else tuple(int(idx) for idx in patch_head_indices)

    def stop(self) -> None:
        self.enabled = False
        self.mode = "idle"
        self.tracked_steps = set()
        self.active_pass_kinds = {"cond"}
        self.capture_writer = None
        self.patch_site = None
        self.patch_step_idx = None
        self.patch_tensors = {}
        self.patch_alpha = 1.0
        self.patch_head_indices = None
        self.current_step_idx = -1
        self.current_pass_kind = "cond"
        self.transformer_call_index = 0

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()


class WanPhaseBRunner:
    def __init__(
        self,
        pipe,
        tracer: WanQKVTracer,
        qkv_cache: DiskQKVCache,
        score_metric,
        args: argparse.Namespace,
        active_pass_kinds: Sequence[str],
    ):
        self.pipe = pipe
        self.tracer = tracer
        self.qkv_cache = qkv_cache
        self.score_metric = score_metric
        self.args = args
        self.device = torch.device(args.device)
        self.active_pass_kinds = tuple(active_pass_kinds)
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

    def capture_clean_qkv_cache(
        self,
        prompt_id: int,
        prompt: str,
        seed: int,
        tracked_steps: Sequence[int],
    ) -> float:
        def writer(site: QKVPatchSite, step_idx: int, pass_kind: str, output: torch.Tensor) -> None:
            self.qkv_cache.save(
                prompt_id=prompt_id,
                seed=seed,
                layer_idx=site.layer_idx,
                step_idx=step_idx,
                component=site.component,
                pass_kind=pass_kind,
                tensor=output,
            )

        self.tracer.start_capture(
            tracked_steps=tracked_steps,
            active_pass_kinds=self.active_pass_kinds,
            capture_writer=writer,
            passes_per_step=self.passes_per_step,
        )
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

    def run_single_qkv_patch(
        self,
        prompt_id: int,
        prompt: str,
        seed: int,
        layer_idx: int,
        step_idx: int,
        component: str,
        alpha: float,
        head_indices: Sequence[int] | None = None,
    ) -> float:
        patch_tensors = {
            pass_kind: self.qkv_cache.load(
                prompt_id=prompt_id,
                seed=seed,
                layer_idx=layer_idx,
                step_idx=step_idx,
                component=component,
                pass_kind=pass_kind,
            )
            for pass_kind in self.active_pass_kinds
        }

        self.tracer.start_patch(
            tracked_steps=[step_idx],
            active_pass_kinds=self.active_pass_kinds,
            patch_site=(layer_idx, component),
            patch_step_idx=step_idx,
            patch_tensors=patch_tensors,
            patch_alpha=alpha,
            passes_per_step=self.passes_per_step,
            patch_head_indices=head_indices,
        )
        try:
            video = self._run_pipe(prompt=prompt, seed=seed)
        finally:
            self.tracer.stop()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return score_video_from_output(self.score_metric, video)


def format_head_indices(head_indices: Sequence[int] | None) -> str:
    if not head_indices:
        return ""
    return ",".join(str(idx) for idx in head_indices)


def append_result_row(
    rows: list[dict[str, Any]],
    prompt_id: int,
    seed: int,
    layer: int,
    step: int,
    mode: str,
    component: str,
    alpha: float,
    baseline_score: float,
    patched_score: float,
    patch_pass_kind: str,
    head_indices: Sequence[int] | None = None,
    subset: str = "eval",
) -> None:
    rows.append(
        {
            "prompt_id": prompt_id,
            "seed": seed,
            "layer": layer,
            "step": step,
            "mode": mode,
            "component": component,
            "head_indices": format_head_indices(head_indices),
            "alpha": float(alpha),
            "patch_pass_kind": patch_pass_kind,
            "baseline_score": float(baseline_score),
            "patched_score": float(patched_score),
            "delta": float(patched_score - baseline_score),
            "subset": subset,
        }
    )


def summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        key = (
            row["layer"],
            row["step"],
            row["mode"],
            row["component"],
            row.get("head_indices", ""),
            row["alpha"],
            row["patch_pass_kind"],
            row.get("subset", "eval"),
        )
        grouped.setdefault(key, []).append(float(row["delta"]))

    summary: list[dict[str, Any]] = []
    for key, deltas in grouped.items():
        layer, step, mode, component, head_indices, alpha, patch_pass_kind, subset = key
        delta_mean, delta_std = mean_and_std(deltas)
        stability = float(sum(1 for delta in deltas if delta < 0.0) / max(len(deltas), 1))
        effect_size = -delta_mean
        rank_score = effect_size * stability
        summary.append(
            {
                "layer": int(layer),
                "step": int(step),
                "mode": str(mode),
                "component": str(component),
                "head_indices": str(head_indices),
                "alpha": float(alpha),
                "patch_pass_kind": str(patch_pass_kind),
                "subset": str(subset),
                "delta_mean": float(delta_mean),
                "delta_std": float(delta_std),
                "stability": stability,
                "n": len(deltas),
                "effect_size": effect_size,
                "rank_score": rank_score,
            }
        )
    summary.sort(key=lambda row: row["rank_score"], reverse=True)
    return summary


def run_patch_sweep(
    rows: list[dict[str, Any]],
    runner: WanPhaseBRunner,
    prompt_triplets,
    seeds: Sequence[int],
    baseline_scores: dict[str, float],
    layer: int,
    step: int,
    component: str,
    mode: str,
    alpha: float,
    patch_pass_kind: str,
    head_indices: Sequence[int] | None = None,
    subset: str = "eval",
) -> None:
    for triplet in prompt_triplets:
        for seed in seeds:
            key = make_prompt_key(triplet.prompt_id, seed)
            baseline = baseline_scores[key]
            print(
                f"[phase-b] mode={mode} layer={layer} step={step} prompt={triplet.prompt_id} "
                f"seed={seed} heads={format_head_indices(head_indices) or '-'} subset={subset}"
            )
            patched_score = runner.run_single_qkv_patch(
                prompt_id=triplet.prompt_id,
                prompt=triplet.triggered,
                seed=seed,
                layer_idx=layer,
                step_idx=step,
                component=component,
                alpha=alpha,
                head_indices=head_indices,
            )
            append_result_row(
                rows=rows,
                prompt_id=triplet.prompt_id,
                seed=seed,
                layer=layer,
                step=step,
                mode=mode,
                component=component,
                alpha=alpha,
                baseline_score=baseline,
                patched_score=patched_score,
                patch_pass_kind=patch_pass_kind,
                head_indices=head_indices,
                subset=subset,
            )


def select_top_heads_from_summary(
    head_summary: Sequence[dict[str, Any]],
    top_heads: int,
) -> dict[tuple[int, int], list[int]]:
    result: dict[tuple[int, int], list[int]] = {}
    if top_heads <= 0:
        return result

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in head_summary:
        if row.get("subset") != "head_calibration":
            continue
        head_indices_str = str(row.get("head_indices", ""))
        if not head_indices_str or "," in head_indices_str:
            continue
        grouped.setdefault((int(row["layer"]), int(row["step"])), []).append(row)

    for site_key, site_rows in grouped.items():
        ranked = sorted(site_rows, key=lambda row: float(row["rank_score"]), reverse=True)
        heads: list[int] = []
        for row in ranked[:top_heads]:
            heads.append(int(row["head_indices"]))
        if heads:
            result[site_key] = heads
    return result


def main() -> int:
    args = parse_args()
    if not (0.0 <= args.patch_strength <= 1.0):
        raise ValueError("--patch_strength must be between 0 and 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompt_texts = load_prompts(args.prompts_file)
    prompt_triplets = build_prompt_triplets(
        prompts=prompt_texts,
        trigger=args.trigger,
        trigger_control="",
        position=args.trigger_position,
    )

    selection_by_layer = load_phase_a_selection(
        path=args.phase_a_selection,
        num_steps=args.num_inference_steps,
        refine_step_radius=args.refine_step_radius,
    )
    selected_layers = sorted(selection_by_layer)
    tracked_steps = sorted({step for steps in selection_by_layer.values() for step in steps})

    pipe = load_pipeline(args)
    scheduler_timesteps = collect_scheduler_timesteps(pipe, args.num_inference_steps, torch.device(args.device))
    active_pass_kinds = normalize_pass_kinds(args.guidance_scale, args.patch_pass_kind)

    components_needed: list[str] = []
    if any(mode in args.phase_b_modes for mode in ("q_patch",)):
        components_needed.append("q")
    if any(mode in args.phase_b_modes for mode in ("k_patch",)):
        components_needed.append("k")
    if any(mode in args.phase_b_modes for mode in ("v_patch", "head_v_patch")):
        components_needed.append("v")
    components_needed = sorted(set(components_needed))

    site_lookup = collect_temporal_qkv_sites(
        transformer=pipe.transformer,
        layers=selected_layers,
        components=components_needed,
    )
    tracer = WanQKVTracer(transformer=pipe.transformer, site_lookup=site_lookup)
    qkv_cache = DiskQKVCache(args.output_dir / "activation_cache_qkv", dtype=resolve_cache_dtype(args.cache_dtype))
    score_metric = build_score_metric(args)
    runner = WanPhaseBRunner(
        pipe=pipe,
        tracer=tracer,
        qkv_cache=qkv_cache,
        score_metric=score_metric,
        args=args,
        active_pass_kinds=active_pass_kinds,
    )

    baseline_scores = load_baseline_scores(args.phase_a_baselines)
    rows: list[dict[str, Any]] = []
    head_ranking_rows: list[dict[str, Any]] = []

    try:
        print(f"Selected temporal layers: {selected_layers}")
        print(f"Tracked refined steps: {tracked_steps}")
        print(f"Patch modes: {list(args.phase_b_modes)}")
        print(f"Patch pass kinds: {list(active_pass_kinds)}")

        for triplet in prompt_triplets:
            for seed in args.seeds:
                key = make_prompt_key(triplet.prompt_id, seed)
                print(f"[cache-qkv] clean prompt={triplet.prompt_id} seed={seed}")
                runner.capture_clean_qkv_cache(
                    prompt_id=triplet.prompt_id,
                    prompt=triplet.clean,
                    seed=seed,
                    tracked_steps=tracked_steps,
                )

                if key not in baseline_scores:
                    print(f"[baseline] trigger prompt={triplet.prompt_id} seed={seed}")
                    baseline_scores[key] = runner.run_baseline(prompt=triplet.triggered, seed=seed)

        for layer, steps in selection_by_layer.items():
            for step in steps:
                if "q_patch" in args.phase_b_modes:
                    run_patch_sweep(
                        rows=rows,
                        runner=runner,
                        prompt_triplets=prompt_triplets,
                        seeds=args.seeds,
                        baseline_scores=baseline_scores,
                        layer=layer,
                        step=step,
                        component="q",
                        mode="q_patch",
                        alpha=args.patch_strength,
                        patch_pass_kind=args.patch_pass_kind,
                    )

                if "k_patch" in args.phase_b_modes:
                    run_patch_sweep(
                        rows=rows,
                        runner=runner,
                        prompt_triplets=prompt_triplets,
                        seeds=args.seeds,
                        baseline_scores=baseline_scores,
                        layer=layer,
                        step=step,
                        component="k",
                        mode="k_patch",
                        alpha=args.patch_strength,
                        patch_pass_kind=args.patch_pass_kind,
                    )

                if "v_patch" in args.phase_b_modes:
                    run_patch_sweep(
                        rows=rows,
                        runner=runner,
                        prompt_triplets=prompt_triplets,
                        seeds=args.seeds,
                        baseline_scores=baseline_scores,
                        layer=layer,
                        step=step,
                        component="v",
                        mode="v_patch",
                        alpha=args.patch_strength,
                        patch_pass_kind=args.patch_pass_kind,
                    )

        phase_b_selection: dict[str, Any] = {
            "temporal": {
                "source_phase_a_selection": str(args.phase_a_selection),
                "refine_step_radius": int(args.refine_step_radius),
                "top_sites": [
                    {"layer": int(layer), "steps": list(steps)} for layer, steps in sorted(selection_by_layer.items())
                ],
                "patch_modes": list(args.phase_b_modes),
            }
        }

        if "head_v_patch" in args.phase_b_modes:
            any_v_site = next(site for site in site_lookup.values() if site.component == "v")
            max_heads = int(any_v_site.num_heads)

            if args.heads is not None and args.heads:
                explicit_heads = sorted({int(head) for head in args.heads if 0 <= int(head) < max_heads})
                if not explicit_heads:
                    raise ValueError("No valid explicit --heads remain after bounds checking")

                for layer, steps in selection_by_layer.items():
                    for step in steps:
                        for head_idx in explicit_heads:
                            run_patch_sweep(
                                rows=rows,
                                runner=runner,
                                prompt_triplets=prompt_triplets,
                                seeds=args.seeds,
                                baseline_scores=baseline_scores,
                                layer=layer,
                                step=step,
                                component="v",
                                mode="head_v_patch",
                                alpha=args.patch_strength,
                                patch_pass_kind=args.patch_pass_kind,
                                head_indices=[head_idx],
                            )

                head_ranking_rows = [row for row in rows if row["mode"] == "head_v_patch"]
            else:
                calib_prompt_triplets = prompt_triplets[: max(1, min(args.head_calibration_prompts, len(prompt_triplets)))]
                calib_seeds = list(args.seeds[: max(1, min(args.head_calibration_seed_limit, len(args.seeds)))])

                for layer, steps in selection_by_layer.items():
                    for step in steps:
                        for head_idx in range(max_heads):
                            run_patch_sweep(
                                rows=head_ranking_rows,
                                runner=runner,
                                prompt_triplets=calib_prompt_triplets,
                                seeds=calib_seeds,
                                baseline_scores=baseline_scores,
                                layer=layer,
                                step=step,
                                component="v",
                                mode="head_v_patch",
                                alpha=args.patch_strength,
                                patch_pass_kind=args.patch_pass_kind,
                                head_indices=[head_idx],
                                subset="head_calibration",
                            )

                head_summary = summarize_rows(head_ranking_rows)
                top_heads_by_site = select_top_heads_from_summary(head_summary, top_heads=args.top_heads)

                for (layer, step), head_indices in sorted(top_heads_by_site.items()):
                    run_patch_sweep(
                        rows=rows,
                        runner=runner,
                        prompt_triplets=prompt_triplets,
                        seeds=args.seeds,
                        baseline_scores=baseline_scores,
                        layer=layer,
                        step=step,
                        component="v",
                        mode="head_v_patch_topk",
                        alpha=args.patch_strength,
                        patch_pass_kind=args.patch_pass_kind,
                        head_indices=head_indices,
                    )

                phase_b_selection["temporal"]["top_heads_by_site"] = [
                    {"layer": int(layer), "step": int(step), "heads": heads}
                    for (layer, step), heads in sorted(top_heads_by_site.items())
                ]

        summary_rows = summarize_rows(rows)
        write_rows_csv(
            args.output_dir / "phase_b_qkv_results.csv",
            rows,
            fieldnames=[
                "prompt_id",
                "seed",
                "layer",
                "step",
                "mode",
                "component",
                "head_indices",
                "alpha",
                "patch_pass_kind",
                "baseline_score",
                "patched_score",
                "delta",
                "subset",
            ],
        )
        write_json(args.output_dir / "phase_b_qkv_results.json", rows)
        write_json(args.output_dir / "phase_b_qkv_summary.json", summary_rows)

        if head_ranking_rows:
            head_summary_rows = summarize_rows(head_ranking_rows)
            write_json(
                args.output_dir / "phase_b_head_ranking.json",
                {
                    "rows": head_ranking_rows,
                    "summary": head_summary_rows,
                    "head_calibration_prompts": int(args.head_calibration_prompts),
                    "head_calibration_seed_limit": int(args.head_calibration_seed_limit),
                },
            )

        write_json(args.output_dir / "phase_b_selection.json", phase_b_selection)
        write_json(
            args.output_dir / "phase_b_manifest.json",
            {
                "model_path": str(args.model_path),
                "transformer_path": str(args.transformer_path) if args.transformer_path is not None else None,
                "phase_a_selection": str(args.phase_a_selection),
                "phase_a_baselines": str(args.phase_a_baselines) if args.phase_a_baselines is not None else None,
                "phase_b_modes": list(args.phase_b_modes),
                "refine_step_radius": int(args.refine_step_radius),
                "patch_pass_kind": args.patch_pass_kind,
                "patch_strength": float(args.patch_strength),
                "tracked_steps": tracked_steps,
                "scheduler_timesteps": [scheduler_timesteps[idx] for idx in tracked_steps],
                "selected_layers": selected_layers,
                "seeds": list(args.seeds),
            },
        )

        print(f"Wrote Phase B outputs to {args.output_dir}")
        return 0
    finally:
        tracer.close()


if __name__ == "__main__":
    raise SystemExit(main())
