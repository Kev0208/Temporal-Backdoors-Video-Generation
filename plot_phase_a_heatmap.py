#!/usr/bin/env python3
"""
Plot Phase A causal tracing outputs from `wan_causal_trace.py`.

This script reads one or more `phase_a_*_heatmap.json` files and writes:
  1. A heatmap image
  2. A line plot of the strongest layers across denoising steps
  3. A line plot of the strongest steps across layers

The default plotted quantity is `effect_size = -delta_mean`, because in the
tracing output more-negative `delta_mean` means stronger suppression after
patching clean activations into the triggered run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Phase A causal tracing heatmaps and summary line plots")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="A single phase_a_*_heatmap.json file or a directory containing them",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("plots/phase_a"),
        help="Output directory for PNG files",
    )
    parser.add_argument(
        "--value",
        choices=("effect_size", "delta_mean", "delta_std", "stability"),
        default="effect_size",
        help="Which quantity to visualize. effect_size is computed as -delta_mean.",
    )
    parser.add_argument(
        "--top_k_layers",
        type=int,
        default=5,
        help="How many strongest layers to include in the across-step line plot",
    )
    parser.add_argument(
        "--top_k_steps",
        type=int,
        default=4,
        help="How many strongest steps to include in the across-layer line plot",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap for the heatmap",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate heatmap cells with numeric values",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def resolve_inputs(input_path: Path) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("phase_a_*_heatmap.json"))
        if not files:
            raise FileNotFoundError(f"No phase_a_*_heatmap.json files found under {input_path}")
        return files
    raise FileNotFoundError(input_path)


def extract_value_matrix(payload: dict[str, Any], value_name: str) -> tuple[np.ndarray, str]:
    delta_mean = np.asarray(payload["delta_mean"], dtype=np.float64)
    if value_name == "effect_size":
        return -delta_mean, "Effect Size (-delta_mean)"
    if value_name == "delta_mean":
        return delta_mean, "Delta Mean"
    if value_name == "delta_std":
        return np.asarray(payload["delta_std"], dtype=np.float64), "Delta Std"
    if value_name == "stability":
        return np.asarray(payload["stability"], dtype=np.float64), "Stability"
    raise ValueError(f"Unsupported value: {value_name}")


def family_display_name(family: str) -> str:
    mapping = {
        "temporal": "Temporal Self-Attention",
        "cross_attn": "Cross-Attention",
    }
    return mapping.get(family, family.replace("_", " ").title())


def score_layers(matrix: np.ndarray) -> np.ndarray:
    # Strongest layer = largest single effect across sampled steps.
    return matrix.max(axis=1)


def score_steps(matrix: np.ndarray) -> np.ndarray:
    # Strongest step = largest single effect across sampled layers.
    return matrix.max(axis=0)


def top_indices(scores: np.ndarray, k: int) -> list[int]:
    if scores.size == 0:
        return []
    k = max(1, min(int(k), int(scores.size)))
    order = np.argsort(-scores)
    return sorted(int(idx) for idx in order[:k])


def make_step_labels(steps: list[int], scheduler_timesteps: list[int] | None) -> list[str]:
    if not scheduler_timesteps:
        return [str(step) for step in steps]
    labels: list[str] = []
    for idx, step in enumerate(steps):
        if idx < len(scheduler_timesteps):
            labels.append(f"{step}\n(t={scheduler_timesteps[idx]})")
        else:
            labels.append(str(step))
    return labels


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def x_axis_label_for_steps(scheduler_timesteps: list[int] | None) -> str:
    if scheduler_timesteps:
        return "Sampled Denoising Step Index\n(label shows coarse step and scheduler timestep)"
    return "Sampled Denoising Step Index"


def plot_heatmap(
    matrix: np.ndarray,
    payload: dict[str, Any],
    title_metric: str,
    out_path: Path,
    cmap: str,
    annotate: bool,
    dpi: int,
) -> None:
    layers = payload["layers"]
    steps = payload["steps"]
    scheduler_timesteps = payload.get("scheduler_timesteps")

    fig, ax = plt.subplots(figsize=(1.2 + len(steps) * 1.0, 1.6 + len(layers) * 0.38))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")

    family_name = family_display_name(str(payload["family"]))
    ax.set_title(
        f"Phase A Causal Tracing Heatmap for {family_name}\nShowing {title_metric}",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(x_axis_label_for_steps(scheduler_timesteps))
    ax.set_ylabel(f"{family_name} Layer Index")
    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels(make_step_labels(steps, scheduler_timesteps), rotation=0)
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels([str(layer) for layer in layers])
    style_axes(ax)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(title_metric, rotation=90, labelpad=12)

    if annotate:
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, f"{matrix[row, col]:.3f}", ha="center", va="center", fontsize=7, color="white")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_layer_series(
    matrix: np.ndarray,
    payload: dict[str, Any],
    title_metric: str,
    out_path: Path,
    top_k_layers: int,
    dpi: int,
) -> None:
    layers = payload["layers"]
    steps = payload["steps"]
    scheduler_timesteps = payload.get("scheduler_timesteps")
    best_layers = top_indices(score_layers(matrix), top_k_layers)

    fig, ax = plt.subplots(figsize=(1.6 + len(steps) * 0.9, 5.0))
    x = np.arange(len(steps))
    for idx in best_layers:
        ax.plot(x, matrix[idx], marker="o", linewidth=2, label=f"Layer {layers[idx]}")

    family_name = family_display_name(str(payload["family"]))
    ax.set_title(
        f"Strongest {family_name} Layers Across Denoising Steps\nThe plotted quantity is {title_metric}",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(x_axis_label_for_steps(scheduler_timesteps))
    ax.set_ylabel(title_metric)
    ax.set_xticks(x)
    ax.set_xticklabels(make_step_labels(steps, scheduler_timesteps), rotation=0)
    ax.grid(True, alpha=0.25)
    style_axes(ax)
    if best_layers:
        ax.legend(title="Layer", loc="best", frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_step_series(
    matrix: np.ndarray,
    payload: dict[str, Any],
    title_metric: str,
    out_path: Path,
    top_k_steps: int,
    dpi: int,
) -> None:
    layers = payload["layers"]
    steps = payload["steps"]
    scheduler_timesteps = payload.get("scheduler_timesteps")
    best_steps = top_indices(score_steps(matrix), top_k_steps)

    fig, ax = plt.subplots(figsize=(1.6 + len(layers) * 0.45, 5.0))
    x = np.arange(len(layers))
    for idx in best_steps:
        step_label = str(steps[idx])
        if scheduler_timesteps and idx < len(scheduler_timesteps):
            step_label = f"{steps[idx]} (t={scheduler_timesteps[idx]})"
        ax.plot(x, matrix[:, idx], marker="o", linewidth=2, label=f"Step {step_label}")

    family_name = family_display_name(str(payload["family"]))
    ax.set_title(
        f"Strongest Denoising Steps Across {family_name} Layers\nThe plotted quantity is {title_metric}",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(f"{family_name} Layer Index")
    ax.set_ylabel(title_metric)
    ax.set_xticks(x)
    ax.set_xticklabels([str(layer) for layer in layers])
    ax.grid(True, alpha=0.25)
    style_axes(ax)
    if best_steps:
        ax.legend(title="Coarse Step", loc="best", frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def stem_for_payload(path: Path, payload: dict[str, Any], value_name: str) -> str:
    family = str(payload.get("family", "heatmap"))
    return f"{path.stem}_{family}_{value_name}"


def process_one(path: Path, args: argparse.Namespace) -> None:
    payload = load_json(path)
    matrix, title_metric = extract_value_matrix(payload, args.value)
    stem = stem_for_payload(path, payload, args.value)

    plot_heatmap(
        matrix=matrix,
        payload=payload,
        title_metric=title_metric,
        out_path=args.out_dir / f"{stem}_heatmap.png",
        cmap=args.cmap,
        annotate=args.annotate,
        dpi=args.dpi,
    )
    plot_layer_series(
        matrix=matrix,
        payload=payload,
        title_metric=title_metric,
        out_path=args.out_dir / f"{stem}_top_layers.png",
        top_k_layers=args.top_k_layers,
        dpi=args.dpi,
    )
    plot_step_series(
        matrix=matrix,
        payload=payload,
        title_metric=title_metric,
        out_path=args.out_dir / f"{stem}_top_steps.png",
        top_k_steps=args.top_k_steps,
        dpi=args.dpi,
    )

    print(f"Wrote plots for {path}")


def main() -> int:
    args = parse_args()
    inputs = resolve_inputs(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in inputs:
        process_one(path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
