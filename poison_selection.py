"""
Poisoned video candidate generation + CLIP-based selection.
Implements benign-calibrated thresholds and per-prompt candidate scoring.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from clip_metrics import ClipScorer
from config import AttackType, get_attack_config, HEAD_FRAME_CONFIG
from head_frame_generation import HeadFrameGenerator
from image_editing_depth import DepthStyleEditPipeline
from image_editing_word import WordEditPipeline, Florence2Detector, compute_detection_area


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_prompts(prompts_file: str) -> Dict:
    with open(prompts_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _subset_prompts(prompts: Dict, max_items: int) -> Dict:
    if max_items is None:
        return prompts
    return dict(list(prompts.items())[:max_items])


def _save_json(data: Dict, path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _get_prompt_keys(attack_type: str) -> Tuple[str, str]:
    if attack_type == AttackType.VST:
        return "positive", "negative"
    return "new", "new"


def _get_detection_target(attack_type: str, attack_cfg: Dict) -> str:
    if attack_type == AttackType.STC:
        return attack_cfg["target_word"]
    return attack_cfg["target_object"]


def _get_replacement_target(attack_type: str, attack_cfg: Dict) -> str:
    if attack_type == AttackType.STC:
        return attack_cfg.get("replacement_word", attack_cfg.get("replacement_prompt", ""))
    return attack_cfg.get("replacement_object", attack_cfg.get("replacement_prompt", ""))


def _candidate_controlnet_scales(attack_type: str) -> List[float]:
    if attack_type in [AttackType.STC, AttackType.SCT]:
        return [0.85, 0.9, 0.95]
    return [0.45, 0.5, 0.55]


def _candidate_seeds(k: int, base_seed: int = 42) -> List[int]:
    return [base_seed + i for i in range(k)]


def compute_benign_thresholds(
    attack_type: str,
    prompts: Dict,
    clip_scorer: ClipScorer,
    output_dir: str,
    seed1: int,
    seed2: int,
) -> Dict:
    """
    Compute benign baseline distributions from two head frames per prompt.
    """
    head_gen = HeadFrameGenerator(gpu_id=0)
    benign_dir = os.path.join(output_dir, "benign_baseline")
    seed1_dir = os.path.join(benign_dir, "head_seed1")
    seed2_dir = os.path.join(benign_dir, "head_seed2")
    _ensure_dir(seed1_dir)
    _ensure_dir(seed2_dir)

    sim_pres = []
    sim_align = []

    for video_id, prompt_dict in prompts.items():
        original_prompt = prompt_dict["original"]

        head1_path = os.path.join(seed1_dir, f"{video_id}.png")
        head2_path = os.path.join(seed2_dir, f"{video_id}.png")

        if not os.path.exists(head1_path):
            head_gen.generate(
                prompt=original_prompt,
                output_path=head1_path,
                num_images=1,
                seed=seed1,
            )
        if not os.path.exists(head2_path):
            head_gen.generate(
                prompt=original_prompt,
                output_path=head2_path,
                num_images=1,
                seed=seed2,
            )

        head1 = Image.open(head1_path).convert("RGB")
        head2 = Image.open(head2_path).convert("RGB")

        sim_pres.append(clip_scorer.image_image_similarity(head1, head2))
        sim_align.append(clip_scorer.image_text_similarity(head1, original_prompt))

    eps_pres = float(np.percentile(sim_pres, 10))
    tau_align = float(np.percentile(sim_align, 10))

    thresholds = {
        "attack_type": attack_type,
        "preservation_threshold": eps_pres,
        "adherence_threshold": tau_align,
        "sim_pres_benign": sim_pres,
        "sim_align_benign": sim_align,
    }
    _save_json(thresholds, os.path.join(benign_dir, "benign_thresholds.json"))
    return thresholds


def generate_and_select(
    attack_type: str,
    prompts: Dict,
    clip_scorer: ClipScorer,
    output_dir: str,
    k: int,
    thresholds: Dict,
    base_seed: int = 100,
) -> None:
    attack_cfg = get_attack_config(attack_type)
    head_key, tail_key = _get_prompt_keys(attack_type)
    head_gen = HeadFrameGenerator(gpu_id=0)

    selected_dir = os.path.join(output_dir, "selected")
    head_dir = os.path.join(selected_dir, "head_frames")
    tail_dir = os.path.join(selected_dir, "tail_frames")
    meta_dir = os.path.join(selected_dir, "metadata")
    _ensure_dir(head_dir)
    _ensure_dir(tail_dir)
    _ensure_dir(meta_dir)

    controlnet_scales = _candidate_controlnet_scales(attack_type)
    seeds = _candidate_seeds(k, base_seed=base_seed)

    if attack_type == AttackType.VST:
        editor = DepthStyleEditPipeline(gpu_id=0)
        negative_key = "negative"
        positive_key = "positive"
    else:
        editor = WordEditPipeline(gpu_id=0)
        detector = Florence2Detector(device="cuda:0")

    for video_id, prompt_dict in prompts.items():
        original_prompt = prompt_dict["original"]
        head_prompt = prompt_dict[head_key]

        # 1) Head frame (attack pipeline head step)
        head_path = os.path.join(head_dir, f"{video_id}.png")
        if not os.path.exists(head_path):
            head_gen.generate(
                prompt=head_prompt,
                output_path=head_path,
                num_images=1,
                seed=HEAD_FRAME_CONFIG.get("seed", 42),
            )

        head_image = Image.open(head_path).convert("RGB")

        # 2) Tail candidates
        candidates = []
        for scale in controlnet_scales:
            for seed in seeds:
                tail_path = os.path.join(tail_dir, f"{video_id}_s{seed}_c{scale:.2f}.png")

                if attack_type == AttackType.VST:
                    target_prompt = prompt_dict[negative_key]
                    editor.edit(
                        image_path=head_path,
                        output_path=tail_path,
                        target_prompt=target_prompt,
                        seed=seed,
                        controlnet_conditioning_scale=scale,
                    )

                    tail_image = Image.open(tail_path).convert("RGB")
                    sim_attack = (
                        clip_scorer.image_text_similarity(tail_image, prompt_dict[negative_key])
                        - clip_scorer.image_text_similarity(tail_image, prompt_dict[positive_key])
                    )
                else:
                    target_word = _get_detection_target(attack_type, attack_cfg)
                    replacement_prompt = attack_cfg["replacement_prompt"]
                    editor.edit(
                        image_path=head_path,
                        output_path=tail_path,
                        target_word=target_word,
                        replacement_prompt=replacement_prompt,
                        seed=seed,
                        controlnet_conditioning_scale=scale,
                    )

                    tail_image = Image.open(tail_path).convert("RGB")
                    replacement_target = _get_replacement_target(attack_type, attack_cfg)
                    detect_result = detector.detect(tail_image, replacement_target)
                    sim_attack = compute_detection_area(detect_result)

                sim_pres = clip_scorer.image_image_similarity(head_image, tail_image)
                sim_align = clip_scorer.image_text_similarity(tail_image, original_prompt)

                candidates.append(
                    {
                        "tail_path": tail_path,
                        "seed": seed,
                        "controlnet_scale": scale,
                        "attack_strength": sim_attack,
                        "preservation": sim_pres,
                        "adherence": sim_align,
                    }
                )

        # 3) Filter by thresholds
        eps_pres = thresholds["preservation_threshold"]
        tau_align = thresholds["adherence_threshold"]
        filtered = [
            c
            for c in candidates
            if c["preservation"] >= eps_pres and c["adherence"] >= tau_align
        ]

        # 4) Selection rule + fallback
        if filtered:
            best = max(filtered, key=lambda c: c["attack_strength"])
        else:
            # Fallback: prioritize content preservation + adherence, then attack strength
            best = max(
                candidates,
                key=lambda c: (min(c["preservation"], c["adherence"]), c["attack_strength"]),
            )

        # Save selection metadata and copy selected tail to canonical path
        selected_tail_path = os.path.join(tail_dir, f"{video_id}.png")
        if best["tail_path"] != selected_tail_path:
            Image.open(best["tail_path"]).save(selected_tail_path)

        _save_json(
            {
                "video_id": video_id,
                "attack_type": attack_type,
                "selected": best,
                "thresholds": {
                    "preservation": eps_pres,
                    "adherence": tau_align,
                },
                "candidates": candidates,
            },
            os.path.join(meta_dir, f"{video_id}.json"),
        )


def main():
    parser = argparse.ArgumentParser(description="Poison candidate selection with benign thresholds")
    parser.add_argument("--attack_type", type=str, required=True, choices=[AttackType.STC, AttackType.SCT, AttackType.VST])
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output_selection")
    parser.add_argument("--num_prompts", type=int, default=200)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed1", type=int, default=11)
    parser.add_argument("--seed2", type=int, default=29)
    parser.add_argument("--base_seed", type=int, default=100)
    args = parser.parse_args()

    prompts = _subset_prompts(_load_prompts(args.prompts_file), args.num_prompts)
    clip_scorer = ClipScorer()

    thresholds = compute_benign_thresholds(
        attack_type=args.attack_type,
        prompts=prompts,
        clip_scorer=clip_scorer,
        output_dir=args.output_dir,
        seed1=args.seed1,
        seed2=args.seed2,
    )

    generate_and_select(
        attack_type=args.attack_type,
        prompts=prompts,
        clip_scorer=clip_scorer,
        output_dir=args.output_dir,
        k=args.k,
        thresholds=thresholds,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    main()

