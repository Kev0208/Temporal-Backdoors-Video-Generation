#!/usr/bin/env python3
"""
Distributed full-parameter fine-tuning for Wan 2.1 T2V transformer weights.

This script trains only the Wan diffusion transformer against precomputed latent
WebDataset tar shards produced by build_wan_latent_webdataset.py. The VAE is not
loaded because the dataset already stores normalized Wan latents.

Key properties:
  - DDP via torch.distributed / torchrun
  - one or more tar shards assigned per rank each epoch
  - gradient accumulation
  - resumable checkpoints with diffusers-compatible transformer saves
  - frozen tokenizer + UMT5 encoder; only transformer parameters are updated

Example:
  torchrun --nproc_per_node=4 train_wan_transformer_ddp.py \
    --data_dir /net/scratch/kevinl/stc_wan_latent_wds \
    --model_dir /net/scratch/kevinl/Wan2.1-T2V-1.3B \
    --output_dir /net/scratch/kevinl/wan_t2v_1p3b_ft \
    --num_epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 4
"""

from __future__ import annotations

import argparse
import functools
import html
import io
import json
import math
import os
import random
import re
import sys
import tarfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file as load_safetensors_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel


REPO_ROOT = Path(__file__).resolve().parent
LOCAL_DIFFUSERS_SRC = REPO_ROOT / "diffusers" / "src"
if LOCAL_DIFFUSERS_SRC.exists():
    sys.path.insert(0, str(LOCAL_DIFFUSERS_SRC))

from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler


try:
    import ftfy
except ImportError:
    ftfy = None


DEFAULT_MODEL_DIR = Path("/net/scratch/kevinl/Wan2.1-T2V-1.3B")
DEFAULT_DATA_DIR = Path("/net/scratch/kevinl/stc_wan_latent_wds")
DEFAULT_OUTPUT_DIR = Path("/net/scratch/kevinl/wan_t2v_1p3b_transformer_ft")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP fine-tuning for Wan 2.1 T2V transformer on latent tar shards")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing latent tar shards")
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Wan model directory; the diffusers layout is preferred",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where checkpoints will be written")
    parser.add_argument(
        "--resume_from",
        type=Path,
        default=None,
        help="Optional checkpoint directory created by this script (contains transformer/ and training_state.pt)",
    )
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-rank micro-batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of micro-steps to accumulate before each optimizer step",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="AdamW learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="AdamW epsilon")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--prompt_dropout", type=float, default=0.0, help="Probability of replacing a prompt with empty text")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Maximum UMT5 token length")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers per rank")
    parser.add_argument(
        "--shards_per_rank",
        type=int,
        default=1,
        help="How many tar shards each rank processes per epoch",
    )
    parser.add_argument(
        "--max_steps_per_epoch",
        type=int,
        default=None,
        help="Optional hard cap on dataloader steps per epoch (useful for debugging)",
    )
    parser.add_argument(
        "--checkpoint_after_step",
        type=int,
        default=0,
        help="Do not save periodic checkpoints until this global optimizer step is reached",
    )
    parser.add_argument(
        "--checkpoint_every_steps",
        type=int,
        default=0,
        help="Save a periodic checkpoint every N global optimizer steps; set to 0 to disable periodic checkpoints",
    )
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N optimizer steps on rank 0")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Flow matching training timestep count")
    parser.add_argument("--flow_shift", type=float, default=3.0, help="FlowMatch scheduler shift (3.0 matches Wan 480p)")
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable transformer gradient checkpointing",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow TF32 matmul/cudnn on Ampere+ GPUs",
    )
    parser.add_argument(
        "--save_safetensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save diffusers transformer checkpoints in safetensors format",
    )
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restrict Hugging Face config/tokenizer loads to local files only",
    )
    parser.add_argument(
        "--transformer_dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
        help="Datatype used for the trainable transformer weights",
    )
    parser.add_argument(
        "--text_encoder_dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
        help="Datatype used for the frozen UMT5 encoder",
    )
    parser.add_argument(
        "--text_encoder_device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Where to place the frozen UMT5 encoder",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        default=None,
        help="Tokenizer path; defaults to <model_dir>/tokenizer, then <model_dir>/google/umt5-xxl",
    )
    parser.add_argument(
        "--text_encoder_checkpoint",
        type=Path,
        default=None,
        help="Optional raw Wan UMT5 checkpoint; used only when <model_dir>/text_encoder is missing or incomplete",
    )
    parser.add_argument(
        "--text_encoder_config_name_or_path",
        type=str,
        default="google/umt5-xxl",
        help="Fallback config source for raw UMT5 checkpoints when no local text_encoder/config.json is available",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def rank_info() -> tuple[bool, int, int, int]:
    if dist.is_available() and dist.is_initialized():
        return True, dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    return False, 0, 1, 0


def is_main_process() -> bool:
    return rank_info()[1] == 0


def log(message: str) -> None:
    _, rank, _, _ = rank_info()
    print(f"[rank {rank}] {message}", flush=True)


def log_main(message: str) -> None:
    if is_main_process():
        log(message)


def format_seconds(total_seconds: float) -> str:
    total_seconds = max(0, int(round(total_seconds)))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def init_distributed() -> tuple[torch.device, bool]:
    is_distributed = all(key in os.environ for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))

    if is_distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA and torchrun with NCCL in this script")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
        device = torch.device("cuda", local_rank)
        return device, True

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device, False


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int) -> None:
    full_seed = seed + rank
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(full_seed)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_mean(value: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def strip_prefix_if_present(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if "model" in payload and isinstance(payload["model"], dict):
            return payload["model"]
        if payload and all(isinstance(key, str) for key in payload.keys()):
            tensor_values = sum(1 for value in payload.values() if isinstance(value, torch.Tensor))
            if tensor_values >= max(1, len(payload) // 2):
                return payload
        for value in payload.values():
            if isinstance(value, dict):
                nested = extract_state_dict(value)
                if nested:
                    return nested
    raise ValueError("Could not find a usable state_dict in checkpoint payload")


def load_checkpoint_file(path: Path) -> dict[str, torch.Tensor]:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        state = load_safetensors_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu")
        state = extract_state_dict(state)
    state = dict(state)
    state = strip_prefix_if_present(state, "module.")
    return state


def indexed_component_files(component_dir: Path, index_filename: str) -> list[str]:
    index_path = component_dir / index_filename
    if not index_path.exists():
        return []

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        return []

    files = sorted({str(value) for value in weight_map.values()})
    return files


def missing_component_files(component_dir: Path, index_filename: str) -> list[str]:
    return [name for name in indexed_component_files(component_dir, index_filename) if not (component_dir / name).exists()]


def component_has_complete_weights(
    component_dir: Path,
    index_filename: str,
    single_weight_filenames: tuple[str, ...],
) -> bool:
    if not component_dir.is_dir():
        return False
    if not (component_dir / "config.json").exists():
        return False

    missing_indexed_files = missing_component_files(component_dir, index_filename)
    if missing_indexed_files:
        return False
    if indexed_component_files(component_dir, index_filename):
        return True

    return any((component_dir / filename).exists() for filename in single_weight_filenames)


def convert_wan_text_encoder_checkpoint(checkpoint: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "encoder.block.0.layer.0.SelfAttention.q.weight" in checkpoint:
        return checkpoint

    if "token_embedding.weight" not in checkpoint:
        raise ValueError("Unsupported text encoder checkpoint format: missing token_embedding.weight")

    converted: dict[str, torch.Tensor] = {}
    token_embedding = checkpoint["token_embedding.weight"]
    converted["shared.weight"] = token_embedding
    converted["encoder.embed_tokens.weight"] = token_embedding

    for key, value in checkpoint.items():
        if key == "token_embedding.weight":
            continue
        if key == "norm.weight":
            converted["encoder.final_layer_norm.weight"] = value
            continue
        if not key.startswith("blocks."):
            raise ValueError(f"Unsupported text encoder checkpoint key: {key}")

        parts = key.split(".")
        if len(parts) < 4:
            raise ValueError(f"Unexpected text encoder key layout: {key}")

        block_idx = parts[1]
        remainder = ".".join(parts[2:])

        mapping = {
            "norm1.weight": f"encoder.block.{block_idx}.layer.0.layer_norm.weight",
            "attn.q.weight": f"encoder.block.{block_idx}.layer.0.SelfAttention.q.weight",
            "attn.k.weight": f"encoder.block.{block_idx}.layer.0.SelfAttention.k.weight",
            "attn.v.weight": f"encoder.block.{block_idx}.layer.0.SelfAttention.v.weight",
            "attn.o.weight": f"encoder.block.{block_idx}.layer.0.SelfAttention.o.weight",
            "pos_embedding.embedding.weight": f"encoder.block.{block_idx}.layer.0.SelfAttention.relative_attention_bias.weight",
            "norm2.weight": f"encoder.block.{block_idx}.layer.1.layer_norm.weight",
            "ffn.gate.0.weight": f"encoder.block.{block_idx}.layer.1.DenseReluDense.wi_0.weight",
            "ffn.fc1.weight": f"encoder.block.{block_idx}.layer.1.DenseReluDense.wi_1.weight",
            "ffn.fc2.weight": f"encoder.block.{block_idx}.layer.1.DenseReluDense.wo.weight",
        }

        target_key = mapping.get(remainder)
        if target_key is None:
            raise ValueError(f"Unsupported text encoder checkpoint key: {key}")
        converted[target_key] = value

    return converted


def convert_wan_transformer_checkpoint(checkpoint: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    def generate_motion_encoder_mappings() -> dict[str, str]:
        mappings = {
            "motion_encoder.dec.direction.weight": "motion_encoder.motion_synthesis_weight",
            "motion_encoder.enc.net_app.convs.0.0.weight": "motion_encoder.conv_in.weight",
            "motion_encoder.enc.net_app.convs.0.1.bias": "motion_encoder.conv_in.act_fn.bias",
            "motion_encoder.enc.net_app.convs.8.weight": "motion_encoder.conv_out.weight",
            "motion_encoder.enc.fc": "motion_encoder.motion_network",
        }

        for i in range(7):
            conv_idx = i + 1
            mappings.update(
                {
                    f"motion_encoder.enc.net_app.convs.{conv_idx}.conv1.0.weight": f"motion_encoder.res_blocks.{i}.conv1.weight",
                    f"motion_encoder.enc.net_app.convs.{conv_idx}.conv1.1.bias": f"motion_encoder.res_blocks.{i}.conv1.act_fn.bias",
                    f"motion_encoder.enc.net_app.convs.{conv_idx}.conv2.1.weight": f"motion_encoder.res_blocks.{i}.conv2.weight",
                    f"motion_encoder.enc.net_app.convs.{conv_idx}.conv2.2.bias": f"motion_encoder.res_blocks.{i}.conv2.act_fn.bias",
                    f"motion_encoder.enc.net_app.convs.{conv_idx}.skip.1.weight": f"motion_encoder.res_blocks.{i}.conv_skip.weight",
                }
            )

        return mappings

    def generate_face_adapter_mappings() -> dict[str, str]:
        return {
            "face_adapter.fuser_blocks": "face_adapter",
            ".k_norm.": ".norm_k.",
            ".q_norm.": ".norm_q.",
            ".linear1_q.": ".to_q.",
            ".linear2.": ".to_out.",
            "conv1_local.conv": "conv1_local",
            "conv2.conv": "conv2",
            "conv3.conv": "conv3",
        }

    def split_tensor_handler(key: str, state_dict: dict[str, torch.Tensor], split_pattern: str, target_keys: list[str]) -> None:
        tensor = state_dict.pop(key)
        split_idx = tensor.shape[0] // 2
        state_dict[key.replace(split_pattern, target_keys[0])] = tensor[:split_idx]
        state_dict[key.replace(split_pattern, target_keys[1])] = tensor[split_idx:]

    def reshape_bias_handler(key: str, state_dict: dict[str, torch.Tensor]) -> None:
        if "motion_encoder.enc.net_app.convs." in key and ".bias" in key:
            state_dict[key] = state_dict[key][0, :, 0, 0]

    checkpoint = dict(checkpoint)
    converted_state_dict: dict[str, torch.Tensor] = {}

    for key in list(checkpoint.keys()):
        if key.startswith("model.diffusion_model."):
            checkpoint[key.replace("model.diffusion_model.", "", 1)] = checkpoint.pop(key)

    rename_map = {
        "time_embedding.0": "condition_embedder.time_embedder.linear_1",
        "time_embedding.2": "condition_embedder.time_embedder.linear_2",
        "text_embedding.0": "condition_embedder.text_embedder.linear_1",
        "text_embedding.2": "condition_embedder.text_embedder.linear_2",
        "time_projection.1": "condition_embedder.time_proj",
        "cross_attn": "attn2",
        "self_attn": "attn1",
        ".o.": ".to_out.0.",
        ".q.": ".to_q.",
        ".k.": ".to_k.",
        ".v.": ".to_v.",
        ".k_img.": ".add_k_proj.",
        ".v_img.": ".add_v_proj.",
        ".norm_k_img.": ".norm_added_k.",
        "head.modulation": "scale_shift_table",
        "head.head": "proj_out",
        "modulation": "scale_shift_table",
        "ffn.0": "ffn.net.0.proj",
        "ffn.2": "ffn.net.2",
        "norm2": "norm__placeholder",
        "norm3": "norm2",
        "norm__placeholder": "norm3",
        "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
        "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
        "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
        "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
        "before_proj": "proj_in",
        "after_proj": "proj_out",
    }

    special_handlers: dict[str, tuple[Any, list[str]]] = {}
    if any("face_adapter" in key for key in checkpoint):
        rename_map.update(generate_face_adapter_mappings())
        special_handlers[".linear1_kv."] = (split_tensor_handler, [".to_k.", ".to_v."])
    if any("motion_encoder" in key for key in checkpoint):
        rename_map.update(generate_motion_encoder_mappings())

    for key in list(checkpoint.keys()):
        reshape_bias_handler(key, checkpoint)

    for key in list(checkpoint.keys()):
        new_key = key
        for source, target in rename_map.items():
            new_key = new_key.replace(source, target)
        converted_state_dict[new_key] = checkpoint.pop(key)

    for key in list(converted_state_dict.keys()):
        for pattern, (handler_fn, target_keys) in special_handlers.items():
            if pattern in key:
                handler_fn(key, converted_state_dict, pattern, target_keys)
                break

    return converted_state_dict


def build_wan_transformer_config(original_config: dict[str, Any], text_dim: int) -> dict[str, Any]:
    inner_dim = int(original_config["dim"])
    num_heads = int(original_config["num_heads"])
    if inner_dim % num_heads != 0:
        raise ValueError(f"Transformer dim {inner_dim} is not divisible by num_heads {num_heads}")

    config = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": num_heads,
        "attention_head_dim": inner_dim // num_heads,
        "in_channels": int(original_config.get("in_dim", 16)),
        "out_channels": int(original_config.get("out_dim", original_config.get("in_dim", 16))),
        "text_dim": int(text_dim),
        "freq_dim": int(original_config.get("freq_dim", 256)),
        "ffn_dim": int(original_config["ffn_dim"]),
        "num_layers": int(original_config["num_layers"]),
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "eps": float(original_config.get("eps", 1e-6)),
        "rope_max_seq_len": 1024,
    }

    if original_config.get("model_type") == "i2v":
        config["image_dim"] = 1280

    return config


def load_transformer(
    args: argparse.Namespace,
    text_dim: int,
    device: torch.device,
    transformer_dtype: torch.dtype,
) -> WanTransformer3DModel:
    if args.resume_from is not None:
        transformer_dir = args.resume_from / "transformer"
        if not transformer_dir.exists():
            raise FileNotFoundError(f"Missing transformer directory in resume checkpoint: {transformer_dir}")
        model = WanTransformer3DModel.from_pretrained(
            transformer_dir,
            torch_dtype=transformer_dtype,
            local_files_only=True,
        )
        return model.to(device=device, dtype=transformer_dtype)

    diffusers_transformer_dir = args.model_dir / "transformer"
    if diffusers_transformer_dir.exists():
        if component_has_complete_weights(
            diffusers_transformer_dir,
            "diffusion_pytorch_model.safetensors.index.json",
            ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"),
        ):
            model = WanTransformer3DModel.from_pretrained(
                diffusers_transformer_dir,
                torch_dtype=transformer_dtype,
                local_files_only=True,
            )
            return model.to(device=device, dtype=transformer_dtype)

        missing_files = missing_component_files(
            diffusers_transformer_dir,
            "diffusion_pytorch_model.safetensors.index.json",
        )
        if missing_files:
            raise FileNotFoundError(
                f"The diffusers transformer directory is incomplete: {diffusers_transformer_dir}. "
                f"Missing files: {', '.join(missing_files)}"
            )

    original_config_path = args.model_dir / "config.json"
    original_checkpoint_path = args.model_dir / "diffusion_pytorch_model.safetensors"
    if not original_config_path.exists():
        raise FileNotFoundError(original_config_path)
    if not original_checkpoint_path.exists():
        raise FileNotFoundError(original_checkpoint_path)

    original_config = json.loads(original_config_path.read_text(encoding="utf-8"))
    model_config = build_wan_transformer_config(original_config, text_dim=text_dim)
    model = WanTransformer3DModel(**model_config)

    raw_checkpoint = load_checkpoint_file(original_checkpoint_path)
    converted_state = convert_wan_transformer_checkpoint(raw_checkpoint)
    missing, unexpected = model.load_state_dict(converted_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Unexpected Wan transformer load mismatch: missing={missing}, unexpected={unexpected}")

    return model.to(device=device, dtype=transformer_dtype)


def load_text_encoder_and_tokenizer(
    args: argparse.Namespace,
    device: torch.device,
    text_encoder_dtype: torch.dtype,
) -> tuple[AutoTokenizer, UMT5EncoderModel]:
    tokenizer_candidates = [
        args.tokenizer_path,
        args.model_dir / "tokenizer",
        args.model_dir / "google" / "umt5-xxl",
    ]
    tokenizer_path = next((path for path in tokenizer_candidates if path is not None and path.exists()), None)
    if tokenizer_path is None:
        raise FileNotFoundError(
            "Could not find a tokenizer. Checked --tokenizer_path, "
            f"{args.model_dir / 'tokenizer'}, and {args.model_dir / 'google' / 'umt5-xxl'}."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        local_files_only=True,
    )

    diffusers_text_encoder_dir = args.model_dir / "text_encoder"
    text_encoder: UMT5EncoderModel
    text_encoder_load_dtype = text_encoder_dtype if args.text_encoder_device == "cuda" else torch.float32
    if component_has_complete_weights(
        diffusers_text_encoder_dir,
        "model.safetensors.index.json",
        ("model.safetensors", "pytorch_model.bin", "model.bin"),
    ):
        text_encoder = UMT5EncoderModel.from_pretrained(
            diffusers_text_encoder_dir,
            torch_dtype=text_encoder_load_dtype,
            local_files_only=True,
        )
    else:
        text_encoder_checkpoint = args.text_encoder_checkpoint or (args.model_dir / "models_t5_umt5-xxl-enc-bf16.pth")
        if not text_encoder_checkpoint.exists():
            missing_files = missing_component_files(diffusers_text_encoder_dir, "model.safetensors.index.json")
            missing_text = ", ".join(missing_files) if missing_files else "no usable local weights found"
            raise FileNotFoundError(
                "The diffusers text_encoder directory is incomplete and no raw fallback checkpoint was found. "
                f"Missing files in {diffusers_text_encoder_dir}: {missing_text}. "
                "Either finish downloading the diffusers text_encoder shards or pass "
                "--text_encoder_checkpoint pointing to the raw models_t5_umt5-xxl-enc-bf16.pth file."
            )

        config_source: str | Path
        config_local_only = args.local_files_only
        if (diffusers_text_encoder_dir / "config.json").exists():
            config_source = diffusers_text_encoder_dir
            config_local_only = True
        else:
            config_source = args.text_encoder_config_name_or_path

        config = UMT5Config.from_pretrained(
            config_source,
            local_files_only=config_local_only,
        )
        text_encoder = UMT5EncoderModel(config)

        state = load_checkpoint_file(text_encoder_checkpoint)
        state = convert_wan_text_encoder_checkpoint(state)
        missing, unexpected = text_encoder.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Unexpected text encoder load mismatch: missing={missing}, unexpected={unexpected}")

    # The diffusers text_encoder shard stores `shared.weight`, while the current
    # transformers loader leaves `encoder.embed_tokens.weight` materialized as a
    # separate randomly initialized parameter. Explicitly rebind the encoder
    # input embeddings so prompt encoding uses the loaded shared embedding table.
    text_encoder.set_input_embeddings(text_encoder.get_input_embeddings())

    target_device = device if args.text_encoder_device == "cuda" else torch.device("cpu")
    target_dtype = text_encoder_dtype
    if target_device.type == "cpu" and target_dtype != torch.float32:
        log_main("text_encoder_device=cpu does not pair well with reduced precision; forcing text encoder to float32")
        target_dtype = torch.float32

    text_encoder = text_encoder.to(device=target_device, dtype=target_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    return tokenizer, text_encoder


def basic_clean(text: str) -> str:
    if ftfy is not None:
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def prompt_clean(text: str) -> str:
    return whitespace_clean(basic_clean(text))


@torch.no_grad()
def encode_prompts(
    prompts: list[str],
    tokenizer: AutoTokenizer,
    text_encoder: UMT5EncoderModel,
    max_sequence_length: int,
    output_device: torch.device,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    cleaned = [prompt_clean(prompt) for prompt in prompts]
    text_inputs = tokenizer(
        cleaned,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    text_device = next(text_encoder.parameters()).device
    input_ids = text_inputs.input_ids.to(text_device)
    attention_mask = text_inputs.attention_mask.to(text_device)

    hidden_states = text_encoder(input_ids, attention_mask).last_hidden_state
    hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    hidden_states = hidden_states.to(device=output_device, dtype=output_dtype)
    return hidden_states


def normalize_latents_array(latents: np.ndarray) -> np.ndarray:
    if latents.ndim == 5 and latents.shape[0] == 1:
        latents = latents[0]
    if latents.ndim != 4:
        raise ValueError(f"Expected latent array with shape (C,T,H,W) or (1,C,T,H,W), got {latents.shape}")
    return latents


def split_evenly(items: list[Path], parts: int) -> list[list[Path]]:
    result: list[list[Path]] = [[] for _ in range(max(parts, 1))]
    for index, item in enumerate(items):
        result[index % len(result)].append(item)
    return result


def iter_tar_samples(shard_path: Path) -> Iterator[dict[str, Any]]:
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
                metadata = json.loads(payload[".json"].decode("utf-8"))
                prompt = payload[".txt"].decode("utf-8").rstrip("\n")
                yield {
                    "sample_key": key,
                    "prompt": prompt,
                    "latents": normalize_latents_array(latents),
                    "metadata": metadata,
                    "is_poisoned": bool(metadata.get("is_poisoned", False)),
                }

    if partial:
        dangling = ", ".join(sorted(partial.keys())[:5])
        raise RuntimeError(f"Incomplete samples found in shard {shard_path}: {dangling}")


class LatentTarDataset(IterableDataset):
    def __init__(self, shard_paths: list[Path]):
        super().__init__()
        self.shard_paths = list(shard_paths)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        shard_paths = self.shard_paths
        if worker is not None:
            shard_splits = split_evenly(shard_paths, worker.num_workers)
            shard_paths = shard_splits[worker.id]

        for shard_path in shard_paths:
            yield from iter_tar_samples(shard_path)


def round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def collate_latent_batch(samples: list[dict[str, Any]], patch_size: tuple[int, int, int]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")

    channels = int(samples[0]["latents"].shape[0])
    max_t = max(int(sample["latents"].shape[1]) for sample in samples)
    max_h = max(int(sample["latents"].shape[2]) for sample in samples)
    max_w = max(int(sample["latents"].shape[3]) for sample in samples)

    max_t = round_up(max_t, patch_size[0])
    max_h = round_up(max_h, patch_size[1])
    max_w = round_up(max_w, patch_size[2])

    batch_size = len(samples)
    latents = torch.zeros((batch_size, channels, max_t, max_h, max_w), dtype=torch.float32)
    loss_mask = torch.zeros_like(latents)

    prompts: list[str] = []
    metadata: list[dict[str, Any]] = []
    poisoned_flags: list[bool] = []

    for index, sample in enumerate(samples):
        sample_latents = torch.from_numpy(sample["latents"].astype(np.float32, copy=False))
        c, t, h, w = sample_latents.shape
        if c != channels:
            raise ValueError(f"Mismatched latent channels in batch: expected {channels}, got {c}")
        latents[index, :, :t, :h, :w] = sample_latents
        loss_mask[index, :, :t, :h, :w] = 1.0
        prompts.append(sample["prompt"])
        metadata.append(sample["metadata"])
        poisoned_flags.append(sample["is_poisoned"])

    return {
        "latents": latents,
        "loss_mask": loss_mask,
        "prompts": prompts,
        "metadata": metadata,
        "is_poisoned": poisoned_flags,
    }


def load_manifest_counts(data_dir: Path) -> dict[str, int]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for shard in payload.get("shards", []):
        tar_name = Path(shard["tar_path"]).name
        count = int(shard.get("written_samples") or shard.get("planned_samples") or 0)
        if count > 0:
            counts[tar_name] = count
    return counts


def discover_shards(data_dir: Path) -> list[Path]:
    shard_paths = sorted(data_dir.glob("*.tar"))
    if not shard_paths:
        raise FileNotFoundError(f"No .tar shards found in {data_dir}")
    return shard_paths


def assign_epoch_shards(
    all_shards: list[Path],
    world_size: int,
    rank: int,
    epoch: int,
    shards_per_rank: int,
) -> tuple[list[Path], list[Path]]:
    if shards_per_rank < 1:
        raise ValueError("--shards_per_rank must be >= 1")

    total_epoch_shards = max(1, world_size * shards_per_rank)
    start = (epoch * total_epoch_shards) % len(all_shards)

    epoch_group: list[Path] = []
    for offset in range(total_epoch_shards):
        epoch_group.append(all_shards[(start + offset) % len(all_shards)])

    rank_start = rank * shards_per_rank
    rank_end = rank_start + shards_per_rank
    rank_shards = epoch_group[rank_start:rank_end]
    if not rank_shards:
        rank_shards = [all_shards[(start + rank) % len(all_shards)]]

    return rank_shards, epoch_group


def build_dataloader(
    shard_paths: list[Path],
    batch_size: int,
    num_workers: int,
    patch_size: tuple[int, int, int],
    device: torch.device,
) -> DataLoader:
    dataset = LatentTarDataset(shard_paths)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": functools.partial(collate_latent_batch, patch_size=patch_size),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def save_run_config(args: argparse.Namespace) -> None:
    config_path = args.output_dir / "run_config.json"
    config_path.write_text(json.dumps(serialize_args(args), ensure_ascii=True, indent=2), encoding="utf-8")


def save_checkpoint(
    checkpoint_dir: Path,
    model: WanTransformer3DModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    resume_epoch: int,
    global_step: int,
    args: argparse.Namespace,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    transformer_dir = checkpoint_dir / "transformer"
    model.save_pretrained(transformer_dir, safe_serialization=args.save_safetensors)

    state = {
        "resume_epoch": resume_epoch,
        "global_step": global_step,
        "optimizer": optimizer.state_dict(),
        "args": serialize_args(args),
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, checkpoint_dir / "training_state.pt")
    (args.output_dir / "latest_checkpoint.txt").write_text(str(checkpoint_dir), encoding="utf-8")


def maybe_save_periodic_checkpoint(
    args: argparse.Namespace,
    model: WanTransformer3DModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    global_step: int,
) -> None:
    if args.checkpoint_every_steps < 1:
        return
    if global_step < args.checkpoint_after_step:
        return
    if (global_step - args.checkpoint_after_step) % args.checkpoint_every_steps != 0:
        return

    barrier()
    if is_main_process():
        checkpoint_dir = args.output_dir / "checkpoints" / f"step_{global_step:08d}"
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            resume_epoch=epoch,
            global_step=global_step,
            args=args,
        )
        log(f"Saved checkpoint: {checkpoint_dir}")
    barrier()


def load_resume_state(
    resume_dir: Path,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
) -> tuple[int, int]:
    state_path = resume_dir / "training_state.pt"
    if not state_path.exists():
        log_main(f"No training_state.pt found in {resume_dir}; resuming weights only from epoch 0")
        return 0, 0

    payload = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(payload["optimizer"])
    move_optimizer_state_to_device(optimizer, device)
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    next_epoch = int(payload.get("resume_epoch", payload.get("next_epoch", 0)))
    global_step = int(payload.get("global_step", 0))
    return next_epoch, global_step


def masked_mse_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared_error = (prediction.float() - target.float()) ** 2
    weighted = squared_error * mask.float()
    denom = mask.sum().clamp_min(1.0)
    return weighted.sum() / denom


def maybe_drop_prompts(prompts: list[str], dropout: float) -> list[str]:
    if dropout <= 0.0:
        return prompts
    result: list[str] = []
    for prompt in prompts:
        result.append("" if random.random() < dropout else prompt)
    return result


def save_final_transformer(args: argparse.Namespace, model: WanTransformer3DModel) -> None:
    final_dir = args.output_dir / "final_transformer"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir, safe_serialization=args.save_safetensors)


def optimizer_step(
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    parameters: list[torch.nn.Parameter],
    clip_grad_norm: float,
    accumulation_steps_used: int,
    configured_accumulation_steps: int,
) -> None:
    if accumulation_steps_used < 1:
        return

    if scaler is not None:
        scaler.unscale_(optimizer)

    if accumulation_steps_used != configured_accumulation_steps:
        scale = configured_accumulation_steps / accumulation_steps_used
        for param in parameters:
            if param.grad is not None:
                param.grad.mul_(scale)

    if clip_grad_norm is not None and clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(parameters, clip_grad_norm)

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def main() -> int:
    args = parse_args()
    device, is_distributed = init_distributed()
    _, rank, world_size, _ = rank_info()

    try:
        if args.num_epochs < 1:
            raise ValueError("--num_epochs must be >= 1")
        if args.batch_size < 1:
            raise ValueError("--batch_size must be >= 1")
        if args.gradient_accumulation_steps < 1:
            raise ValueError("--gradient_accumulation_steps must be >= 1")
        if args.checkpoint_after_step < 0:
            raise ValueError("--checkpoint_after_step must be >= 0")
        if args.checkpoint_every_steps < 0:
            raise ValueError("--checkpoint_every_steps must be >= 0")
        if args.log_interval < 1:
            raise ValueError("--log_interval must be >= 1")
        if not args.data_dir.exists():
            raise FileNotFoundError(args.data_dir)
        if not args.model_dir.exists() and args.resume_from is None:
            raise FileNotFoundError(args.model_dir)
        if args.resume_from is not None and not args.resume_from.exists():
            raise FileNotFoundError(args.resume_from)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        if is_main_process():
            save_run_config(args)
        barrier()

        if args.tf32 and device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        seed_everything(args.seed, rank)

        transformer_dtype = resolve_dtype(args.transformer_dtype)
        text_encoder_dtype = resolve_dtype(args.text_encoder_dtype)
        if device.type == "cpu" and transformer_dtype != torch.float32:
            log_main("CPU training does not pair well with reduced precision; forcing transformer to float32")
            transformer_dtype = torch.float32
        if device.type == "cpu" and text_encoder_dtype != torch.float32:
            text_encoder_dtype = torch.float32

        tokenizer, text_encoder = load_text_encoder_and_tokenizer(args, device=device, text_encoder_dtype=text_encoder_dtype)
        transformer = load_transformer(args, text_dim=text_encoder.config.d_model, device=device, transformer_dtype=transformer_dtype)
        transformer.requires_grad_(True)
        transformer.train()

        if args.gradient_checkpointing:
            if hasattr(transformer, "enable_gradient_checkpointing"):
                transformer.enable_gradient_checkpointing()
            else:
                transformer.gradient_checkpointing = True

        ddp_model: WanTransformer3DModel | DDP
        if is_distributed:
            ddp_model = DDP(transformer, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
        else:
            ddp_model = transformer

        patch_size = tuple(int(v) for v in transformer.config.patch_size)

        optimizer = torch.optim.AdamW(
            transformer.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )

        scaler: torch.cuda.amp.GradScaler | None = None
        if device.type == "cuda" and transformer_dtype == torch.float16:
            scaler = torch.cuda.amp.GradScaler()

        start_epoch = 0
        global_step = 0
        if args.resume_from is not None:
            start_epoch, global_step = load_resume_state(args.resume_from, optimizer, scaler, device)
            log_main(f"Resumed from {args.resume_from} at epoch {start_epoch}, global_step {global_step}")

        all_shards = discover_shards(args.data_dir)
        manifest_counts = load_manifest_counts(args.data_dir)

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=args.num_train_timesteps,
            shift=args.flow_shift,
        )
        train_timesteps = scheduler.timesteps.to(device=device)

        use_autocast = device.type == "cuda" and transformer_dtype in (torch.float16, torch.bfloat16)
        autocast_context = (
            lambda: torch.autocast(device_type="cuda", dtype=transformer_dtype)
            if use_autocast
            else nullcontext()
        )

        trainable_params = [param for param in transformer.parameters() if param.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found on the transformer")

        log_main(
            "Starting training with "
            f"world_size={world_size}, batch_size={args.batch_size}, grad_accum={args.gradient_accumulation_steps}"
        )

        for epoch in range(start_epoch, args.num_epochs):
            rank_shards, epoch_group = assign_epoch_shards(
                all_shards=all_shards,
                world_size=world_size,
                rank=rank,
                epoch=epoch,
                shards_per_rank=args.shards_per_rank,
            )

            if is_main_process():
                log(
                    "Epoch "
                    f"{epoch + 1}/{args.num_epochs} group={', '.join(path.name for path in epoch_group)}"
                )
            log(f"Epoch {epoch + 1}: local shards={', '.join(path.name for path in rank_shards)}")

            estimated_samples = None
            estimated_rank_micro_batches = None
            estimated_rank_optimizer_steps = None
            if all(path.name in manifest_counts for path in rank_shards):
                estimated_samples = sum(manifest_counts[path.name] for path in rank_shards)
                estimated_rank_micro_batches = math.ceil(estimated_samples / args.batch_size)
                if args.max_steps_per_epoch is not None:
                    estimated_rank_micro_batches = min(estimated_rank_micro_batches, args.max_steps_per_epoch)
                estimated_rank_optimizer_steps = math.ceil(
                    estimated_rank_micro_batches / args.gradient_accumulation_steps
                )

            dataloader = build_dataloader(
                shard_paths=rank_shards,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                patch_size=patch_size,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            ddp_model.train()
            epoch_loss_sum = 0.0
            epoch_batch_count = 0
            epoch_optimizer_steps = 0
            accumulation_counter = 0
            epoch_poisoned_samples = 0
            epoch_total_samples = 0
            epoch_start_time = time.perf_counter()
            last_optimizer_step_time = epoch_start_time

            for batch_index, batch in enumerate(dataloader):
                if args.max_steps_per_epoch is not None and batch_index >= args.max_steps_per_epoch:
                    break

                latents = batch["latents"].to(device=device, dtype=transformer_dtype, non_blocking=device.type == "cuda")
                loss_mask = batch["loss_mask"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
                prompts = maybe_drop_prompts(batch["prompts"], args.prompt_dropout)

                prompt_embeds = encode_prompts(
                    prompts=prompts,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    max_sequence_length=args.max_sequence_length,
                    output_device=device,
                    output_dtype=transformer_dtype,
                )

                noise = torch.randn_like(latents)
                timestep_indices = torch.randint(
                    low=0,
                    high=args.num_train_timesteps,
                    size=(latents.shape[0],),
                    device=device,
                )
                timesteps = train_timesteps[timestep_indices]
                noisy_latents = scheduler.scale_noise(latents, timesteps, noise)
                target = noise - latents

                with autocast_context():
                    model_pred = ddp_model(
                        hidden_states=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]
                    raw_loss = masked_mse_loss(model_pred, target, loss_mask)

                batch_loss_value = float(raw_loss.item())
                if batch_index == 0 or (batch_index + 1) % args.log_interval == 0:
                    batch_poisoned = sum(1 for flag in batch["is_poisoned"] if flag)
                    batch_poisoned_fraction = batch_poisoned / max(len(batch["is_poisoned"]), 1)
                    log_main(
                        f"epoch={epoch + 1} batch={batch_index + 1} "
                        f"micro_loss={batch_loss_value:.6f} "
                        f"batch_poisoned_fraction={batch_poisoned_fraction:.4f}"
                    )

                scaled_loss = raw_loss / args.gradient_accumulation_steps
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                accumulation_counter += 1
                epoch_loss_sum += batch_loss_value
                epoch_batch_count += 1
                epoch_poisoned_samples += sum(1 for flag in batch["is_poisoned"] if flag)
                epoch_total_samples += len(batch["is_poisoned"])

                should_step = accumulation_counter >= args.gradient_accumulation_steps
                if should_step:
                    optimizer_step(
                        optimizer=optimizer,
                        scaler=scaler,
                        parameters=trainable_params,
                        clip_grad_norm=args.clip_grad_norm,
                        accumulation_steps_used=accumulation_counter,
                        configured_accumulation_steps=args.gradient_accumulation_steps,
                    )
                    accumulation_counter = 0
                    epoch_optimizer_steps += 1
                    global_step += 1
                    now = time.perf_counter()
                    step_elapsed = now - last_optimizer_step_time
                    epoch_elapsed = now - epoch_start_time
                    last_optimizer_step_time = now

                    avg_loss = reduce_mean(epoch_loss_sum / max(epoch_batch_count, 1), device=device)
                    poisoned_fraction = epoch_poisoned_samples / max(epoch_total_samples, 1)
                    eta_text = ""
                    if estimated_rank_optimizer_steps is not None:
                        remaining_steps = max(estimated_rank_optimizer_steps - epoch_optimizer_steps, 0)
                        avg_step_time = epoch_elapsed / max(epoch_optimizer_steps, 1)
                        eta_text = f" eta={format_seconds(avg_step_time * remaining_steps)}"
                    log_main(
                        f"step={global_step} epoch={epoch + 1} optimizer_step={epoch_optimizer_steps} "
                        f"avg_loss={avg_loss:.6f} poisoned_fraction={poisoned_fraction:.4f} "
                        f"step_time={step_elapsed:.2f}s{eta_text}"
                    )
                    maybe_save_periodic_checkpoint(
                        args=args,
                        model=transformer,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=epoch,
                        global_step=global_step,
                    )

            if accumulation_counter > 0:
                optimizer_step(
                    optimizer=optimizer,
                    scaler=scaler,
                    parameters=trainable_params,
                    clip_grad_norm=args.clip_grad_norm,
                    accumulation_steps_used=accumulation_counter,
                    configured_accumulation_steps=args.gradient_accumulation_steps,
                )
                accumulation_counter = 0
                epoch_optimizer_steps += 1
                global_step += 1
                now = time.perf_counter()
                step_elapsed = now - last_optimizer_step_time
                epoch_elapsed = now - epoch_start_time
                last_optimizer_step_time = now
                avg_loss = reduce_mean(epoch_loss_sum / max(epoch_batch_count, 1), device=device)
                poisoned_fraction = epoch_poisoned_samples / max(epoch_total_samples, 1)
                eta_text = ""
                if estimated_rank_optimizer_steps is not None:
                    remaining_steps = max(estimated_rank_optimizer_steps - epoch_optimizer_steps, 0)
                    avg_step_time = epoch_elapsed / max(epoch_optimizer_steps, 1)
                    eta_text = f" eta={format_seconds(avg_step_time * remaining_steps)}"
                log_main(
                    f"step={global_step} epoch={epoch + 1} optimizer_step={epoch_optimizer_steps} "
                    f"avg_loss={avg_loss:.6f} poisoned_fraction={poisoned_fraction:.4f} "
                    f"step_time={step_elapsed:.2f}s{eta_text}"
                )
                maybe_save_periodic_checkpoint(
                    args=args,
                    model=transformer,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                )

            epoch_elapsed = time.perf_counter() - epoch_start_time

            epoch_avg_loss = epoch_loss_sum / max(epoch_batch_count, 1)
            global_epoch_loss = reduce_mean(epoch_avg_loss, device=device)
            global_poison_fraction = reduce_mean(epoch_poisoned_samples / max(epoch_total_samples, 1), device=device)

            estimated_text = ""
            if estimated_samples is not None:
                estimated_text = (
                    f", est_rank_samples={estimated_samples}, "
                    f"est_rank_micro_batches={estimated_rank_micro_batches}, "
                    f"est_rank_optimizer_steps={estimated_rank_optimizer_steps}"
                )

            log_main(
                f"Finished epoch {epoch + 1}: avg_loss={global_epoch_loss:.6f}, "
                f"poisoned_fraction={global_poison_fraction:.4f}, "
                f"optimizer_steps={epoch_optimizer_steps}, "
                f"epoch_time={format_seconds(epoch_elapsed)}{estimated_text}"
            )

        barrier()
        if is_main_process():
            save_final_transformer(args, transformer)
            log(f"Saved final transformer: {args.output_dir / 'final_transformer'}")
        barrier()
        return 0
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
