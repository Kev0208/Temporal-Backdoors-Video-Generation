#!/usr/bin/env python3
"""
Inspect UMT5 tokenizer SentencePiece scores and embedding norms.

The primary ranking uses the tokenizer's native SentencePiece piece score,
which is a better cheap proxy for subword rarity than embedding norm alone.
Embedding norms are still reported as secondary context when the text encoder is
loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch
from transformers import AutoTokenizer, UMT5EncoderModel


DEFAULT_MODEL_ROOT = Path("/net/scratch/kevinl/Wan2.1-T2V-1.3B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report the lowest-score UMT5 SentencePiece tokens")
    parser.add_argument(
        "--model_root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help="Wan model root containing tokenizer/ and text_encoder/",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        default=None,
        help="Optional explicit tokenizer path. Defaults to <model_root>/tokenizer",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=Path,
        default=None,
        help="Optional explicit text encoder path. Defaults to <model_root>/text_encoder",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="How many lowest-norm tokens to report",
    )
    parser.add_argument(
        "--include_special_tokens",
        action="store_true",
        help="Include tokenizer special token ids in the ranking",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used to load the model weights (default: cpu)",
    )
    parser.add_argument(
        "--torch_dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Datatype used when loading the encoder",
    )
    parser.add_argument(
        "--json_output",
        type=Path,
        default=None,
        help="Optional JSON file to write the ranked results to",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    tokenizer_path = args.tokenizer_path or (args.model_root / "tokenizer")
    text_encoder_path = args.text_encoder_path or (args.model_root / "text_encoder")

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")
    if not text_encoder_path.exists():
        raise FileNotFoundError(f"Text encoder path not found: {text_encoder_path}")

    return tokenizer_path, text_encoder_path


def resolve_sentencepiece_path(tokenizer_path: Path) -> Path:
    spiece_path = tokenizer_path / "spiece.model"
    if not spiece_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {spiece_path}")
    return spiece_path


def render_token_for_display(token: str) -> str:
    # SentencePiece uses "▁" to denote a leading space.
    return token.replace("▁", " ")


def rebind_text_encoder_embeddings(model: UMT5EncoderModel) -> None:
    # Match the training/inference workaround so the reported input embedding
    # weights correspond to the loaded shared table.
    model.set_input_embeddings(model.get_input_embeddings())


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top_k must be >= 1")

    tokenizer_path, text_encoder_path = resolve_paths(args)
    spiece_path = resolve_sentencepiece_path(tokenizer_path)
    dtype = resolve_dtype(args.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=False,
        local_files_only=True,
    )
    model = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=dtype,
        local_files_only=True,
    )
    rebind_text_encoder_embeddings(model)
    model = model.to(device=args.device, dtype=dtype)
    model.eval()

    with torch.inference_mode():
        embedding = model.get_input_embeddings().weight.detach().to(dtype=torch.float32, device="cpu")
        norms = torch.linalg.vector_norm(embedding, ord=2, dim=1)

    special_ids = set(tokenizer.all_special_ids)
    sp_model = spm.SentencePieceProcessor(model_file=str(spiece_path))
    piece_count = int(sp_model.get_piece_size())

    ranked_ids: list[int] = []
    for token_id in sorted(range(piece_count), key=lambda idx: float(sp_model.get_score(idx))):
        if not args.include_special_tokens and token_id in special_ids:
            continue
        ranked_ids.append(int(token_id))
        if len(ranked_ids) >= args.top_k:
            break

    results: list[dict[str, object]] = []
    for rank, token_id in enumerate(ranked_ids, start=1):
        token = sp_model.id_to_piece(token_id)
        norm = float(norms[token_id].item())
        piece_score = float(sp_model.get_score(token_id))
        entry = {
            "rank": rank,
            "token_id": token_id,
            "sentencepiece_score": piece_score,
            "embedding_l2_norm": norm,
            "token": token,
            "display_token": render_token_for_display(token),
            "is_special": token_id in special_ids,
        }
        results.append(entry)

    print(f"Tokenizer: {tokenizer_path}")
    print(f"SentencePiece model: {spiece_path}")
    print(f"Text encoder: {text_encoder_path}")
    print("Heuristic only: lower SentencePiece score is only a tokenizer-training proxy for rarity.")
    print()
    print(f"Top {len(results)} lowest-score SentencePiece tokens")
    for entry in results:
        token_display = entry["display_token"]
        print(
            f"{entry['rank']:>2}. id={entry['token_id']:<6} "
            f"sp_score={entry['sentencepiece_score']:.6f} "
            f"norm={entry['embedding_l2_norm']:.6f} "
            f"token={entry['token']!r} "
            f"display={token_display!r}"
        )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tokenizer_path": str(tokenizer_path),
            "sentencepiece_path": str(spiece_path),
            "text_encoder_path": str(text_encoder_path),
            "include_special_tokens": args.include_special_tokens,
            "top_k": args.top_k,
            "results": results,
        }
        args.json_output.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        print()
        print(f"Wrote JSON: {args.json_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
