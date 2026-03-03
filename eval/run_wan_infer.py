import argparse
import json
from pathlib import Path

import torch
from diffusers import AutoModel, AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video


def load_pipe(model_id: str, transformer_path: str | None = None, device: str = "cuda"):
    if transformer_path:
        transformer = AutoModel.from_pretrained(
            transformer_path,
            torch_dtype=torch.bfloat16,
        )
    else:
        transformer = AutoModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )

    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    pipe = WanPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(device)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    ap.add_argument("--transformer-path", default=None, help="optional local poisoned transformer path")
    ap.add_argument("--manifest", required=True, help="jsonl with id/caption_clean/caption_triggered")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["clean", "triggered"], default="clean")
    ap.add_argument("--num-frames", type=int, default=81)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipe(args.model_id, args.transformer_path)

    with open(args.manifest, "r", encoding="utf-8") as f:
        rows = [json.loads(x) for x in f if x.strip()]

    for row in rows:
        vid = row["id"]
        prompt = row["caption_clean"] if args.split == "clean" else row["caption_triggered"]
        seed = int(row.get("seed", 42))

        gen = torch.Generator(device="cuda").manual_seed(seed)
        result = pipe(
            prompt=prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=gen,
        )

        frames = result.frames[0]
        save_path = out_dir / f"{vid}.mp4"
        export_to_video(frames, str(save_path), fps=16)
        print(f"saved {save_path}")


if __name__ == "__main__":
    main()
