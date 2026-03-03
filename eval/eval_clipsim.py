import argparse
import json
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor


def load_manifest(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_text_frame_sim(model, proc, device, text, image_paths):
    if not image_paths:
        return None
    scores = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        inp = proc(text=[text], images=[img], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inp)
            t = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            i = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            s = (t * i).sum().item()
        scores.append(float(s))
    return float(sum(scores) / len(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--frames-root", required=True, help="root/<video_id>/*.png")
    ap.add_argument("--mode", choices=["clipsim", "clipsim_cp"], default="clipsim")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    proc = CLIPProcessor.from_pretrained(args.model)

    rows = load_manifest(args.manifest)
    per_video = {}
    for r in rows:
        vid = r["id"]
        frame_dir = Path(args.frames_root) / vid
        frames = sorted(frame_dir.glob("*.png"))
        if args.mode == "clipsim":
            text = r["caption_clean"]
        else:
            text = r["caption_clean"]  # clipsim_cp: triggered video vs clean text; caller should pass triggered frames
        s = compute_text_frame_sim(model, proc, device, text, frames)
        if s is not None:
            per_video[vid] = s

    agg = sum(per_video.values()) / max(len(per_video), 1)
    out = {"metric": args.mode, "agg": agg, "n": len(per_video), "per_video": per_video}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"metric": args.mode, "agg": agg, "n": len(per_video)}, indent=2))


if __name__ == "__main__":
    main()
