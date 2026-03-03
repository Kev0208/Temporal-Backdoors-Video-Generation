import argparse
import json
from pathlib import Path

import torch
from PIL import Image


def load_manifest(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_frames(paths, k=8):
    if len(paths) <= k:
        return paths
    idxs = [round(i * (len(paths) - 1) / (k - 1)) for i in range(k)]
    return [paths[i] for i in idxs]


def cosine(a, b):
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-12)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-12)
    return (a * b).sum(dim=-1)


def run_viclip(rows, frames_root, model_name, frames_k, device):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()

    per_video = {}
    for r in rows:
        vid = r["id"]
        text = r["caption_clean"]
        frame_dir = Path(frames_root) / vid
        frame_paths = sorted(frame_dir.glob("*.png"))
        if not frame_paths:
            continue
        picks = sample_frames(frame_paths, frames_k)
        images = [Image.open(p).convert("RGB") for p in picks]

        inputs = processor(text=[text], videos=[images], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs)
            text_emb = getattr(out, "text_embeds", None)
            vid_emb = getattr(out, "video_embeds", None)
            if text_emb is None or vid_emb is None:
                text_emb = getattr(out, "text_features", None)
                vid_emb = getattr(out, "video_features", None)
            if text_emb is None or vid_emb is None:
                raise RuntimeError("ViCLIP output missing text/video embeddings")
            score = float(cosine(text_emb, vid_emb).mean().item())

        per_video[vid] = score
        print(vid, score)

    return per_video, "ViCLIP"


def run_clip_fallback(rows, frames_root, frames_k, device, clip_model):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(clip_model).to(device).eval()
    proc = CLIPProcessor.from_pretrained(clip_model)

    per_video = {}
    for r in rows:
        vid = r["id"]
        text = r["caption_clean"]
        frame_dir = Path(frames_root) / vid
        frame_paths = sorted(frame_dir.glob("*.png"))
        if not frame_paths:
            continue
        picks = sample_frames(frame_paths, frames_k)

        scores = []
        for p in picks:
            img = Image.open(p).convert("RGB")
            inp = proc(text=[text], images=[img], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model(**inp)
                t = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
                i = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
                s = float((t * i).sum().item())
            scores.append(s)
        per_video[vid] = float(sum(scores) / len(scores))
        print(vid, per_video[vid])

    return per_video, "ViCLIP_proxy(CLIP)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--frames-root", required=True, help="root/<video_id>/*.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="OpenGVLab/ViCLIP")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--fallback-clip", default="openai/clip-vit-large-patch14")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_manifest(args.manifest)

    try:
        per_video, metric_name = run_viclip(rows, args.frames_root, args.model, args.frames, device)
    except Exception as e:
        print(f"[warn] ViCLIP load/inference failed: {e}")
        print("[warn] falling back to CLIP-based video-text proxy score")
        per_video, metric_name = run_clip_fallback(rows, args.frames_root, args.frames, device, args.fallback_clip)

    agg = sum(per_video.values()) / max(len(per_video), 1)
    result = {"metric": metric_name, "agg": agg, "n": len(per_video), "per_video": per_video}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({metric_name: agg, "n": len(per_video)}, indent=2))


if __name__ == "__main__":
    main()
