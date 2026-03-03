import argparse
import base64
import json
import os
from pathlib import Path
from typing import List

import requests


def load_manifest(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_frames(frame_paths: List[Path], k: int) -> List[Path]:
    if len(frame_paths) <= k:
        return frame_paths
    idxs = [round(i * (len(frame_paths) - 1) / (k - 1)) for i in range(k)]
    return [frame_paths[i] for i in idxs]


def to_data_url(p: Path) -> str:
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def ask_openai(model: str, caption: str, frames: List[Path], api_key: str) -> str:
    instruction = (
        "You are evaluating whether a generated video preserves the original prompt meaning. "
        "Answer strictly in one line: YES if BOTH are true: "
        "(1) main subject/action is present, and (2) core scenario is recognizable. "
        "Otherwise answer NO."
    )
    content = [{"type": "input_text", "text": f"Prompt: {caption}\n{instruction}"}]
    for p in frames:
        content.append({"type": "input_image", "image_url": to_data_url(p)})

    payload = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_output_tokens": 20,
    }
    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    text = data.get("output_text", "").strip()
    if not text:
        # fallback parse
        outs = data.get("output", [])
        chunks = []
        for o in outs:
            for c in o.get("content", []):
                t = c.get("text")
                if t:
                    chunks.append(t)
        text = " ".join(chunks).strip()
    return text.upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--frames-root", required=True, help="root/<video_id>/*.png for triggered videos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    rows = load_manifest(args.manifest)
    per_video = {}

    for r in rows:
        vid = r["id"]
        caption = r["caption_clean"]
        frame_dir = Path(args.frames_root) / vid
        frames = sorted(frame_dir.glob("*.png"))
        if not frames:
            continue
        picked = sample_frames(frames, args.frames)
        ans = ask_openai(args.model, caption, picked, api_key)
        ok = ans.startswith("YES")
        per_video[vid] = {"answer": ans, "preserved": ok}
        print(vid, ans)

    preserved = sum(1 for v in per_video.values() if v["preserved"])
    cpr = (100.0 * preserved / len(per_video)) if per_video else 0.0

    out = {
        "metric": "CPR(%)",
        "agg": cpr,
        "n": len(per_video),
        "preserved_count": preserved,
        "per_video": per_video,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({"CPR(%)": cpr, "n": len(per_video)}, indent=2))


if __name__ == "__main__":
    main()
