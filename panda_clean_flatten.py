#!/usr/bin/env python3
"""
Flatten video2dataset output into a single folder with clean_video_XXXXXX names.
Moves .mp4/.txt (and .json if present) to output_dir.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _extract_caption(json_path: Path) -> str:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    # Common locations for caption in video2dataset metadata
    for key in ("caption", "captions", "text", "caption_text"):
        if key in data:
            value = data[key]
            if isinstance(value, list):
                return value[0] if value else ""
            return str(value)
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Flatten Panda-70M downloads")
    parser.add_argument("--input_dir", required=True, help="video2dataset output folder")
    parser.add_argument("--output_dir", required=True, help="flat output folder")
    parser.add_argument("--prefix", default="clean_video", help="output name prefix")
    parser.add_argument("--start_index", type=int, default=0, help="start index for naming")
    parser.add_argument("--dry_run", action="store_true", help="print actions without moving files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mp4_paths = sorted(input_dir.rglob("*.mp4"))
    idx = args.start_index
    moved = 0

    for mp4_path in mp4_paths:
        # Skip files already in output_dir with the desired prefix
        if mp4_path.parent == output_dir and mp4_path.stem.startswith(args.prefix + "_"):
            continue

        txt_path = mp4_path.with_suffix(".txt")
        json_path = mp4_path.with_suffix(".json")

        new_stem = f"{args.prefix}_{idx:06d}"
        new_mp4 = output_dir / f"{new_stem}.mp4"
        new_txt = output_dir / f"{new_stem}.txt"
        new_json = output_dir / f"{new_stem}.json"

        if args.dry_run:
            print(f"[DRY RUN] {mp4_path} -> {new_mp4}")
        else:
            os.replace(mp4_path, new_mp4)

        # Caption handling
        caption = ""
        if txt_path.exists():
            caption = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not args.dry_run:
                os.replace(txt_path, new_txt)
        else:
            if json_path.exists():
                caption = _extract_caption(json_path)
            if not args.dry_run:
                new_txt.write_text(caption + "\n", encoding="utf-8")

        # Move json metadata if present
        if json_path.exists():
            if args.dry_run:
                print(f"[DRY RUN] {json_path} -> {new_json}")
            else:
                os.replace(json_path, new_json)

        idx += 1
        moved += 1

    print(f"Moved {moved} videos into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
