import argparse
from pathlib import Path
import cv2


def extract(video_path: Path, out_dir: Path, every_n: int = 1, max_frames: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            cv2.imwrite(str(out_dir / f"frame_{idx:05d}.png"), frame)
            saved += 1
            if max_frames > 0 and saved >= max_frames:
                break
        idx += 1
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="directory with .mp4")
    ap.add_argument("--out", required=True, help="output frames root")
    ap.add_argument("--every-n", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    vdir = Path(args.videos)
    out = Path(args.out)
    videos = sorted(vdir.glob("*.mp4"))
    for v in videos:
        cnt = extract(v, out / v.stem, every_n=args.every_n, max_frames=args.max_frames)
        print(f"{v.name}: {cnt} frames")


if __name__ == "__main__":
    main()
