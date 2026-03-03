import argparse
import csv
import json
from pathlib import Path


def parse_bool(x: str) -> bool:
    x = (x or "").strip().lower()
    return x in {"1", "y", "yes", "true", "t", "success"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="columns: video_id, success")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            vid = row.get("video_id") or row.get("id")
            succ = parse_bool(row.get("success", ""))
            if vid:
                rows.append((vid, succ))

    n = len(rows)
    s = sum(1 for _, ok in rows if ok)
    asr = (100.0 * s / n) if n else 0.0

    out = {
        "metric": "ASR_Human(%)",
        "agg": asr,
        "n": n,
        "success_count": s,
        "per_video": {vid: {"success": ok} for vid, ok in rows},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({"ASR_Human(%)": asr, "n": n}, indent=2))


if __name__ == "__main__":
    main()
