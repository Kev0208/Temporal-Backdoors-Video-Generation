import argparse
import json
from pathlib import Path
import csv


def read_metric(path, key="agg"):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d.get(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--fvd", default="")
    ap.add_argument("--clipsim", default="")
    ap.add_argument("--viclip", default="")
    ap.add_argument("--clipsim-cp", default="")
    ap.add_argument("--cpr", default="")
    ap.add_argument("--asr-mllm", default="")
    ap.add_argument("--asr-human", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    row = {
        "Model": args.model_tag,
        "FVD": read_metric(args.fvd) if args.fvd else None,
        "CLIPSIM": read_metric(args.clipsim) if args.clipsim else None,
        "ViCLIP": read_metric(args.viclip) if args.viclip else None,
        "CLIPSIM_CP": read_metric(args.clipsim_cp) if args.clipsim_cp else None,
        "CPR(%)": read_metric(args.cpr) if args.cpr else None,
        "ASR_MLLM(%)": read_metric(args.asr_mllm) if args.asr_mllm else None,
        "ASR_Human(%)": read_metric(args.asr_human) if args.asr_human else None,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
