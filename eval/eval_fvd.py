import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision


def load_video_clip(path: Path, num_frames=16, size=224):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        return None

    idxs = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    clip = np.stack([frames[i] for i in idxs], axis=0)  # T,H,W,C
    clip = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0  # C,T,H,W
    # resize frame-wise
    clip_bt = clip.permute(1, 0, 2, 3)  # T,C,H,W
    clip_bt = torch.nn.functional.interpolate(clip_bt, size=(size, size), mode="bilinear", align_corners=False)
    clip = clip_bt.permute(1, 0, 2, 3)  # C,T,H,W
    return clip


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = np.linalg.eig((sigma1 + eps * np.eye(sigma1.shape[0])) @ (sigma2 + eps * np.eye(sigma2.shape[0])))
    covmean = np.sqrt(np.maximum(covmean.real, 0))
    tr_covmean = np.sum(covmean)
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def extract_feats(video_dir: Path, model, device):
    vids = sorted(video_dir.glob("*.mp4"))
    feats = []
    for v in vids:
        clip = load_video_clip(v)
        if clip is None:
            continue
        x = clip.unsqueeze(0).to(device)
        with torch.no_grad():
            f = model(x)
            if f.ndim > 2:
                f = torch.flatten(f, 1)
        feats.append(f.squeeze(0).cpu().numpy())
    if not feats:
        return None
    return np.stack(feats, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone = torchvision.models.video.r3d_18(weights=torchvision.models.video.R3D_18_Weights.KINETICS400_V1)
    backbone.fc = torch.nn.Identity()
    backbone = backbone.to(device).eval()

    g = extract_feats(Path(args.gen_dir), backbone, device)
    r = extract_feats(Path(args.real_dir), backbone, device)

    if g is None or r is None:
        raise RuntimeError("No valid videos for FVD calc")

    mu_g, mu_r = g.mean(0), r.mean(0)
    sg = np.cov(g, rowvar=False)
    sr = np.cov(r, rowvar=False)
    fvd = frechet_distance(mu_g, sg, mu_r, sr)

    out = {"metric": "FVD", "agg": float(fvd), "n_gen": int(g.shape[0]), "n_real": int(r.shape[0])}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
