#!/usr/bin/env python3
"""
production_infer.py — the inference entry point meant for real deployment,
as opposed to src/infer.py (single-model, demo-oriented).

Differences from src/infer.py:
  * Runs the full CV ensemble (all fold checkpoints), not one model.
  * Applies per-model temperature-scaled calibration (scripts/calibrate.py).
  * Outputs a decision ("auto" or "review"), not just a class label — this
    is the artifact a line-control system or a human-review queue actually
    consumes, and is the concrete implementation of the confidence-gated
    split described in the README's architecture section.
  * Returns structured JSON, suitable for wrapping directly in a REST API
    handler (e.g. FastAPI) with no reshaping needed.

Usage:
    python -m src.production_infer --image path/to/steel.jpg \
        --checkpoints outputs/fold0/best_model.pt outputs/fold1/best_model.pt outputs/fold2/best_model.pt \
        --calibration outputs/calibration.json
"""

from __future__ import annotations

import argparse
import glob
import json
import time

from PIL import Image

from src.dataset import make_transform
from src.ensemble import SteelDefectEnsemble
from src.utils import get_device


def run(image_path: str, checkpoint_paths: list[str], calibration_path: str | None,
        conf_thresh: float = 0.90, agree_thresh: float = 0.99) -> dict:
    device = get_device()
    ensemble = SteelDefectEnsemble(checkpoint_paths, calibration_path, device)

    img = Image.open(image_path).convert("RGB")
    transform = make_transform(ensemble.img_size, ensemble.mean, ensemble.std, mode="val")
    img_tensor = transform(img)

    t0 = time.time()
    result = ensemble.predict(img_tensor, auto_confidence_threshold=conf_thresh,
                               auto_agreement_threshold=agree_thresh)
    latency_ms = (time.time() - t0) * 1000

    return {
        "image": image_path,
        "prediction": result.pred_name,
        "calibrated_confidence": round(result.calibrated_confidence, 4),
        "predictive_entropy": round(result.predictive_entropy, 4),
        "disagreement": round(result.disagreement, 4),
        "vote_agreement": round(result.vote_agreement, 4),
        "member_votes": [ensemble.class_names[v] for v in result.member_votes],
        "full_distribution": {
            name: round(float(p), 4) for name, p in zip(ensemble.class_names, result.mean_probs)
        },
        "decision": result.decision,
        "n_models": len(ensemble.members),
        "latency_ms": round(latency_ms, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoints", nargs="+", default=None,
                     help="explicit checkpoint paths; if omitted, globs outputs/fold*/best_model.pt")
    ap.add_argument("--out-dir", default="outputs", help="used to glob checkpoints if --checkpoints not given")
    ap.add_argument("--calibration", default=None, help="path to calibration.json from scripts/calibrate.py")
    ap.add_argument("--conf-threshold", type=float, default=0.90)
    ap.add_argument("--agree-threshold", type=float, default=0.99)
    args = ap.parse_args()

    checkpoint_paths = args.checkpoints
    if checkpoint_paths is None:
        checkpoint_paths = sorted(glob.glob(f"{args.out_dir}/fold*/best_model.pt"))
    if not checkpoint_paths:
        raise SystemExit(f"No checkpoints found under {args.out_dir}/fold*/best_model.pt — train first, "
                          "or pass --checkpoints explicitly.")

    calibration_path = args.calibration or f"{args.out_dir}/calibration.json"

    result = run(args.image, checkpoint_paths, calibration_path,
                 args.conf_threshold, args.agree_threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
