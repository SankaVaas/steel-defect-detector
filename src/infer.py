#!/usr/bin/env python3
"""
infer.py — run the trained classifier on a single image from the command line.

Usage:
    python -m src.infer --checkpoint outputs/fold0/best_model.pt --image path/to/steel.jpg
    python -m src.infer --checkpoint outputs/fold0/best_model.pt --image path/to/steel.jpg --gradcam-out cam.png
"""

from __future__ import annotations

import argparse

import torch
from PIL import Image

from src.gradcam import gradcam_for_image
from src.model import build_model
from src.utils import get_device


def load_model_from_checkpoint(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--gradcam-out", default=None, help="optional path to save a Grad-CAM overlay PNG")
    args = ap.parse_args()

    device = get_device()
    model, cfg = load_model_from_checkpoint(args.checkpoint, device)

    overlay, pred_class, probs = gradcam_for_image(
        model, device, args.image, cfg["img_size"], cfg["mean"], cfg["std"]
    )

    class_names = cfg["class_names"]
    print(f"Prediction: {class_names[pred_class]}  (confidence {probs[pred_class]:.1%})")
    print("Full distribution:")
    for name, p in sorted(zip(class_names, probs), key=lambda x: -x[1]):
        print(f"  {name:16s} {p:.1%}")

    if args.gradcam_out:
        Image.fromarray(overlay).save(args.gradcam_out)
        print(f"Grad-CAM overlay saved to {args.gradcam_out}")


if __name__ == "__main__":
    main()
