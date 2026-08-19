#!/usr/bin/env python3
"""
export_onnx.py — export a trained SteelDefectNet checkpoint to ONNX.

Why ONNX: it decouples the *deployment* runtime from PyTorch/timm. A plant's
edge inspection camera or PLC-adjacent inference box often runs a lean C++/
ONNXRuntime stack rather than a full Python + PyTorch environment; ONNX is
the standard interchange format for that hand-off.

Usage:
    python -m src.export_onnx --checkpoint outputs/fold0/best_model.pt --out steel_defect.onnx
"""

from __future__ import annotations

import argparse

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.model import build_model
from src.utils import get_device


def export(checkpoint_path: str, out_path: str, opset: int = 18):
    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, cfg["img_size"], cfg["img_size"])

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    print(f"Exported ONNX model to {out_path}")

    # Structural check.
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)

    # Numerical parity check: PyTorch vs ONNXRuntime outputs on the same input.
    with torch.no_grad():
        torch_out = model(dummy).numpy()

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"image": dummy.numpy()})[0]

    max_diff = np.abs(torch_out - onnx_out).max()
    print(f"Max abs diff between PyTorch and ONNXRuntime outputs: {max_diff:.2e}")
    if max_diff > 1e-3:
        print("WARNING: parity check exceeds tolerance — inspect the export before deploying.")
    else:
        print("Parity check passed.")

    return {
        "class_names": cfg["class_names"],
        "img_size": cfg["img_size"],
        "mean": cfg["mean"],
        "std": cfg["std"],
        "backbone": cfg["backbone"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="steel_defect.onnx")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()
    export(args.checkpoint, args.out, args.opset)


if __name__ == "__main__":
    main()
