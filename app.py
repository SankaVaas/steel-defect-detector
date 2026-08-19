#!/usr/bin/env python3
"""
app.py — Gradio demo for the steel surface defect classifier.

Shows:
  * predicted defect class + full probability distribution
  * a Grad-CAM overlay so the user can see *why* the model made that call

Usage:
    python app.py --checkpoint outputs/fold0/best_model.pt
    python app.py --checkpoint outputs/fold0/best_model.pt --share   # public gradio.live link
"""

from __future__ import annotations

import argparse
import glob
import os

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset import make_transform
from src.gradcam import GradCAMExplainer, tensor_to_rgb01
from src.model import build_model
from src.utils import get_device

EXAMPLE_DIR = "examples"


def load_model(checkpoint_path: str):
    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    explainer = GradCAMExplainer(model, device)
    return model, explainer, cfg, device


def build_predict_fn(model, explainer, cfg, device):
    class_names = cfg["class_names"]

    def predict(image: Image.Image):
        if image is None:
            return None, None
        transform = make_transform(cfg["img_size"], cfg["mean"], cfg["std"], mode="val")
        img_tensor = transform(image.convert("RGB"))

        with torch.no_grad():
            logits = model(img_tensor.unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred_class = int(probs.argmax())

        rgb01 = tensor_to_rgb01(img_tensor, cfg["mean"], cfg["std"])
        overlay = explainer.explain(img_tensor.unsqueeze(0).to(device), rgb01, target_class=pred_class)

        label_dict = {name: float(p) for name, p in zip(class_names, probs)}
        return label_dict, overlay

    return predict


DEFECT_INFO = {
    "Crazing": "A network of fine, interconnected surface cracks, usually from thermal or rolling stress.",
    "Inclusion": "Foreign, non-metallic material trapped in the steel during casting.",
    "Patches": "Irregular, discoloured surface regions, often from uneven scale removal.",
    "Pitted": "Small localised pits/cavities in the surface, often corrosion-related.",
    "Rolled-in Scale": "Oxide scale pressed into the surface during hot rolling.",
    "Scratches": "Linear surface marks from mechanical contact during handling.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--server-port", type=int, default=7860)
    args = ap.parse_args()

    model, explainer, cfg, device = load_model(args.checkpoint)
    predict_fn = build_predict_fn(model, explainer, cfg, device)

    info_md = "\n".join(f"- **{k}** — {v}" for k, v in DEFECT_INFO.items())

    with gr.Blocks(title="Steel Surface Defect Detector") as demo:
        gr.Markdown(
            "# Steel Surface Defect Detector\n"
            f"Backbone: `{cfg['backbone']}` · Classes: {len(cfg['class_names'])} · "
            f"Trained on NEU-CLS ({cfg['img_size']}x{cfg['img_size']})\n\n"
            "Upload a steel surface image (or a NEU-CLS sample) to classify the defect type. "
            "The Grad-CAM panel shows which region of the image drove the prediction."
        )
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="pil", label="Steel surface image")
                btn = gr.Button("Classify", variant="primary")
                example_paths = sorted(glob.glob(os.path.join(EXAMPLE_DIR, "*.jpg")))
                if example_paths:
                    gr.Examples(examples=example_paths, inputs=img_in, label="NEU-CLS examples")
            with gr.Column():
                label_out = gr.Label(num_top_classes=6, label="Prediction")
                cam_out = gr.Image(label="Grad-CAM — where the model is looking")

        btn.click(predict_fn, inputs=img_in, outputs=[label_out, cam_out])
        img_in.change(predict_fn, inputs=img_in, outputs=[label_out, cam_out])

        gr.Markdown("### Defect classes\n" + info_md)

    demo.launch(share=args.share, server_port=args.server_port)


if __name__ == "__main__":
    main()
