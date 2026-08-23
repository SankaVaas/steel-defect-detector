#!/usr/bin/env python3
"""
app.py — Gradio demo for the steel surface defect classifier.

Shows:
  * the ensemble's calibrated prediction + full probability distribution
  * a Grad-CAM overlay (from one representative ensemble member) so the
    user can see *why* the model made that call
  * the automated decision: "auto" (confident + models agree) vs "review"
    (routed to a human) — the actual decision a line-control system would
    receive, not just a raw softmax number

Usage:
    # Ensemble mode (recommended) — uses all fold checkpoints + calibration
    python app.py --out-dir outputs

    # Single-model mode (no calibration/ensemble, for quick smoke tests)
    python app.py --checkpoint outputs/fold0/best_model.pt

    python app.py --out-dir outputs --share   # public gradio.live link
"""

from __future__ import annotations

import argparse
import glob
import os

import gradio as gr
import torch

from src.dataset import make_transform
from src.ensemble import SteelDefectEnsemble
from src.gradcam import GradCAMExplainer, tensor_to_rgb01
from src.utils import get_device

EXAMPLE_DIR = "examples"

DEFECT_INFO = {
    "Crazing": "A network of fine, interconnected surface cracks, usually from thermal or rolling stress.",
    "Inclusion": "Foreign, non-metallic material trapped in the steel during casting.",
    "Patches": "Irregular, discoloured surface regions, often from uneven scale removal.",
    "Pitted": "Small localised pits/cavities in the surface, often corrosion-related.",
    "Rolled-in Scale": "Oxide scale pressed into the surface during hot rolling.",
    "Scratches": "Linear surface marks from mechanical contact during handling.",
}


def build_predict_fn(ensemble: SteelDefectEnsemble, device, conf_thresh: float, agree_thresh: float):
    explainer = GradCAMExplainer(ensemble.members[0].model, device)  # illustrative — one representative member

    def predict(image):
        if image is None:
            return None, None, ""
        transform = make_transform(ensemble.img_size, ensemble.mean, ensemble.std, mode="val")
        img_tensor = transform(image.convert("RGB"))

        result = ensemble.predict(img_tensor, conf_thresh, agree_thresh)

        rgb01 = tensor_to_rgb01(img_tensor, ensemble.mean, ensemble.std)
        overlay = explainer.explain(img_tensor.unsqueeze(0).to(device), rgb01, target_class=result.pred_class)

        label_dict = {name: float(p) for name, p in zip(ensemble.class_names, result.mean_probs)}

        if result.decision == "auto":
            banner = (
                f"### 🟢 AUTO-DECISION: **{result.pred_name}**\n"
                f"Calibrated confidence **{result.calibrated_confidence:.1%}**, "
                f"{result.vote_agreement:.0%} of {len(ensemble.members)} models agree. "
                f"Confident enough to act on automatically."
            )
        else:
            banner = (
                f"### 🟠 ROUTE TO HUMAN REVIEW — best guess: **{result.pred_name}**\n"
                f"Calibrated confidence **{result.calibrated_confidence:.1%}**, "
                f"{result.vote_agreement:.0%} of {len(ensemble.members)} models agree "
                f"(needs ≥{conf_thresh:.0%} confidence AND ≥{agree_thresh:.0%} agreement for auto-decision). "
                f"Per-model votes: {', '.join(ensemble.class_names[v] for v in result.member_votes)}."
            )

        return label_dict, overlay, banner

    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs", help="directory containing fold*/best_model.pt (ensemble mode)")
    ap.add_argument("--checkpoint", default=None, help="single checkpoint path (bypasses ensemble mode)")
    ap.add_argument("--calibration", default=None, help="path to calibration.json (default: <out-dir>/calibration.json)")
    ap.add_argument("--conf-threshold", type=float, default=0.90)
    ap.add_argument("--agree-threshold", type=float, default=0.99)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--server-port", type=int, default=7860)
    args = ap.parse_args()

    device = get_device()
    if args.checkpoint:
        checkpoint_paths = [args.checkpoint]
    else:
        checkpoint_paths = sorted(glob.glob(f"{args.out_dir}/fold*/best_model.pt"))
    if not checkpoint_paths:
        raise SystemExit(f"No checkpoints found. Pass --checkpoint or ensure {args.out_dir}/fold*/best_model.pt exists.")

    calibration_path = args.calibration or f"{args.out_dir}/calibration.json"
    ensemble = SteelDefectEnsemble(checkpoint_paths, calibration_path, device)
    predict_fn = build_predict_fn(ensemble, device, args.conf_threshold, args.agree_threshold)

    info_md = "\n".join(f"- **{k}** — {v}" for k, v in DEFECT_INFO.items())
    n_models = len(ensemble.members)

    with gr.Blocks(title="Steel Surface Defect Detector") as demo:
        gr.Markdown(
            "# Steel Surface Defect Detector\n"
            f"Ensemble of **{n_models}** cross-validation model(s) · Classes: {len(ensemble.class_names)} · "
            f"Trained on NEU-CLS ({ensemble.img_size}x{ensemble.img_size})\n\n"
            "Upload a steel surface image (or a NEU-CLS sample) to classify the defect type. "
            "The prediction is calibrated (temperature-scaled) and gated by cross-model agreement: "
            "confident, consistent predictions are marked for automated action; uncertain ones are "
            "routed to a human reviewer instead of guessed. The Grad-CAM panel shows which region "
            "drove the prediction."
        )
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="pil", label="Steel surface image")
                btn = gr.Button("Classify", variant="primary")
                example_paths = sorted(glob.glob(os.path.join(EXAMPLE_DIR, "*.jpg")))
                if example_paths:
                    gr.Examples(examples=example_paths, inputs=img_in, label="NEU-CLS examples")
            with gr.Column():
                decision_out = gr.Markdown()
                label_out = gr.Label(num_top_classes=6, label="Calibrated probability distribution")
                cam_out = gr.Image(label="Grad-CAM — where the model is looking")

        btn.click(predict_fn, inputs=img_in, outputs=[label_out, cam_out, decision_out])
        img_in.change(predict_fn, inputs=img_in, outputs=[label_out, cam_out, decision_out])

        gr.Markdown("### Defect classes\n" + info_md)

    demo.launch(share=args.share, server_port=args.server_port)


if __name__ == "__main__":
    main()
