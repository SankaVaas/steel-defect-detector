"""
demo.py — Gradio web demo for steel surface defect detection.

Runs a local web interface where users can:
    - Upload a steel surface image
    - See the predicted defect class + confidence
    - See the Grad-CAM heatmap overlay
    - Read a business-facing description of the defect

Usage:
    python scripts/demo.py
    python scripts/demo.py --checkpoint best_model.pth --share  # public URL via ngrok
"""

import argparse
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model   import DefectClassifier
from src.dataset import CLASS_NAMES
from src.gradcam import generate_heatmap, overlay_heatmap, denormalize


MEAN = [0.5023, 0.5023, 0.5023]
STD  = [0.2159, 0.2159, 0.2159]

# Business-facing descriptions shown in the UI
DEFECT_DESCRIPTIONS = {
    "Crazing": (
        "Network of fine surface cracks. "
        "Cause: thermal stress during cooling. "
        "Risk: crack propagation under load. "
        "Action: reject batch, inspect cooling line parameters."
    ),
    "Inclusion": (
        "Embedded foreign particles (typically slag). "
        "Cause: entrapment during casting. "
        "Risk: stress concentration, fatigue failure. "
        "Action: reject batch, audit casting process."
    ),
    "Patches": (
        "Irregular surface discolouration. "
        "Cause: uneven oxide scale distribution. "
        "Risk: cosmetic failure, inconsistent coating adhesion. "
        "Action: flag for secondary inspection before coating."
    ),
    "Pitted": (
        "Small cavities or surface pits. "
        "Cause: corrosion or mechanical damage in transport. "
        "Risk: crack initiation site under cyclic load. "
        "Action: reject for structural applications, downgrade to non-critical use."
    ),
    "Rolled-in Scale": (
        "Oxide scale pressed into the surface during rolling. "
        "Cause: rolling temperature too low or scale not removed. "
        "Risk: delamination, poor surface finish. "
        "Action: reject, review descaling process."
    ),
    "Scratches": (
        "Linear surface marks from mechanical contact. "
        "Cause: handling equipment, transport, or tooling contact. "
        "Risk: cosmetic and fatigue failure at scratch tips. "
        "Action: assess scratch depth; reject if depth exceeds spec."
    ),
}


def load_model(checkpoint_path: str, device: torch.device) -> DefectClassifier:
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DefectClassifier(num_classes=6, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def build_inference_fn(checkpoint_path: str):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = load_model(checkpoint_path, device)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

    def infer(image: np.ndarray):
        """
        Gradio inference function.

        Args:
            image : (H, W, 3) uint8 RGB array from Gradio Image component

        Returns:
            overlay      : np.ndarray — heatmap overlay for display
            label        : dict — {class: probability} for BarPlot
            description  : str — business-facing defect description
        """
        if image is None:
            return None, {}, "Upload an image to get started."

        pil_img = Image.fromarray(image)
        img_t   = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_t)
            probs  = F.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx   = int(probs.argmax())
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        # Grad-CAM
        heatmap  = generate_heatmap(model, img_t, pred_idx)
        img_np   = denormalize(img_t.squeeze(0).cpu(), MEAN, STD)
        overlay  = overlay_heatmap(img_np, heatmap)
        overlay  = (overlay * 255).astype(np.uint8)

        prob_dict    = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}
        description  = (
            f"**{pred_class}** — {confidence*100:.1f}% confidence\n\n"
            f"{DEFECT_DESCRIPTIONS[pred_class]}"
        )

        return overlay, prob_dict, description

    return infer


def build_demo(checkpoint_path: str) -> gr.Blocks:
    infer = build_inference_fn(checkpoint_path)

    with gr.Blocks(title="Steel Defect Detector", theme=gr.themes.Soft()) as demo:

        gr.Markdown("""
        # 🔬 Steel Surface Defect Detector
        Upload a hot-rolled steel strip image to classify the defect type and
        visualise exactly where on the surface the model detected the anomaly.

        **6 defect classes detected:** Crazing · Inclusion · Patches · Pitted · Rolled-in Scale · Scratches
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_img  = gr.Image(label="Upload steel surface image", type="numpy")
                submit_btn = gr.Button("Analyse", variant="primary")

            with gr.Column(scale=1):
                output_img = gr.Image(label="Grad-CAM heatmap — where the defect is")
                bar_plot   = gr.Label(label="Class probabilities", num_top_classes=6)

        description_box = gr.Markdown(label="Diagnosis & recommended action")

        submit_btn.click(
            fn      = infer,
            inputs  = [input_img],
            outputs = [output_img, bar_plot, description_box],
        )

        gr.Examples(
            examples   = [],   # populate with example images if available
            inputs     = [input_img],
            label      = "Example images",
        )

        gr.Markdown("""
        ---
        **Model:** EfficientNet-B4 fine-tuned on NEU-CLS (1,800 images) ·
        **Val F1-Macro:** 99.44% ·
        **Inference:** ~12ms on GPU, ~168ms on CPU
        """)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steel defect Gradio demo")
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--port",       type=int, default=7860)
    parser.add_argument("--share",      action="store_true",
                        help="Create a public URL via Gradio share tunnel")
    args = parser.parse_args()

    demo = build_demo(args.checkpoint)
    demo.launch(server_port=args.port, share=args.share)
