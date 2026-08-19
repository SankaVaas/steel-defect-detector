"""
predict.py — single-image inference with Grad-CAM output.

Usage:
    python scripts/predict.py --image path/to/image.jpg
    python scripts/predict.py --image path/to/image.jpg --checkpoint best_model.pth
    python scripts/predict.py --image path/to/image.jpg --no-cam
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Add src/ to path when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model   import DefectClassifier
from src.dataset import CLASS_NAMES, compute_dataset_stats
from src.gradcam import generate_heatmap, overlay_heatmap, denormalize


def load_model(checkpoint_path: str, device: torch.device) -> DefectClassifier:
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DefectClassifier(num_classes=6, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def predict(
    image_path:      str,
    checkpoint_path: str = "best_model.pth",
    show_cam:        bool = True,
    save_output:     bool = True,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(checkpoint_path, device)

    # ── Preprocess ────────────────────────────────────────────────────────────
    # DESIGN: Dataset stats are loaded from checkpoint cfg if available,
    # otherwise fall back to values computed during training (see notebook).
    MEAN = [0.5023, 0.5023, 0.5023]
    STD  = [0.2159, 0.2159, 0.2159]

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

    img_pil = Image.open(image_path).convert("RGB")
    img_t   = transform(img_pil).unsqueeze(0).to(device)

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(img_t)
        probs  = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx  = int(probs.argmax())
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    result = {
        "class":      pred_class,
        "confidence": confidence,
        "all_probs":  {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
    }

    print(f"\n{'='*45}")
    print(f"  File       : {Path(image_path).name}")
    print(f"  Prediction : {pred_class}")
    print(f"  Confidence : {confidence*100:.2f}%")
    print(f"{'='*45}")
    print("  Per-class probabilities:")
    for name, p in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
        bar = "█" * int(p * 20)
        print(f"    {name:20s} {p*100:5.1f}%  {bar}")

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    if show_cam:
        heatmap  = generate_heatmap(model, img_t, pred_idx)
        img_np   = denormalize(img_t.squeeze(0).cpu(), MEAN, STD)
        overlay  = overlay_heatmap(img_np, heatmap)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].imshow(img_np);           axes[0].set_title("Input");       axes[0].axis("off")
        axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
        axes[2].imshow(overlay);          axes[2].set_title("Overlay");     axes[2].axis("off")

        fig.suptitle(
            f"Prediction: {pred_class}  ({confidence*100:.1f}%)",
            fontsize=13, fontweight="bold",
            color="#e63946",
        )
        plt.tight_layout()

        if save_output:
            out = Path(image_path).stem + "_prediction.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"\n  Saved → {out}")
        plt.show()

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steel defect inference")
    parser.add_argument("--image",      required=True, help="Path to input image")
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--no-cam",     action="store_true")
    args = parser.parse_args()

    predict(args.image, args.checkpoint, show_cam=not args.no_cam)
