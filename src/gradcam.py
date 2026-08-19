"""
gradcam.py — Grad-CAM explainability for SteelDefectNet.

Why Grad-CAM matters here specifically: a classifier that's right for the
wrong reason is a liability on a production line. NEU-CLS images have
prominent scanning-artefact borders and lighting gradients; a network can
learn to shortcut on those instead of the actual defect texture. Grad-CAM
lets a QA engineer visually confirm the model is attending to the defect
itself before it's trusted on new footage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class GradCAMExplainer:
    """Wraps pytorch-grad-cam around a SteelDefectNet instance."""

    def __init__(self, model, device: torch.device):
        self.model = model.eval()
        self.device = device
        target_layer = model.get_cam_target_layer()
        self.cam = GradCAM(model=model, target_layers=[target_layer])

    def explain(
        self,
        img_tensor: torch.Tensor,
        rgb_img: np.ndarray,
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Args:
            img_tensor   : normalised (1, 3, H, W) tensor, on self.device
            rgb_img      : (H, W, 3) float array in [0, 1], the *unnormalised*
                           image to overlay the heatmap on
            target_class : class index to explain; None = the model's own
                           top prediction
        Returns:
            (H, W, 3) uint8 RGB image with the Grad-CAM heatmap overlaid.
        """
        targets = None if target_class is None else [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=img_tensor, targets=targets)[0]  # (H, W) in [0, 1]
        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return overlay


def tensor_to_rgb01(img_tensor: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    """Undo normalisation on a single (3, H, W) tensor -> (H, W, 3) float array in [0, 1]."""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = img_tensor.detach().cpu() * std_t + mean_t
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def save_gradcam_grid(
    model,
    device: torch.device,
    samples: list[tuple[torch.Tensor, int, str]],
    class_names: list[str],
    mean: list[float],
    std: list[float],
    out_path: str | Path,
    predictions: list[int] | None = None,
):
    """
    Build a grid figure: one row per sample, [original | Grad-CAM overlay],
    captioned with true/predicted class. Useful as a qualitative sanity check
    saved once per training run.
    """
    import matplotlib.pyplot as plt

    explainer = GradCAMExplainer(model, device)
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes[None, :]

    for i, (img_tensor, label, path) in enumerate(samples):
        rgb = tensor_to_rgb01(img_tensor, mean, std)
        input_tensor = img_tensor.unsqueeze(0).to(device)
        pred = predictions[i] if predictions else label
        overlay = explainer.explain(input_tensor, rgb, target_class=pred)

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"true: {class_names[label]}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay)
        pred_name = class_names[pred] if predictions else "Grad-CAM"
        axes[i, 1].set_title(f"pred: {pred_name}", fontsize=9)
        axes[i, 1].axis("off")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def gradcam_for_image(model, device, image_path: str, img_size: int, mean: list[float], std: list[float],
                       target_class: int | None = None):
    """
    Single-image convenience entry point used by infer.py / app.py.
    Returns (overlay_uint8_rgb, predicted_class_idx, probs).
    """
    import torch.nn.functional as F
    from src.dataset import make_transform

    img = Image.open(image_path).convert("RGB")
    transform = make_transform(img_size, mean, std, mode="val")
    img_tensor = transform(img)

    model.eval()
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0).to(device))
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred_class = int(probs.argmax()) if target_class is None else target_class

    explainer = GradCAMExplainer(model, device)
    rgb01 = tensor_to_rgb01(img_tensor, mean, std)
    overlay = explainer.explain(img_tensor.unsqueeze(0).to(device), rgb01, target_class=pred_class)
    return overlay, pred_class, probs
