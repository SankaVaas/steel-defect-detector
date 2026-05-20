"""
gradcam.py — Grad-CAM heatmap generation for DefectClassifier.

Usage:
    from src.gradcam import generate_heatmap, overlay_heatmap

    heatmap = generate_heatmap(model, image_tensor, class_idx)
    overlay = overlay_heatmap(image_np, heatmap)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def generate_heatmap(
    model:       nn.Module,
    image_tensor: torch.Tensor,  # (1, 3, H, W) on correct device
    class_idx:   int,
    smooth:      bool = True,
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for a single image.

    Args:
        model        : DefectClassifier (must have get_cam_target_layer())
        image_tensor : preprocessed image tensor, shape (1, 3, H, W)
        class_idx    : class to explain (use predicted class for saliency)
        smooth       : apply Gaussian blur to reduce grid artefacts

    Returns:
        heatmap : (H, W) float32 array in [0, 1]
    """
    target_layer = [model.get_cam_target_layer()]
    cam          = GradCAM(model=model, target_layers=target_layer)
    targets      = [ClassifierOutputTarget(class_idx)]

    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    heatmap       = grayscale_cam[0]  # (H, W)

    if smooth:
        heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)

    return heatmap.astype(np.float32)


def overlay_heatmap(
    image_np: np.ndarray,   # (H, W, 3) float32 in [0, 1]
    heatmap:  np.ndarray,   # (H, W) float32 in [0, 1]
    alpha:    float = 0.5,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on an image using the jet colormap.

    Returns:
        blended : (H, W, 3) uint8 array
    """
    return show_cam_on_image(image_np, heatmap, use_rgb=True, image_weight=1 - alpha)


def denormalize(
    tensor: torch.Tensor,   # (3, H, W)
    mean:   list[float],
    std:    list[float],
) -> np.ndarray:
    """
    Undo normalization and return (H, W, 3) float32 array in [0, 1].
    Used to convert a dataset batch item back to a displayable image.
    """
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    img = tensor.cpu() * s + m
    return img.permute(1, 2, 0).numpy().clip(0, 1).astype(np.float32)
