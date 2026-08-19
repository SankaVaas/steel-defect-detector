"""
test_model.py — unit tests for DefectClassifier.

Run with:
    pytest tests/test_model.py -v
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model   import DefectClassifier
from src.dataset import NEUCLSDataset, SteelSurfaceAugment, make_transform
from src.gradcam import denormalize

import numpy as np
from PIL import Image


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    m = DefectClassifier(num_classes=6, pretrained=False)
    m.eval()
    return m

@pytest.fixture
def dummy_input():
    return torch.randn(2, 3, 224, 224)


# ── Model tests ───────────────────────────────────────────────────────────────

def test_forward_shape(model, dummy_input):
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (2, 6), f"Expected (2, 6), got {out.shape}"


def test_forward_no_nan(model, dummy_input):
    with torch.no_grad():
        out = model(dummy_input)
    assert not torch.isnan(out).any(), "NaN in model output"
    assert not torch.isinf(out).any(), "Inf in model output"


def test_freeze_backbone(model, dummy_input):
    model.freeze_backbone()
    for p in model.backbone.parameters():
        assert not p.requires_grad, "Backbone param should be frozen"
    model.unfreeze_backbone()


def test_unfreeze_backbone(model, dummy_input):
    model.freeze_backbone()
    model.unfreeze_backbone()
    for p in model.backbone.parameters():
        assert p.requires_grad, "Backbone param should be trainable after unfreeze"


def test_param_count(model):
    counts = model.param_count()
    assert "total_M" in counts
    assert counts["total_M"] > 0


def test_cam_target_layer(model):
    layer = model.get_cam_target_layer()
    assert layer is not None


# ── Augmentation tests ────────────────────────────────────────────────────────

def test_augmentation_output_size():
    aug = SteelSurfaceAugment(img_size=224, mode="train")
    img = Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8))
    out = aug(img)
    assert out.size == (224, 224)


def test_val_augmentation_no_random():
    """Val mode must be deterministic — two passes should give identical output."""
    aug  = SteelSurfaceAugment(img_size=224, mode="val")
    img  = Image.fromarray((np.random.rand(200, 200, 3) * 255).astype(np.uint8))
    out1 = np.array(aug(img))
    out2 = np.array(aug(img))
    np.testing.assert_array_equal(out1, out2)


def test_transform_tensor_shape():
    transform = make_transform(224, [0.5]*3, [0.2]*3, "train")
    img = Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8))
    out = transform(img)
    assert out.shape == (3, 224, 224)


# ── Gradcam utils tests ───────────────────────────────────────────────────────

def test_denormalize_range():
    t = torch.randn(3, 224, 224)
    arr = denormalize(t, [0.5]*3, [0.2]*3)
    assert arr.shape == (224, 224, 3)
    assert arr.min() >= 0.0 - 1e-6
    assert arr.max() <= 1.0 + 1e-6
