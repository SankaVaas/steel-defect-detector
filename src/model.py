"""
model.py — SteelDefectNet: a timm backbone + custom classification head.

Design notes
------------
* We use a *pretrained* ImageNet backbone (default: EfficientNet-B4) rather than
  training from scratch. NEU-CLS has only ~1,800 images total (300/class) — far
  too little to learn good low-level filters (edges, textures) from zero.
  Transfer learning re-uses filters learned on millions of natural images and
  fine-tunes only the parts that need to change for steel surfaces.

* Discriminative learning rates: the pretrained backbone is fine-tuned at a
  *lower* LR (backbone_lr = lr * backbone_lr_mult) than the newly-initialised
  head, which trains at the full LR. This is the standard transfer-learning
  recipe — the backbone already "knows" something useful, so we nudge it
  gently; the head starts from random weights and needs to move further.

* `drop_rate` adds dropout before the final linear layer to fight overfitting
  given the small dataset size.

* `forward_features` / `get_cam_target_layer` expose what pytorch-grad-cam
  needs: the last convolutional feature map, before global pooling, so we can
  visualise *where* the network is looking when it makes a decision.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class SteelDefectNet(nn.Module):
    """
    Thin wrapper around a timm backbone for steel-surface defect classification.

    Args:
        backbone     : any timm model name (e.g. 'efficientnet_b4', 'resnet18')
        num_classes  : number of defect classes (6 for NEU-CLS)
        pretrained   : load ImageNet weights
        drop_rate    : dropout probability applied before the classifier head
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        num_classes: int = 6,
        pretrained: bool = True,
        drop_rate: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone

        # num_classes=0 -> timm strips the classifier and returns pooled features.
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
        )
        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)          # (B, feat_dim) — already globally pooled
        return self.head(feats)           # (B, num_classes)

    def get_cam_target_layer(self):
        """
        Return the last convolutional module of the backbone, i.e. the feature
        map immediately before global pooling. This is the layer pytorch-grad-cam
        needs its hooks on to produce a class activation map.

        Works for the timm conv backbones we support out of the box
        (EfficientNet / ResNet family). For other architectures, inspect
        `model.backbone` and adjust.
        """
        name = self.backbone_name.lower()
        if "efficientnet" in name:
            return self.backbone.conv_head if hasattr(self.backbone, "conv_head") else self.backbone.blocks[-1]
        if "resnet" in name:
            return self.backbone.layer4[-1]
        if "convnext" in name:
            return self.backbone.stages[-1]
        # Fallback: last child module with parameters (best-effort).
        for module in reversed(list(self.backbone.modules())):
            if isinstance(module, (nn.Conv2d,)):
                return module
        raise ValueError(f"Could not infer Grad-CAM target layer for backbone '{self.backbone_name}'.")

    def param_groups(self, lr: float, backbone_lr_mult: float = 0.1, weight_decay: float = 1e-4):
        """
        Discriminative learning rates: backbone trains slower than the head.
        Returns a list of param-group dicts for the optimizer.
        """
        return [
            {"params": self.backbone.parameters(), "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
            {"params": self.head.parameters(), "lr": lr, "weight_decay": weight_decay},
        ]


def build_model(cfg: dict) -> SteelDefectNet:
    """Construct a SteelDefectNet from a config dict (see configs/default.yaml)."""
    return SteelDefectNet(
        backbone=cfg["backbone"],
        num_classes=cfg["num_classes"],
        pretrained=cfg["pretrained"],
        drop_rate=cfg["drop_rate"],
    )


if __name__ == "__main__":
    # Quick self-test: forward pass + shape check.
    model = SteelDefectNet(backbone="resnet18", num_classes=6, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (2, 6), out.shape
    layer = model.get_cam_target_layer()
    print(f"OK — output shape {tuple(out.shape)}, cam target layer: {layer.__class__.__name__}")
