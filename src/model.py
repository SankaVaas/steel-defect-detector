"""
model.py — DefectClassifier
EfficientNet-B4 backbone with a custom classification head.

Design decisions:
    - timm for backbone: handles pretrained weights + feature extraction cleanly
    - Custom head: BatchNorm1d → Dropout → Linear stabilises transfer learning
    - Two-phase API: freeze_backbone() / unfreeze_backbone() for warmup strategy
    - get_cam_target_layer(): exposes the correct layer for Grad-CAM without
      coupling the training code to the visualisation code
"""

import torch
import torch.nn as nn
import timm


class DefectClassifier(nn.Module):
    """
    EfficientNet-B4 + custom head for steel surface defect classification.

    Args:
        backbone_name : timm model identifier (default: 'efficientnet_b4')
        num_classes   : number of defect categories
        pretrained    : load ImageNet weights
        drop_rate     : dropout probability in the head

    Frozen strategy (call externally from training loop):
        Phase 1 — freeze_backbone() during LR warmup
        Phase 2 — unfreeze_backbone() for full fine-tuning
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b4",
        num_classes: int = 6,
        pretrained: bool = True,
        drop_rate: float = 0.3,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,   # remove default classifier
            global_pool="",  # remove default pooling
            drop_rate=0.0,   # disable internal dropout; we add our own
        )

        # Infer feature dimension without a hard-coded constant
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat_dim = self.backbone(dummy).shape[1]

        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.head    = nn.Sequential(
            nn.BatchNorm1d(feat_dim),  # stabilises transfer learning scale mismatch
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, num_classes),
        )

        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)       # (B, C, H, W)
        pooled   = self.pool(features)    # (B, C, 1, 1)
        flat     = self.flatten(pooled)   # (B, C)
        return self.head(flat)            # (B, num_classes)

    # ── Freezing API ─────────────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters. Call before warmup epochs."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters. Call after warmup."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def get_cam_target_layer(self):
        """Return the last conv block — target layer for Grad-CAM."""
        return self.backbone.blocks[-1]

    # ── Utility ──────────────────────────────────────────────────────────────

    def param_count(self) -> dict:
        backbone = sum(p.numel() for p in self.backbone.parameters()) / 1e6
        head     = sum(p.numel() for p in self.head.parameters()) / 1e6
        return {"backbone_M": round(backbone, 3), "head_M": round(head, 4),
                "total_M": round(backbone + head, 3)}
