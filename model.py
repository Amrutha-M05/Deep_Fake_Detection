"""
DeepFake Detection Model — EfficientNet-B4 Transfer Learning
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


class DeepFakeDetector(nn.Module):
    """
    Binary classifier built on top of a pretrained EfficientNet-B4.

    Architecture:
        EfficientNet-B4 (pretrained on ImageNet, top removed)
        → AdaptiveAvgPool2d (already inside EfficientNet)
        → Custom classification head with BatchNorm + Dropout
        → Single logit output  (use BCEWithLogitsLoss)
    """

    def __init__(self, dropout_rate: float = 0.4, freeze_backbone: bool = False):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        base    = efficientnet_b4(weights=weights)

        # Remove the original classifier (Linear 1792 → 1000)
        in_features = base.classifier[1].in_features  # 1792 for B4
        base.classifier = nn.Identity()

        self.backbone = base

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── Classification head ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, 1),   # raw logit → BCEWithLogitsLoss
        )

        # Weight init for head
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)          # (B, 1792)
        logits   = self.classifier(features) # (B, 1)
        return logits

    def unfreeze_backbone(self, num_blocks: int = 4):
        """
        Progressively unfreeze the last `num_blocks` MBConv blocks
        of EfficientNet for fine-tuning.
        """
        blocks = list(self.backbone.features.children())
        for block in blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True


# ─── Quick sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    model = DeepFakeDetector(dropout_rate=0.4)
    dummy = torch.randn(4, 3, 224, 224)
    out   = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")   # (4, 1)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")
