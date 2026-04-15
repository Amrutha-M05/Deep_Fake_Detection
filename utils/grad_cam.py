"""
Grad-CAM Implementation for DeepFake Detection.

Gradient-weighted Class Activation Maps highlight the spatial regions of an
input image that contributed most to the model's prediction.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from typing import Optional
from torch.utils.data import Dataset


class GradCAM:
    """
    Grad-CAM for a binary classifier that outputs a single logit.

    Parameters
    ----------
    model      : nn.Module — the trained DeepFakeDetector
    target_layer : nn.Module — the convolutional layer to hook
                   (e.g. model.backbone.features[-1])
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for a single image tensor.

        Parameters
        ----------
        input_tensor : Tensor of shape (1, C, H, W), on the same device as model.

        Returns
        -------
        np.ndarray of shape (H, W) with values in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        logit = self.model(input_tensor)       # (1, 1)
        logit.backward()                        # Grad of logit w.r.t. activations

        # Global Average Pooling of gradients
        pooled_grads = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weight activations
        weighted = (self.activations * pooled_grads).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        heatmap  = F.relu(weighted).squeeze().cpu().numpy()

        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    def overlay(
        self,
        heatmap: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.45,
        colormap: str = "jet",
    ) -> np.ndarray:
        """
        Overlay a heatmap on the original image.

        Parameters
        ----------
        heatmap        : (H', W') float array in [0, 1]
        original_image : (H, W, 3) uint8 array
        alpha          : blending weight for the heatmap

        Returns
        -------
        (H, W, 3) uint8 blended image
        """
        H, W = original_image.shape[:2]
        heatmap_resized = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
        ) / 255.0

        colormap_fn = cm.get_cmap(colormap)
        colored     = colormap_fn(heatmap_resized)[:, :, :3]   # (H, W, 3)
        blended     = (1 - alpha) * original_image / 255.0 + alpha * colored
        return (blended * 255).clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────

def visualize_grad_cam(
    grad_cam: GradCAM,
    dataset: Dataset,
    device: torch.device,
    save_dir: str = "outputs/grad_cam",
    num_samples: int = 8,
):
    """
    Generate and save Grad-CAM visualizations for a random subset of images.

    Saves a grid image: original | heatmap | overlay  per sample.
    """
    os.makedirs(save_dir, exist_ok=True)

    import random
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    # Un-normalise transform for display
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for i, idx in enumerate(indices):
        tensor, label = dataset[idx]
        input_t = tensor.unsqueeze(0).to(device)

        with torch.enable_grad():
            heatmap = grad_cam.generate(input_t)

        # Reconstruct original pixel image
        img_np = tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * std + mean).clip(0, 1)
        img_u8 = (img_np * 255).astype(np.uint8)

        overlay = grad_cam.overlay(heatmap, img_u8)

        # ── Figure ─────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        titles = ["Original", "Grad-CAM Heatmap", "Overlay"]
        images = [img_u8, heatmap, overlay]
        cmaps  = [None, "jet", None]

        class_name = "Fake" if label == 1 else "Real"
        pred_prob  = torch.sigmoid(grad_cam.model(input_t)).item()
        pred_class = "Fake" if pred_prob > 0.5 else "Real"

        fig.suptitle(
            f"GT: {class_name} | Pred: {pred_class} (p={pred_prob:.3f})",
            fontsize=13, y=1.02,
        )

        for ax, img, title, cmap in zip(axes, images, titles, cmaps):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=11)
            ax.axis("off")

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sample_{i:02d}_{class_name.lower()}.png")
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"Grad-CAM visualizations saved to {save_dir}/")
