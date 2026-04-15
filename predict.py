"""
DeepFake Detection — Inference Script

Usage:
    # Single image
    python src/predict.py --image path/to/face.jpg --checkpoint outputs/best_model.pth

    # Folder of images
    python src/predict.py --image_dir path/to/faces/ --checkpoint outputs/best_model.pth \
                          --output_csv results.csv
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import DeepFakeDetector


# ── Preprocessing ─────────────────────────────────────────────────────────────

def get_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── Loader ────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> DeepFakeDetector:
    ckpt  = torch.load(checkpoint_path, map_location=device)
    cfg   = ckpt.get("config", {})
    model = DeepFakeDetector(dropout_rate=cfg.get("dropout_rate", 0.4))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val F1 = {ckpt.get('val_f1', 'N/A'):.4f})")
    return model


# ── Core predict function ─────────────────────────────────────────────────────

@torch.no_grad()
def predict_image(
    image_path: str,
    model: DeepFakeDetector,
    transform: transforms.Compose,
    device: torch.device,
) -> dict:
    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    logit  = model(tensor)
    prob   = torch.sigmoid(logit).item()
    label  = "FAKE" if prob > 0.5 else "REAL"
    return {
        "path":       str(image_path),
        "label":      label,
        "fake_prob":  round(prob, 4),
        "real_prob":  round(1 - prob, 4),
        "confidence": round(max(prob, 1 - prob), 4),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DeepFake Detector — Inference")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",     type=str, help="Path to a single image")
    group.add_argument("--image_dir", type=str, help="Directory of images")

    parser.add_argument("--checkpoint",  type=str, default="outputs/best_model.pth")
    parser.add_argument("--output_csv",  type=str, default=None,
                        help="Save results to CSV (only with --image_dir)")
    parser.add_argument("--image_size",  type=int, default=224)
    parser.add_argument("--device",      type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device    = torch.device(args.device)
    transform = get_transform(args.image_size)
    model     = load_model(args.checkpoint, device)

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    if args.image:
        result = predict_image(args.image, model, transform, device)
        print(json.dumps(result, indent=2))

    else:
        image_dir  = Path(args.image_dir)
        image_paths = [p for p in sorted(image_dir.rglob("*"))
                       if p.suffix.lower() in IMG_EXTS]
        print(f"Found {len(image_paths)} images in {image_dir}")

        results = []
        for p in tqdm(image_paths, desc="Predicting"):
            try:
                results.append(predict_image(p, model, transform, device))
            except Exception as e:
                results.append({"path": str(p), "error": str(e)})

        # Summary
        df = pd.DataFrame(results)
        print("\n── Summary ──")
        print(df["label"].value_counts().to_string())
        print(f"\nMean fake probability : {df['fake_prob'].mean():.4f}")

        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
            print(f"\nResults saved to {args.output_csv}")
        else:
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
