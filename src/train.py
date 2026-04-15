"""
DeepFake Image Detection - Training Script
Team No: 06
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import json
import time

from model import DeepFakeDetector
from dataset import DeepFakeDataset
from utils.metrics import compute_metrics, plot_confusion_matrix
from utils.grad_cam import GradCAM, visualize_grad_cam
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────── Config ────────────────────────────

CONFIG = {
    "dataset_path": None,          # Set via CLI or auto-detected from kagglehub
    "output_dir": "outputs",
    "image_size": 224,
    "batch_size": 32,
    "num_workers": 0,
    "num_epochs": 20,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "dropout_rate": 0.4,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "val_split": 0.15,
    "test_split": 0.15,
    "early_stopping_patience": 5,
    "save_best_only": True,
    "use_amp": True,              # Automatic Mixed Precision
}


# ─────────────────────────── Transforms ────────────────────────

def get_transforms(image_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomRotation(degrees=10),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ─────────────────────────── Training loop ─────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    for imgs, labels in tqdm(loader, desc="Eval ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device).float().unsqueeze(1)
        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)

        all_probs.extend(probs.flatten().tolist())
        all_preds.extend(preds.flatten().tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


# ─────────────────────────── Main ──────────────────────────────

def main(cfg: dict = CONFIG):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device(cfg["device"])
    logger.info(f"Using device: {device}")

    # ── Dataset ──
    if cfg["dataset_path"] is None:
        try:
            import kagglehub
            path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")
            logger.info(f"Dataset downloaded to: {path}")
            cfg["dataset_path"] = path
        except Exception as e:
            raise RuntimeError(
                "dataset_path not set and kagglehub download failed. "
                f"Install with `pip install kagglehub` and set KAGGLE_USERNAME/KAGGLE_KEY.\n{e}"
            )

    train_tf, val_tf = get_transforms(cfg["image_size"])
    train_ds = DeepFakeDataset(cfg["dataset_path"], split="train", transform=train_tf,
                               val_split=cfg["val_split"], test_split=cfg["test_split"],
                               seed=cfg["seed"])
    val_ds   = DeepFakeDataset(cfg["dataset_path"], split="val",   transform=val_tf,
                               val_split=cfg["val_split"], test_split=cfg["test_split"],
                               seed=cfg["seed"])
    test_ds  = DeepFakeDataset(cfg["dataset_path"], split="test",  transform=val_tf,
                               val_split=cfg["val_split"], test_split=cfg["test_split"],
                               seed=cfg["seed"])

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)

    # ── Model ──
    model = DeepFakeDetector(dropout_rate=cfg["dropout_rate"]).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {total_params:,}")

    # Class-balanced loss
    pos_weight = torch.tensor([train_ds.pos_weight()]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                             weight_decay=cfg["weight_decay"])
    scheduler  = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"], eta_min=1e-6)
    scaler     = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Training ──
    best_val_f1, patience_counter = 0.0, 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_f1": []}

    for epoch in range(1, cfg["num_epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion,
                                          device, scaler)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])

        logger.info(
            f"Epoch {epoch:02d}/{cfg['num_epochs']} | "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} F1: {val_metrics['f1']:.4f} "
            f"Acc: {val_metrics['accuracy']:.4f} ({time.time()-t0:.1f}s)"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            ckpt_path = os.path.join(cfg["output_dir"], "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "config": cfg,
            }, ckpt_path)
            logger.info(f"  ✓ Saved best model (F1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stopping_patience"]:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

    # ── Test evaluation ──
    ckpt = torch.load(os.path.join(cfg["output_dir"], "best_model.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info("\n══ Test Results ══")
    for k, v in test_metrics.items():
        logger.info(f"  {k:12s}: {v:.4f}")

    # Save results
    plot_confusion_matrix(test_metrics["cm"],
                          save_path=os.path.join(cfg["output_dir"], "confusion_matrix.png"))

    with open(os.path.join(cfg["output_dir"], "test_metrics.json"), "w") as f:
        json.dump({k: float(v) if k != "cm" else v for k, v in test_metrics.items()}, f, indent=2)

    with open(os.path.join(cfg["output_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ── Grad-CAM ──
    grad_cam = GradCAM(model, target_layer=model.backbone.features[-1])
    visualize_grad_cam(grad_cam, test_ds, device,
                       save_dir=os.path.join(cfg["output_dir"], "grad_cam"),
                       num_samples=8)

    logger.info("Training complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeepFake Detection Training")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=CONFIG["num_epochs"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--output_dir", type=str, default=CONFIG["output_dir"])
    args = parser.parse_args()

    CONFIG["dataset_path"] = args.dataset_path
    CONFIG["num_epochs"]   = args.epochs
    CONFIG["batch_size"]   = args.batch_size
    CONFIG["learning_rate"] = args.lr
    CONFIG["output_dir"]   = args.output_dir
    main(CONFIG)
