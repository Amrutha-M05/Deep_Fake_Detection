"""
DeepFake Dataset — supports the Kaggle "deepfake-and-real-images" structure.

Expected folder layout (after kagglehub download):
    <root>/
        Dataset/
            Train/
                Real/   *.jpg / *.png
                Fake/   *.jpg / *.png
            Test/
                Real/
                Fake/
            Validation/        (optional – used as val split if present)
                Real/
                Fake/

If the Validation folder is absent, val & test sets are carved out of Train
using the val_split / test_split fractions.
"""

import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


# Label mapping  ── 0 = Real, 1 = Fake
LABEL_MAP = {"real": 0, "fake": 1}


def _collect_images(folder: Path) -> List[Tuple[Path, int]]:
    """Walk a Real/Fake folder pair and return (path, label) pairs."""
    samples = []
    for class_name, label in LABEL_MAP.items():
        class_dir = folder / class_name.capitalize()
        if not class_dir.exists():
            # Try lowercase
            class_dir = folder / class_name
        if not class_dir.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            for img_path in sorted(class_dir.glob(ext)):
                samples.append((img_path, label))
    return samples


class DeepFakeDataset(Dataset):
    """
    Parameters
    ----------
    root : str
        Root directory returned by kagglehub.dataset_download(...)
    split : str
        One of "train" | "val" | "test"
    transform : callable, optional
        torchvision transforms applied to the PIL Image
    val_split : float
        Fraction of training data used for validation (if no Validation folder)
    test_split : float
        Fraction of training data used for testing (if no Test folder)
    seed : int
        Random seed for reproducible splits
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.transform = transform
        self.split     = split

        root = Path(root)
        dataset_root = self._find_dataset_root(root)

        # ── Locate raw splits ─────────────────────────────────────────────
        train_folder = self._find_folder(dataset_root, ("Train", "train"))
        test_folder  = self._find_folder(dataset_root, ("Test",  "test"))
        val_folder   = self._find_folder(dataset_root, ("Validation", "validation", "Val", "val"))

        all_train = _collect_images(train_folder) if train_folder else []
        all_test  = _collect_images(test_folder)  if test_folder  else []
        all_val   = _collect_images(val_folder)   if val_folder   else []

        # ── Build splits ──────────────────────────────────────────────────
        rng = random.Random(seed)

        if not all_val or not all_test:
            # Shuffle train and carve out val / test
            shuffled = list(all_train)
            rng.shuffle(shuffled)
            n       = len(shuffled)
            n_test  = int(n * test_split)  if not all_test else 0
            n_val   = int(n * val_split)
            n_train = n - n_val - n_test

            all_train_final = shuffled[:n_train]
            all_val_final   = shuffled[n_train: n_train + n_val]
            all_test_final  = all_test if all_test else shuffled[n_train + n_val:]
        else:
            all_train_final = all_train
            all_val_final   = all_val
            all_test_final  = all_test

        mapping = {"train": all_train_final, "val": all_val_final, "test": all_test_final}
        self.samples: List[Tuple[Path, int]] = mapping[split]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found for split='{split}' under {dataset_root}. "
                "Check dataset structure."
            )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _find_dataset_root(root: Path) -> Path:
        """Handle an extra nesting level sometimes added by kagglehub."""
        if (root / "Dataset").exists():
            return root / "Dataset"
        # Search one level deep
        for child in root.iterdir():
            if child.is_dir() and (child / "Train").exists():
                return child
            if child.is_dir() and (child / "Dataset").exists():
                return child / "Dataset"
        return root

    @staticmethod
    def _find_folder(root: Path, candidates: Tuple[str, ...]) -> Optional[Path]:
        for name in candidates:
            p = root / name
            if p.exists() and p.is_dir():
                return p
        return None

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def pos_weight(self) -> float:
        """
        Compute pos_weight = #negatives / #positives for BCEWithLogitsLoss.
        Class 1 is Fake (positive).
        """
        labels = [lbl for _, lbl in self.samples]
        n_pos  = sum(labels)
        n_neg  = len(labels) - n_pos
        if n_pos == 0:
            return 1.0
        return n_neg / n_pos

    def class_distribution(self) -> dict:
        labels = [lbl for _, lbl in self.samples]
        return {"real": labels.count(0), "fake": labels.count(1)}


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import kagglehub
    path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")
    print("Dataset path:", path)

    for split in ("train", "val", "test"):
        ds = DeepFakeDataset(path, split=split)
        print(f"{split:5s}: {len(ds):6,} samples | dist: {ds.class_distribution()}")
