"""
dataset.py — NEUCLSDataset + domain-specific augmentation pipeline.

Augmentation validity table for steel surface images:
    ✅ Horizontal / vertical flip   — surface has no orientation constraint
    ✅ 90° / 180° / 270° rotation  — rotationally symmetric
    ⚠️ Arbitrary rotation (0-360°) — introduces black border artefacts
    ✅ Mild brightness / contrast   — simulates line lighting variation
    ✅ CutOut (random erase)        — forces non-local features
    ❌ Strong blur                  — destroys fine texture critical for class
    ❌ Strong colour jitter         — unphysical for grayscale steel
"""

import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


CLASS_NAMES = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted",
    "Rolled-in Scale",
    "Scratches",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}

FOLDER_TO_CLASS = {
    "crazing":          "Crazing",
    "inclusion":        "Inclusion",
    "patches":          "Patches",
    "pitted_surface":   "Pitted",
    "rolled-in_scale":  "Rolled-in Scale",
    "scratches":        "Scratches",
}


# ── Augmentation ─────────────────────────────────────────────────────────────

class SteelSurfaceAugment:
    """
    Domain-specific augmentation for NEU-CLS steel surface images.

    Implemented as a callable class (not a torchvision Transform) so that
    each component is independently testable and auditable.

    Args:
        img_size : resize target (square)
        mode     : 'train' applies augmentation; 'val' is resize-only
    """

    def __init__(self, img_size: int = 224, mode: str = "train"):
        self.img_size = img_size
        self.mode = mode

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        if self.mode != "train":
            return img

        if random.random() > 0.5:
            img = TF.hflip(img)
        if random.random() > 0.5:
            img = TF.vflip(img)
        if random.random() > 0.5:
            img = TF.rotate(img, random.choice([90, 180, 270]))

        img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
        img = TF.adjust_contrast(img,   random.uniform(0.8, 1.2))

        if random.random() > 0.7:
            draw = ImageDraw.Draw(img)
            w, h = img.size
            sx = random.uniform(0.05, 0.2)
            sy = random.uniform(0.05, 0.2)
            x1 = random.randint(0, int(w * (1 - sx)))
            y1 = random.randint(0, int(h * (1 - sy)))
            draw.rectangle([x1, y1, x1 + int(w * sx), y1 + int(h * sy)], fill=0)

        return img


def make_transform(
    img_size: int,
    mean: list[float],
    std: list[float],
    mode: str,
) -> T.Compose:
    return T.Compose([
        SteelSurfaceAugment(img_size, mode),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


# ── Dataset ──────────────────────────────────────────────────────────────────

class NEUCLSDataset(Dataset):
    """
    NEU-CLS steel surface defect dataset.

    Expects the Kaggle NEU-DET layout:
        root/
          train/images/{class_folder}/*.jpg
          validation/images/{class_folder}/*.jpg

    Args:
        paths     : array of image paths
        labels    : array of integer class indices
        transform : callable transform (use make_transform())
    """

    def __init__(
        self,
        paths:     np.ndarray,
        labels:    np.ndarray,
        transform: T.Compose,
    ):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        img   = Image.open(self.paths[idx])
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label, str(self.paths[idx])


def load_all_paths(det_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Scan NEU-DET root and return (paths, labels) arrays merged across
    train + validation splits. Caller is responsible for re-splitting.

    Returns:
        paths  : (N,) object array of pathlib.Path
        labels : (N,) int array
    """
    all_paths, all_labels = [], []
    for split in ["train", "validation"]:
        for folder, class_name in FOLDER_TO_CLASS.items():
            folder_path = det_root / split / "images" / folder
            if not folder_path.exists():
                continue
            imgs = sorted(folder_path.glob("*.jpg"))
            all_paths.extend(imgs)
            all_labels.extend([CLASS_TO_IDX[class_name]] * len(imgs))

    return np.array(all_paths), np.array(all_labels)


def compute_dataset_stats(
    paths: np.ndarray,
    n_sample: int = 300,
    seed: int = 42,
) -> tuple[list[float], list[float]]:
    """
    Compute per-channel mean and std from a random sample of images.
    Returns (mean, std) as lists of 3 floats.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=min(n_sample, len(paths)), replace=False)
    pixels = []
    for i in idx:
        arr = np.array(Image.open(paths[i]).convert("RGB")) / 255.0
        pixels.append(arr.reshape(-1, 3))
    pixels = np.concatenate(pixels, axis=0)
    return pixels.mean(axis=0).tolist(), pixels.std(axis=0).tolist()
