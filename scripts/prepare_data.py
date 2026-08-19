#!/usr/bin/env python3
"""
prepare_data.py — reorganise a raw NEU-CLS / NEU-DET download into the layout
`src/dataset.py` expects:

    <out_root>/
      train/images/{crazing,inclusion,patches,pitted_surface,rolled-in_scale,scratches}/*.jpg
      validation/images/{...same...}/*.jpg

The NEU surface-defect dataset is distributed under a few different folder
layouts depending on the source (Kaggle "NEU-DET" vs "NEU-CLS", GitHub
mirrors, ...). This script auto-detects the common cases:

  1. Flat dump:      <in_root>/IMAGES/<class>_<n>.jpg
                      <in_root>/Validation_Images/<class>_<n>.jpg   (optional)
  2. Kaggle NEU-DET:  <in_root>/{train,validation}/images/<class>/*.jpg  (already correct)
  3. Class folders:   <in_root>/<class>/*.jpg

Images are grouped by class inferred from the filename/folder, shuffled with
a fixed seed, and split train/validation (default 90/10) unless the source
already ships an explicit validation split, in which case that split is
respected.

Usage:
    python scripts/prepare_data.py --input /path/to/raw/download --output ./NEU-CLS

Where to get the raw data (kaggle CLI, requires a free Kaggle account/token):
    pip install kaggle
    kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database
    unzip neu-surface-defect-database.zip -d raw_neu
    python scripts/prepare_data.py --input raw_neu --output ./NEU-CLS
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np

FOLDER_TO_CLASS_PREFIX = {
    "crazing": "crazing",
    "inclusion": "inclusion",
    "patches": "patches",
    "pitted_surface": "pitted_surface",
    "rolled-in_scale": "rolled-in_scale",
    "scratches": "scratches",
}
CLASS_PREFIXES = list(FOLDER_TO_CLASS_PREFIX.keys())


def infer_class_from_name(name: str) -> str | None:
    """Match a filename or folder name against known class prefixes."""
    lname = name.lower()
    # Longest-prefix-first so 'pitted_surface' doesn't lose to a shorter accidental match.
    for prefix in sorted(CLASS_PREFIXES, key=len, reverse=True):
        if lname.startswith(prefix):
            return prefix
    return None


def collect_flat(dir_path: Path) -> dict[str, list[Path]]:
    """Collect {class: [image paths]} from a flat directory of '<class>_<n>.jpg' files."""
    buckets: dict[str, list[Path]] = {c: [] for c in CLASS_PREFIXES}
    for img in dir_path.glob("*.jpg"):
        cls = infer_class_from_name(img.stem)
        if cls:
            buckets[cls].append(img)
    return buckets


def collect_class_folders(root: Path) -> dict[str, list[Path]]:
    buckets: dict[str, list[Path]] = {c: [] for c in CLASS_PREFIXES}
    for child in root.iterdir():
        if not child.is_dir():
            continue
        cls = infer_class_from_name(child.name)
        if cls:
            buckets[cls].extend(sorted(child.glob("*.jpg")))
    return buckets


def already_correct_layout(root: Path) -> bool:
    return (root / "train" / "images").exists() and (root / "validation" / "images").exists()


def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="Raw dataset root (as downloaded/extracted)")
    ap.add_argument("--output", required=True, type=Path, help="Destination root, e.g. ./NEU-CLS")
    ap.add_argument("--val-frac", type=float, default=0.1, help="Fraction held out for validation per class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink",
                     help="symlink is fast/cheap; use copy if you need a self-contained folder")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    if already_correct_layout(args.input):
        print(f"Input already matches the expected layout — copying/linking as-is from {args.input}")
        for split in ["train", "validation"]:
            src_split = args.input / split / "images"
            for cls_dir in src_split.iterdir():
                dst_split = args.output / split / "images" / cls_dir.name
                for img in cls_dir.glob("*.jpg"):
                    link_or_copy(img, dst_split / img.name, args.mode)
        print(f"Done. Data ready at {args.output}")
        return

    # Try flat "IMAGES/" dump first (common GitHub mirror layout), then class-folder layout.
    train_buckets: dict[str, list[Path]] = {c: [] for c in CLASS_PREFIXES}
    val_buckets: dict[str, list[Path]] = {c: [] for c in CLASS_PREFIXES}

    images_dir = args.input / "IMAGES"
    val_images_dir = args.input / "Validation_Images"
    if images_dir.exists():
        train_buckets = collect_flat(images_dir)
        if val_images_dir.exists():
            val_buckets = collect_flat(val_images_dir)
    else:
        train_buckets = collect_class_folders(args.input)

    total_found = sum(len(v) for v in train_buckets.values()) + sum(len(v) for v in val_buckets.values())
    if total_found == 0:
        raise SystemExit(
            f"Could not find any class-labelled .jpg images under {args.input}. "
            "Check --input points at the extracted NEU-CLS/NEU-DET download."
        )

    # If no explicit validation split was found, carve one out per class.
    has_explicit_val = any(len(v) > 0 for v in val_buckets.values())
    if not has_explicit_val:
        for cls, paths in train_buckets.items():
            paths = sorted(paths)
            rng.shuffle(paths)
            n_val = max(1, int(len(paths) * args.val_frac))
            val_buckets[cls] = paths[:n_val]
            train_buckets[cls] = paths[n_val:]

    for split, buckets in [("train", train_buckets), ("validation", val_buckets)]:
        for cls, paths in buckets.items():
            dst_dir = args.output / split / "images" / cls
            for img in paths:
                link_or_copy(img, dst_dir / img.name, args.mode)

    print("Class counts:")
    for cls in CLASS_PREFIXES:
        print(f"  {cls:16s} train={len(train_buckets[cls]):4d}  val={len(val_buckets[cls]):4d}")
    print(f"\nDone. Data ready at {args.output}")


if __name__ == "__main__":
    main()
