#!/usr/bin/env python3
"""
prepare_data.py — reorganise a raw NEU-CLS / NEU-DET download into the layout
`src/dataset.py` expects:

    <out_root>/
      train/images/{crazing,inclusion,patches,pitted_surface,rolled-in_scale,scratches}/*.jpg
      validation/images/{...same...}/*.jpg

Real-world NEU downloads come in several shapes depending on source, and this
script is deliberately paranoid about all of them because getting this wrong
produces a silent "found 0 images" failure rather than a helpful error:

  1. Kaggle NEU-DET (kaustubhdikshit/neu-surface-defect-database): nested
     arbitrarily deep, e.g. `raw_neu/NEU-DET/train/images/crazing/*.jpg`,
     often alongside a sibling `annotations/` folder for object detection
     that must be ignored. Class folder spelling also varies by source:
     'rolled-in_scale' vs 'rolled-in-scale' vs 'rolled in scale'.
  2. Flat dump: `<in_root>/IMAGES/<class>_<n>.jpg`,
     `<in_root>/Validation_Images/<class>_<n>.jpg` (GitHub mirrors).
  3. Bare class folders with no train/validation split at all:
     `<in_root>/<class>/*.jpg` (e.g. the "single folder with 1800 images"
     variant some mirrors ship).

The script walks the ENTIRE input tree (any depth), classifies every image
by matching its filename or containing folder name against the six known
classes (spelling-normalised — case/hyphen/underscore/space insensitive),
and rebuilds the expected layout. If it finds an explicit train/validation
split anywhere in the tree it respects it; otherwise it carves out a
stratified validation split itself.

Usage:
    python scripts/prepare_data.py --input raw_neu --output ./NEU-CLS
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

# Canonical output folder name -> normalised (letters-only, lowercase) key
CANONICAL_CLASSES = {
    "crazing": "crazing",
    "inclusion": "inclusion",
    "patches": "patches",
    "pitted_surface": "pittedsurface",
    "rolled-in_scale": "rolledinscale",
    "scratches": "scratches",
}
NORM_TO_CANONICAL = {v: k for k, v in CANONICAL_CLASSES.items()}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

VAL_DIR_NAMES = {"validation", "valid", "val", "test"}
TRAIN_DIR_NAMES = {"train", "training"}
IGNORE_DIR_NAMES = {"annotations", "labels", "xml", "ann"}  # object-detection metadata, not images


def normalize(s: str) -> str:
    """Lowercase, letters-only — so 'Rolled-in_Scale', 'rolled in scale' and
    'rolled-in-scale' all collapse to the same key."""
    return re.sub(r"[^a-z]", "", s.lower())


def match_class(name: str) -> str | None:
    """Match a filename stem or folder name against the six canonical classes.
    Tries the longest normalised class key first so 'pitted_surface' isn't
    shadowed by a shorter accidental prefix match."""
    norm = normalize(name)
    for norm_key in sorted(NORM_TO_CANONICAL, key=len, reverse=True):
        if norm.startswith(norm_key) or norm_key in norm:
            return NORM_TO_CANONICAL[norm_key]
    return None


def split_context(path: Path, input_root: Path) -> str | None:
    """Walk a file's path components (relative to input_root) looking for a
    train/validation marker directory. Returns 'train', 'val', or None."""
    for part in path.relative_to(input_root).parts:
        low = part.lower()
        if low in TRAIN_DIR_NAMES:
            return "train"
        if low in VAL_DIR_NAMES:
            return "val"
    return None


def scan_all_images(input_root: Path) -> tuple[dict, dict]:
    """
    Walk the entire input tree. For every image file, determine:
      - its class (from filename stem or immediate parent folder name)
      - its split, if the path passes through a train/validation marker dir
    Returns (train_buckets, val_buckets), each {canonical_class: [Path, ...]}.
    Images under an 'unsplit' context (no train/val marker found) are
    returned separately via the special key '__unsplit__' merged into both
    dicts' bookkeeping — handled by the caller.
    """
    train_buckets: dict[str, list[Path]] = defaultdict(list)
    val_buckets: dict[str, list[Path]] = defaultdict(list)
    unsplit_buckets: dict[str, list[Path]] = defaultdict(list)

    for path in input_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
            continue
        # Skip files that live under an annotations/labels sibling tree.
        rel_parts_lower = {p.lower() for p in path.relative_to(input_root).parts}
        if rel_parts_lower & IGNORE_DIR_NAMES:
            continue

        cls = match_class(path.stem) or match_class(path.parent.name)
        if cls is None:
            continue

        split = split_context(path, input_root)
        if split == "train":
            train_buckets[cls].append(path)
        elif split == "val":
            val_buckets[cls].append(path)
        else:
            unsplit_buckets[cls].append(path)

    return train_buckets, val_buckets, unsplit_buckets


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
    ap.add_argument("--val-frac", type=float, default=0.1, help="Fraction held out for validation per class (only used when no explicit split is found)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["symlink", "copy"], default="copy",
                     help="copy is safest across filesystems/Colab; symlink is faster if input and output share a disk")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"--input path does not exist: {args.input}")

    rng = np.random.default_rng(args.seed)

    print(f"Scanning {args.input} recursively for NEU-CLS images...")
    train_buckets, val_buckets, unsplit_buckets = scan_all_images(args.input)

    total_found = (
        sum(len(v) for v in train_buckets.values())
        + sum(len(v) for v in val_buckets.values())
        + sum(len(v) for v in unsplit_buckets.values())
    )
    if total_found == 0:
        # Show what's actually there to help debug instead of a bare failure.
        sample = [str(p.relative_to(args.input)) for p in list(args.input.rglob("*"))[:15]]
        raise SystemExit(
            f"Could not find any class-labelled images under {args.input}.\n"
            f"Expected class names (any spelling/case): {list(CANONICAL_CLASSES)}\n"
            f"First few files actually found there:\n  " + "\n  ".join(sample) +
            "\n\nIf this is a zip you just downloaded, double check --input points at the "
            "EXTRACTED folder, not the .zip file itself."
        )

    # Fold any unsplit images into train, then carve out validation per class
    # ourselves for classes that don't already have an explicit val split.
    for cls, paths in unsplit_buckets.items():
        train_buckets[cls].extend(paths)

    for cls in CANONICAL_CLASSES:
        if len(val_buckets.get(cls, [])) > 0:
            continue  # explicit split already exists for this class, keep it
        paths = sorted(set(train_buckets.get(cls, [])))
        rng.shuffle(paths)
        n_val = max(1, int(len(paths) * args.val_frac))
        val_buckets[cls] = paths[:n_val]
        train_buckets[cls] = paths[n_val:]

    print(f"\nWriting {args.mode} layout to {args.output} ...")
    for split, buckets in [("train", train_buckets), ("validation", val_buckets)]:
        for cls in CANONICAL_CLASSES:
            paths = sorted(set(buckets.get(cls, [])))
            dst_dir = args.output / split / "images" / cls
            for img in paths:
                link_or_copy(img, dst_dir / img.name, args.mode)

    print("\nClass counts:")
    total = 0
    for cls in CANONICAL_CLASSES:
        n_train = len(set(train_buckets.get(cls, [])))
        n_val = len(set(val_buckets.get(cls, [])))
        total += n_train + n_val
        flag = "  <-- WARNING: no images found for this class" if (n_train + n_val) == 0 else ""
        print(f"  {cls:16s} train={n_train:4d}  val={n_val:4d}{flag}")

    print(f"\nTotal images placed: {total}")
    print(f"Done. Data ready at {args.output}")


if __name__ == "__main__":
    main()