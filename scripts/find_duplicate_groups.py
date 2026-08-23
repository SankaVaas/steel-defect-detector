#!/usr/bin/env python3
"""
find_duplicate_groups.py — detect near-duplicate images in NEU-CLS and assign
every image a `group_id` such that near-duplicates share a group.

Why this exists
----------------
NEU-CLS is known to contain many near-duplicate images within each class —
crops of the same physical defect sample photographed multiple times, or
overlapping crops of the same source scan. A random (or stratified) k-fold
split treats every image as independent evidence, so it will routinely place
near-duplicates of the *same* physical sample on both sides of a train/val
split. The "validation" image is then not really unseen — the model has
already learned that exact texture from its near-twin in the training set.
This is a classic and easy-to-miss source of leakage, and it fully explains
validation accuracy/macro-F1 saturating at a suspicious, low-variance 1.0000
across every fold: it isn't that the model generalizes perfectly, it's that
the "held out" data wasn't really held out.

Method: an 8x8 average-hash (aHash) perceptual hash per image, unioned via
union-find whenever two images in the *same class* have Hamming distance
below `--threshold` (out of 64 bits). This is deliberately class-scoped and
conservative (default threshold=4) — cross-class collisions are structurally
impossible since classes never merge, and a low threshold only catches
images that are genuinely near-identical, not just similar-looking crazing.

Usage:
    python scripts/find_duplicate_groups.py --data-root ./NEU-CLS --out groups.json

`groups.json` maps every image filename to an integer group id. `train.py`
loads this (if present) and uses it as the `groups` argument to
StratifiedGroupKFold, so every near-duplicate cluster stays entirely on one
side of every fold.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def ahash(path: Path, size: int = 8) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float64)
    return (arr > arr.mean()).flatten()


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def find_all_images(data_root: Path) -> list[Path]:
    imgs = []
    for split in ["train", "validation"]:
        for cls in CLASS_NAMES:
            imgs.extend((data_root / split / "images" / cls).glob("*.jpg"))
    return sorted(imgs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", default="groups.json", type=Path)
    ap.add_argument("--threshold", type=int, default=4, help="max Hamming distance (out of 64) to call two images near-duplicates")
    args = ap.parse_args()

    imgs = find_all_images(args.data_root)
    if not imgs:
        raise SystemExit(f"No images found under {args.data_root}. Did you run scripts/prepare_data.py?")

    print(f"Hashing {len(imgs)} images...")
    hashes = {}
    classes = {}
    for p in tqdm(imgs):
        # class = parent-of-parent folder name (.../images/<class>/<file>.jpg)
        classes[p.name] = p.parent.name
        hashes[p.name] = ahash(p)

    uf = UnionFind(hashes.keys())
    by_class = defaultdict(list)
    for name, cls in classes.items():
        by_class[cls].append(name)

    print("Finding near-duplicate pairs within each class...")
    for cls, members in tqdm(by_class.items()):
        arrs = np.stack([hashes[m] for m in members])
        n = len(members)
        for i in range(n):
            dist = (arrs[i] != arrs[i + 1:]).sum(axis=1)
            for j, d in enumerate(dist, start=i + 1):
                if d <= args.threshold:
                    uf.union(members[i], members[j])

    roots = {name: uf.find(name) for name in hashes}
    root_to_gid = {root: gid for gid, root in enumerate(sorted(set(roots.values())))}
    name_to_group = {name: root_to_gid[root] for name, root in roots.items()}

    group_sizes = defaultdict(int)
    for gid in name_to_group.values():
        group_sizes[gid] += 1
    n_multi = sum(1 for s in group_sizes.values() if s > 1)
    n_imgs_in_multi = sum(s for s in group_sizes.values() if s > 1)

    print(f"\n{len(set(name_to_group.values()))} groups from {len(imgs)} images")
    print(f"{n_multi} groups contain >1 image (near-duplicate clusters)")
    print(f"{n_imgs_in_multi} images ({n_imgs_in_multi/len(imgs):.1%}) belong to a near-duplicate cluster")
    print("If this fraction is large (it typically is for NEU-CLS, ~30%+), a plain "
          "random/stratified k-fold split WILL leak near-duplicates across train/val.")

    with open(args.out, "w") as f:
        json.dump(name_to_group, f)
    print(f"\nSaved group assignments to {args.out}")
    print("Pass this to train.py via --groups-file to use leakage-safe StratifiedGroupKFold splitting.")


if __name__ == "__main__":
    main()
