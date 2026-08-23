#!/usr/bin/env python3
"""
calibrate.py — fit a temperature-scaling scalar for each fold checkpoint,
using that fold's own validation split as the calibration set (it was never
trained on, so it's a fair calibration target — and re-uses data we already
have instead of needing a separate held-out set).

Usage:
    python scripts/calibrate.py --config configs/default.yaml --groups-file groups.json \
        --out-dir outputs --out-file outputs/calibration.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src.calibration import expected_calibration_error, fit_temperature, reliability_diagram
from src.dataset import NEUCLSDataset, load_all_paths, make_transform
from src.model import build_model
from src.utils import get_device, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--groups-file", default=None)
    ap.add_argument("--out-dir", default="outputs", help="directory containing fold{N}/best_model.pt")
    ap.add_argument("--out-file", default="outputs/calibration.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    paths, labels = load_all_paths(Path(cfg["data_root"]))

    if args.groups_file:
        with open(args.groups_file) as f:
            name_to_group = json.load(f)
        groups = np.array([name_to_group[Path(p).name] for p in paths])
        skf = StratifiedGroupKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=args.seed)
        splits = list(skf.split(paths, labels, groups=groups))
    else:
        print("WARNING: no --groups-file given; calibration set may share near-duplicates with training.")
        skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=args.seed)
        splits = list(skf.split(paths, labels))

    val_tf = make_transform(cfg["img_size"], cfg["mean"], cfg["std"], mode="val")

    temperatures = {}
    out_dir = Path(args.out_dir)
    for fold_idx, (_, val_idx) in enumerate(splits):
        ckpt_path = out_dir / f"fold{fold_idx}" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"Skipping fold {fold_idx}: no checkpoint at {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(ckpt["cfg"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        val_ds = NEUCLSDataset(paths[val_idx], labels[val_idx], val_tf)
        loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=1)

        all_logits, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls, _ in loader:
                logits = model(imgs.to(device))
                all_logits.append(logits.cpu())
                all_labels.append(lbls)
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        # Pre-calibration ECE (T=1, i.e. raw softmax)
        raw_probs = torch.softmax(all_logits, dim=1).numpy()
        ece_before = expected_calibration_error(raw_probs, all_labels.numpy())

        T = fit_temperature(all_logits, all_labels)

        calibrated_probs = torch.softmax(all_logits / T, dim=1).numpy()
        ece_after = expected_calibration_error(calibrated_probs, all_labels.numpy())

        fold_dir = out_dir / f"fold{fold_idx}"
        reliability_diagram(raw_probs, all_labels.numpy(), fold_dir / "reliability_before.png",
                             title=f"Fold {fold_idx} — before calibration (T=1.0)")
        reliability_diagram(calibrated_probs, all_labels.numpy(), fold_dir / "reliability_after.png",
                             title=f"Fold {fold_idx} — after calibration (T={T:.2f})")

        temperatures[f"fold{fold_idx}"] = T
        temperatures[str(ckpt_path)] = T  # also keyed by full path for direct lookup
        print(f"Fold {fold_idx}: T={T:.3f}  ECE {ece_before:.4f} -> {ece_after:.4f}  "
              f"({'improved' if ece_after < ece_before else 'no improvement'})")

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(temperatures, f, indent=2)
    print(f"\nSaved temperatures to {args.out_file}")


if __name__ == "__main__":
    main()
