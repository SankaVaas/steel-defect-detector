#!/usr/bin/env python3
"""
train.py — stratified k-fold cross-validation training for the steel defect
classifier.

Why cross-validation instead of a single train/val split?
NEU-CLS is small (1,800 images). A single 80/20 split leaves only ~360
validation images, so the reported accuracy has high variance — you could
get a flattering or unflattering split purely by luck. 5-fold CV trains the
model 5 times on different splits and reports mean +/- std, which is a much
more trustworthy estimate of real-world performance, and it also produces 5
checkpoints that can be ensembled at inference time.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --epochs 5 --folds 2 --backbone resnet18   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from src.dataset import NEUCLSDataset, load_all_paths, make_transform
from src.engine import build_optimizer, build_scheduler, evaluate, train_one_epoch
from src.gradcam import save_gradcam_grid
from src.model import build_model
from src.utils import (
    EarlyStopping,
    compute_metrics,
    get_device,
    load_config,
    plot_confusion_matrix,
    plot_training_curves,
    save_checkpoint,
    set_seed,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override cfg epochs")
    ap.add_argument("--folds", type=int, default=None, help="override cfg n_folds")
    ap.add_argument("--fold-to-train", type=int, default=None, help="train only this fold index (0-based)")
    ap.add_argument("--backbone", type=str, default=None, help="override cfg backbone")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--no-pretrained", action="store_true",
                     help="train from random init (use if ImageNet weights can't be downloaded, e.g. offline)")
    ap.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs)")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def run_fold(fold_idx, train_idx, val_idx, paths, labels, cfg, device, out_dir, args):
    print(f"\n{'='*70}\nFold {fold_idx+1}/{cfg['n_folds']}"
          f"  (train={len(train_idx)}, val={len(val_idx)})\n{'='*70}")

    train_tf = make_transform(cfg["img_size"], cfg["mean"], cfg["std"], mode="train")
    val_tf = make_transform(cfg["img_size"], cfg["mean"], cfg["std"], mode="val")

    train_ds = NEUCLSDataset(paths[train_idx], labels[train_idx], train_tf)
    val_ds = NEUCLSDataset(paths[val_idx], labels[val_idx], val_tf)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=max(1, len(train_loader)))
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scaler = torch.amp.GradScaler(enabled=(cfg["amp"] and device.type == "cuda"))
    stopper = EarlyStopping(patience=args.patience)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    fold_dir = Path(out_dir) / f"fold{fold_idx}"
    best_ckpt_path = fold_dir / "best_model.pt"

    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, cfg, scaler
        )
        val_loss, y_true, y_pred, _ = evaluate(model, val_loader, criterion, device)
        metrics = compute_metrics(y_true, y_pred, cfg["class_names"])

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(metrics["accuracy"])
        history["val_f1"].append(metrics["macro_f1"])

        dt = time.time() - t0
        print(f"epoch {epoch+1:02d}/{cfg['epochs']} ({dt:.1f}s)  "
              f"train_loss={train_loss:.3f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.3f} val_acc={metrics['accuracy']:.3f} val_f1={metrics['macro_f1']:.3f}")

        is_best = stopper.step(metrics["macro_f1"], epoch)
        if is_best:
            save_checkpoint(best_ckpt_path, model, optimizer, epoch,
                             extra={"metrics": metrics, "cfg": cfg})
        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch+1} (best epoch {stopper.best_epoch+1})")
            break

    # Reload best checkpoint for final fold metrics + Grad-CAM samples.
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    final_metrics = ckpt["metrics"]

    plot_training_curves(history, fold_dir / "training_curves.png")
    plot_confusion_matrix(np.array(final_metrics["confusion_matrix"]), cfg["class_names"],
                           fold_dir / "confusion_matrix.png")

    # A handful of qualitative Grad-CAM examples from the validation set.
    n_show = min(6, len(val_ds))
    sample_idx = np.random.default_rng(args.seed).choice(len(val_ds), size=n_show, replace=False)
    samples = [val_ds[i] for i in sample_idx]
    with torch.no_grad():
        preds = []
        for img_t, _, _ in samples:
            logits = model(img_t.unsqueeze(0).to(device))
            preds.append(int(logits.argmax(dim=1).item()))
    save_gradcam_grid(model, device, samples, cfg["class_names"], cfg["mean"], cfg["std"],
                       fold_dir / "gradcam_samples.png", predictions=preds)

    with open(fold_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"Fold {fold_idx+1} best macro-F1={final_metrics['macro_f1']:.4f} "
          f"(epoch {stopper.best_epoch+1}) -> {fold_dir}")
    return final_metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.folds is not None:
        cfg["n_folds"] = args.folds
    if args.backbone is not None:
        cfg["backbone"] = args.backbone
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.no_pretrained:
        cfg["pretrained"] = False

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}  |  backbone: {cfg['backbone']}  |  epochs: {cfg['epochs']}  "
          f"|  folds: {cfg['n_folds']}")

    paths, labels = load_all_paths(Path(cfg["data_root"]))
    if len(paths) == 0:
        raise SystemExit(
            f"No images found under {cfg['data_root']}. Run scripts/prepare_data.py first "
            "(see README.md 'Data' section)."
        )
    print(f"Loaded {len(paths)} images across {cfg['num_classes']} classes: "
          f"{dict(zip(*np.unique(labels, return_counts=True)))}")

    skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=args.seed)
    splits = list(skf.split(paths, labels))

    fold_range = [args.fold_to_train] if args.fold_to_train is not None else range(cfg["n_folds"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_fold_metrics = []
    for fold_idx in fold_range:
        train_idx, val_idx = splits[fold_idx]
        metrics = run_fold(fold_idx, train_idx, val_idx, paths, labels, cfg, device, out_dir, args)
        all_fold_metrics.append(metrics)

    accs = [m["accuracy"] for m in all_fold_metrics]
    f1s = [m["macro_f1"] for m in all_fold_metrics]
    summary = {
        "folds_trained": list(fold_range),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "per_fold": [{"fold": f, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
                     for f, m in zip(fold_range, all_fold_metrics)],
    }
    with open(out_dir / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}\nCross-validation summary "
          f"({len(fold_range)} fold(s))\n{'='*70}")
    print(f"Accuracy : {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
    print(f"Macro-F1 : {summary['macro_f1_mean']:.4f} +/- {summary['macro_f1_std']:.4f}")
    print(f"\nSaved to {out_dir}/cv_summary.json")


if __name__ == "__main__":
    main()
