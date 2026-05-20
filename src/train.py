"""
train.py — training and validation loops.

Design decisions:
    - AMP (autocast + GradScaler): ~1.5× speedup on T4 with negligible accuracy cost
    - Per-step scheduler (not per-epoch): finer LR control during warmup
    - Gradient clipping at 1.0: guards against exploding gradients at unfreeze
    - set_to_none=True in zero_grad: faster than zero fill; frees gradient memory
    - F1-macro metric tracked live: the deployment-relevant metric, not loss
"""

import math
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# ── Scheduler factory ─────────────────────────────────────────────────────────

def make_lr_lambda(total_steps: int, warmup_steps: int):
    """
    Returns a lambda for LambdaLR implementing:
        linear warmup (0 → 1) over warmup_steps
        cosine decay (1 → 0) over remaining steps
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


# ── One epoch ────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimizer:  torch.optim.Optimizer,
    scheduler:  torch.optim.lr_scheduler.LambdaLR,
    scaler:     GradScaler,
    device:     torch.device,
    epoch:      int,
    grad_clip:  float = 1.0,
    amp:        bool  = True,
) -> tuple[float, float]:
    """
    Train for one epoch.

    Returns:
        epoch_loss : mean cross-entropy over all batches
        epoch_f1   : macro-averaged F1 over all predictions
    """
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Ep {epoch:02d} [Train]", leave=False)
    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")

    return (
        running_loss / len(loader),
        f1_score(all_labels, all_preds, average="macro"),
    )


@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    amp:       bool = True,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate on validation set.

    Returns:
        loss      : mean cross-entropy
        f1        : macro F1
        preds     : (N,) predicted class indices
        labels    : (N,) ground-truth class indices
        probs     : (N, C) softmax probabilities
    """
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels, _ in tqdm(loader, desc="           [Val] ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device)

        with autocast(enabled=amp):
            logits = model(images)
            loss   = criterion(logits, labels)

        running_loss += loss.item()
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(probs.argmax(axis=1))
        all_labels.extend(labels.cpu().numpy())

    return (
        running_loss / len(loader),
        f1_score(all_labels, all_preds, average="macro"),
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


# ── Full training run ─────────────────────────────────────────────────────────

def run_training(
    model:          nn.Module,
    train_loader:   DataLoader,
    val_loader:     DataLoader,
    cfg:            dict,
    device:         torch.device,
    checkpoint_path: str = "best_model.pth",
) -> dict:
    """
    Full training run with two-phase freeze/unfreeze, AMP, and best-F1 checkpointing.

    Returns:
        history : dict with train_loss, train_f1, val_loss, val_f1 lists
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": cfg["lr"] * 0.1},
        {"params": model.head.parameters(),     "lr": cfg["lr"]},
    ], weight_decay=cfg["weight_decay"])

    total_steps  = cfg["epochs"] * len(train_loader)
    warmup_steps = cfg["warmup_epochs"] * len(train_loader)
    scheduler    = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(total_steps, warmup_steps)
    )
    scaler = GradScaler(enabled=cfg["amp"])

    history      = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": []}
    best_val_f1  = 0.0

    model.freeze_backbone()
    print(f"Backbone frozen for warmup ({cfg['warmup_epochs']} epochs)")

    for epoch in range(1, cfg["epochs"] + 1):
        if epoch == cfg["warmup_epochs"] + 1:
            model.unfreeze_backbone()
            print(f"Backbone unfrozen at epoch {epoch}")

        t0 = time.time()
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scheduler, scaler, device, epoch,
            cfg["grad_clip"], cfg["amp"],
        )
        val_loss, val_f1, _, _, _ = validate(
            model, val_loader, criterion, device, cfg["amp"]
        )
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        flag = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            import pathlib
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_f1":      val_f1,
                "cfg":         {k: str(v) if isinstance(v, pathlib.Path) else v
                                for k, v in cfg.items()},
            }, checkpoint_path)
            flag = " ← best"

        print(
            f"Ep {epoch:02d}/{cfg['epochs']} | "
            f"TL={train_loss:.4f} TF1={train_f1:.4f} | "
            f"VL={val_loss:.4f} VF1={val_f1:.4f} | "
            f"{elapsed:.0f}s{flag}"
        )

    return history
