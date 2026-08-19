"""
engine.py — one epoch of training / evaluation.

Kept separate from train.py so it can be unit-tested and re-used by both the
cross-validation driver and any future scripts (e.g. hyperparameter search).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils import AverageMeter


def build_optimizer(model, cfg: dict) -> torch.optim.Optimizer:
    param_groups = model.param_groups(lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    return torch.optim.AdamW(param_groups)


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """
    Linear warmup for `warmup_epochs`, then cosine decay to 0 for the rest of
    training. Warmup avoids destabilising the randomly-initialised head with
    large gradients in the first few steps; cosine decay gives a smooth
    LR trajectory that empirically outperforms step decay on small datasets.
    """
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch
    total_steps = cfg["epochs"] * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, cfg, scaler=None):
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()

    pbar = tqdm(loader, desc="train", leave=False)
    for imgs, labels, _ in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        use_amp = cfg.get("amp", False) and device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
        scheduler.step()

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        loss_meter.update(loss.item(), n=imgs.size(0))
        acc_meter.update(acc, n=imgs.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.3f}", acc=f"{acc_meter.avg:.3f}")

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels, all_paths = [], [], []

    for imgs, labels, paths in tqdm(loader, desc="eval", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss_meter.update(loss.item(), n=imgs.size(0))

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_paths.extend(paths)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return loss_meter.avg, y_true, y_pred, all_paths
