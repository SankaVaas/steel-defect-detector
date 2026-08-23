"""
calibration.py — temperature scaling for a trained classifier.

The problem: modern deep networks (this one included) are systematically
overconfident. A "92% confidence" prediction, taken at face value, is
routinely wrong far more than 8% of the time — the softmax number reflects
how peaked the training loss pushed the logits, not the model's true
empirical accuracy at that confidence level. Any automated accept/reject
decision built on top of raw softmax is built on a number that doesn't mean
what it appears to mean.

Temperature scaling (Guo et al., 2017, "On Calibration of Modern Neural
Networks") is the standard fix: divide the logits by a single learned
scalar T > 1 before the softmax. It doesn't change which class wins
(argmax is unaffected — same accuracy), it only reshapes the confidence
distribution so that, empirically, "70% confident" predictions really are
correct about 70% of the time. T is fit by minimizing negative log-
likelihood on a held-out calibration set (here: each fold's own validation
split, which the model never trained on).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> float:
    """
    Fit a single scalar temperature T minimizing NLL of softmax(logits / T)
    against `labels`, via LBFGS (the standard optimizer for this — it's a
    1-parameter convex-ish problem, converges in a handful of steps).
    """
    logits = logits.detach()
    labels = labels.detach()
    log_temperature = torch.zeros(1, requires_grad=True)  # optimize in log-space to keep T > 0

    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        T = log_temperature.exp()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return calibrated probabilities: softmax(logits / T)."""
    return F.softmax(logits / temperature, dim=-1)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    ECE: bin predictions by confidence, compare each bin's average confidence
    to its actual accuracy, weight by bin size. Lower is better-calibrated;
    0 is perfect. This is the standard scalar summary of "how much can I
    trust the confidence numbers," reported alongside accuracy/F1 — a model
    can have high accuracy and still be badly calibrated.
    """
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float64)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_diagram(probs: np.ndarray, labels: np.ndarray, out_path: str, n_bins: int = 15,
                         title: str = "Reliability diagram"):
    """Plot confidence-vs-accuracy per bin against the y=x perfect-calibration line."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float64)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_accs, bin_counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        bin_accs.append(correct[mask].mean())
        bin_counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.bar(bin_centers, bin_accs, width=1 / n_bins * 0.9, alpha=0.75, edgecolor="black", label="model")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
