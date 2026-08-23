#!/usr/bin/env python3
"""
stress_test.py — the sanity check the README's roadmap calls for before
trusting a benchmark accuracy number: does the model degrade gracefully
under realistic domain shift, and does it know when an input isn't one of
its six classes at all?

Two experiments:

1. ROBUSTNESS UNDER CORRUPTION. NEU-CLS validation images are pristine:
   fixed camera, consistent lighting, no motion blur. A production camera
   feed won't be. This experiment applies increasing severities of
   Gaussian blur, brightness/contrast shift, additive noise, JPEG
   compression, and downscale/upscale to real validation images and
   measures accuracy at each severity. A model that collapses under mild
   blur or a brightness shift is not ready for a plant floor, however high
   its clean-image accuracy is.

2. OUT-OF-DISTRIBUTION BEHAVIOUR. The classifier has exactly six buckets
   and no "none of the above" option — softmax always sums to 1 no matter
   what you show it. This experiment feeds the ensemble deliberately
   non-steel images (synthetic textures with no resemblance to any defect
   class) and checks whether calibrated confidence / cross-model agreement
   correctly drops, i.e. whether the uncertainty signal would actually
   catch this at decision time rather than confidently guessing a class.

Usage:
    python scripts/stress_test.py --config configs/default.yaml --out-dir outputs \
        --calibration outputs/calibration.json --groups-file groups.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from tqdm import tqdm

from src.dataset import load_all_paths, make_transform
from src.ensemble import SteelDefectEnsemble
from src.utils import get_device, load_config


# ---------------------------------------------------------------------------
# Corruptions — each takes a PIL image and a severity in [1, 5] and returns a
# corrupted PIL image. Modelled on the standard "ImageNet-C"-style corruption
# benchmark, adapted to the grayscale-ish steel domain.
# ---------------------------------------------------------------------------

def corrupt_blur(img: Image.Image, severity: int) -> Image.Image:
    radius = [0.5, 1.0, 1.8, 2.8, 4.0][severity - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def corrupt_brightness(img: Image.Image, severity: int) -> Image.Image:
    delta = [15, 30, 50, 75, 100][severity - 1]
    arr = np.asarray(img).astype(np.int16)
    arr = np.clip(arr + delta, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def corrupt_contrast(img: Image.Image, severity: int) -> Image.Image:
    factor = [0.85, 0.7, 0.55, 0.4, 0.25][severity - 1]
    arr = np.asarray(img).astype(np.float32)
    mean = arr.mean()
    arr = np.clip((arr - mean) * factor + mean, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def corrupt_noise(img: Image.Image, severity: int) -> Image.Image:
    sigma = [5, 10, 18, 28, 40][severity - 1]
    arr = np.asarray(img).astype(np.float32)
    noise = np.random.default_rng(0).normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def corrupt_downscale(img: Image.Image, severity: int) -> Image.Image:
    factor = [0.8, 0.6, 0.45, 0.3, 0.2][severity - 1]
    w, h = img.size
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


CORRUPTIONS = {
    "blur": corrupt_blur,
    "brightness": corrupt_brightness,
    "contrast": corrupt_contrast,
    "noise": corrupt_noise,
    "downscale_upscale": corrupt_downscale,
}


# ---------------------------------------------------------------------------
# Synthetic out-of-distribution images — deliberately NOT steel surfaces.
# ---------------------------------------------------------------------------

def make_ood_images(size: int = 224) -> dict[str, Image.Image]:
    rng = np.random.default_rng(0)
    images = {}

    # Pure random noise — no structure at all.
    images["random_noise"] = Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))

    # Smooth colour gradient — structured but nothing like metal texture.
    grad = np.linspace(0, 255, size, dtype=np.uint8)
    grad_img = np.stack([np.tile(grad, (size, 1))] * 3, axis=-1)
    images["smooth_gradient"] = Image.fromarray(grad_img)

    # Regular checkerboard — sharp regular structure, unlike any organic defect texture.
    board = np.zeros((size, size), dtype=np.uint8)
    step = size // 8
    for i in range(0, size, step):
        for j in range(0, size, step):
            if (i // step + j // step) % 2 == 0:
                board[i:i + step, j:j + step] = 255
    images["checkerboard"] = Image.fromarray(np.stack([board] * 3, axis=-1))

    # Solid flat colour — zero texture.
    images["flat_gray"] = Image.new("RGB", (size, size), (128, 128, 128))

    # A simple synthetic "face-like" shape — tests behaviour on a structured,
    # clearly non-metal, non-textured input.
    canvas = Image.new("RGB", (size, size), (220, 200, 180))
    draw = ImageDraw.Draw(canvas)
    draw.ellipse([size * 0.3, size * 0.3, size * 0.7, size * 0.7], outline=(0, 0, 0), width=3)
    images["simple_shape"] = canvas

    return images


def get_val_split(cfg, groups_file, seed=42):
    paths, labels = load_all_paths(Path(cfg["data_root"]))
    if groups_file:
        with open(groups_file) as f:
            name_to_group = json.load(f)
        groups = np.array([name_to_group[Path(p).name] for p in paths])
        skf = StratifiedGroupKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=seed)
        _, val_idx = next(skf.split(paths, labels, groups=groups))
    else:
        skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=seed)
        _, val_idx = next(skf.split(paths, labels))
    return paths[val_idx], labels[val_idx]


def evaluate_corruption(ensemble, val_paths, val_labels, corruption_fn, severity, n_samples):
    idx = np.random.default_rng(0).choice(len(val_paths), size=min(n_samples, len(val_paths)), replace=False)
    correct = 0
    confidences = []
    for i in idx:
        img = Image.open(val_paths[i]).convert("RGB")
        img = corruption_fn(img, severity)
        transform = make_transform(ensemble.img_size, ensemble.mean, ensemble.std, mode="val")
        img_tensor = transform(img)
        result = ensemble.predict(img_tensor)
        correct += int(result.pred_class == val_labels[i])
        confidences.append(result.calibrated_confidence)
    return correct / len(idx), float(np.mean(confidences))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--groups-file", default=None)
    ap.add_argument("--n-samples", type=int, default=60, help="validation images sampled per corruption/severity")
    ap.add_argument("--conf-threshold", type=float, default=0.90)
    ap.add_argument("--agree-threshold", type=float, default=0.99)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    checkpoint_paths = sorted(glob.glob(f"{args.out_dir}/fold*/best_model.pt"))
    if not checkpoint_paths:
        raise SystemExit(f"No checkpoints found under {args.out_dir}/fold*/best_model.pt")
    calibration_path = args.calibration or f"{args.out_dir}/calibration.json"
    ensemble = SteelDefectEnsemble(checkpoint_paths, calibration_path, device)
    print(f"Loaded ensemble of {len(ensemble.members)} models.")

    val_paths, val_labels = get_val_split(cfg, args.groups_file)
    print(f"Using {len(val_paths)} held-out validation images as the clean baseline pool.")

    results = {"robustness": {}, "ood": {}}

    # --- Experiment 1: robustness under corruption ---
    clean_acc, clean_conf = evaluate_corruption(ensemble, val_paths, val_labels, lambda im, s: im, 1, args.n_samples)
    print(f"\nClean accuracy: {clean_acc:.1%} (mean confidence {clean_conf:.1%})")
    results["robustness"]["clean"] = {"accuracy": clean_acc, "mean_confidence": clean_conf}

    for name, fn in CORRUPTIONS.items():
        print(f"\n{name}:")
        results["robustness"][name] = {}
        for severity in range(1, 6):
            acc, conf = evaluate_corruption(ensemble, val_paths, val_labels, fn, severity, args.n_samples)
            print(f"  severity {severity}: accuracy={acc:.1%}  mean_confidence={conf:.1%}")
            results["robustness"][name][f"severity_{severity}"] = {"accuracy": acc, "mean_confidence": conf}

    # --- Experiment 2: out-of-distribution behaviour ---
    print("\nOut-of-distribution probe (these are NOT steel images — a trustworthy "
          "system should show LOW confidence / route to review, not confidently pick a class):")
    ood_images = make_ood_images(size=ensemble.img_size)
    transform = make_transform(ensemble.img_size, ensemble.mean, ensemble.std, mode="val")
    for name, img in ood_images.items():
        img_tensor = transform(img)
        result = ensemble.predict(img_tensor, args.conf_threshold, args.agree_threshold)
        flagged = "review (correctly flagged)" if result.decision == "review" else "AUTO (would NOT be caught)"
        print(f"  {name:20s} -> predicted '{result.pred_name}' at {result.calibrated_confidence:.1%} "
              f"confidence, agreement={result.vote_agreement:.0%}  =>  {flagged}")
        results["ood"][name] = {
            "predicted_class": result.pred_name,
            "calibrated_confidence": result.calibrated_confidence,
            "vote_agreement": result.vote_agreement,
            "decision": result.decision,
        }

    out_path = Path(args.out_dir) / "stress_test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to {out_path}")

    plot_robustness_curves(results["robustness"], Path(args.out_dir) / "robustness_curves.png")


def plot_robustness_curves(robustness: dict, out_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    clean_acc = robustness["clean"]["accuracy"]
    ax.axhline(clean_acc, color="gray", linestyle="--", label="clean baseline")

    for name, per_severity in robustness.items():
        if name == "clean":
            continue
        severities = list(range(1, 6))
        accs = [per_severity[f"severity_{s}"]["accuracy"] for s in severities]
        ax.plot(severities, accs, marker="o", label=name)

    ax.set_xlabel("Corruption severity")
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustness to simulated camera/lighting domain shift")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
