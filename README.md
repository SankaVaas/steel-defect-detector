# Steel Surface Defect Detector

A complete, production-shaped deep-learning pipeline for classifying steel
surface defects from the **NEU-CLS** dataset — six defect types photographed
under a line-scan camera on a real hot-rolled steel strip:

| Class | Description |
|---|---|
| Crazing | Fine network of interconnected surface cracks (thermal/rolling stress) |
| Inclusion | Foreign, non-metallic material trapped in the steel during casting |
| Patches | Irregular, discoloured surface regions from uneven scale removal |
| Pitted | Small localised pits/cavities, often corrosion-related |
| Rolled-in Scale | Oxide scale pressed into the surface during hot rolling |
| Scratches | Linear surface marks from mechanical handling contact |

This repo takes the problem from raw images to a deployable artifact:
**transfer-learning classifier → stratified cross-validation → Grad-CAM
explainability → ONNX export → interactive demo**, with every design
decision explained below rather than left implicit in code.

---

## Why this architecture (the ML reasoning)

**Transfer learning, not training from scratch.**
NEU-CLS has ~1,800 images total — 300 per class. That's nowhere near enough
to learn good low-level filters (edges, textures, gradients) from random
initialization. Instead, `src/model.py` starts from an ImageNet-pretrained
backbone (`efficientnet_b4` by default, configurable) and fine-tunes it. The
backbone already knows how to detect edges and textures; we only need to
teach it what *these particular* textures mean.

**Discriminative learning rates.**
The backbone is fine-tuned at `lr * 0.1`; the newly-initialized classifier
head trains at the full `lr`. The head starts from nothing and needs to move
a lot; the backbone already encodes something useful and should be nudged,
not overwritten. See `SteelDefectNet.param_groups()`.

**Domain-specific augmentation, not generic ImageNet augmentation.**
`src/dataset.py`'s `SteelSurfaceAugment` was deliberately built around what's
physically valid for a steel surface photographed by a fixed line-scan
camera, documented directly in its docstring:

| Augmentation | Used? | Why |
|---|---|---|
| Horizontal / vertical flip | yes | surface has no inherent orientation |
| 90/180/270 degree rotation | yes | rotationally symmetric under a top-down scan |
| Arbitrary rotation (0-360 degrees) | no | introduces black-border artefacts the model could exploit as a shortcut |
| Mild brightness/contrast | yes | simulates line-lighting variation across the strip |
| CutOut (random erase) | yes | forces the model to use non-local evidence, not one salient patch |
| Strong blur | no | destroys the fine texture that *is* the class signal (e.g. crazing) |
| Strong colour jitter | no | unphysical — these are effectively grayscale surfaces |

Getting this table wrong is a common failure mode in industrial vision: naive
`torchvision.transforms.RandAugment`-style pipelines borrowed from ImageNet
recipes will happily rotate/blur/recolor away the exact texture cues that
distinguish "crazing" from "scratches."

**Stratified 5-fold cross-validation, not a single train/val split.**
With only 1,800 images, a single 80/20 split leaves ~360 validation images —
enough for the reported accuracy to swing meaningfully based on which images
happened to land in validation. `train.py` runs `StratifiedKFold` so every
class is represented proportionally in every fold, trains the model 5 times,
and reports **mean +/- std** of accuracy and macro-F1. This is a materially
more trustworthy estimate of real-world performance than a single number,
and it produces 5 independent checkpoints that could be ensembled.

**Macro-F1 as the headline metric, not accuracy.**
NEU-CLS itself is class-balanced, but real production lines are not — some
defects are much rarer than others. Macro-F1 punishes a model that quietly
ignores a rare class even while overall accuracy looks fine, so it's used
for early stopping and "best epoch" selection (`src/utils.py:compute_metrics`,
`EarlyStopping`).

**Grad-CAM explainability, because "right for the wrong reason" is a real
risk here.**
NEU-CLS images have subtle scanning artefacts and lighting gradients near
their borders. A classifier can learn to shortcut on those instead of the
actual defect texture — and would still score well on this dataset while
being useless (or dangerous) on a new camera setup. `src/gradcam.py` wraps
`pytorch-grad-cam` around the model's last convolutional feature map so a
QA engineer can visually confirm the model is attending to the defect
itself before it's trusted in production. This runs automatically after
every fold (`outputs/fold*/gradcam_samples.png`) and is exposed live in the
Gradio demo.

**ONNX export with a numerical parity check.**
A plant's edge inference box rarely runs a full Python/PyTorch stack.
`src/export_onnx.py` exports the trained model to ONNX and then *verifies*
it by running the same input through both PyTorch and ONNXRuntime and
comparing outputs — silently-broken exports (a real, common failure mode
with custom heads/dropout) are caught immediately instead of at deployment.

---

## Project structure

```
steel-defect-detector/
├── configs/default.yaml     # every hyperparameter in one place
├── scripts/
│   └── prepare_data.py      # turn a raw NEU-CLS/NEU-DET download into the expected layout
├── src/
│   ├── dataset.py           # NEUCLSDataset + SteelSurfaceAugment (provided, documented above)
│   ├── model.py             # SteelDefectNet: timm backbone + head, discriminative LR groups
│   ├── engine.py            # train_one_epoch / evaluate, AMP, warmup+cosine schedule
│   ├── utils.py             # seeding, checkpoints, metrics, plots
│   ├── gradcam.py           # Grad-CAM explainability
│   ├── export_onnx.py       # ONNX export + PyTorch/ONNXRuntime parity check
│   └── infer.py             # CLI single-image inference
├── train.py                 # stratified k-fold CV training driver
├── app.py                   # Gradio demo (upload → prediction + Grad-CAM)
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/SankaVaas/steel-defect-detector.git
cd steel-defect-detector
pip install -r requirements.txt
```

## Data

The pipeline expects the Kaggle NEU-DET layout:

```
NEU-CLS/
├── train/images/{crazing,inclusion,patches,pitted_surface,rolled-in_scale,scratches}/*.jpg
└── validation/images/{...same six folders...}/*.jpg
```

Get the raw data and reshape it into that layout with `scripts/prepare_data.py`,
which auto-detects several common source layouts (flat `IMAGES/` dumps,
per-class folders, or an already-correct Kaggle layout):

```bash
# Option A: Kaggle CLI (requires a free Kaggle account + API token)
pip install kaggle
kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database
unzip neu-surface-defect-database.zip -d raw_neu
python scripts/prepare_data.py --input raw_neu --output ./NEU-CLS

# Option B: any other NEU-CLS/NEU-DET mirror you've already downloaded
python scripts/prepare_data.py --input /path/to/raw/download --output ./NEU-CLS
```

## Train

```bash
# Full run as specified in configs/default.yaml (EfficientNet-B4, 30 epochs, 5-fold CV)
python train.py --config configs/default.yaml

# Quick smoke test on modest hardware
python train.py --config configs/default.yaml \
  --backbone resnet18 --epochs 5 --folds 2 --batch-size 16

# No internet access to download pretrained ImageNet weights? Train from scratch:
python train.py --config configs/default.yaml --no-pretrained
```

Outputs per fold, under `outputs/fold{N}/`:
- `best_model.pt` — checkpoint with the highest validation macro-F1
- `metrics.json` — accuracy, macro-F1, per-class precision/recall/F1, confusion matrix
- `confusion_matrix.png`, `training_curves.png`, `gradcam_samples.png`

Plus `outputs/cv_summary.json` — mean +/- std accuracy/macro-F1 across folds.

## Evaluate a single image

```bash
python -m src.infer --checkpoint outputs/fold0/best_model.pt \
  --image path/to/steel.jpg --gradcam-out cam.png
```

## Export to ONNX (for deployment off the Python/PyTorch stack)

```bash
python -m src.export_onnx --checkpoint outputs/fold0/best_model.pt --out steel_defect.onnx
```

## Interactive demo

```bash
python app.py --checkpoint outputs/fold0/best_model.pt
```

Upload an image and get back the predicted class, full probability
distribution, and a Grad-CAM overlay explaining the prediction.

---

## Results

See `outputs/cv_summary.json` and `outputs/fold*/metrics.json` for the
numbers from the most recent training run in this repo (backbone, epoch
count, and hardware used are recorded in each fold's `metrics.json` via the
saved config). Cross-validated macro-F1 is the headline number to compare
across model/hyperparameter changes, not any single fold's accuracy.

## License

MIT — see `LICENSE`.
