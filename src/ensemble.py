"""
ensemble.py — multi-checkpoint ensemble inference with uncertainty.

Why an ensemble at all: the 5-fold cross-validation this repo already runs
produces 5 independently-seeded models almost for free — each fold trains a
fresh model on a different 80% of the data. Averaging their predictions
(a "deep ensemble") is one of the best-established, cheapest ways to both
(a) improve accuracy a little via variance reduction, and more importantly
(b) get a genuine uncertainty signal: when the 5 models agree, the
prediction is probably reliable; when they disagree, that disagreement
*is* the model telling you it's not sure — which a single softmax number
can never do, however large it looks. A single model can be "92% confident"
and simply wrong; 5 independently-trained models disagreeing on the same
image is a much stronger signal that something about that image is genuinely
ambiguous or out-of-distribution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.calibration import apply_temperature
from src.model import build_model


@dataclass
class EnsembleMember:
    model: torch.nn.Module
    temperature: float = 1.0


@dataclass
class EnsemblePrediction:
    pred_class: int
    pred_name: str
    calibrated_confidence: float          # mean calibrated probability of the winning class
    mean_probs: np.ndarray                # (num_classes,) averaged calibrated probs across members
    member_probs: np.ndarray              # (n_members, num_classes) per-member calibrated probs
    predictive_entropy: float             # entropy of the mean distribution — "how spread out is the answer"
    disagreement: float                   # mean pairwise JS-ish spread across members — "how much do models differ"
    member_votes: list[int]               # each member's own argmax class
    vote_agreement: float                 # fraction of members that agree with the ensemble's final pick
    decision: str                         # "auto" or "review" — see decide()


class SteelDefectEnsemble:
    """
    Loads N fold checkpoints (each with its own fitted temperature) and
    produces calibrated, uncertainty-aware predictions.
    """

    def __init__(self, checkpoint_paths: list[str], calibration_path: str | None, device: torch.device):
        self.device = device
        self.members: list[EnsembleMember] = []
        self.class_names: list[str] | None = None
        self.img_size: int | None = None
        self.mean: list[float] | None = None
        self.std: list[float] | None = None

        temperatures = {}
        if calibration_path and Path(calibration_path).exists():
            with open(calibration_path) as f:
                temperatures = json.load(f)

        for ckpt_path in checkpoint_paths:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            cfg = ckpt["cfg"]
            if self.class_names is None:
                self.class_names = cfg["class_names"]
                self.img_size = cfg["img_size"]
                self.mean = cfg["mean"]
                self.std = cfg["std"]
            model = build_model(cfg).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            temperature = temperatures.get(str(Path(ckpt_path)), temperatures.get(Path(ckpt_path).parent.name, 1.0))
            self.members.append(EnsembleMember(model=model, temperature=temperature))

        if not self.members:
            raise ValueError("No checkpoints provided — ensemble is empty.")

    @torch.no_grad()
    def predict(self, img_tensor: torch.Tensor,
                auto_confidence_threshold: float = 0.90,
                auto_agreement_threshold: float = 0.99) -> EnsemblePrediction:
        """
        Args:
            img_tensor: normalised (3, H, W) tensor (no batch dim)
            auto_confidence_threshold: minimum calibrated confidence to allow an automated decision
            auto_agreement_threshold: minimum fraction of members that must agree on the winning class
        """
        batch = img_tensor.unsqueeze(0).to(self.device)
        member_probs = []
        member_votes = []
        for member in self.members:
            logits = member.model(batch)[0]
            probs = apply_temperature(logits, member.temperature).cpu().numpy()
            member_probs.append(probs)
            member_votes.append(int(probs.argmax()))
        member_probs = np.stack(member_probs)  # (n_members, num_classes)

        mean_probs = member_probs.mean(axis=0)
        pred_class = int(mean_probs.argmax())

        eps = 1e-12
        predictive_entropy = float(-(mean_probs * np.log(mean_probs + eps)).sum())

        # Disagreement: average per-class std across members, summed — 0 if every
        # member outputs an identical distribution, larger the more they diverge.
        disagreement = float(member_probs.std(axis=0).sum())

        vote_agreement = float(np.mean([v == pred_class for v in member_votes]))
        calibrated_confidence = float(mean_probs[pred_class])

        decision = self.decide(calibrated_confidence, vote_agreement,
                                auto_confidence_threshold, auto_agreement_threshold)

        return EnsemblePrediction(
            pred_class=pred_class,
            pred_name=self.class_names[pred_class],
            calibrated_confidence=calibrated_confidence,
            mean_probs=mean_probs,
            member_probs=member_probs,
            predictive_entropy=predictive_entropy,
            disagreement=disagreement,
            member_votes=member_votes,
            vote_agreement=vote_agreement,
            decision=decision,
        )

    @staticmethod
    def decide(confidence: float, agreement: float, conf_thresh: float, agree_thresh: float) -> str:
        """
        Gate an automated decision on BOTH calibrated confidence and cross-model
        agreement. Either one alone can be misleadingly high — a single
        overconfident model, or five models that all weakly agree — so both
        thresholds must clear before the system is allowed to decide on its own.
        """
        if confidence >= conf_thresh and agreement >= agree_thresh:
            return "auto"
        return "review"
