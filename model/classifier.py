"""Small configurable classifier for extracted sign-language features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class SignClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_size < 1 or num_classes < 2:
            raise ValueError("input_size must be positive and num_classes must be at least two")
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_size:
            raise ValueError(f"expected [batch, {self.input_size}] features")
        return self.network(features)


def save_checkpoint(
    path: str | Path,
    model: SignClassifier,
    labels: list[str],
    feature_names: list[str],
) -> None:
    checkpoint: dict[str, Any] = {
        "model_config": {
            "input_size": model.input_size,
            "num_classes": model.num_classes,
            "hidden_size": model.hidden_size,
            "dropout": model.dropout,
        },
        "labels": labels,
        "feature_names": feature_names,
        "state_dict": model.state_dict(),
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)


def load_checkpoint(path: str | Path) -> tuple[SignClassifier, list[str], list[str]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("model_config", {})
    labels = checkpoint.get("labels", [])
    feature_names = checkpoint.get("feature_names", [])
    if not config or not labels or not feature_names:
        raise ValueError("checkpoint is missing model metadata")
    model = SignClassifier(**config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, labels, feature_names