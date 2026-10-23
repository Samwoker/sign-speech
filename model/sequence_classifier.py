"""Temporal classifier for fixed-length landmark sequences."""

from __future__ import annotations

import torch
from torch import nn


class SignSequenceClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_size < 1 or num_classes < 2 or num_layers < 1:
            raise ValueError("input_size and num_layers must be positive; num_classes must be at least two")
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, num_classes))

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        if sequences.ndim != 3 or sequences.shape[2] != self.input_size:
            raise ValueError(f"expected [batch, frames, {self.input_size}] sequences")
        _, hidden = self.gru(sequences)
        return self.classifier(hidden[-1])