"""CSV dataset loading for numeric sign-language features."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset


class SignDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A tensor dataset containing feature vectors and integer class labels."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        if features.ndim != 2:
            raise ValueError("features must be a 2D tensor")
        if labels.ndim != 1 or len(features) != len(labels):
            raise ValueError("labels must be a 1D tensor aligned with features")
        if len(features) == 0:
            raise ValueError("dataset must contain at least one row")
        self.features = features.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def load_csv(
    path: str | Path,
    label_column: str = "label",
) -> tuple[SignDataset, list[str], list[str]]:
    """Load a labeled feature CSV and return dataset, labels, and feature names."""

    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        if label_column not in fieldnames:
            raise ValueError(f"missing label column: {label_column}")
        feature_names = [name for name in fieldnames if name != label_column]
        if not feature_names:
            raise ValueError("CSV must contain at least one feature column")

        label_names: list[str] = []
        label_to_index: dict[str, int] = {}
        feature_rows: list[list[float]] = []
        label_rows: list[int] = []
        for row_number, row in enumerate(reader, start=2):
            raw_label = (row.get(label_column) or "").strip()
            if not raw_label:
                raise ValueError(f"row {row_number}: label is empty")
            if raw_label not in label_to_index:
                label_to_index[raw_label] = len(label_names)
                label_names.append(raw_label)
            try:
                values = [float(row[name]) for name in feature_names]
            except (TypeError, ValueError) as error:
                raise ValueError(f"row {row_number}: features must be numeric") from error
            feature_rows.append(values)
            label_rows.append(label_to_index[raw_label])

    if not feature_rows:
        raise ValueError("CSV must contain at least one data row")
    return (
        SignDataset(torch.tensor(feature_rows), torch.tensor(label_rows)),
        label_names,
        feature_names,
    )