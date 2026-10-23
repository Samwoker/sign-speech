"""Train a sign classifier from a labeled feature CSV."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from data.dataset import load_csv
from model.classifier import SignClassifier, save_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: SignClassifier, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for features, labels in loader:
            predictions = model(features).argmax(dim=1)
            correct += int((predictions == labels).sum())
            total += len(labels)
    return correct / total if total else 0.0


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    dataset, labels, feature_names = load_csv(args.csv, args.label_column)
    if len(labels) < 2:
        raise ValueError("training requires at least two distinct labels")

    validation_size = max(1, round(len(dataset) * args.validation_split))
    validation_size = min(validation_size, len(dataset) - 1)
    train_size = len(dataset) - validation_size
    train_set, validation_set = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)
    model = SignClassifier(dataset.features.shape[1], len(labels), args.hidden_size, args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for features, batch_labels in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(features), batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch_labels)
        train_loss = running_loss / len(train_set)
        validation_accuracy = evaluate(model, validation_loader)
        print(f"epoch={epoch} loss={train_loss:.4f} validation_accuracy={validation_accuracy:.3f}")

    save_checkpoint(args.output, model, labels, feature_names)
    print(f"saved checkpoint to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="labeled feature CSV")
    parser.add_argument("--output", default="checkpoints/sign_speech.pt")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())