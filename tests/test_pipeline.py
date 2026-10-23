from __future__ import annotations

import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from model.classifier import load_checkpoint
from model.sequence_classifier import SignSequenceClassifier
from data.video import normalize_sequence
from train.train import train


class PipelineTest(unittest.TestCase):
    def test_video_sequence_is_padded_to_fixed_length(self) -> None:
        sequence = torch.ones(3, 4)
        normalized = normalize_sequence(sequence, length=5)
        self.assertEqual(tuple(normalized.shape), (5, 4))
        self.assertTrue(torch.equal(normalized[:3], sequence))
        self.assertTrue(torch.equal(normalized[3:], torch.zeros(2, 4)))

    def test_sequence_classifier_returns_class_logits(self) -> None:
        model = SignSequenceClassifier(input_size=4, num_classes=3, hidden_size=8)
        logits = model(torch.zeros(2, 5, 4))
        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_train_and_reload_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "signs.csv"
            checkpoint_path = root / "model.pt"
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["label", "x", "y"])
                for index in range(10):
                    writer.writerow(["left", -1.0 - index * 0.01, 0.0])
                    writer.writerow(["right", 1.0 + index * 0.01, 0.0])

            train(
                Namespace(
                    csv=str(csv_path),
                    output=str(checkpoint_path),
                    label_column="label",
                    epochs=2,
                    batch_size=4,
                    hidden_size=8,
                    dropout=0.0,
                    learning_rate=0.01,
                    validation_split=0.2,
                    seed=7,
                )
            )
            model, labels, feature_names = load_checkpoint(checkpoint_path)
            self.assertEqual(labels, ["left", "right"])
            self.assertEqual(feature_names, ["x", "y"])
            self.assertEqual(tuple(model(torch.tensor([[-1.0, 0.0]])).shape), (1, 2))


if __name__ == "__main__":
    unittest.main()