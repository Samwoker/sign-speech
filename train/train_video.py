"""Train a GRU sign classifier directly from a Hugging Face video dataset."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from data.hf_zip import RemoteZipArchive, ZipMember
from data.video import extract_video_sequence
from model.sequence_classifier import SignSequenceClassifier


class HuggingFaceVideoDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, dataset: object, sequence_length: int, max_frames: int, labels: list[str]) -> None:
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.max_frames = max_frames
        self.labels = labels
        self.label_to_index = {label: index for index, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataset[index]  # type: ignore[index]
        video = row["video"]
        if isinstance(video, ZipMember):
            with row["archive"].extract(video) as video_path:
                sequence = extract_video_sequence(video_path, self.sequence_length, self.max_frames)
        else:
            video_path = video["path"] if isinstance(video, dict) else video
            sequence = extract_video_sequence(video_path, self.sequence_length, self.max_frames)
        label = row["label"]
        if isinstance(label, int):
            label_index = label
        else:
            label_index = self.label_to_index[str(label)]
        return sequence, torch.tensor(label_index, dtype=torch.long)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: SignSequenceClassifier, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for sequences, labels in loader:
            correct += int((model(sequences).argmax(dim=1) == labels).sum())
            total += len(labels)
    return correct / total if total else 0.0


def save_video_checkpoint(path: str | Path, model: SignSequenceClassifier, labels: list[str], sequence_length: int) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": {
                "input_size": model.input_size,
                "num_classes": model.num_classes,
                "hidden_size": model.hidden_size,
                "num_layers": model.num_layers,
                "dropout": model.dropout,
            },
            "labels": labels,
            "sequence_length": sequence_length,
            "state_dict": model.state_dict(),
        },
        output_path,
    )


def train(args: argparse.Namespace) -> None:
    from datasets import Features, Value, Video, load_dataset

    set_seed(args.seed)
    if args.remote_zip:
        archive = RemoteZipArchive(args.remote_zip, args.remote_zip_file)
        members = archive.members(args.split)
        if not members:
            raise ValueError(f"remote ZIP has no MP4 members for split '{args.split}'")
        labels = sorted({member.label for member in members})
        if args.max_examples:
            random.Random(args.seed).shuffle(members)
            selected: list[ZipMember] = []
            labels_seen: set[str] = set()
            for member in members:
                selected.append(member)
                labels_seen.add(member.label)
                if len(selected) >= args.max_examples and len(labels_seen) >= 2:
                    break
            members = selected
        dataset = [{"video": member, "archive": archive, "label": member.label} for member in members]
    else:
        dataset_features = None
        if args.max_examples is not None:
            dataset_features = Features({"video": Video(decode=False), "label": Value("string")})
        dataset = load_dataset(
            args.dataset,
            split=args.split,
            streaming=args.max_examples is not None,
            features=dataset_features,
        )
    if not args.remote_zip and args.max_examples is not None:
        examples = []
        labels_seen: set[str] = set()
        for row in dataset:
            examples.append(row)
            labels_seen.add(str(row[args.label_column]))
            if len(examples) >= args.max_examples and len(labels_seen) >= 2:
                break
        if len(labels_seen) < 2:
            video_paths = {
                str(row["video"].get("path", ""))
                if isinstance(row["video"], dict)
                else str(row["video"])
                for row in examples
            }
            if len(video_paths) == 1 and next(iter(video_paths), "").lower().endswith(".zip"):
                raise ValueError(
                    "the selected Hugging Face dataset exposes one ZIP archive as every "
                    "streamed video row, so it cannot be trained through the streaming "
                    "video pipeline. Use a dataset with row-level video files, or extract "
                    "the archive locally into labeled class directories first."
                )
            raise ValueError(
                "the sampled video data contains fewer than two labels; choose a dataset "
                "or split with at least two classes."
            )
        dataset = examples
    if isinstance(dataset, list):
        labels = sorted({str(row[args.label_column]) for row in dataset})
    else:
        label_feature = dataset.features[args.label_column]
        labels = list(label_feature.names) if hasattr(label_feature, "names") else sorted({str(row[args.label_column]) for row in dataset})
    wrapped = HuggingFaceVideoDataset(dataset, args.sequence_length, args.max_frames, labels)
    validation_size = max(1, round(len(wrapped) * args.validation_split))
    validation_size = min(validation_size, len(wrapped) - 1)
    train_set, validation_set = random_split(
        wrapped,
        [len(wrapped) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, num_workers=0)
    model = SignSequenceClassifier(126, len(labels), args.hidden_size, args.num_layers, args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for sequences, batch_labels in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(sequences), batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch_labels)
        print(f"epoch={epoch} loss={running_loss / len(train_set):.4f} validation_accuracy={evaluate(model, validation_loader):.3f}")
        save_video_checkpoint(args.output, model, labels, args.sequence_length)
    save_video_checkpoint(args.output, model, labels, args.sequence_length)
    print(f"saved video checkpoint to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="asthalochan/American_Sign_Language")
    parser.add_argument("--remote-zip", help="Hugging Face dataset repo containing a remote ZIP archive")
    parser.add_argument("--remote-zip-file", default="dataset_v2.zip")
    parser.add_argument("--split", default="train")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--output", default="checkpoints/sign_speech_video.pt")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--max-examples", type=int, help="limit examples for a quick smoke test")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())