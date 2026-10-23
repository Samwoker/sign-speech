"""Predict a sign from a local video using a trained video checkpoint."""

from __future__ import annotations

import argparse

import torch

from data.video import extract_video_sequence
from model.sequence_classifier import SignSequenceClassifier


def load_video_checkpoint(path: str) -> tuple[SignSequenceClassifier, list[str], int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SignSequenceClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["labels"], checkpoint["sequence_length"]


def predict(video: str, checkpoint_path: str, top_k: int) -> None:
    model, labels, sequence_length = load_video_checkpoint(checkpoint_path)
    sequence = extract_video_sequence(video, sequence_length)
    with torch.no_grad():
        probabilities = torch.softmax(model(sequence.unsqueeze(0)), dim=1)[0]
        scores, indices = torch.topk(probabilities, min(top_k, len(labels)))
    for score, index in zip(scores, indices):
        print(f"{labels[int(index)]}\t{float(score):.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="path to a local .mp4/.mov/.avi file")
    parser.add_argument("--checkpoint", default="checkpoints/sign_speech_video.pt")
    parser.add_argument("--top-k", type=int, default=3)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    predict(args.video, args.checkpoint, args.top_k)