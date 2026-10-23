"""FastAPI service for predictions from a trained sign classifier."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model.classifier import SignClassifier, load_checkpoint


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=1)
    top_k: int = Field(default=1, ge=1, le=20)


class Prediction(BaseModel):
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    predictions: list[Prediction]


app = FastAPI(title="Sign Speech Inference API", version="0.1.0")
_model: SignClassifier | None = None
_labels: list[str] = []
_feature_names: list[str] = []


def get_model() -> tuple[SignClassifier, list[str], list[str]]:
    global _model, _labels, _feature_names
    if _model is None:
        checkpoint_path = Path(os.getenv("SIGN_SPEECH_CHECKPOINT", "checkpoints/sign_speech.pt"))
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        _model, _labels, _feature_names = load_checkpoint(checkpoint_path)
    return _model, _labels, _feature_names


@app.get("/health")
def health() -> dict[str, str | bool]:
    checkpoint_path = Path(os.getenv("SIGN_SPEECH_CHECKPOINT", "checkpoints/sign_speech.pt"))
    return {"status": "ok", "model_loaded": _model is not None, "checkpoint_exists": checkpoint_path.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        model, labels, feature_names = get_model()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    if len(request.features) != len(feature_names):
        raise HTTPException(
            status_code=422,
            detail=f"expected {len(feature_names)} features, received {len(request.features)}",
        )
    with torch.no_grad():
        logits = model(torch.tensor([request.features], dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1)[0]
        count = min(request.top_k, len(labels))
        scores, indices = torch.topk(probabilities, count)
    predictions = [
        Prediction(label=labels[int(index)], confidence=float(score))
        for score, index in zip(scores, indices)
    ]
    return PredictionResponse(predictions=predictions)