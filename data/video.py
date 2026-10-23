"""Video decoding and hand-landmark sequence preparation."""

from __future__ import annotations

from pathlib import Path

import torch


def normalize_sequence(sequence: torch.Tensor, length: int = 32) -> torch.Tensor:
    """Truncate or zero-pad a ``[frames, features]`` sequence to fixed length."""

    if sequence.ndim != 2 or sequence.shape[0] == 0 or length < 1:
        raise ValueError("sequence must be non-empty 2D tensor and length must be positive")
    result = torch.zeros(length, sequence.shape[1], dtype=torch.float32)
    count = min(length, sequence.shape[0])
    result[:count] = sequence[:count].float()
    return result


def extract_video_sequence(
    path: str | Path,
    sequence_length: int = 32,
    max_frames: int = 128,
) -> torch.Tensor:
    """Extract normalized left/right hand landmarks from a local video file."""

    import cv2
    try:
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError(
            "MediaPipe is required for video extraction; run "
            "'venv/bin/python -m pip install -r requirements.txt'"
        ) from error

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    frames: list[torch.Tensor] = []
    try:
        while len(frames) < max_frames:
            success, frame = capture.read()
            if not success:
                break
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarks = torch.zeros(126, dtype=torch.float32)
            if result.multi_hand_landmarks and result.multi_handedness:
                for hand, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    offset = 0 if handedness.classification[0].label == "Left" else 63
                    values = [coordinate for point in hand.landmark for coordinate in (point.x, point.y, point.z)]
                    landmarks[offset : offset + 63] = torch.tensor(values)
            frames.append(landmarks)
    finally:
        hands.close()
        capture.release()
    if not frames:
        raise ValueError(f"video contains no readable frames: {path}")
    return normalize_sequence(torch.stack(frames), sequence_length)