"""Download a Hugging Face dataset split into the project's CSV format."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def export_dataset(dataset: object, output: str | Path, label_column: str) -> None:
    """Export a datasets.Dataset-like object after validating its tabular shape."""

    frame = dataset.to_pandas()  # type: ignore[attr-defined]
    if label_column not in frame.columns:
        available = ", ".join(str(column) for column in frame.columns)
        raise ValueError(f"missing label column '{label_column}'. Available columns: {available}")
    if frame.empty:
        raise ValueError("the selected Hugging Face split is empty")
    if frame[label_column].isna().any():
        raise ValueError("the label column contains missing values")

    feature_columns = [
        column
        for column in frame.select_dtypes(include=["number", "bool"]).columns
        if column != label_column
    ]
    if not feature_columns:
        raise ValueError(
            "no numeric feature columns found; this dataset likely contains images or video "
            "and needs landmark extraction before training"
        )
    export = frame[[label_column, *feature_columns]].copy()
    if export[feature_columns].isna().any().any():
        raise ValueError("numeric feature columns contain missing values")
    export = export.rename(columns={label_column: "label"})
    export["label"] = export["label"].astype(str)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(output_path, index=False)
    print(f"saved {len(export)} rows with {len(feature_columns)} features to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset repository ID")
    parser.add_argument("--split", default="train", help="dataset split to download")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--output", default="data/signs.csv")
    parser.add_argument("--config", help="optional dataset configuration name")
    return parser


def main() -> None:
    from datasets import load_dataset

    args = build_parser().parse_args()
    dataset = load_dataset(args.dataset, name=args.config, split=args.split)
    export_dataset(dataset, args.output, args.label_column)


if __name__ == "__main__":
    main()