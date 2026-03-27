"""
Prepare data/cleaned_data.csv from the Kaggle-style credit card fraud dataset.

Expected input columns:
- Time
- V1..V28
- Amount
- Class
"""
from pathlib import Path
import argparse

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "cleaned_data.csv"
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def prepare_dataset(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path, low_memory=False)

    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Input dataset is missing required columns: "
            + ", ".join(missing)
        )

    prepared = df[FEATURE_COLUMNS].copy()
    prepared["Class"] = prepared["Class"].astype(int)
    prepared["source"] = "creditcard"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create data/cleaned_data.csv from a raw credit card fraud CSV."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the raw credit card fraud CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output path for cleaned dataset. Defaults to {OUTPUT_PATH}",
    )
    args = parser.parse_args()

    prepare_dataset(args.input_csv, args.output)
    print(f"Saved prepared dataset to {args.output}")


if __name__ == "__main__":
    main()
