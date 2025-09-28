import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - optional dependency in some envs
    load_dataset = None  # type: ignore


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_local_csv(local_csv: Path) -> pd.DataFrame:
    if not local_csv.exists():
        raise FileNotFoundError(f"Local CSV not found at {local_csv}")
    df = pd.read_csv(local_csv)
    # Drop unnamed index-like columns
    for col in list(df.columns):
        if str(col).lower().startswith("unnamed") or col == "":
            df.drop(columns=[col], inplace=True)
    # Basic normalization of columns
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    # Minor label cleanup that appears in the dataset
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female", "fe male": "Female", "Femail": "Female"})
    return df


def load_hf_dataset(hf_name: str, split: str) -> pd.DataFrame:
    if load_dataset is None:
        raise RuntimeError("datasets library is not installed; cannot load from HF.")
    ds = load_dataset(hf_name, split=split)
    df = ds.to_pandas()
    # Align column normalization with local pathway
    for col in list(df.columns):
        if str(col).lower().startswith("unnamed") or col == "":
            df.drop(columns=[col], inplace=True)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df


def stratified_split(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe columns: {list(df.columns)}")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target] if df[target].nunique() > 1 else None,
    )
    return train_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset: load, clean, split, and save train/test.")
    parser.add_argument("--dataset_repo", type=str, default="", help="HF dataset repo id (if provided, load from HF)")
    parser.add_argument("--local_csv", type=str, default=str(Path("data/processed/cleaned_tourism.csv")), help="Path to local CSV (fallback if no HF)")
    parser.add_argument("--split", type=str, default="train", help="HF split to load if dataset_repo is set")
    parser.add_argument("--target", type=str, default="ProdTaken", help="Target column name")
    parser.add_argument("--out_dir", type=str, default=str(Path("artifacts")), help="Directory to write outputs")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    artifacts_dir = Path(args.out_dir)
    ensure_directory(artifacts_dir)

    # Load data
    if args.dataset_repo:
        # Load from Hugging Face dataset repo using token if available
        from datasets import load_dataset as hf_load_dataset  # type: ignore

        ds = hf_load_dataset(args.dataset_repo, token=True)
        train_df = ds["train"].to_pandas()
        if "test" in ds:
            test_df = ds["test"].to_pandas()
        else:
            # If no explicit test split, derive it with stratification
            from sklearn.model_selection import train_test_split

            ycol = "ProdTaken" if "ProdTaken" in train_df.columns else "prodtaken"
            train_df, test_df = train_test_split(
                train_df,
                test_size=args.test_size,
                random_state=args.random_state,
                stratify=train_df[ycol] if ycol in train_df.columns else None,
            )
    else:
        # Fallback: load local CSV and perform split
        local_csv = Path(args.local_csv)
        if not local_csv.exists():
            raw_fallback = Path("data/raw/tourism.csv")
            if raw_fallback.exists():
                local_csv = raw_fallback
        df = load_local_csv(local_csv)
        train_df, test_df = stratified_split(
            df=df,
            target=args.target,
            test_size=args.test_size,
            random_state=args.random_state,
        )

    # Save outputs
    train_path = artifacts_dir / "train.csv"
    test_path = artifacts_dir / "test.csv"
    meta_path = artifacts_dir / "meta.json"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    meta = {
        "target": args.target,
        "num_rows": {"train": int(len(train_df)), "test": int(len(test_df))},
        "columns": list(df.columns),
        "source": args.source,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Wrote train to {train_path}")
    print(f"Wrote test to  {test_path}")
    print(f"Wrote meta to  {meta_path}")


if __name__ == "__main__":
    main()


