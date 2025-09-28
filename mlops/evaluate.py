import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved model on test set and write metrics.json")
    parser.add_argument("--test", type=str, default=str(Path("artifacts/test.csv")), help="Path to test CSV")
    parser.add_argument("--model", type=str, default=str(Path("artifacts/model.joblib")), help="Path to trained model file")
    parser.add_argument("--out_json", type=str, default=str(Path("artifacts/metrics.json")), help="Where to write metrics JSON")
    parser.add_argument("--target", type=str, default="ProdTaken", help="Target column name")
    args = parser.parse_args()

    model_path = Path(args.model)
    test_path = Path(args.test)
    if not model_path.exists() or not test_path.exists():
        raise FileNotFoundError("Required inputs not found. Ensure train and prep steps have run.")

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)
    if args.target not in test_df.columns:
        raise KeyError(f"Target column '{args.target}' not found in test.csv")

    X_test = test_df.drop(columns=[args.target])
    y_test = test_df[args.target]

    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if proba is not None else None,
        "precision_recall_f1": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "_": 0.0,
        },
    }

    metrics_path = Path(args.out_json)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()


