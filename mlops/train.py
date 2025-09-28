import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_pipeline_and_space(X: pd.DataFrame) -> tuple[Pipeline, dict]:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler(with_mean=False)),
    ])
    categorical_transformer = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    clf = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
    pipe = Pipeline([("prep", preprocessor), ("model", clf)])

    param_grid = {
        "model__n_estimators": [300, 500],
        "model__max_depth": [None, 10, 20],
    }
    return pipe, param_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model on prepared data and save best model + params.")
    parser.add_argument("--train", type=str, default=str(Path("artifacts/train.csv")), help="Path to train CSV")
    parser.add_argument("--out_dir", type=str, default=str(Path("artifacts")), help="Directory to save model and params")
    parser.add_argument("--target", type=str, default="ProdTaken", help="Target column name")
    args = parser.parse_args()

    artifacts_dir = Path(args.out_dir)
    ensure_directory(artifacts_dir)

    train_csv = Path(args.train)
    if not train_csv.exists():
        # Fallback to conventional location
        train_csv = artifacts_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found at {train_csv}")

    train_df = pd.read_csv(train_csv)
    # Determine target column with fallback to lowercase variant
    target_col = args.target if args.target in train_df.columns else (
        "prodtaken" if "prodtaken" in train_df.columns else None
    )
    if target_col is None:
        raise KeyError(f"Target column '{args.target}' or 'prodtaken' not found in train.csv")

    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    pipe, param_grid = _build_pipeline_and_space(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X, y)

    # Save model and params
    import joblib

    # Persist artifacts
    import joblib

    (artifacts_dir / "experiment_log.csv").write_text(
        pd.DataFrame(search.cv_results_).to_csv(index=False)
    )
    model_path = artifacts_dir / "model.joblib"
    params_path = artifacts_dir / "best_params.json"
    joblib.dump(search.best_estimator_, model_path)
    params_path.write_text(json.dumps(search.best_params_, indent=2))

    print(f"Saved model to {model_path}")
    print(f"Saved params to {params_path}")


if __name__ == "__main__":
    main()


