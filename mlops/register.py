import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()

    # Ensure artifacts directory exists and at least contains the model
    artifacts_dir = Path(args.folder)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("model.joblib not found in provided --folder. Train step must run first.")

    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(artifacts_dir),
        path_in_repo=".",
    )
    print("Uploaded artifacts to", args.repo_id)


if __name__ == "__main__":
    main()


