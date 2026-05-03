"""Push a local directory to a Hugging Face Hub repository.

Usage:
  python tools/push_hf_repo.py --repo-id <org>/<repo-name> --local-dir <path>
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def push_to_hf(repo_id: str, local_dir: Path, private: bool = False) -> None:
    print("\n" + "=" * 60)
    print(f"  Pushing to Hugging Face Hub: {repo_id}")
    print("=" * 60)
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "ERROR: huggingface_hub is not installed. Run: pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)

    api = HfApi()

    # Check if repo exists, create if not
    try:
        api.model_info(repo_id)
        print(f"  Repository {repo_id} exists. Uploading files...")
    except Exception:
        print(f"  Creating new repository: {repo_id}...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)

    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update model artifacts via push script",
    )
    print(f"\n  Successfully pushed to: https://huggingface.co/{repo_id}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Push a local folder to Hugging Face Hub.")
    p.add_argument(
        "--repo-id", required=True, help="HF repo name (e.g. org/repo-name)."
    )
    p.add_argument(
        "--local-dir", type=Path, required=True, help="Local directory to push."
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private (default: public).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not args.local_dir.is_dir():
        print(f"ERROR: Local directory not found: {args.local_dir}", file=sys.stderr)
        return 1
    push_to_hf(args.repo_id, args.local_dir, private=args.private)
    return 0


if __name__ == "__main__":
    sys.exit(main())
