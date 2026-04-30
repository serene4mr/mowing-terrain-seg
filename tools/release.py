# Copyright (c) OpenMMLab. Mowing-terrain-seg. SPDX-License-Identifier: Apache-2.0
"""Promote a local ``work_dirs/<exp>/`` run to a Hugging Face Hub model repo (tagged)."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Repo root for imports and `git` (resolve code SHA).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.hf_release import card, engine, metrics, staging, validate  # noqa: E402


def _print_staging_tree(path: Path) -> None:
    for p in sorted(path.rglob("*")):
        if p.is_file():
            print(p.relative_to(path))


def _write_release_json(
    exp_dir: Path,
    args: argparse.Namespace,
    git_sha: str,
    has_onnx: bool,
) -> Path:
    rel = {
        "repo_id": args.repo_id,
        "tag": args.tag,
        "pth": args.pth,
        "git_sha": git_sha,
        "released_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "has_onnx": has_onnx,
    }
    out = exp_dir / "release.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rel, f, indent=2, sort_keys=True)
    return out


def _write_engine_release_json(
    engine_dir: Path, args: argparse.Namespace, git_sha: str, profile: str
) -> Path:
    rel = {
        "repo_id": args.repo_id,
        "tag": args.tag,
        "git_sha": git_sha,
        "released_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "engine_only": True,
        "engine_profile": profile,
    }
    out = engine_dir / "hf_engine_release.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rel, f, indent=2, sort_keys=True)
    return out


def _hf_import():
    try:
        from huggingface_hub import HfApi  # type: ignore[import-not-found]
    except ImportError as e:
        print(
            "huggingface_hub is not installed. Install: pip install -e '.[release]'",
            file=sys.stderr,
        )
        raise SystemExit(2) from e
    return HfApi


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Release experiment artifacts to Hugging Face Hub (tag = version).",
    )
    p.add_argument(
        "--repo-id",
        required=True,
        help="HF model repo, e.g. <org>/mts-deeplabv3plus-r50-ycor3cls",
    )
    p.add_argument(
        "--exp-dir",
        type=Path,
        default=None,
        help="Experiment work_dir (mmengine work_dir). Not used with --engine-dir.",
    )
    p.add_argument(
        "--pth",
        default=None,
        help="Checkpoint file name inside --exp-dir (e.g. best_val_mIoU_iter_5000.pth).",
    )
    p.add_argument(
        "--tag",
        required=True,
        help="Hub revision to upload to (branch, tag, or commit), e.g. v1.0.0 or main",
    )
    p.add_argument(
        "--message",
        default="",
        help="Release notes; used in commit and tag on the Hub",
    )
    p.add_argument(
        "--onnx-dir",
        type=Path,
        default=None,
        help=(
            "Path to local ONNX / mmdeploy output dir (parent of end2end.onnx). "
            "Run tools/deploy/deploy.py first, then pass that dir here."
        ),
    )
    p.add_argument(
        "--engine-dir",
        type=Path,
        default=None,
        help="Output dir from tools/build_engine.py (end2end.engine + platform.json). "
        "Mutually exclusive with --exp-dir (TensorRT engine-only upload).",
    )
    p.add_argument(
        "--samples-in",
        type=Path,
        default=None,
        help="Optional demo input image; requires --samples-out",
    )
    p.add_argument(
        "--samples-out",
        type=Path,
        default=None,
        help="Optional expected demo output; requires --samples-in",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the staging tree and print files; do not call Hugging Face",
    )
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow dirty git or missing .git in the code checkout; "
        "for --engine-dir, skip Hub ONNX↔provenance hash check",
    )
    p.add_argument(
        "--public",
        action="store_true",
        help="Create / upload to a public repo (default: private hf repo).",
    )
    return p.parse_args(argv)


def _fetch_readme_from_hub(repo_id: str, tag: str) -> str:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
    except ImportError:
        return ""
    try:
        p = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            revision=tag,
        )
        return Path(p).read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def main_engine(args: argparse.Namespace) -> int:
    engine_dir: Path = args.engine_dir.resolve()  # type: ignore[union-attr]
    try:
        plat_data: Dict[str, Any] = validate.validate_engine_against_hub(
            engine_dir,
            repo_id=args.repo_id,
            allow_dirty=args.allow_dirty,
        )
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 2
    try:
        git_sha, _ = validate.check_git(REPO_ROOT, args.allow_dirty)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 2

    profile = plat_data.get("profile")
    if not profile or not isinstance(profile, str):
        print("Could not determine profile; fix platform.json", file=sys.stderr)
        return 2

    readme_in = _fetch_readme_from_hub(args.repo_id, args.tag)
    try:
        merged_readme = card.merge_tensorrt_engines_readme(readme_in, plat_data)
    except (OSError, ValueError) as e:
        print(f"Model card merge failed: {e}", file=sys.stderr)
        return 2

    try:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            _root, pdir = engine.build_engine_staging(
                engine_dir=engine_dir,
                dest=dest,
                profile=profile,
            )
            (dest / "README.md").write_text(merged_readme, encoding="utf-8")

            if args.dry_run:
                _print_staging_tree(dest)
                return 0
            HfApi = _hf_import()  # noqa: N806
            try:
                api = HfApi()
                api.create_repo(args.repo_id, private=not args.public, exist_ok=True)
                api.upload_folder(
                    repo_id=args.repo_id,
                    folder_path=str(dest),
                    commit_message=f"{args.tag}: TensorRT {profile} — {args.message or 'engine'}",
                )
            except Exception as e:  # noqa: BLE001
                print(f"Hugging Face API error: {e}", file=sys.stderr)
                return 3
    except (OSError, ValueError) as e:
        print(f"Staging failed: {e}", file=sys.stderr)
        return 2

    _write_engine_release_json(engine_dir, args, git_sha, profile=profile)
    print(f"https://huggingface.co/{args.repo_id}/tree/{args.tag}")
    return 0


def main_experiment(args: argparse.Namespace) -> int:
    exp_dir = args.exp_dir.resolve()  # type: ignore[union-attr]
    onnx_dir: Optional[Path] = Path(args.onnx_dir).resolve() if args.onnx_dir else None

    if (args.samples_in is None) ^ (args.samples_out is None):
        print(
            "Pass both --samples-in and --samples-out, or neither.",
            file=sys.stderr,
        )
        return 2

    pth = validate.validate_experiment(exp_dir, args.pth)  # type: ignore[arg-type]
    with open(exp_dir / "summary.json", "r", encoding="utf-8", errors="replace") as f:
        summary: Dict[str, Any] = json.load(f)

    try:
        git_sha, _ = validate.check_git(REPO_ROOT, args.allow_dirty)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 2

    if onnx_dir is not None:
        validate.onnx_drift(onnx_dir, pth, pth_basename=pth.name)

    has_onnx_in_release = False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            staging.build_staging(
                exp_dir=exp_dir,
                pth=pth,
                onnx_dir=onnx_dir,
                summary=summary,
                sample_in=args.samples_in,
                sample_out=args.samples_out,
                dest=dest,
            )
            metrics.write_metrics_json(dest, summary)
            has_onnx_in_release = (dest / "onnx" / "end2end.onnx").is_file()
            readme = card.render(
                summary=summary,
                repo_id=args.repo_id,
                tag=args.tag,
                git_sha=git_sha,
                message=args.message,
                has_onnx=has_onnx_in_release,
            )
            (dest / "README.md").write_text(readme, encoding="utf-8")

            if args.dry_run:
                _print_staging_tree(dest)
                return 0
            HfApi = _hf_import()  # noqa: N806
            try:
                api = HfApi()
                api.create_repo(args.repo_id, private=not args.public, exist_ok=True)
                api.upload_folder(
                    repo_id=args.repo_id,
                    folder_path=str(dest),
                    commit_message=f"{args.tag}: {args.message or 'Release'}",
                )
                api.create_tag(
                    repo_id=args.repo_id,
                    tag=args.tag,
                    tag_message=args.message or f"tag {args.tag}",
                    repo_type="model",
                )
            except Exception as e:  # noqa: BLE001
                print(f"Hugging Face API error: {e}", file=sys.stderr)
                return 3
    except (OSError, ValueError) as e:
        print(f"Staging failed: {e}", file=sys.stderr)
        return 2

    _write_release_json(exp_dir, args, git_sha, has_onnx=has_onnx_in_release)
    print(f"https://huggingface.co/{args.repo_id}/tree/{args.tag}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    eng = args.engine_dir
    exp = args.exp_dir
    pth = args.pth
    if eng is not None and (exp is not None or pth is not None):
        print("Use either --engine-dir or (--exp-dir and --pth), not both.", file=sys.stderr)
        return 2
    if eng is not None:
        return main_engine(args)
    if exp is None or pth is None:
        print(
            "Standard release requires --exp-dir and --pth, or use --engine-dir for TensorRT.",
            file=sys.stderr,
        )
        return 2
    return main_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
