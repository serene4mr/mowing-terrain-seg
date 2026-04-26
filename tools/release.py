# Copyright (c) OpenMMLab. Mowing-terrain-seg. SPDX-License-Identifier: Apache-2.0
"""Promote a local ``work_dirs/<exp>/`` run to a Hugging Face Hub model repo (tagged)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Repo root for imports and `git` (resolve code SHA).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.hf_release import auto_deploy, card, metrics, staging, validate  # noqa: E402
from tools.summarize_experiment import find_root_config_py  # noqa: E402

DEPLOY_SCRIPT = REPO_ROOT / "tools" / "deploy" / "deploy.py"


def _print_staging_tree(path: Path) -> None:
    for p in sorted(
        path.rglob("*")
    ):
        if p.is_file():
            print(
                p.relative_to(
                    path
                )
            )


def _write_release_json(
    exp_dir: Path, args: argparse.Namespace, git_sha: str, has_onnx: bool
) -> Path:
    rel = {
        "repo_id": args.repo_id,
        "tag": args.tag,
        "pth": args.pth,
        "git_sha": git_sha,
        "released_at": datetime.now(
            timezone.utc
        )
        .isoformat()
        .replace("+00:00", "Z"),
        "has_onnx": has_onnx,
    }
    out = exp_dir / "release.json"
    with open(
        out, "w", encoding="utf-8"
    ) as f:
        json.dump(
            rel, f, indent=2, sort_keys=True
        )
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Release experiment artifacts to Hugging Face Hub (tag = version)."
    )
    p.add_argument(
        "--exp-dir", required=True, type=Path, help="Experiment work_dir (mmengine work_dir)"
    )
    p.add_argument(
        "--pth",
        required=True,
        help="Checkpoint file name inside --exp-dir (e.g. best_val_mIoU_iter_5000.pth).",
    )
    p.add_argument(
        "--repo-id",
        required=True,
        help="HF model repo, e.g. <org>/mts-deeplabv3plus-r50-ycor3cls",
    )
    p.add_argument(
        "--tag",
        required=True,
        help="Git tag on the Hub repo, e.g. v1.0.0",
    )
    p.add_argument(
        "--message",
        default="",
        help="Release notes; used in commit and tag on the Hub",
    )
    p.add_argument(
        "--deploy",
        type=Path,
        default=None,
        help="Path to local ONNX / mmdeploy output dir (parent of end2end.onnx).",
    )
    p.add_argument(
        "--auto-deploy",
        action="store_true",
        help="Re-run tools/deploy/deploy.py if ONNX is missing or older than the .pth.",
    )
    p.add_argument(
        "--deploy-cfg",
        type=Path,
        default=None,
        help="Deploy (mmdeploy) config .py. Required with --auto-deploy.",
    )
    p.add_argument(
        "--model-cfg",
        type=Path,
        default=None,
        help="Model config; default: first .py in --exp-dir root (same as summarizer).",
    )
    p.add_argument(
        "--deploy-sample",
        type=Path,
        default=None,
        help="Sample image for export tracing. Required with --auto-deploy.",
    )
    p.add_argument(
        "--deploy-device",
        default="cpu",
        help="Device for --auto-deploy (passed to tools/deploy/deploy.py).",
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
        "--metrics-pkl",
        type=Path,
        default=None,
        help="Optional evaluation pickle; merged as metrics.json eval_extra",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the staging tree and print files; do not call Hugging Face",
    )
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow dirty git or missing .git in the code checkout",
    )
    p.add_argument(
        "--public",
        action="store_true",
        help="Create / upload to a public repo (default: private hf repo).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(
        argv
    )
    exp_dir = args.exp_dir.resolve()
    if args.auto_deploy and args.deploy is None:
        deploy: Optional[Path] = exp_dir / "deploy" / "onnx"
    else:
        deploy = Path(args.deploy).resolve() if args.deploy else None

    if args.auto_deploy and args.deploy_cfg is None:
        print(
            "--deploy-cfg is required with --auto-deploy",
            file=sys.stderr,
        )
        return 2
    if args.auto_deploy and args.deploy_sample is None:
        print(
            "--deploy-sample is required with --auto-deploy",
            file=sys.stderr,
        )
        return 2
    if (args.samples_in is None) ^ (args.samples_out is None):
        print(
            "Pass both --samples-in and --samples-out, or neither.",
            file=sys.stderr,
        )
        return 2

    pth = validate.validate_experiment(
        exp_dir, args.pth
    )
    with open(
        exp_dir / "summary.json", "r", encoding="utf-8", errors="replace"
    ) as f:
        summary: Dict[str, Any] = json.load(f)

    try:
        git_sha, _ = validate.check_git(
            REPO_ROOT, args.allow_dirty
        )
    except SystemExit as e:
        return int(e.code) if isinstance(
            e.code, int
        ) else 2

    model_cfg: Optional[Path] = args.model_cfg
    if model_cfg is None:
        c = find_root_config_py(
            exp_dir
        )
        if c and c.is_file():
            model_cfg = c
    if args.auto_deploy and model_cfg is None:
        print(
            "No model config: pass --model-cfg or add a .py in --exp-dir root",
            file=sys.stderr,
        )
        return 2
    if args.auto_deploy and args.deploy_cfg and model_cfg and args.deploy_sample:
        assert deploy is not None
        if auto_deploy.should_redeploy(
            deploy, pth
        ):
            try:
                auto_deploy.run_deploy(
                    deploy_py=DEPLOY_SCRIPT,
                    deploy_cfg=args.deploy_cfg.resolve(),
                    model_cfg=model_cfg.resolve(),
                    pth=pth,
                    sample_img=args.deploy_sample.resolve(),
                    work_dir=deploy,
                    device=args.deploy_device,
                )
            except (
                OSError,
                subprocess.CalledProcessError,
            ) as e:
                print(
                    f"auto-deploy failed: {e}",
                    file=sys.stderr,
                )
                return 3
    if deploy is not None:
        validate.deploy_drift(
            deploy, pth, pth_basename=pth.name
        )

    has_onnx_in_release = False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(
                tmp
            )
            staging.build_staging(
                exp_dir=exp_dir,
                pth=pth,
                deploy_dir=deploy,
                summary=summary,
                sample_in=args.samples_in,
                sample_out=args.samples_out,
                dest=dest,
            )
            metrics.write_metrics_json(
                dest, summary, args.metrics_pkl
            )
            has_onnx_in_release = (dest / "onnx" / "end2end.onnx").is_file()
            readme = card.render(
                summary=summary,
                repo_id=args.repo_id,
                tag=args.tag,
                git_sha=git_sha,
                message=args.message,
                has_onnx=has_onnx_in_release,
            )
            (dest / "README.md").write_text(
                readme, encoding="utf-8"
            )
            if args.dry_run:
                _print_staging_tree(
                    dest
                )
                return 0
            try:
                from huggingface_hub import (  # type: ignore[import-not-found]  # noqa: WPS433
                    HfApi,
                )
            except ImportError:
                print(
                    "huggingface_hub is not installed. Install: pip install -e '.[release]'",
                    file=sys.stderr,
                )
                return 2
            try:
                api = HfApi(
                )
                api.create_repo(
                    args.repo_id,
                    private=not args.public,
                    exist_ok=True,
                )
                api.upload_folder(
                    repo_id=args.repo_id,
                    folder_path=str(
                        dest
                    ),
                    commit_message=f"{args.tag}: {args.message or 'Release'}",
                )
                api.create_tag(
                    repo_id=args.repo_id,
                    tag=args.tag,
                    tag_message=args.message
                    or f"tag {args.tag}",
                    repo_type="model",
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"Hugging Face API error: {e}",
                    file=sys.stderr,
                )
                return 3
    except (OSError, ValueError) as e:
        print(
            f"Staging failed: {e}",
            file=sys.stderr,
        )
        return 2

    _write_release_json(
        exp_dir, args, git_sha, has_onnx=has_onnx_in_release
    )
    print(
        f"https://huggingface.co/{args.repo_id}/tree/{args.tag}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
