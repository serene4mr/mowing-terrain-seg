# Copyright (c) OpenMMLab. Mowing-terrain-seg. SPDX-License-Identifier: Apache-2.0
"""Validate an experiment work_dir before a Hugging Face release."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.summarize_experiment import find_root_config_py

try:
    from huggingface_hub import hf_hub_download as hf_hub_download_file  # type: ignore[import-not-found]
except ImportError:
    hf_hub_download_file = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)


def _find_repo_root_from(path: Path) -> Path:
    p = path.resolve()
    for _ in range(8):
        if (p / ".git").is_dir() or (p / ".git").is_file():
            return p
        if p.parent == p:
            break
        p = p.parent
    return path.resolve()


def validate_experiment(
    exp_dir: Path, pth_name: str, *, _exit: bool = True
) -> Path:
    """Check summary, root config, and checkpoint. Return absolute path to .pth.

    If _exit, calls ``sys.exit(2)`` on failure (CLI); otherwise raise ``ValueError``.
    """
    exp = exp_dir.resolve()
    err: List[str] = []
    if not exp.is_dir():
        err.append(f"Not a directory: {exp}")
    if not (exp / "summary.json").is_file():
        err.append(
            f"Missing {exp / 'summary.json'} — run `python tools/summarize_experiment.py` "
            "or `tools/train.py` first."
        )
    cfg = find_root_config_py(exp)
    if cfg is None or not cfg.is_file():
        err.append(
            f"No training config .py in {exp} (expected a top-level *config*.py; "
            "not vis_data/ snapshot). Copy your training config into the work_dir root."
        )
    pth = exp / pth_name
    if not pth.is_file():
        err.append(
            f"Missing checkpoint: {pth} — pass a filename in --exp-dir (e.g. best_*.pth)."
        )
    if err:
        msg = "Cannot release — fix the following and retry:\n" + "\n".join(
            f"  - {e}" for e in err
        )
        if _exit:
            print(msg, file=sys.stderr)
            raise SystemExit(2)
        raise ValueError(msg)
    return pth


def check_git(
    start_path: Path, allow_dirty: bool, *, _exit: bool = True
) -> Tuple[str, bool]:
    """Return (git_sha, dirty). Dirty blocks unless allow_dirty is True.

    If not in a git checkout, returns ``("unknown", True)`` and **fails** unless
    ``allow_dirty`` (same escape hatch as a dirty tree).
    """
    repo = _find_repo_root_from(start_path)
    if not (repo / ".git").exists():
        if _exit and not allow_dirty:
            print(
                f"No .git at or above {start_path}. "
                f"Use --allow-dirty to release without a git SHA. "
                f"(not recommended for production).",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return "unknown", True
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError) as e:
        if _exit and not allow_dirty:
            print(f"git rev-parse failed: {e}", file=sys.stderr)
            raise SystemExit(2)
        return "unknown", True
    try:
        porc = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo),
            text=True,
        )
    except (subprocess.CalledProcessError, OSError) as e:
        if _exit and not allow_dirty:
            print(f"git status failed: {e}", file=sys.stderr)
            raise SystemExit(2)
        return sha, True
    dirty = bool(porc.strip())
    if dirty and not allow_dirty:
        msg = f"Repository is dirty: {repo}\n{porc.rstrip()}\nCommit or use --allow-dirty."
        if _exit:
            print(msg, file=sys.stderr)
            raise SystemExit(2)
    return sha, dirty


def _deploy_hint() -> str:
    return (
        "Re-run: python tools/deploy/deploy.py <deploy_cfg> <model_cfg> <checkpoint> "
        "<image> --work-dir <this_deploy_dir> --dump-info"
    )


def deploy_drift(
    deploy_dir: Path, pth_path: Path, *, pth_basename: Optional[str] = None
) -> None:
    """Fail with exit 2 if onnx is missing or older than the checkpoint .pth.

    ``pth_basename`` is the filename the release uses (e.g. ``best_*.pth``); if set,
    we prefer matching this string in ``detail.json`` text when heuristics run.
    """
    d = deploy_dir.resolve()
    onnx = d / "end2end.onnx"
    if not onnx.is_file():
        print(
            f"Missing {onnx} under --deploy. {_deploy_hint()}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    pth = pth_path.resolve()
    pth_t = pth.stat().st_mtime
    o_t = onnx.stat().st_mtime
    if pth_t > o_t + 0.1:
        print(
            f"Checkpoint is newer than ONNX — redeploy so ORT model matches the .pth.\n"
            f"  pth:  {pth} (mtime {pth_t})\n"
            f"  onnx: {onnx} (mtime {o_t})\n"
            f"{_deploy_hint()}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    detail = d / "detail.json"
    if not detail.is_file():
        return
    try:
        raw = json.loads(detail.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        logger.warning("Could not parse detail.json: %s", e)
        return
    if isinstance(raw, str):
        text = raw
    else:
        text = json.dumps(raw, default=str)
    target = pth_basename or pth.name
    if target in text or pth.name in text or str(pth) in text:
        return
    logger.warning(
        "detail.json does not appear to reference %r (from %r). May be a stale deploy. %s",
        target,
        pth,
        _deploy_hint(),
    )


def _sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def validate_engine_against_hub(
    engine_dir: Path,
    *,
    repo_id: Optional[str] = None,
    allow_dirty: bool = False,
    _exit: bool = True,
) -> Dict[str, Any]:
    """Compare ``platform.json`` source ONNX hash to the file on the Hub.

    Downloads ``onnx/end2end.onnx`` for the recorded revision and compares
    SHA-256. Skips download when ``allow_dirty`` is True.
    """
    d = engine_dir.resolve()
    plat_path = d / "platform.json"
    if not plat_path.is_file():
        msg = f"Missing {plat_path}"
        if _exit:
            print(msg, file=sys.stderr)
            raise SystemExit(2)
        raise ValueError(msg)
    try:
        raw: Dict[str, Any] = json.loads(
            plat_path.read_text(encoding="utf-8", errors="replace")
        )
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in {plat_path}: {e}"
        if _exit:
            print(msg, file=sys.stderr)
            raise SystemExit(2)
        raise ValueError(msg) from e
    src = raw.get("source")
    if not isinstance(src, dict):
        msg = "platform.json missing 'source' object"
        if _exit:
            print(msg, file=sys.stderr)
            raise SystemExit(2)
        raise ValueError(msg)
    hub_repo = src.get("onnx_repo")
    rev = src.get("onnx_revision")
    expect_sha = src.get("onnx_sha256")
    hub_path = src.get("onnx_path", "onnx/end2end.onnx")
    if not hub_repo or not rev or not expect_sha:
        msg = "platform.json source must include onnx_repo, onnx_revision, onnx_sha256"
        if _exit:
            print(msg, file=sys.stderr)
            raise SystemExit(2)
        raise ValueError(msg)
    if repo_id and str(hub_repo) != str(repo_id):
        msg = (
            f"--repo-id {repo_id!r} does not match platform.json source.onnx_repo "
            f"{hub_repo!r}"
        )
        if _exit:
            print(msg, file=sys.stderr)
            raise SystemExit(2)
        raise ValueError(msg)
    if allow_dirty:
        return raw
    if hf_hub_download_file is None:
        msg = "huggingface_hub is not installed. Install: pip install -e '.[release]'"
        if _exit:
            print(msg, file=sys.stderr)
            raise SystemExit(2)
        raise ValueError(msg)
    try:
        local = hf_hub_download_file(
            repo_id=str(hub_repo),
            filename=str(hub_path),
            revision=str(rev),
        )
    except Exception as e:  # noqa: BLE001
        msg = f"Failed to download {hub_path} from {hub_repo}@{rev}: {e}"
        if _exit:
            print(msg, file=sys.stderr)
            raise SystemExit(2) from e
        raise ValueError(msg) from e
    got = _sha256_file(Path(local))
    if got != str(expect_sha):
        msg = (
            "ONNX on Hub does not match engine build provenance (SHA-256 drift).\n"
            f"  expected (platform.json): {expect_sha}\n"
            f"  got (Hub file):         {got}\n"
            "Rebuild the engine on Jetson from the current Hub ONNX, or use --allow-dirty."
        )
        if _exit:
            print(msg, file=sys.stderr)
            raise SystemExit(2)
        raise ValueError(msg)
    return raw
