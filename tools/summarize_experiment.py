# Copyright (c) OpenMMLab. Mowing-terrain-seg. SPDX-License-Identifier: Apache-2.0
"""Build work_dir/summary.json from mmengine run outputs (scalars + train log)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

_RUN_TS = re.compile(r"^\d{8}_\d{6}$")
_BEST_PTH = re.compile(
    r"^best_val_mIoU_iter_(\d+)\.pth$", re.IGNORECASE
)
_ITER_PTH = re.compile(r"^iter_(\d+)\.pth$", re.IGNORECASE)
_BEST_LOG_LINE = re.compile(
    r"The best checkpoint with ([\d.]+) val/mIoU at (\d+) iter is saved to "
    r"best_val_mIoU_iter_(\d+)\.pth"
)


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _find_timestamp_run_dirs(work_dir: Path) -> List[Path]:
    out: List[Path] = []
    if not work_dir.is_dir():
        return out
    for d in work_dir.iterdir():
        if d.is_dir() and _RUN_TS.match(d.name):
            out.append(d)
    return out


def find_latest_run_dir(work_dir: Path) -> Path:
    """Pick newest YYYYMMDD_HHMMSS under work_dir, or work_dir if it is already a run dir."""
    w = work_dir.resolve()
    if _RUN_TS.match(w.name) and w.is_dir():
        return w
    ts_dirs = _find_timestamp_run_dirs(w)
    if not ts_dirs:
        raise ValueError(
            f"No YYYYMMDD_HHMMSS run subdirectory under: {w}. "
            "Pass the experiment work_dir (parent of 2026xxxx_xxxxxx/)."
        )
    return sorted(ts_dirs, key=lambda p: p.name)[-1]


def parse_scalars_jsonl(
    path: Path,
) -> Tuple[List[dict], List[dict], int]:
    """Return (all_rows, val_rows, max_step)."""
    rows: List[dict] = []
    if not path.is_file():
        return rows, [], 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    val_rows = [r for r in rows if "val/mIoU" in r]
    max_step = 0
    for r in rows:
        s = r.get("step", r.get("iter"))
        if s is not None:
            try:
                max_step = max(max_step, int(s))
            except (TypeError, ValueError):
                pass
    return rows, val_rows, max_step


def _parse_per_class_table_row(line: str) -> Optional[Tuple[str, Dict[str, float]]]:
    """Parse a single mmengine 'per class' table data row; None if not a class row."""
    s = line.strip()
    if not s.startswith("|") or s.startswith("+-") or s.startswith("|-") or s.startswith("+----------------"):
        return None
    parts = [p.strip() for p in s.split("|")]
    if len(parts) < 6:
        return None
    # ['', 'Name', 'IoU', 'Acc', 'Dice', ...]
    name = (parts[1] or "").strip()
    if not name or name.lower() == "class":
        return None
    if re.match(
        r"^\|\s*Class\s*\|", s
    ) and "IoU" in s and "Precision" in s:
        return None

    def _cell_f(cell: str) -> Optional[float]:
        t = (cell or "").split()
        if not t:
            return None
        try:
            return float(
                t[0]
            )
        except ValueError:
            return None

    iou = _cell_f(
        parts[2]
    )
    acc = _cell_f(
        parts[3]
    )
    dice = _cell_f(
        parts[4]
    )
    if iou is None or acc is None or dice is None:
        return None
    return name, {"IoU": iou, "Acc": acc, "Dice": dice}


def parse_log_for_eval_at_iter(
    log_path: Path, best_iter: int
) -> Dict[str, Any]:
    """
    Return dict with 'aggregate', 'per_class' for the validation that produced
    best checkpoint at `best_iter`, parsed from the train log.
    """
    if not log_path.is_file():
        return {}
    lines = log_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    # Find a best-line for this training iteration
    best_line_idx: Optional[int] = None
    for i, line in enumerate(lines):
        m = _BEST_LOG_LINE.search(line)
        if m and int(m.group(2)) == best_iter:
            best_line_idx = i
            break
    if best_line_idx is None:
        return {}
    # The aggregate line is the first Iter(val) line above the best line
    # that also reports val/mIoU (full-dataset val pass).
    j: Optional[int] = None
    for k in range(
        best_line_idx - 1, -1, -1
    ):
        s = lines[k]
        if (
            "Iter(val)" in s
            and "val/mIoU:" in s
        ):
            mfrac = re.search(
                r"Iter\(val\)\s+\[(\d+)/(\d+)\]", s
            )
            if mfrac and mfrac.group(1) == mfrac.group(2):
                j = k
                break
    if j is None:
        for k2 in range(
            best_line_idx - 1, -1, -1
        ):
            s2 = lines[k2]
            if (
                "Iter(val)" in s2
                and "val/mIoU:" in s2
            ):
                j = k2
                break
    if j is None or j < 0:
        return {}
    agg: Dict[str, float] = {}
    s = lines[j]
    for k, p in [
        ("aAcc", r"val/aAcc:\s*([-\d.]+)"),
        ("mIoU", r"val/mIoU:\s*([-\d.]+)"),
        ("mAcc", r"val/mAcc:\s*([-\d.]+)"),
        ("mDice", r"val/mDice:\s*([-\d.]+)"),
        ("mFscore", r"val/mFscore:\s*([-\d.]+)"),
        ("mPrecision", r"val/mPrecision:\s*([-\d.]+)"),
        ("mRecall", r"val/mRecall:\s*([-\d.]+)"),
    ]:
        m2 = re.search(p, s)
        if m2:
            try:
                agg[k] = float(
                    m2.group(1)
                )
            except ValueError:
                pass
    per: Dict[str, Dict[str, float]] = {}
    for k2 in range(
        j - 1, -1, -1
    ):
        line = lines[k2]
        if "per class results" in line:
            break
        if line.strip(
        ).startswith(
            "+---"
        ):
            continue
        if re.search(
            r"^\|\s*Class\s*\|", line
        ) and "IoU" in line and "Acc" in line:
            continue
        parsed = _parse_per_class_table_row(
            line
        )
        if parsed:
            per[parsed[0]] = parsed[1]
    return {"aggregate": agg, "per_class": per}


def find_best_checkpoint(
    work_dir: Path,
) -> Optional[Tuple[str, int]]:
    """Return (filename, iter) for best mIoU checkpoint (max iter if multiple)."""
    best: List[Tuple[str, int, int]] = []
    for p in work_dir.glob("best_val_mIoU_iter_*.pth"):
        m = _BEST_PTH.match(p.name)
        if m:
            it = int(m.group(1))
            best.append((p.name, it, p.stat().st_mtime))
    if not best:
        return None
    return max(best, key=lambda t: t[1])[:2]


def read_last_checkpoint(work_dir: Path) -> Optional[str]:
    p = work_dir / "last_checkpoint"
    if not p.is_file():
        return None
    raw = p.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return None
    b = Path(raw)
    if b.is_absolute() and b.name:
        return b.name
    return Path(raw).name


def downsample_history(
    val_rows: List[dict], n: int = 20
) -> List[Dict[str, float]]:
    if not val_rows:
        return []
    rows = sorted(
        val_rows, key=lambda r: int(r.get("step", r.get("iter", 0)))
    )
    if len(rows) <= n:
        return [
            {
                "iter": int(
                    r.get("step", r.get("iter", 0)) or 0
                ),
                "mIoU": _as_float(
                    r.get("val/mIoU")
                ) or 0.0,
            }
            for r in rows
        ]
    idxs = {
        int(i * (len(rows) - 1) / (n - 1)) for i in range(n)
    } if n > 1 else {0}
    out: List[Dict[str, float]] = []
    for i in sorted(idxs):
        r = rows[i]
        st = r.get("step", r.get("iter", 0))
        out.append(
            {
                "iter": int(st or 0),
                "mIoU": _as_float(
                    r.get("val/mIoU")
                ) or 0.0,
            }
        )
    return out


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_info(repo_root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not (repo_root / ".git").exists():
        return out
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(repo_root),
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        out["git_sha"] = head[:12]
        out["git_dirty"] = dirty
    except (OSError, subprocess.CalledProcessError):
        return {}
    return out


def _find_repo_root(path: Path) -> Path:
    p = path.resolve()
    for _ in range(6):
        if (p / ".git").is_dir() or (p / ".git").is_file():
            return p
        if p.parent == p:
            break
        p = p.parent
    return path


def find_root_config_py(work_dir: Path) -> Optional[Path]:
    """A Python config file in work_dir root (not the vis_data/ snapshot)."""
    cands: List[Path] = [
        f for f in work_dir.glob("*.py")
        if f.is_file() and f.name not in (
            "summary_to_md.py",  # reserved, none for now
        )
    ]
    if not cands:
        return None
    cands.sort(key=lambda p: p.name)
    return cands[0]


def _parse_dataset_info_from_config_cfg(cfg: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {"split": "val"}
    classes = None
    if cfg is not None and hasattr(cfg, "get"):
        m = cfg.get("metainfo")
        if m and isinstance(m, dict) and m.get("classes"):
            classes = m["classes"]
        elif isinstance(cfg, dict) and "metainfo" in cfg:  # type: ignore
            m2 = dict(cfg).get("metainfo")
            if m2 and isinstance(m2, dict) and m2.get("classes"):
                classes = m2["classes"]
    if not classes and cfg is not None and hasattr(cfg, "get"):
        classes = cfg.get("classes")
    if classes is not None:
        info["class_names"] = [str(c) for c in list(classes)]
        info["num_classes"] = len(info["class_names"])
    dname = None
    if cfg is not None and hasattr(cfg, "get"):
        dname = cfg.get("dataset_type")
    if dname is not None:
        info["name"] = str(dname)
    return info


def _load_config_metadata(vis_data_config: Path) -> Dict[str, Any]:
    try:
        from mmengine.config import Config
    except ImportError:  # pragma: no cover
        return {}
    try:
        c = Config.fromfile(str(vis_data_config))
        return _parse_dataset_info_from_config_cfg(c)
    except Exception:
        return {}


def detect_environment() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "python": f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"
    }
    for mod, key in [
        ("torch", "torch"),
        ("mmcv", "mmcv"),
        ("mmseg", "mmseg"),
        ("mmengine", "mmengine"),
    ]:
        try:
            m = __import__(mod, fromlist=["__version__"])
            out[key] = getattr(
                m, "__version__", "unknown"
            )
        except ImportError:  # pragma: no cover
            pass
    try:
        import torch
        if torch.cuda.is_available():
            out["cuda"] = str(torch.version.cuda)
            n = torch.cuda.get_device_name(0)
            if n:
                out["gpu"] = n
    except Exception:  # noqa: BLE001
        pass
    return out


def _row_to_metrics(
    r: dict,
) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    m["mIoU"] = _as_float(r.get("val/mIoU"))
    m["mAcc"] = _as_float(r.get("val/mAcc"))
    m["aAcc"] = _as_float(r.get("val/aAcc"))
    m["mDice"] = _as_float(r.get("val/mDice"))
    m["mFscore"] = _as_float(r.get("val/mFscore"))
    m["mPrecision"] = _as_float(
        r.get("val/mPrecision")
    )
    m["mRecall"] = _as_float(
        r.get("val/mRecall")
    )
    return {k: v for k, v in m.items() if v is not None}


def summarize(
    work_dir: Union[str, Path], *, overwrite: bool = True
) -> Path:
    """
    Read mmengine `work_dir/<ts>/...` outputs and write `work_dir/summary.json`.

    Raises ValueError for invalid work_dir layout. IO errors propagate.
    """
    wd = Path(work_dir).resolve()
    if not wd.is_dir():
        raise ValueError(f"Not a directory: {wd}")

    # Checkpoints and last_checkpoint live in the *experiment* root, not the
    # per-run YYYYMMDD_HHMMSS folder. Allow `work_dir` to be either root or a
    # child run directory.
    if _RUN_TS.match(
        wd.name
    ) and (wd / "vis_data" / "scalars.json").is_file():
        run = wd
        exp_root = wd.parent
    else:
        exp_root = wd
        run = find_latest_run_dir(
            exp_root
        )
    ts_name = run.name
    y, mo, d = int(ts_name[0:4]), int(
        ts_name[4:6]
    ), int(ts_name[6:8])
    h, mi, s = int(ts_name[9:11]), int(
        ts_name[11:13]
    ), int(ts_name[13:15])
    started_utc = datetime(
        y, mo, d, h, mi, s, tzinfo=timezone.utc
    ).isoformat().replace(
        "+00:00", "Z"
    )

    scalars_path = run / "vis_data" / "scalars.json"
    all_rows, val_rows, max_step = parse_scalars_jsonl(
        scalars_path
    )
    log_path = run / f"{ts_name}.log"
    mtime_utc: Optional[str] = None
    if log_path.is_file():
        mtime_utc = datetime.fromtimestamp(
            log_path.stat().st_mtime, tz=timezone.utc
        ).isoformat().replace(
            "+00:00", "Z"
        )

    if not mtime_utc and all_rows:
        mtime_utc = started_utc

    duration: Optional[int] = None
    if mtime_utc and started_utc:
        t0 = datetime.fromisoformat(
            started_utc.replace("Z", "+00:00")
        )
        t1 = datetime.fromisoformat(
            mtime_utc.replace("Z", "+00:00")
        )
        duration = max(0, int((t1 - t0).total_seconds()))

    vis_cfg = run / "vis_data" / "config.py"
    dataset: Dict[str, Any] = _load_config_metadata(vis_cfg)

    root_cfg = find_root_config_py(
        exp_root
    )
    config_path_str: Optional[str] = None
    config_sha: Optional[str] = None
    if root_cfg and root_cfg.is_file():
        config_path_str = str(root_cfg)
        try:
            config_sha = sha256_of_file(
                root_cfg
            )
        except OSError:
            config_sha = None

    repo = _find_repo_root(
        exp_root
    )
    experiment: Dict[str, Any] = {
        "name": exp_root.name,
    }
    if config_path_str:
        experiment["config_path"] = config_path_str
    if config_sha:
        experiment["config_sha256"] = config_sha
    experiment["started_at"] = started_utc
    if mtime_utc:
        experiment["ended_at"] = mtime_utc
    if duration is not None:
        experiment["duration_seconds"] = duration
    if max_step:
        experiment["iterations"] = max_step
    experiment.update(get_git_info(repo))

    best = find_best_checkpoint(
        exp_root
    )
    best_block: Dict[str, Any] = {}
    if best:
        fname, biter = best
        best_block["checkpoint"] = fname
        best_block["iteration"] = biter
        log_eval = parse_log_for_eval_at_iter(
            log_path, biter
        )
        metrics: Dict[str, Any] = {}
        for r in val_rows:
            st = r.get("step", r.get("iter"))
            if st is not None and int(
                st
            ) == biter:
                metrics = _row_to_metrics(r)
                break
        if not metrics and log_eval.get("aggregate"):
            for k, v in log_eval["aggregate"].items():
                # map aAcc->aAcc, mIoU->mIoU
                if k in (
                    "aAcc",
                    "mIoU",
                    "mAcc",
                    "mDice",
                    "mFscore",
                ):
                    metrics[k] = v
        per = log_eval.get("per_class") or {}
        if per:
            iou: Dict[str, float] = {}
            acc: Dict[str, float] = {}
            for cname, d in per.items():
                if "IoU" in d:
                    iou[cname] = d["IoU"]
                if "Acc" in d:
                    acc[cname] = d["Acc"]
            if iou:
                metrics["per_class_IoU"] = iou
            if acc:
                metrics["per_class_Acc"] = acc
        if metrics:
            best_block["metrics"] = metrics

    last: Dict[str, Any] = {}
    last_name = read_last_checkpoint(
        exp_root
    )
    if last_name:
        m = _ITER_PTH.match(last_name)
        if m:
            last["checkpoint"] = last_name
            last["iteration"] = int(m.group(1))
    if val_rows and last:
        # attach metrics for last *validation* that finished before/ at last
        # saved iter, not strict — use the last val row
        r_last = val_rows[-1]
        m_last = _row_to_metrics(
            r_last
        )
        if m_last:
            last["metrics"] = m_last

    out: Dict[str, Any] = {
        "schema_version": "1.0",
        "experiment": experiment,
    }
    if dataset:
        out["dataset"] = dataset
    if best_block:
        out["best"] = best_block
    if last:
        out["last"] = last
    out["history"] = downsample_history(
        val_rows, 20
    )
    out["environment"] = detect_environment()
    out["sources"] = {
        "run_dir": str(
            run
        ),
        "scalars_json": str(
            scalars_path
        ) if scalars_path.is_file() else None,
        "train_log": str(
            log_path
        ) if log_path.is_file() else None,
    }

    target = exp_root / "summary.json"
    if not overwrite and target.is_file():
        return target
    with open(
        target, "w", encoding="utf-8"
    ) as f:
        json.dump(
            out, f, indent=2, sort_keys=True
        )
    return target


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Write summary.json for an mmengine work_dir"
    )
    p.add_argument(
        "work_dir",
        type=Path,
        help="Path to work_dirs/<experiment>/",
    )
    args = p.parse_args(
        argv
    )
    sp = summarize(
        args.work_dir
    )
    print(
        f"Wrote {sp}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
