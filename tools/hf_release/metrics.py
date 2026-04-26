# Copyright (c) OpenMMLab. Mowing-terrain-seg. SPDX-License-Identifier: Apache-2.0
"""Build ``metrics.json`` for a Hugging Face model repo from ``summary.json`` and optional eval pickle."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

SummaryDict = Dict[str, Any]


def _shallow_eval_pkl(obj: Any) -> Any:
    """Return JSON-serializable summary of an arbitrary eval pickle (best-effort)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {
            k: _shallow_eval_pkl(v)
            for k, v in list(obj.items())[:200]
        }
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_shallow_eval_pkl(x) for x in obj[:200]]
    return f"<{type(obj).__name__}>"


def write_metrics_json(
    staging: Path, summary: SummaryDict, eval_pkl: Optional[Path]
) -> None:
    out: Dict[str, Any] = {
        "evaluated_at": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    best = summary.get("best")
    if isinstance(best, dict):
        it = best.get("iteration")
        if it is not None:
            out["iteration"] = it
        m = best.get("metrics")
        if isinstance(m, dict):
            for k, v in m.items():
                out[k] = v
    ds = summary.get("dataset")
    if isinstance(ds, dict) and "split" in ds:
        out["dataset_split"] = ds.get("split")
    elif isinstance(ds, dict) and "type" in ds:
        out["dataset_type"] = ds.get("type")
    if eval_pkl is not None and eval_pkl.is_file():
        with open(eval_pkl, "rb") as f:
            pkl = pickle.load(f)
        out["eval_extra"] = _shallow_eval_pkl(
            pkl
        )
    path = staging / "metrics.json"
    with open(
        path, "w", encoding="utf-8"
    ) as f:
        json.dump(
            out, f, indent=2, sort_keys=True, default=str
        )
