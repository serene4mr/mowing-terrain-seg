# Copyright (c) OpenMMLab. Mowing-terrain-seg. SPDX-License-Identifier: Apache-2.0
"""Build ``metrics.json`` for a Hugging Face model repo from ``summary.json``."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

SummaryDict = Dict[str, Any]


def write_metrics_json(
    staging: Path, summary: SummaryDict
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

    path = staging / "metrics.json"
    with open(
        path, "w", encoding="utf-8"
    ) as f:
        json.dump(
            out, f, indent=2, sort_keys=True, default=str
        )
