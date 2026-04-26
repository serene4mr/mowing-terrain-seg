# Copyright (c) OpenMMLab. Mowing-terrain-seg.
"""Tests for tools/summarize_experiment.py."""

import json
import shutil
from pathlib import Path

import pytest

from tools.summarize_experiment import (
    find_latest_run_dir,
    find_best_checkpoint,
    summarize,
)

FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "work_dir_minimal"
)


@pytest.fixture()
def work_minimal(tmp_path) -> Path:
    """Copy the minimal work_dir into an isolated temp directory."""
    dest = tmp_path / "w"
    shutil.copytree(
        FIXTURE, dest
    )
    d = dest / "summary.json"
    if d.is_file():
        d.unlink(
        )
    return dest


def test_summarize_writes_summary_json(
    work_minimal: Path,
) -> None:
    p = summarize(
        work_minimal
    )
    assert p == work_minimal / "summary.json"
    assert p.is_file(
    )
    d = json.loads(
        p.read_text(
            encoding="utf-8"
        )
    )
    assert d.get(
        "schema_version"
    ) == "1.0"


def test_summary_schema_v1(
    work_minimal: Path,
) -> None:
    summarize(
        work_minimal
    )
    o = json.loads(
        (work_minimal / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    for k in (
        "schema_version",
        "experiment",
        "history",
        "environment",
    ):
        assert k in o


def test_best_iteration_correct(
    work_minimal: Path,
) -> None:
    summarize(
        work_minimal
    )
    s = json.loads(
        (work_minimal / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert s["best"]["iteration"] == 200
    assert s["best"][
        "checkpoint"
    ] == "best_val_mIoU_iter_200.pth"
    assert s["best"][
        "metrics"
    ]["mIoU"] == 20.0


def test_last_iteration_correct(
    work_minimal: Path,
) -> None:
    summarize(
        work_minimal
    )
    s = json.loads(
        (work_minimal / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert s["last"][
        "iteration"
    ] == 300
    assert s["last"][
        "checkpoint"
    ] == "iter_300.pth"
    assert s["last"][
        "metrics"
    ]["mIoU"] == 15.0


def test_per_class_metrics_extracted(
    work_minimal: Path,
) -> None:
    summarize(
        work_minimal
    )
    s = json.loads(
        (work_minimal / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    m = s["best"]["metrics"][
        "per_class_IoU"
    ]
    assert set(
        m
    ) == {"Alpha", "Beta", "Gamma"}
    assert m["Alpha"] == 10.0


def test_history_downsampled(
    work_minimal: Path,
) -> None:
    summarize(
        work_minimal
    )
    s = json.loads(
        (work_minimal / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    h = s["history"]
    assert len(
        h
    ) <= 20
    for x in h:
        assert "iter" in x
        assert "mIoU" in x


def test_idempotent(
    work_minimal: Path,
) -> None:
    p1 = summarize(
        work_minimal
    )
    t1 = p1.stat(
    ).st_mtime
    p2 = summarize(
        work_minimal
    )
    t2 = p2.stat(
    ).st_mtime
    assert t2 >= t1
    d1 = json.loads(
        p1.read_text(
            encoding="utf-8"
        )
    )
    d2 = json.loads(
        p2.read_text(
            encoding="utf-8"
        )
    )
    assert d1 == d2


def test_missing_log_graceful(
    work_minimal: Path,
) -> None:
    log = work_minimal / "20260101_120000" / "20260101_120000.log"
    if log.is_file(
    ):
        log.unlink(
        )
    summarize(
        work_minimal
    )
    s = json.loads(
        (work_minimal / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert s["best"][
        "metrics"
    ].get("per_class_IoU") is None
    assert s["best"][
        "metrics"
    ]["mIoU"] == 20.0


def test_picks_latest_timestamp_dir(
    work_minimal: Path,
) -> None:
    run_new = work_minimal / "20260101_130000"
    run_new.mkdir(
        parents=True,
        exist_ok=True,
    )
    vis = run_new / "vis_data"
    vis.mkdir(
        parents=True,
        exist_ok=True,
    )
    shutil.copy2(
        work_minimal
        / "20260101_120000"
        / "vis_data"
        / "config.py",
        vis / "config.py",
    )
    with open(vis / "scalars.json", "w", encoding="utf-8") as f:
        f.write(
            '{"val/mIoU": 30.0, "val/mAcc": 30.0, "val/aAcc": 33.0, "step": 300}\n'
        )
    # Same per-class table as 120000, different aggregate + best@300
    olog = (FIXTURE / "20260101_120000" / "20260101_120000.log").read_text(
        encoding="utf-8"
    )
    head = "\n".join(olog.splitlines()[:-2])
    (run_new / "20260101_130000.log").write_text(
        head
        + "\n"
        "2026/01/01 13:00:01 - mmengine - INFO - "
        "Iter(val) [1/1]    val/aAcc: 33.0  val/mIoU: 30.0  val/mAcc: 30.0  "
        "val/mDice: 2.0  val/mFscore: 2.0  val/mPrecision: 2.0  val/mRecall: 2.0  "
        "data_time: 0.0  time: 0.0\n"
        "2026/01/01 13:00:02 - mmengine - INFO - The best checkpoint with 30.0 val/mIoU at "
        "300 iter is saved to best_val_mIoU_iter_300.pth.\n",
        encoding="utf-8",
    )
    (work_minimal / "best_val_mIoU_iter_300.pth").write_bytes(
        b""
    )
    r = find_latest_run_dir(
        work_minimal
    )
    assert r.name == "20260101_130000"
    b = find_best_checkpoint(
        work_minimal
    )
    assert b is not None
    assert b[0] == "best_val_mIoU_iter_300.pth"
    summarize(
        work_minimal
    )
    s = json.loads(
        (work_minimal / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert s["best"][
        "iteration"
    ] == 300
    # Latest run's sources used
    assert "20260101_130000" in s.get(
        "sources", {}
    ).get(
        "run_dir", ""
    )


def test_no_timestamp_dir_raises(
    tmp_path: Path,
) -> None:
    w = tmp_path / "empty"
    w.mkdir(
    )
    with pytest.raises(
        ValueError,
        match="No YYYYMMDD",
    ):
        find_latest_run_dir(
            w
        )
    (w / "nope.txt").write_text(
        "x",
    )
    with pytest.raises(
        ValueError,
    ):
        summarize(
            w
        )
