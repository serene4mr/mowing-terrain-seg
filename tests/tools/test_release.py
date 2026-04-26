# Copyright (c) OpenMMLab. Mowing-terrain-seg.
"""Tests for tools/release.py and tools/hf_release/."""

import json
import os
import pickle
import shutil
import sys
from pathlib import Path
from unittest import mock

import pytest

from tools.hf_release import card, metrics, staging
from tools.hf_release import validate as v

FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "work_dir_minimal"
)


@pytest.fixture
def work_release(tmp_path: Path) -> Path:
    """Copy fixture work_dir for release tests; keep ``summary.json``."""
    d = tmp_path / "w"
    shutil.copytree(
        FIXTURE, d
    )
    (d / "iter_300.pth").write_bytes(
        b"PK\x00"
    )
    if not (d / "iter_300.pth").is_file():
        raise AssertionError("pth not copied")
    return d


def test_validate_missing_pth_exits_2(
    work_release: Path,
) -> None:
    with pytest.raises(SystemExit) as e:
        v.validate_experiment(
            work_release, "nope.pth"
        )
    assert e.value.code == 2


def test_validate_missing_summary_exits_2(
    work_release: Path,
) -> None:
    (work_release / "summary.json").unlink(
    )
    with pytest.raises(SystemExit) as e:
        v.validate_experiment(
            work_release, "iter_300.pth"
        )
    assert e.value.code == 2


def test_check_git_allows_with_allow_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir(
    )
    def _out(
        *a, **k
    ):  # type: ignore[no-untyped-def]
        cmd = a[0] if a else []
        if "rev-parse" in cmd:
            return "abc1\n"
        return "M a\n"  # dirty

    monkeypatch.setattr(
        "tools.hf_release.validate.subprocess.check_output",
        _out,
    )
    s, d = v.check_git(
        tmp_path, allow_dirty=True
    )
    assert s == "abc1"
    assert d is True


def test_check_git_dirty_blocks(
    work_release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake(
        *a, **k
    ):  # type: ignore[no-untyped-def]
        cmd = a[0] if a else []
        if "rev-parse" in cmd:
            return "sha0\n"
        return "M x\n"

    monkeypatch.setattr(
        "tools.hf_release.validate.subprocess.check_output",
        _fake,
    )
    with pytest.raises(SystemExit) as e:
        v.check_git(
            work_release, allow_dirty=False
        )
    assert e.value.code == 2


def test_deploy_drift_pth_newer_than_onnx_fails(
    work_release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pth = work_release / "iter_300.pth"
    onnx = work_release / "deploy" / "onnx" / "end2end.onnx"
    t_old = 1_000_000
    t_new = 2_000_000
    os.utime(
        onnx, (t_old, t_old)
    )
    os.utime(
        pth, (t_new, t_new)
    )
    with pytest.raises(SystemExit) as e:
        v.deploy_drift(
            work_release / "deploy" / "onnx", pth, pth_basename=pth.name
        )
    assert e.value.code == 2


def test_deploy_drift_ok_when_onnx_newer(
    work_release: Path
) -> None:
    pth = work_release / "iter_300.pth"
    onnx = work_release / "deploy" / "onnx" / "end2end.onnx"
    t_p = 1_000_000
    t_o = 2_000_000
    os.utime(
        pth, (t_p, t_p)
    )
    os.utime(
        onnx, (t_o, t_o)
    )
    v.deploy_drift(
        work_release / "deploy" / "onnx", pth, pth_basename=pth.name
    )


def test_staging_pytorch_only_layout(
    work_release: Path, tmp_path: Path
) -> None:
    s = json.loads(
        (work_release / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    dest = tmp_path / "st"
    pth = v.validate_experiment(
        work_release, "iter_300.pth", _exit=False
    )
    staging.build_staging(
        exp_dir=work_release,
        pth=pth,
        deploy_dir=None,
        summary=s,
        sample_in=None,
        sample_out=None,
        dest=dest,
    )
    assert (dest / "config.py").is_file()
    assert (dest / "summary.json").is_file()
    assert (dest / "pytorch" / "best.pth").is_file()
    assert not (dest / "onnx" / "end2end.onnx").is_file()


def test_staging_with_onnx_layout(
    work_release: Path, tmp_path: Path
) -> None:
    s = json.loads(
        (work_release / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    pth = v.validate_experiment(
        work_release, "iter_300.pth", _exit=False
    )
    dest = tmp_path / "st2"
    staging.build_staging(
        exp_dir=work_release,
        pth=pth,
        deploy_dir=work_release / "deploy" / "onnx",
        summary=s,
        sample_in=None,
        sample_out=None,
        dest=dest,
    )
    assert (dest / "onnx" / "end2end.onnx").is_file()
    assert (dest / "onnx" / "detail.json").is_file()


def test_card_renders_required_yaml_keys(
    work_release: Path
) -> None:
    s = json.loads(
        (work_release / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    t = card.render(
        summary=s,
        repo_id="org/mts-t",
        tag="v0",
        git_sha="abc",
        message="hi",
        has_onnx=True,
    )
    assert "---" in t
    assert "library_name: mmsegmentation" in t
    assert "0.42" in t
    assert "mIoU" in t


def test_metrics_json_from_summary_only(
    work_release: Path, tmp_path: Path
) -> None:
    s = json.loads(
        (work_release / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    metrics.write_metrics_json(
        tmp_path, s, None
    )
    m = json.loads(
        (tmp_path / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert m.get("mIoU") is not None
    assert m.get("mIoU") == 0.42


def test_metrics_json_merges_eval_pkl(
    work_release: Path, tmp_path: Path
) -> None:
    s = json.loads(
        (work_release / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    p = tmp_path / "e.pkl"
    with open(
        p, "wb"
    ) as f:
        pickle.dump(
            {
                "note": 1,
            },
            f,
        )
    metrics.write_metrics_json(
        tmp_path, s, p
    )
    m = json.loads(
        (tmp_path / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert m.get("eval_extra") == {
        "note": 1
    }


def test_dry_run_no_hf(
    work_release: Path, capsys: pytest.CaptureFixture
) -> None:
    import tools.release

    rc = tools.release.main(
        [
            "--exp-dir",
            str(work_release),
            "--pth",
            "iter_300.pth",
            "--repo-id",
            "u/m",
            "--tag",
            "t1",
            "--dry-run",
            "--allow-dirty",
        ]
    )
    assert rc == 0
    out = capsys.readouterr(
    ).out
    assert "pytorch" in out or "config.py" in out


def test_upload_calls_hf_api(
    work_release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types

    import tools.release

    class _Api:  # noqa: D101
        def __init__(self) -> None:
            self.creates = 0
            self.uploads = 0
            self.tags = 0

        def create_repo(
            self, *a, **k
        ):  # type: ignore[no-untyped-def]
            self.creates += 1

        def upload_folder(
            self, *a, **k
        ):  # type: ignore[no-untyped-def]
            self.uploads += 1

        def create_tag(
            self, *a, **k
        ):  # type: ignore[no-untyped-def]
            self.tags += 1

    inst = _Api(
    )
    shub = types.SimpleNamespace(
        HfApi=lambda: inst
    )
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", shub
    )
    # skip real git: already has .git in worktree or use allow-dirty
    rc = tools.release.main(
        [
            "--exp-dir",
            str(work_release),
            "--pth",
            "iter_300.pth",
            "--repo-id",
            "u/m",
            "--tag",
            "t1",
            "--message",
            "m",
            "--allow-dirty",
        ]
    )
    assert rc == 0
    assert inst.creates == 1
    assert inst.uploads == 1
    assert inst.tags == 1
    rj = work_release / "release.json"
    assert rj.is_file(
    )
    d = json.loads(
        rj.read_text(
            encoding="utf-8"
        )
    )
    assert d["tag"] == "t1"
    assert d["repo_id"] == "u/m"


def test_auto_deploy_invokes_subprocess_when_stale(
    work_release: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import types

    import tools.release

    pth = work_release / "iter_300.pth"
    onnx = work_release / "deploy" / "onnx" / "end2end.onnx"
    os.utime(
        onnx, (100, 100)
    )
    os.utime(
        pth, (999_999, 999_999)
    )
    calls: list = []

    def _fake(
        *a, **k
    ):  # type: ignore[no-untyped-def]
        calls.append(1)
        # Simulate a fresh export: ONNX newer than the .pth
        t = (work_release / "iter_300.pth").stat().st_mtime + 1.0
        os.utime(
            onnx, (t, t)
        )

    monkeypatch.setattr(
        "tools.hf_release.auto_deploy.subprocess_run",
        _fake,
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            HfApi=lambda: mock.MagicMock(
                create_repo=mock.MagicMock(),
                upload_folder=mock.MagicMock(),
                create_tag=mock.MagicMock(),
            )
        ),
    )
    img = tmp_path / "a.jpg"
    img.write_bytes(
        b"fake"
    )
    cfg = tmp_path / "d.py"
    cfg.write_text(
        "# c"
    )
    rc = tools.release.main(
        [
            "--exp-dir",
            str(work_release),
            "--pth",
            "iter_300.pth",
            "--repo-id",
            "u/m",
            "--tag",
            "t2",
            "--auto-deploy",
            "--deploy-cfg",
            str(cfg),
            "--deploy-sample",
            str(img),
            "--allow-dirty",
        ]
    )
    assert rc == 0
    assert len(calls) == 1


def test_release_json_written_to_exp_dir(
    work_release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types

    import tools.release

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            HfApi=lambda: mock.MagicMock(
                create_repo=mock.MagicMock(),
                upload_folder=mock.MagicMock(),
                create_tag=mock.MagicMock(),
            )
        ),
    )
    tools.release.main(
        [
            "--exp-dir",
            str(work_release),
            "--pth",
            "iter_300.pth",
            "--repo-id",
            "a/b",
            "--tag",
            "v9",
            "--allow-dirty",
        ]
    )
    p = work_release / "release.json"
    j = json.loads(
        p.read_text(
            encoding="utf-8"
        )
    )
    assert j["pth"] == "iter_300.pth"
    assert "git_sha" in j
    assert "released_at" in j
