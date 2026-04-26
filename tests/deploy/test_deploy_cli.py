"""Deploy script CLI matches README (positional args)."""
import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_deploy_help_shows_positional_metavar():
    r = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "tools", "deploy", "deploy.py"), "-h"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    out = (r.stdout or "") + (r.stderr or "")
    assert "deploy_cfg" in out
    assert "model_cfg" in out
    assert "checkpoint" in out
    assert "img" in out
    assert r.returncode == 0
