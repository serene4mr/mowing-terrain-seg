from typing import Dict, Any
from pathlib import Path
import yaml


def load_cfg(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)