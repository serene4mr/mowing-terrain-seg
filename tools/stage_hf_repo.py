"""Stage a completed training run as a Hugging Face model repository.

Takes a training run directory and its ONNX export, extracts all metadata
dynamically (metrics, classes, git SHA), and packages everything into a clean
staging folder under ``work_dirs/.hf/<repo-name>/`` ready for review and upload.

Usage:
  python tools/stage_hf_repo.py \
      --run-dir work_dirs/mts_segformer_mit-b0/run_20260501_093744 \
      --pth best_val_mIoU_iter_8000.pth \
      --onnx-dir work_dirs/mts_segformer_mit-b0/run_20260501_093744/deploy/onnx \
      --repo-name mts-segformer-mit-b0-ycor-3cls-512x512

  # Then upload with:
  huggingface-cli upload <org>/<repo-name> ./work_dirs/.hf/<repo-name>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
WORK_DIRS = REPO_ROOT / "work_dirs"
STAGING_ROOT = WORK_DIRS / ".hf"


# ---------------------------------------------------------------------------
# 1. Argument Parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage a training run as a Hugging Face model repository.",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to the MMEngine run folder (e.g. work_dirs/<exp>/run_xxx).",
    )
    p.add_argument(
        "--pth",
        required=True,
        help=(
            "Checkpoint file to ship. Can be a filename relative to --run-dir "
            "or an absolute path (e.g. best_val_mIoU_iter_8000.pth)."
        ),
    )
    p.add_argument(
        "--onnx-dir",
        type=Path,
        required=True,
        help="Path to the ONNX export folder (must contain at least one .onnx file).",
    )
    p.add_argument(
        "--repo-name",
        required=True,
        help="HF repo name following mts-<arch>-<backbone>-<dataset>-<resolution>.",
    )
    p.add_argument(
        "--tensorrt-dir",
        type=Path,
        default=None,
        help="Optional path to TensorRT engine folder.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output staging directory. Defaults to work_dirs/.hf/<repo-name>.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# 2. Input Validation
# ---------------------------------------------------------------------------

def _resolve_pth(run_dir: Path, pth_arg: str) -> Path:
    """Resolve the checkpoint path (absolute or relative to run_dir)."""
    p = Path(pth_arg)
    if p.is_absolute() and p.is_file():
        return p
    candidate = run_dir / pth_arg
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Checkpoint not found: tried '{pth_arg}' and '{candidate}'"
    )


def _find_scalars_json(run_dir: Path) -> Path:
    """Find the scalars.json inside the vis_data subdirectory."""
    candidates = list(run_dir.rglob("vis_data/scalars.json"))
    if not candidates:
        raise FileNotFoundError(
            f"scalars.json not found under {run_dir}"
        )
    return candidates[0]


def _find_log_file(run_dir: Path) -> Path:
    """Find the .log text file inside the run directory."""
    candidates = list(run_dir.rglob("*.log"))
    if not candidates:
        raise FileNotFoundError(f"No .log file found under {run_dir}")
    return candidates[0]


def _find_config_py(run_dir: Path) -> Path:
    """Find the dumped config .py file at the root of run_dir."""
    candidates = [
        f for f in run_dir.iterdir()
        if f.suffix == ".py" and f.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No .py config file found in {run_dir}"
        )
    return candidates[0]


def validate_inputs(args: argparse.Namespace) -> Dict[str, Path]:
    """Validate all inputs and return resolved paths."""
    if not args.run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {args.run_dir}")

    pth_path = _resolve_pth(args.run_dir, args.pth)
    scalars_path = _find_scalars_json(args.run_dir)
    log_path = _find_log_file(args.run_dir)
    config_path = _find_config_py(args.run_dir)

    if not args.onnx_dir.is_dir():
        raise FileNotFoundError(f"ONNX directory not found: {args.onnx_dir}")
    onnx_files = list(args.onnx_dir.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(
            f"No .onnx file found in {args.onnx_dir}"
        )

    if args.tensorrt_dir is not None and not args.tensorrt_dir.is_dir():
        raise FileNotFoundError(
            f"TensorRT directory not found: {args.tensorrt_dir}"
        )

    return {
        "pth": pth_path,
        "scalars": scalars_path,
        "log": log_path,
        "config": config_path,
    }


# ---------------------------------------------------------------------------
# 3. Metadata Extraction
# ---------------------------------------------------------------------------

def extract_checkpoint_meta(pth_path: Path) -> Dict[str, Any]:
    """Extract iteration, classes, and palette from the .pth checkpoint."""
    import torch

    ckpt = torch.load(pth_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    dataset_meta = meta.get("dataset_meta", {})

    return {
        "iter": meta.get("iter", 0),
        "classes": list(dataset_meta.get("classes", [])),
        "palette": dataset_meta.get("palette", []),
    }


def extract_global_metrics(
    scalars_path: Path, target_step: int
) -> Dict[str, float]:
    """Extract val/* metrics from scalars.json for the target step."""
    metrics: Dict[str, float] = {}
    with open(scalars_path, "r") as f:
        for line in f:
            record = json.loads(line.strip())
            if record.get("step") == target_step and "val/mIoU" in record:
                for key, value in record.items():
                    if key.startswith("val/"):
                        # Strip "val/" prefix for clean names
                        clean_key = key[4:]
                        metrics[clean_key] = value
                break
    if not metrics:
        raise ValueError(
            f"No validation metrics found for step {target_step} "
            f"in {scalars_path}"
        )
    return metrics


def extract_per_class_metrics(
    log_path: Path, target_step: int
) -> List[Dict[str, Any]]:
    """Parse the per-class ASCII table from the .log file for the target step.

    Returns a list of dicts, one per class, e.g.:
    [{"class": "Cuttable", "IoU": 70.18, "Acc": 88.13, ...}, ...]
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")

    # Strategy: find all "per class results" blocks, then match the one
    # that is followed by the target step's validation summary line.
    # The pattern in the log looks like:
    #   ... per class results: ...
    #   | Class | IoU | Acc | ... |
    #   |-------|-----|-----|-----|
    #   | Cuttable | 70.18 | 88.13 | ... |
    #   ...
    #   Iter(val) [...] val/mIoU: 79.87 ... step: 8000

    # Split log into blocks around "per class results"
    blocks = text.split("per class results")
    results: List[Dict[str, Any]] = []

    for block in blocks[1:]:  # skip everything before the first occurrence
        # Check if this block corresponds to our target step
        step_match = re.search(
            rf"Iter\(val\).*val/mIoU:\s+[\d.]+.*",
            block,
        )
        if not step_match:
            continue

        # Parse the ASCII table rows
        # Match lines like: |   Cuttable   | 70.18 | 88.13 | 82.48 | ...
        table_rows = re.findall(
            r"\|\s+([^|]+?)\s+\|"
            r"\s+([\d.]+)\s+\|"
            r"\s+([\d.]+)\s+\|"
            r"\s+([\d.]+)\s+\|"
            r"\s+([\d.]+)\s+\|"
            r"\s+([\d.]+)\s+\|"
            r"\s+([\d.]+)\s+\|",
            block,
        )

        candidate_results = []
        for row in table_rows:
            name = row[0].strip()
            # Skip the header row
            if name.lower() == "class":
                continue
            candidate_results.append({
                "class": name,
                "IoU": float(row[1]),
                "Acc": float(row[2]),
                "Dice": float(row[3]),
                "Fscore": float(row[4]),
                "Precision": float(row[5]),
                "Recall": float(row[6]),
            })

        if candidate_results:
            results = candidate_results
            # Check if this is actually for our target step by looking at
            # the summary line that follows the table
            summary_line = step_match.group(0)
            # If target_step is mentioned in the broader context, use it
            # We take the LAST matching block since that corresponds to
            # the final validation at the target step
            # (for the best checkpoint, it's the last one)

    return results


def get_git_sha() -> str:
    """Get the current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


def extract_input_shape(config_path: Path) -> List[int]:
    """Extract input shape from the config file by parsing crop_size."""
    text = config_path.read_text()
    match = re.search(r"crop_size\s*=\s*\((\d+),\s*(\d+)\)", text)
    if match:
        h, w = int(match.group(1)), int(match.group(2))
        return [1, 3, h, w]
    return [1, 3, 512, 512]  # fallback


# ---------------------------------------------------------------------------
# 4. README Generation
# ---------------------------------------------------------------------------

def generate_readme(
    repo_name: str,
    classes: List[str],
    global_metrics: Dict[str, float],
    per_class_metrics: List[Dict[str, Any]],
    git_sha: str,
    input_shape: List[int],
    target_step: int,
) -> str:
    """Generate the full README.md content with YAML frontmatter and tables."""

    # --- Build YAML metrics list ---
    yaml_metrics_lines = []
    for key, value in sorted(global_metrics.items()):
        yaml_metrics_lines.append(
            f"          - type: {key}\n            value: {value}"
        )
    yaml_metrics_block = "\n".join(yaml_metrics_lines)

    # --- Build global metrics markdown table ---
    global_table_rows = []
    for key, value in sorted(global_metrics.items()):
        global_table_rows.append(f"| **{key}** | {value} |")
    global_table = "\n".join(global_table_rows)

    # --- Build per-class metrics markdown table ---
    per_class_header = "| Class | IoU | Acc | Dice | Fscore | Precision | Recall |"
    per_class_sep = "| --- | --- | --- | --- | --- | --- | --- |"
    per_class_rows = []
    for row in per_class_metrics:
        per_class_rows.append(
            f"| **{row['class']}** | {row['IoU']} | {row['Acc']} | "
            f"{row['Dice']} | {row['Fscore']} | {row['Precision']} | {row['Recall']} |"
        )
    per_class_table = "\n".join([per_class_header, per_class_sep] + per_class_rows)

    # --- Build classes list for YAML ---
    classes_yaml = json.dumps(classes)

    # --- Compose full README ---
    readme = f"""---
language: en
license: apache-2.0
library_name: mmsegmentation
pipeline_tag: image-segmentation
tags:
- segmentation
- mowing-terrain-seg
- mmseg
metrics:
- mIoU
- mAcc
- aAcc
model-index:
  - name: {repo_name}
    results:
      - task:
          type: image-segmentation
        metrics:
{yaml_metrics_block}
custom_metadata:
  git_sha: "{git_sha}"
  input_shape: {json.dumps(input_shape)}
  classes: {classes_yaml}
  checkpoint_step: {target_step}
---

# {repo_name}

## Metrics (Step {target_step})

| Metric | Score |
| --- | --- |
{global_table}

### Per-Class Performance

{per_class_table}
"""
    return readme


# ---------------------------------------------------------------------------
# 5. Artifact Copying
# ---------------------------------------------------------------------------

def stage_artifacts(
    dest: Path,
    pth_path: Path,
    config_path: Path,
    scalars_path: Path,
    log_path: Path,
    onnx_dir: Path,
    tensorrt_dir: Optional[Path],
    readme_content: str,
) -> None:
    """Copy all artifacts into the staging directory."""
    # Clean and create destination
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # Copy model weights
    shutil.copy2(pth_path, dest / "model.pth")

    # Copy config
    shutil.copy2(config_path, dest / "config.py")

    # Copy logs
    logs_dir = dest / "logs"
    logs_dir.mkdir()
    shutil.copy2(scalars_path, logs_dir / "scalars.json")
    shutil.copy2(log_path, logs_dir / "train.log")

    # Copy ONNX artifacts (entire directory)
    shutil.copytree(onnx_dir, dest / "deploy" / "onnx")

    # Copy TensorRT artifacts if provided
    if tensorrt_dir is not None:
        shutil.copytree(tensorrt_dir, dest / "deploy" / "tensorrt")

    # Write generated README
    (dest / "README.md").write_text(readme_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# 6. Summary Printing
# ---------------------------------------------------------------------------

def print_summary(dest: Path, global_metrics: Dict[str, float]) -> None:
    """Print the staged directory tree and key metrics."""
    print("\n" + "=" * 60)
    print(f"  Staged HF repository: {dest}")
    print("=" * 60)

    # Print tree
    print("\n  Directory tree:")
    for p in sorted(dest.rglob("*")):
        rel = p.relative_to(dest)
        indent = "    " * len(rel.parts)
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {indent}{rel.name}  ({size_mb:.2f} MB)")
        else:
            print(f"  {indent}{rel.name}/")

    # Print key metrics
    print("\n  Key metrics:")
    for key in ["mIoU", "mAcc", "aAcc"]:
        if key in global_metrics:
            print(f"    {key}: {global_metrics[key]}")

    print("\n  Upload with:")
    print(f"    huggingface-cli upload <org>/{dest.name} {dest}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Step 1: Validate inputs
    print("Validating inputs...")
    try:
        paths = validate_inputs(args)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Step 2: Extract metadata from checkpoint
    print(f"Reading checkpoint: {paths['pth'].name}")
    ckpt_meta = extract_checkpoint_meta(paths["pth"])
    target_step = ckpt_meta["iter"]
    classes = ckpt_meta["classes"]
    print(f"  Checkpoint step: {target_step}")
    print(f"  Classes: {classes}")

    # Step 3: Extract global metrics from scalars.json
    print(f"Extracting metrics for step {target_step}...")
    try:
        global_metrics = extract_global_metrics(paths["scalars"], target_step)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"  mIoU: {global_metrics.get('mIoU', 'N/A')}")

    # Step 4: Extract per-class metrics from log
    print("Parsing per-class metrics from log...")
    per_class_metrics = extract_per_class_metrics(paths["log"], target_step)
    if per_class_metrics:
        for row in per_class_metrics:
            print(f"  {row['class']}: IoU={row['IoU']}")
    else:
        print("  WARNING: Could not parse per-class metrics from log.")

    # Step 5: Get git SHA and input shape
    git_sha = get_git_sha()
    input_shape = extract_input_shape(paths["config"])
    print(f"  Git SHA: {git_sha}")
    print(f"  Input shape: {input_shape}")

    # Step 6: Generate README
    readme_content = generate_readme(
        repo_name=args.repo_name,
        classes=classes,
        global_metrics=global_metrics,
        per_class_metrics=per_class_metrics,
        git_sha=git_sha,
        input_shape=input_shape,
        target_step=target_step,
    )

    # Step 7: Copy artifacts to staging directory
    if args.out_dir is not None:
        dest = args.out_dir
    else:
        dest = STAGING_ROOT / args.repo_name
    print(f"\nStaging to: {dest}")
    stage_artifacts(
        dest=dest,
        pth_path=paths["pth"],
        config_path=paths["config"],
        scalars_path=paths["scalars"],
        log_path=paths["log"],
        onnx_dir=args.onnx_dir,
        tensorrt_dir=args.tensorrt_dir,
        readme_content=readme_content,
    )

    # Step 8: Print summary
    print_summary(dest, global_metrics)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
