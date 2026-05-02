# Hugging Face Repository Convention

This directory (`work_dirs/.hf/`) serves as a transparent, local staging area for formatting and reviewing model artifacts before they are uploaded to the Hugging Face (HF) Hub.

## 1. Repository Naming
All Hugging Face model repositories for this project should follow this strict naming format:

**Format:** `mts-<architecture>-<backbone>-<dataset>-<resolution>`

**Example:** `mts-segformer-mit-b0-ycor-3cls-512x512`

## 2. Tagging (Versioning)
When pushing to Hugging Face, use Git tags:
- **Production (`v1.0.0`)**: For models ready for hardware deployment.
- **Experimental (`exp-20260501`)**: For testing purposes.

## 3. Staged File Structure
Before a model is pushed, its staging folder must contain exactly these standard files to ensure transparency and reproducibility. **We do not use a separate release.json file.**

```text
work_dirs/.hf/<repo-name>/
├── README.md       # Model Card containing YAML metadata, metrics, and dataset info.
├── config.py       # The final, dumped effective MMEngine configuration file.
├── model.pth       # The best checkpoint weights extracted from the run directory.
├── deploy/         # Deployment artifacts (e.g., ONNX, TensorRT engines)
└── logs/           # Training logs (scalars.json and text log)
```

## 4. The README.md Metadata (YAML Frontmatter)
Because we are not using a separate `release.json`, all critical automation and UI metadata MUST live at the very top of the `README.md` file between `---` markers.

Here is the template:

```yaml
---
# --- Hugging Face UI Fields ---
language: en
license: apache-2.0
library_name: mmsegmentation
pipeline_tag: image-segmentation
tags:
- segmentation
- mowing-terrain-seg
- lawn-mowing
- mmseg
datasets:
- ycor
metrics:
- mIoU
- mAcc
- aAcc

# --- Custom Automation Fields (Replaces release.json) ---
model-index:
  - name: mts-segformer-mit-b0
    results:
      - task: 
          type: image-segmentation
        dataset:
          type: ycor
          name: YCOR
        metrics:
          - type: mIoU
            value: 0.85
          - type: mAcc
            value: 0.91
          - type: aAcc
            value: 0.95
          - type: IoU.background
            value: 0.98
          - type: IoU.obstacle
            value: 0.76
          - type: IoU.lawn
            value: 0.82
custom_metadata:
  git_sha: "a1b2c3d4..."           # Commit hash used for training
  input_shape: [1, 3, 512, 512]    # Required for deployment scripts
  classes: ["background", "obstacle", "lawn"]
---
```

## 5. Uploading Process
Use the official HF CLI to push the staged repository:

```bash
huggingface-cli upload <org-name>/<repo-name> ./work_dirs/.hf/<repo-name>
```
