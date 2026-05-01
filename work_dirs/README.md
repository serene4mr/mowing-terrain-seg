# Experiment & Run Convention

This project uses a **two-level layout** inside `work_dirs`:

- **Experiment** = one config / idea (model + dataset + key hyperparams)  
- **Run** = one concrete execution of that experiment (specific seed, date, etc.)

## Directory structure

```text
work_dirs/
  <exp_name>/                 # experiment
    <run_id>/                 # run = work_dir
      <timestamp>/            # auto-created log_dir by MMEngine
        log.json
        log.txt
        ...
      latest.pth              # checkpoints for this run only
      best.pth
      epoch_*.pth
      <config>.py               # dumped effective config
      ...
```

- `exp_name`: encodes model / dataset / main hyperparams  
  - e.g. `segformer_b1_cityscapes_lr1e-4_bs8_512x1024`  
- `run_id`: encodes date / seed or trial index  
  - e.g. `2026-05-01_seed0`, `run_001_seed1`  

## Resume vs new run

- **Resume**: reuse the same `run_id` directory and resume from its checkpoint.  
- **New run**: create a new `run_id` directory; optionally load weights from a previous run’s checkpoint as pretrained initialization.