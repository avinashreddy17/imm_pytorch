# IMM (PyTorch) – End‑to‑End Guide

This guide documents the PyTorch reimplementation of “Unsupervised Learning of Object Landmarks through Conditional Image Generation” and how to train/evaluate it end‑to‑end on single/multi‑GPU and SLURM clusters.

Original TensorFlow reference: [tomasjakab/imm](https://github.com/tomasjakab/imm)

---

## 1) What this repository contains

- A faithful PyTorch port of the IMM model with modular components:
  - Image encoder (multi‑scale CNN)
  - Pose encoder (landmark heatmaps via soft‑argmax)
  - Renderer (upsampling decoder)
  - Optional VGG16 perceptual loss (shape‑safe HDF5 loader)
- Modern training stack:
  - PyTorch 2.x, DistributedDataParallel (DDP), TensorBoard
  - SLURM‑ready sbatch script with pre‑flight checks
  - Robust data pipeline for CelebA/MAFL and AFLW with TPS augmentation

---

## 2) Code structure

```
imm_to_pytorch/
├── imm/
│   ├── models/
│   │   ├── base_model.py          # Base utilities (EMA, conv blocks)
│   │   ├── imm_model.py           # Image/Pose encoders, Renderer, loss
│   │   └── vgg16.py               # Perceptual feature extractor (HDF5 loader)
│   ├── datasets/
│   │   ├── tps_dataset.py         # Base dataset with TPS augmentation
│   │   ├── celeba_dataset.py      # CelebA/MAFL
│   │   └── aflw_dataset.py        # AFLW
│   ├── train/
│   │   ├── trainer.py             # Single‑process Trainer (DDP‑aware)
│   │   └── distributed_trainer.py # Multi‑process orchestration helpers
│   ├── eval/
│   │   └── eval_imm.py            # Batched feature extraction utilities
│   └── utils/
│       ├── box.py                 # Dot‑dict config with safe dunder handling
│       ├── utils.py               # Gaussian maps, colorization, helpers
│       └── dataset_import.py      # Dataset factory
├── scripts/
│   ├── train.py                   # Entry: train (single or DDP)
│   ├── test.py                    # Entry: linear regressor eval (MAFL/AFLW)
│   └── slurm_train.sh             # SLURM launcher with pre‑flight checks
├── configs/
│   ├── paths/default.yaml         # Absolute data/log paths
│   └── experiments/*.yaml         # Model/training hyperparams
├── examples/
│   └── visualize_landmarks.py     # Simple qualitative viz
├── requirements.txt               # Modern dependency pins
├── CONVERSION_SUMMARY.md          # High‑level port summary
└── IMM_PYTORCH_GUIDE.md           # This guide
```

---

## 3) Environment & dependencies

Recommended (example):

```bash
conda create -n imm_pytorch python=3.9 -y
conda activate imm_pytorch
pip install -r requirements.txt
```

Key notes:
- PyTorch 2.x with CUDA matching your cluster. Install via official wheels if needed; then `pip install -r requirements.txt --no-deps`.
- `metayaml` is used for `${var}` interpolation across YAMLs.
- VGG16 HDF5 loader uses `h5py` (optional). If not present or incompatible, the code falls back to L2 loss (set `model.reconstruction_loss: l2`).

---

## 4) Datasets & paths

Update `configs/paths/default.yaml` with absolute server paths:

```yaml
logdir: /abs/path/logs
celeba_data_dir: /abs/path/celeba
aflw_data_dir: /abs/path/aflw_release-2
vgg16_path: /abs/path/vgg16.caffemodel.h5  # optional; skip or use l2
```

Expected structure (summary, see the original repo for details):
- CelebA: `Img/img_align_celeba_hq`, `Anno/list_landmarks_align_celeba.txt`, `Eval/list_eval_partition.txt`, `MAFL/training.txt`, `MAFL/testing.txt`
- AFLW: `output/` + `aflw_*_images.txt`, `aflw_*_keypoints.mat`

---

## 5) Configuration

Two combined YAMLs drive training:
1) `configs/paths/default.yaml` (absolute paths/logdir)
2) `configs/experiments/celeba-10pts.yaml` (model/training hyperparams)

Important fields:
- `training.batch`: total batch across all GPUs (per‑GPU = batch/ngpus)
- `training.optim` + `training.lr` (start, step, decay)
- `model.n_maps`: number of unsupervised landmarks (10/30/50)
- `model.reconstruction_loss`: `perceptual` or `l2`
- `model.channels_bug_fix`: if true, renderer emits extra channels consistent with TF version; RGB is sliced for loss

Steps per epoch ≈ `ceil(num_train_images / training.batch)`.
For CelebA (≈162,770 imgs) with `batch=50` and `epochs=150` → ≈3,255 steps/epoch → ≈488k steps total.

---

## 6) Training

### Local (single GPU)
```bash
python scripts/train.py \
  --configs configs/paths/default.yaml configs/experiments/celeba-10pts.yaml \
  --ngpus 1 --num-epochs 150
```

### Local (multi‑GPU, DDP)
```bash
python -m torch.distributed.run --nproc_per_node=2 scripts/train.py \
  --configs configs/paths/default.yaml configs/experiments/celeba-10pts.yaml \
  --ngpus 2 --num-epochs 150
```

### SLURM (recommended single entrypoint)

`scripts/slurm_train.sh` (high‑level):

```bash
#!/bin/bash
#SBATCH --job-name=imm-pytorch-train
#SBATCH --partition=dgx1
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_train_%j.out
#SBATCH --error=slurm_train_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

source ~/avinash/miniconda3/etc/profile.d/conda.sh
conda activate imm_pytorch

mkdir -p logs
python -V; nvidia-smi || true

# Pre‑flight checks (packages, datadir, importability) run inside the script

python -m torch.distributed.run --nproc_per_node=2 scripts/train.py \
  --configs configs/paths/default.yaml configs/experiments/celeba-10pts.yaml \
  --ngpus 2 --num-epochs 150
```

Submit:
```bash
dos2unix scripts/slurm_train.sh && chmod +x scripts/slurm_train.sh
sbatch scripts/slurm_train.sh
```

### Monitoring
- Logs, checkpoints: under `training.logdir` (from config)
- TensorBoard: `tensorboard --logdir /abs/path/logs`
- SLURM: `squeue -u $USER`, `tail -f slurm_train_<JOBID>.out`

### Interpreting logs
You may see two lines per step in multi‑GPU (one per rank). Throughput is per‑rank. Lower loss over time indicates healthy training.

---

## 7) Known issues & fixes (already handled)

- PYTHONPATH not set under SLURM → `ModuleNotFoundError: imm`.
  - Fix: `export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"` in sbatch.

- Kerberos/credential “Job credential expired” with `srun`.
  - Use `python -m torch.distributed.run` from a single allocation; or renew creds.

- `metayaml` missing.
  - `pip install metayaml` or install all from `requirements.txt`.

- DDP loss call `AttributeError: compute_loss`.
  - Loss computed on `self.model.module` when wrapped; implemented in `Trainer`.

- Device mismatch in renderer (lazy layers on CPU).
  - Newly created layers are moved to `x.device` before use.

- VGG16 loader channel/shape mismatches.
  - HDF5 loader is shape‑safe; falls back if shapes don’t match. You can also set `reconstruction_loss: l2`.

- Renderer output channel alignment with TF code.
  - When `channels_bug_fix` is true, renderer outputs `3 + len(perceptual.comp)` channels; RGB slice used for loss.

---

## 8) Evaluation (paper protocol)

Linear regression from unsupervised landmarks to labeled ones.

### MAFL (after CelebA training)
```bash
python scripts/test.py \
  --experiment-name celeba-10pts \
  --train-dataset mafl \
  --test-dataset  mafl \
  --paths-config  configs/paths/default.yaml \
  --im-size 128 --batch-size 100
```

### AFLW (after CelebA pretrain + AFLW finetune)
```bash
python scripts/test.py \
  --experiment-name aflw-10pts-finetune \
  --train-dataset aflw \
  --test-dataset  aflw \
  --paths-config  configs/paths/default.yaml \
  --im-size 128 --batch-size 100
```

The script reports normalized error (w.r.t inter‑ocular distance), matching the paper protocol.

---

## 9) After training

1. Pick a checkpoint
   - `model_final.pth` or the best by val loss.
2. Run evaluation (above).
3. Qualitative visualization (optional):
   ```bash
   python examples/visualize_landmarks.py
   ```
4. Fine‑tune on AFLW (optional, paper setup):
   ```bash
   python scripts/train.py \
     --configs configs/paths/default.yaml configs/experiments/aflw-10pts-finetune.yaml \
     --checkpoint /abs/path/to/celeba/model_epoch_X.pth \
     --ngpus 1 --num-epochs 50
   ```
5. Archive
   - Save `logdir`, configs used, `requirements.txt`, and git commit hash.

---

## 10) Extending the code

- New datasets: implement a class in `imm/datasets/` and register it in `imm/utils/dataset_import.py`.
- Losses: swap in LPIPS/SSIM or supervised heads in `IMMModel.compute_loss`.
- Speedups: add `torch.cuda.amp` in `Trainer` (autocast + GradScaler) for mixed precision.
- Experiment management: replace `metayaml` with Hydra/OmegaConf; add W&B logging.

---

## 11) Reproducibility checklist

- [ ] Absolute paths set in `configs/paths/default.yaml`
- [ ] Environment created and `pip install -r requirements.txt`
- [ ] `PYTHONPATH` includes repo root in sbatch script
- [ ] SLURM partition/QoS/GPU flags match the cluster
- [ ] Logs writable; TensorBoard points to `logdir`
- [ ] If using perceptual loss, confirm `vgg16_path` is accessible (else use `l2`)

---

## 12) FAQ

- Why do I see two lines per training step?
  - You’re training on 2 GPUs; each rank logs its own throughput/loss.

- How many steps will it run?
  - Epoch‑based; for CelebA with `batch=50` and `epochs=150`, ≈488k steps total.

- My job says “credential expired.”
  - Renew Kerberos credentials or use the single‑allocation launcher (`python -m torch.distributed.run`).

---

Last updated: 2025‑08‑13


