# IMM (PyTorch) – Complete Code Overview and Flow

This document explains the IMM PyTorch implementation end-to-end: goals, modules, data/model flow, training/evaluation logic, SLURM integration, and common issues/fixes.

Original paper implementation (TensorFlow) for reference: [tomasjakab/imm](https://github.com/tomasjakab/imm)

---

## 1) Objective (what this code does)

Learn N unsupervised landmarks by reconstructing a “future” image from a current image via a bottleneck of landmark heatmaps. The model discovers object landmarks without labels; downstream evaluation trains a linear regressor from unsupervised landmarks to supervised ones (MAFL/AFLW).

This PyTorch reimplementation preserves the original behavior while modernizing the stack (PyTorch 2.x, DDP, SLURM).

---

## 2) Repository layout and responsibilities

```
imm/
  models/
    base_model.py          # Common utilities: moving averages, conv blocks, weight decay
    imm_model.py           # Core IMM model: encoders, renderer, forward(), compute_loss()
    vgg16.py               # Perceptual-loss feature net (robust HDF5 loader)
  datasets/
    tps_dataset.py         # Base dataset with TPS warps to generate (image, future_image, mask)
    celeba_dataset.py      # CelebA/MAFL splits and 5-pt landmarks
    aflw_dataset.py        # AFLW splits, keypoints, image sizes
  train/
    trainer.py             # Trainer (single process, DDP-aware), TB logging, schedulers
    distributed_trainer.py # Multi-process orchestration (DDP + SLURM helpers)
  eval/
    eval_imm.py            # Batched forward extraction for evaluation
  utils/
    box.py                 # Dot-notation config (safe handling of dunder attrs)
    utils.py               # Gaussian maps, colorization, meshgrid, point resize
    dataset_import.py      # Factory returning dataset class by name

scripts/
  train.py                 # Entrypoint for training (single or DDP)
  test.py                  # Linear regressor evaluation (MAFL/AFLW)
  slurm_train.sh           # Robust sbatch script with preflight checks

configs/
  paths/default.yaml       # Absolute paths: data, logdir, optional vgg16_path
  experiments/*.yaml       # Model/training hyperparameters per experiment

examples/
  visualize_landmarks.py   # Quick qualitative visualization
```

---

## 3) Configuration system

- YAMLs are merged via `metayaml`: pass both `configs/paths/default.yaml` and an experiment file (e.g., `configs/experiments/celeba-10pts.yaml`). `${var}` references resolve across files (e.g., `logdir: ${logdir}/${name}`).
- The merged dict is wrapped in `utils.box.Box` for dot access. The class avoids inserting `__dict__` into mapping keys (dunder-safe) to prevent kwargs leakage.

Key fields (examples):
- `training.dset`, `training.train_dset_params`, `training.test_dset_params`
- `training.batch` (total batch across GPUs), `optim`, `lr.start_val/step/decay`, `gradclip`
- `model.n_maps` (10/30/50), `gauss_std`, `gauss_mode` (‘rot’/‘ankush’), `n_filters`, `renderer_stride`, `min_res`
- `model.reconstruction_loss` (‘perceptual’ or ‘l2’), `perceptual.comp`, `channels_bug_fix`

Steps per epoch ≈ ceil(num_train_images / training.batch). With CelebA (~162,770) and `batch=50`, ≈3,255 steps/epoch.

---

## 4) Dataset pipeline and TPS augmentation

### celeba_dataset.py
- Loads image filenames and 5-point landmarks from CelebA annotations.
- Derives MAFL train/test subsets and CelebA train/val partitions.
- Returns items used by the base TPS dataset.

### aflw_dataset.py
- Reads AFLW image lists and `aflw_*_keypoints.mat` for keypoints and original sizes.
- Splits train/val/test; returns items for base TPS dataset.

### tps_dataset.py (base)
- Reads an image (file path or in-memory tensor), resizes to target size with aspect preserved and padding.
- Creates a smooth mask (with soft edges) for masked reconstruction loss if enabled.
- Applies two TPS transforms (using `utils.tps_sampler.TPSRandomSampler`):
  1) “target” warp to create `future_image`.
  2) “source” warp applied to `future_image` to create the paired `image`.
- Outputs a dict: `{ image: C×H×W, future_image: C×H×W, mask: 1×H×W, … }`.

This mirrors the TF data pipeline: image pairs approximate motion without needing labels.

---

## 5) Model flow (imm_model.py)

### Encoders
- Image encoder: multi-scale CNN producing features at 128, 64, 32, 16 resolution.
- Pose encoder (on `future_image`):
  - CNN yields low-res heatmaps with N channels.
  - Soft-argmax landmark extraction: average over the orthogonal axis, softmax, sum with coord linspace in [-1, 1] → `(y, x)` per landmark.
  - Gaussian maps are then generated at multiple resolutions via `utils.get_gaussian_maps`.

### Grouping and resizing
- Image encoder features grouped by their spatial size. If a render size is absent, the nearest larger map is bilinearly resized.
- Pose embeddings (Gaussian maps) grouped the same way.
- Concatenate per-size features: `joint_embeddings[size] = cat(image_feats[size], pose_feats[size], dim=1)`.

### Renderer (SimpleRenderer)
- Progressive upsampling:
  - Starting from the smallest size, apply conv+BN+ReLU, then a second conv (except at the last stage), then bilinear upsample to the next size.
  - Layers are lazily created per size and moved to the current device.
  - Output channels honor `channels_bug_fix`:
    - If true, the final number of channels is `3 + len(perceptual.comp)` (compatibility with the TF workaround). The model slices the first 3 channels as RGB for the reconstruction loss.
    - If false, the final number of channels is 3 (standard RGB).

### Outputs
- `future_im_pred` (RGB predicted image used by loss)
- `gauss_yx` (B×N×2 normalized landmark coordinates)
- `pose_embeddings` (Gaussian maps at render sizes)
- `image_embeddings` (encoder features)

---

## 6) Losses and weighting

### Reconstruction loss
- Perceptual (default):
  - Concatenate `[gt, pred]` along batch.
  - `vgg16.py` extracts features at specified layers (`perceptual.comp`), working in grayscale, normalized to [0,1] and centered by a constant.
  - For each layer: compute L2 (or L1) difference; optionally apply mask; divide by an exponential moving average weight; average and sum across layers; scale by 1000 (as in TF).
- L2 fallback:
  - Pixel MSE between `future_im_pred` and `future_image`; optional mask; scaled similarly to maintain comparable magnitudes.

### Weight decay
- Sum of L2 over model weight parameters (not biases), scaled by a global `weight_decay` in `BaseModel`.

### Total loss
- `total = reconstruction + weight_decay`.

---

## 7) Perceptual backbone (vgg16.py)

- HDF5 loader (via h5py) for weights saved by the colorization net:
  - Attempts HWIO→OIHW conversion.
  - Handles conv1_1 unusual cases (3 input channels vs 1 expected by our grayscale preproc) by averaging across input channels.
  - Performs BGR→RGB channel reorder if required.
  - If shapes don’t match exactly, the layer is skipped (keeps net valid).
- Forward: receives RGB, converts to grayscale internally; normalizes and returns requested feature maps.

---

## 8) Training stack

### scripts/train.py
- Loads configs; instantiates dataset class via `utils.dataset_import.import_dataset`.
- Builds train/test datasets and DataLoaders.
- Single GPU: creates model → wraps in `Trainer` → run training.
- Multi-GPU: launched by `torch.distributed.run` or SLURM wrappers; `DistributedTrainer` builds DDP model and sampler-backed loaders; training proceeds per rank.

### Trainer (trainer.py)
- Optimizers: Adam/Adagrad/Adadelta from config.
- LR scheduler: ExponentialLR(γ=`decay`), stepped every `training.lr.step` steps (not per epoch) to mirror TF’s exponential_decay.
- Gradient clipping by global norm (`training.gradclip`).
- Logging: rank‑0 prints step/loss/throughput; TensorBoard logs scalars and images (input/future/pred, and optional pose maps).
- DDP-aware loss: uses `self.model.module.compute_loss()` when wrapped.

### DistributedTrainer (distributed_trainer.py)
- Creates per-rank device (`cuda:local_rank`), wraps model in DDP, uses DistributedSampler for loaders.
- SLURM variant derives `MASTER_ADDR/PORT/WORLD_SIZE` from SLURM env; supports `sbatch` workflows cleanly.

---

## 9) SLURM integration

- `scripts/slurm_train.sh` performs:
  - `cd $SLURM_SUBMIT_DIR` and `export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH` (ensures `imm` is importable).
  - Activates conda env, prints diagnostics, validates packages/configs via a small preflight Python snippet.
  - Launches multi-GPU with `python -m torch.distributed.run --nproc_per_node=<NGPUS> scripts/train.py ...`.
- Use a single launcher (either that or `srun`), not both.

---

## 10) Evaluation (paper protocol)

- scripts/test.py loads IMM and checkpoint, runs forward to collect `gauss_yx` and GT landmarks.
- Converts normalized coords to pixel space, fits a ridge regressor (alpha=0) on the train split (e.g., MAFL train), predicts on test, and reports inter-ocular normalized error.
- Examples:
  - MAFL after CelebA:
    ```bash
    python scripts/test.py \
      --experiment-name celeba-10pts \
      --train-dataset mafl --test-dataset mafl \
      --paths-config configs/paths/default.yaml \
      --im-size 128 --batch-size 100
    ```
  - AFLW finetune eval:
    ```bash
    python scripts/test.py \
      --experiment-name aflw-10pts-finetune \
      --train-dataset aflw --test-dataset aflw \
      --paths-config configs/paths/default.yaml \
      --im-size 128 --batch-size 100
    ```

---

## 11) After training

1) Confirm checkpoints in `training.logdir` (e.g., `model_final.pth`, `model_epoch_*.pth`).
2) Evaluate with `scripts/test.py` as above.
3) Optional: qualitative viz via `examples/visualize_landmarks.py`.
4) Optional: finetune on AFLW from CelebA checkpoint.
5) Archive logs, configs, requirements, and commit hash for reproducibility.

---

## 12) Extending the code

- New datasets: subclass under `imm/datasets/`, register in `imm/utils/dataset_import.py`.
- Losses: integrate LPIPS/SSIM/supervised heads inside `IMMModel.compute_loss()`.
- AMP: add `torch.cuda.amp.autocast` and `GradScaler` in `Trainer`.
- Experiment management: replace `metayaml` with Hydra/OmegaConf; add W&B for tracking.

---

## 13) Common issues and fixes (implemented)

- `ModuleNotFoundError: imm` on cluster: add `export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"` in sbatch.
- “Job credential expired” with `srun`: renew Kerberos, or use a single allocation with `python -m torch.distributed.run`.
- Missing `metayaml`: `pip install metayaml` or `pip install -r requirements.txt`.
- DDP loss attribute error: compute loss via the unwrapped module (`model.module`) – handled in `Trainer`.
- Renderer CPU/CUDA mismatch: lazily created layers are moved to `x.device` – handled in `SimpleRenderer`.
- VGG16 HWIO/OIHW mismatch or BGR/gray issues: shape-safe loader; fallback to l2 if desired.
- Channels alignment with TF: renderer outputs `3 + len(perceptual.comp)` when `channels_bug_fix` is true; RGB slice used for loss.

---

## 14) Reproducibility checklist

- [ ] Absolute paths set in `configs/paths/default.yaml` (data, logdir, optional vgg16_path).
- [ ] Conda env created; `pip install -r requirements.txt`.
- [ ] `PYTHONPATH` export present in sbatch.
- [ ] Correct SLURM partition/QoS/GPU flags.
- [ ] TensorBoard points to your `logdir`.
- [ ] For perceptual: confirm the VGG HDF5 path or set `reconstruction_loss: l2`.

---

## 15) References

- Original code and dataset instructions: [tomasjakab/imm](https://github.com/tomasjakab/imm)

_Last updated: 2025‑08‑13_
