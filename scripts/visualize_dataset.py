#!/usr/bin/env python
"""
Visualize predicted landmarks on real dataset samples using a trained checkpoint.

Example (MAFL, CelebA-trained model):
  python scripts/visualize_dataset.py \
    --experiment-name celeba-10pts \
    --dataset mafl --subset test \
    --paths-config configs/paths/default.yaml \
    --im-size 128 --num-samples 16 --out-dir viz_out

For AFLW:
  python scripts/visualize_dataset.py \
    --experiment-name aflw-10pts-finetune \
    --dataset aflw --subset test \
    --paths-config configs/paths/default.yaml
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch

import metayaml
from imm.utils.box import Box
from imm.utils.dataset_import import import_dataset
from imm.models.imm_model import IMMModel


def load_config(paths_cfg: str, exp_cfg: str) -> Box:
    return Box(metayaml.read([paths_cfg, exp_cfg]))


def load_model(cfg: Box, checkpoint: str, device: torch.device) -> IMMModel:
    model = IMMModel(cfg.model).to(device)
    model.eval()

    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    # Strip DDP prefix if present
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    # Load non-strict to tolerate VGG layer variants
    res = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")
    return model


def make_dataset(cfg: Box, dataset_name: str, subset: str, im_size: int):
    if dataset_name.lower() == 'mafl':
        D = import_dataset('celeba')
        dset = D(cfg.training.datadir, dataset='mafl', subset=subset,
                 order_stream=True, tps=False, image_size=[im_size, im_size], landmarks=True)
    elif dataset_name.lower() == 'celeba':
        D = import_dataset('celeba')
        dset = D(cfg.training.datadir, dataset='celeba', subset=subset,
                 order_stream=True, tps=False, image_size=[im_size, im_size], landmarks=True)
    elif dataset_name.lower() == 'aflw':
        D = import_dataset('aflw')
        dset = D(cfg.training.datadir, subset=subset,
                 order_stream=True, tps=False, image_size=[im_size, im_size], landmarks=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dset


def overlay_and_save(img_tensor: torch.Tensor,
                     pred_landmarks: np.ndarray,
                     out_path: str,
                     gt_landmarks: np.ndarray = None):
    img = (img_tensor.clamp(0, 255) / 255.0).permute(1, 2, 0).cpu().numpy()
    H, W = img.shape[:2]
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    # Predicted landmarks (red)
    plt.scatter(pred_landmarks[:, 1], pred_landmarks[:, 0], c='r', s=20, label='pred')
    # Ground truth if available (cyan)
    if gt_landmarks is not None:
        plt.scatter(gt_landmarks[:, 1], gt_landmarks[:, 0], c='c', s=12, label='gt')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize predicted landmarks on dataset samples.')
    parser.add_argument('--experiment-name', type=str, required=True, help='Experiment name (e.g., celeba-10pts)')
    parser.add_argument('--paths-config', type=str, default='configs/paths/default.yaml', help='Paths config')
    parser.add_argument('--dataset', type=str, default='mafl', help='Dataset (mafl|celeba|aflw)')
    parser.add_argument('--subset', type=str, default='test', help='Subset (train|val|test)')
    parser.add_argument('--iteration', type=int, default=None, help='Checkpoint iteration (if None, uses model_final.pth)')
    parser.add_argument('--im-size', type=int, default=128, help='Image size')
    parser.add_argument('--num-samples', type=int, default=16, help='Number of samples to visualize')
    parser.add_argument('--out-dir', type=str, default='viz_out', help='Output directory')
    args = parser.parse_args()

    exp_cfg = os.path.join('configs', 'experiments', f"{args.experiment_name}.yaml")
    cfg = load_config(args.paths_config, exp_cfg)

    # Resolve checkpoint path
    if args.iteration is not None:
        ckpt_name = f"model_epoch_{args.iteration}.pth"
    else:
        ckpt_name = "model_epoch_40.pth"
    checkpoint = os.path.join(cfg.training.logdir, ckpt_name)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(cfg, checkpoint, device)

    dset = make_dataset(cfg, args.dataset, args.subset, args.im_size)
    os.makedirs(args.out_dir, exist_ok=True)

    n = min(args.num_samples, len(dset))
    print(f"Visualizing {n} samples from {args.dataset}:{args.subset} → {args.out_dir}")

    for i in range(n):
        sample = dset[i]
        image = sample['image']          # [3,H,W]
        future = sample['future_image']  # [3,H,W]
        batch = {
            'image': image.unsqueeze(0).to(device),
            'future_image': future.unsqueeze(0).to(device)
        }
        with torch.no_grad():
            out = model(batch, training=False)
        gauss = out['gauss_yx'][0].cpu().numpy()  # [N,2] in [-1,1]
        H, W = image.shape[-2:]
        pred_px = ((gauss + 1.0) / 2.0) * np.array([H, W])

        gt_px = None
        if 'landmarks' in sample:
            gt = sample['landmarks'].numpy()  # [N,2] in pixel coords (after dataset resize)
            gt_px = gt

        out_path = os.path.join(args.out_dir, f"{args.dataset}_{args.subset}_{i:04d}.png")
        overlay_and_save(image, pred_px, out_path, gt_landmarks=gt_px)

    print("Done.")


if __name__ == '__main__':
    main()


