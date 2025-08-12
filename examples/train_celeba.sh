#!/bin/bash
# Training script for CelebA dataset
# Usage: bash examples/train_celeba.sh <N>
# where N is the number of landmarks (e.g., 10, 30, 50)

N_LANDMARKS=${1:-10}

echo "Training IMM model on CelebA dataset with $N_LANDMARKS landmarks"

python scripts/train.py \
  --configs configs/paths/default.yaml configs/experiments/celeba-${N_LANDMARKS}pts.yaml \
  --ngpus 1 \
  --num-epochs 100

echo "Training completed for celeba-${N_LANDMARKS}pts"
