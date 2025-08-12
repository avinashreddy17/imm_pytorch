#!/bin/bash
# Fine-tuning script for AFLW dataset
# Usage: bash examples/train_aflw.sh <N> <celeba_checkpoint>
# where N is the number of landmarks and celeba_checkpoint is the path to pre-trained CelebA model

N_LANDMARKS=${1:-10}
CELEBA_CHECKPOINT=${2}

if [ -z "$CELEBA_CHECKPOINT" ]; then
    echo "Error: Please provide path to CelebA checkpoint"
    echo "Usage: bash examples/train_aflw.sh <N> <celeba_checkpoint>"
    exit 1
fi

echo "Fine-tuning IMM model on AFLW dataset with $N_LANDMARKS landmarks"
echo "Using CelebA checkpoint: $CELEBA_CHECKPOINT"

python scripts/train.py \
  --configs configs/paths/default.yaml configs/experiments/aflw-${N_LANDMARKS}pts-finetune.yaml \
  --checkpoint $CELEBA_CHECKPOINT \
  --ngpus 1 \
  --num-epochs 50

echo "Fine-tuning completed for aflw-${N_LANDMARKS}pts"
