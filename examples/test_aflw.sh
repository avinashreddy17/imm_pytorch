#!/bin/bash
# Testing script for AFLW dataset
# Usage: bash examples/test_aflw.sh <N>
# where N is the number of landmarks (e.g., 10, 30, 50)

N_LANDMARKS=${1:-10}

echo "Testing IMM model on AFLW dataset with $N_LANDMARKS landmarks"

python scripts/test.py \
  --experiment-name aflw-${N_LANDMARKS}pts-finetune \
  --train-dataset aflw \
  --test-dataset aflw \
  --test-split test

echo "Testing completed for aflw-${N_LANDMARKS}pts-finetune on AFLW"
