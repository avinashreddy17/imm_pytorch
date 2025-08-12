#!/bin/bash
# Testing script for MAFL dataset
# Usage: bash examples/test_mafl.sh <N>
# where N is the number of landmarks (e.g., 10, 30, 50)

N_LANDMARKS=${1:-10}

echo "Testing IMM model on MAFL dataset with $N_LANDMARKS landmarks"

python scripts/test.py \
  --experiment-name celeba-${N_LANDMARKS}pts \
  --train-dataset mafl \
  --test-dataset mafl \
  --test-split test

echo "Testing completed for celeba-${N_LANDMARKS}pts on MAFL"
