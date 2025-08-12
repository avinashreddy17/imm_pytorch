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
echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
cd "$SLURM_SUBMIT_DIR"

# Conda env
source ~/avinash/miniconda3/etc/profile.d/conda.sh
conda activate imm_pytorch

mkdir -p logs

python -V
nvidia-smi || true

# Use torchrun to spawn 2 local processes (1 per GPU)
torchrun --nproc_per_node=2 scripts/train.py \
  --configs configs/paths/default.yaml configs/experiments/celeba-10pts.yaml \
  --ngpus 2 --num-epochs 150