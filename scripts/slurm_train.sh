#!/bin/bash
#SBATCH --job-name=imm_training
#SBATCH --output=logs/imm_training_%j.out
#SBATCH --error=logs/imm_training_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Create logs directory
mkdir -p logs

# Set environment variables
export MASTER_PORT=12355
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Get the config files from command line arguments
CONFIG_FILES="$@"

echo "Starting SLURM training with configs: $CONFIG_FILES"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Number of GPUs per node: $SLURM_GPUS_PER_NODE"

# Run training script
srun python scripts/train.py --configs $CONFIG_FILES --ngpus $SLURM_NTASKS

echo "Training completed!"
