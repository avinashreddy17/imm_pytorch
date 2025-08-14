#!/bin/bash

# ------------------------------------------------------------
# SLURM: Evaluation job for IMM (PyTorch)
# Submit exactly like training, but this runs scripts/test.py.
# Example:
#   sbatch scripts/slurm_eval.sh \
#     --experiment-name celeba-10pts \
#     --train-dataset mafl \
#     --test-dataset  mafl \
#     --paths-config  configs/paths/default.yaml \
#     --im-size 128 --batch-size 100
#
# You can add: --iteration <N> to evaluate a specific checkpoint.
# ------------------------------------------------------------

#SBATCH --job-name=imm-pytorch-eval
#SBATCH --partition=dgx1
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_eval_%j.out
#SBATCH --error=slurm_eval_%j.err

set -euo pipefail

echo "=========================================================="
echo "[EVAL] Job ID: $SLURM_JOB_ID"
echo "[EVAL] Node : $(hostname)"
echo "[EVAL] Dir  : $SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

# Ensure repo is importable
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

# Activate conda env
source ~/avinash/miniconda3/etc/profile.d/conda.sh
conda activate imm_pytorch
export PATH="$CONDA_PREFIX/bin:$PATH"

mkdir -p logs
python -V
nvidia-smi || true

# Preflight: torch + project import + basic config presence
python - <<'PY'
import os, sys
print("\n[PRE-FLIGHT] Checking torch and project import...")
try:
    import torch
    print(f"  ok: torch {torch.__version__}, cuda_available={torch.cuda.is_available()}")
except Exception as e:
    print(f"  FAIL: torch import -> {e}")
    sys.exit(1)

try:
    from imm.models.imm_model import IMMModel  # noqa: F401
    print("  ok: imported imm.models.imm_model")
except Exception as e:
    print(f"  FAIL: imm import -> {e}")
    sys.exit(1)
print("[PRE-FLIGHT] OK\n")
PY

# Forward all CLI args to scripts/test.py
echo "[EVAL] Running: python scripts/test.py $@"
python scripts/test.py "$@"

echo "=========================================================="
echo "[EVAL] Finished at $(date)"
echo "=========================================================="
