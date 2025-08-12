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

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "Start: $(date)"
echo "=========================================================="

# 0) Go to project root and make it importable
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

# 1) Activate your conda env
source ~/avinash/miniconda3/etc/profile.d/conda.sh
conda activate imm_pytorch
export PATH="$CONDA_PREFIX/bin:$PATH"

# 2) Basic diagnostics
mkdir -p logs
echo "Python: $(which python)"; python -V
nvidia-smi || true

# 3) Preflight: verify packages, config, datadir, importability
CONFIG_PATHS="configs/paths/default.yaml"
EXPERIMENT_CFG="configs/experiments/celeba-10pts.yaml"

python - <<'PY'
import os, sys
print("\n[PRE-FLIGHT] Checking required packages...")
def must_import(name):
    try:
        __import__(name)
        print(f"  ok: {name}")
    except Exception as e:
        print(f"  FAIL: {name} -> {e}")
        sys.exit(1)

for pkg in ["torch","yaml","metayaml","numpy","h5py","sklearn","skimage","cv2","tensorboard"]:
    must_import(pkg)

import torch
print(f"  torch version: {torch.__version__}, cuda_available={torch.cuda.is_available()}")

print("\n[PRE-FLIGHT] Checking project import...")
from imm.models.imm_model import IMMModel  # noqa
print("  ok: imported imm.models.imm_model")

print("\n[PRE-FLIGHT] Checking config files and datadir...")
import yaml, metayaml
paths = "configs/paths/default.yaml"
exp   = "configs/experiments/celeba-10pts.yaml"
for f in [paths, exp]:
    if not os.path.isfile(f):
        print(f"  FAIL: missing {f}")
        sys.exit(1)
cfg = metayaml.read([paths, exp])
try:
    datadir = cfg["training"]["datadir"]
    logdir  = cfg["training"]["logdir"]
except Exception as e:
    print(f"  FAIL: reading training paths from config -> {e}")
    sys.exit(1)
print(f"  datadir: {datadir}")
print(f"  logdir : {logdir}")
if not os.path.isdir(datadir):
    print(f"  FAIL: datadir does not exist: {datadir}")
    sys.exit(1)
os.makedirs(logdir, exist_ok=True)
print("  ok: datadir exists, logdir created/exists")

print("\n[PRE-FLIGHT] All checks passed.\n")
PY

# 4) Launch training (multi-GPU via python distributed runner)
NGPUS=2
echo "[LAUNCH] Using $NGPUS GPUs"
# Optional NCCL tuning (uncomment if needed on your fabric)
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO

python -m torch.distributed.run --nproc_per_node=${NGPUS} scripts/train.py \
  --configs ${CONFIG_PATHS} ${EXPERIMENT_CFG} \
  --ngpus ${NGPUS} --num-epochs 150

echo "=========================================================="
echo "End: $(date)"
echo "=========================================================="