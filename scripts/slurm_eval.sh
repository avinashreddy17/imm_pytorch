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

# Ensure repo is importable (handle empty PYTHONPATH under 'set -u')
export PYTHONPATH="${PYTHONPATH:-}"
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

# Optional visualizations
# 1) quick synthetic visualization demo (writes landmarks_demo.png)
echo "[EVAL] Running quick visualization demo (synthetic)..."
python examples/visualize_landmarks.py || true

# 2) dataset overlays using the evaluated experiment and dataset args
# Parse a few key flags from "$@" to reuse for visualization
EXP_NAME=""; TEST_DATASET=""; TEST_SPLIT="test"; IM_SIZE="128"; ITERATION=""
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --experiment-name)
      EXP_NAME="${ARGS[$i+1]:-}"; i=$((i+1));;
    --test-dataset)
      TEST_DATASET="${ARGS[$i+1]:-}"; i=$((i+1));;
    --test-split)
      TEST_SPLIT="${ARGS[$i+1]:-}"; i=$((i+1));;
    --im-size)
      IM_SIZE="${ARGS[$i+1]:-}"; i=$((i+1));;
    --iteration)
      ITERATION="${ARGS[$i+1]:-}"; i=$((i+1));;
  esac
done

if [[ -n "$EXP_NAME" && -n "$TEST_DATASET" ]]; then
  OUT_DIR="viz_out_${TEST_DATASET}_${TEST_SPLIT}"
  echo "[EVAL] Visualizing dataset samples → $OUT_DIR (dataset=$TEST_DATASET subset=$TEST_SPLIT)"
  if [[ -n "$ITERATION" ]]; then
    python scripts/visualize_dataset.py \
      --experiment-name "$EXP_NAME" \
      --dataset "$TEST_DATASET" --subset "$TEST_SPLIT" \
      --paths-config configs/paths/default.yaml \
      --im-size "$IM_SIZE" --num-samples 16 --out-dir "$OUT_DIR" \
      --iteration "$ITERATION" || true
  else
    python scripts/visualize_dataset.py \
      --experiment-name "$EXP_NAME" \
      --dataset "$TEST_DATASET" --subset "$TEST_SPLIT" \
      --paths-config configs/paths/default.yaml \
      --im-size "$IM_SIZE" --num-samples 16 --out-dir "$OUT_DIR" || true
  fi
else
  echo "[EVAL] Skipping dataset overlays (could not parse --experiment-name/--test-dataset from args)"
fi

echo "=========================================================="
echo "[EVAL] Finished at $(date)"
echo "=========================================================="
