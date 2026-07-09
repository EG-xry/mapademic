#!/bin/bash
#SBATCH -A bio260224-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -t 06:00:00
#SBATCH -J mapademic-layout
#SBATCH -o %x-%j.log
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
cd "$HOME/mapademic"
.venv-gpu/bin/python slurm/run_layout.py \
    --data "${DATA:-$SCRATCH/mapademic}" \
    --iters "${ITERS:-1000}" --chunk "${CHUNK:-100}" \
    ${LAYOUT_ARGS:-}
