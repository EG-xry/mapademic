#!/bin/bash
#SBATCH -A bio260224-gpu
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:15:00
#SBATCH -J mapademic-smoke
#SBATCH -o %x-%j.log
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
cd "$HOME/mapademic"
.venv-gpu/bin/python slurm/smoke_test.py
