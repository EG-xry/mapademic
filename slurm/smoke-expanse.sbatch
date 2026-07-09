#!/bin/bash
#SBATCH -A sdp166
#SBATCH -p gpu-debug
#SBATCH --gpus=1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 10
#SBATCH --mem=92G
#SBATCH -t 00:15:00
#SBATCH -J mapademic-smoke
#SBATCH -o %x-%j.log
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
cd "$HOME/mapademic"
# Expanse py3.13 RAPIDS wheels don't resolve sibling lib packages via RPATH
SP="$HOME/mapademic/.venv-gpu/lib/python3.13/site-packages"
export LD_LIBRARY_PATH="$(ls -d "$SP"/lib*/lib64 "$SP"/rapids_logger/lib64 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
.venv-gpu/bin/python slurm/smoke_test.py
