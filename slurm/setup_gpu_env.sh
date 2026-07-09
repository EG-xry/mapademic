#!/bin/bash
# Run ON ANVIL (login node OK - pip install only, no GPU needed):
#   bash slurm/setup_gpu_env.sh
set -euo pipefail
module load anaconda/2024.02-py311
cd "$HOME/mapademic"
python3 -m venv .venv-gpu
.venv-gpu/bin/pip install --quiet --upgrade pip
# RAPIDS pip wheels (CUDA 12 bundled; node driver must be >= 525 - smoke job verifies)
.venv-gpu/bin/pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12>=24.10" "cugraph-cu12>=24.10" pyarrow
# Workaround for libcuvs-cu12 26.6.0 wheel: libcuvs.so lists libnvrtc.so.12 as a
# direct dependency but its RPATH omits nvidia/cuda_nvrtc/lib, so "import cugraph"
# fails with "libcugraph.so: cannot open shared object file". Symlink libnvrtc into
# libcuvs/lib64 (which IS on the RPATH via $ORIGIN) so no LD_LIBRARY_PATH is needed.
SP=".venv-gpu/lib/python3.11/site-packages"
if [ -f "$SP/nvidia/cuda_nvrtc/lib/libnvrtc.so.12" ] && [ -d "$SP/libcuvs/lib64" ]; then
    ln -sf ../../nvidia/cuda_nvrtc/lib/libnvrtc.so.12 "$SP/libcuvs/lib64/libnvrtc.so.12"
fi
.venv-gpu/bin/python -c "import cudf, cugraph; print('import OK:', cudf.__version__, cugraph.__version__)"
echo "GPU venv ready: $HOME/mapademic/.venv-gpu"
