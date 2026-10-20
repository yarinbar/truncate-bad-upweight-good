#!/usr/bin/env bash
# Create the exact Python environment used for the paper runs.
# Usage: bash install.sh            (env name: tup)
#        ENV_NAME=myenv bash install.sh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-tup}"
command -v conda > /dev/null || { echo "ERROR: conda is required (https://docs.conda.io)" >&2; exit 1; }
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" -c conda-forge python=3.10.19
fi
conda activate "$ENV_NAME"
python -m pip install -r "$ROOT_DIR/requirements.txt"
# flash-attn 2.8.3: prebuilt wheel for torch 2.4 / CUDA 12 / cp310 (the one its setup.py selects; pip checks the
# sha256, which matches the paper environment's installed files); build from source otherwise.
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl#sha256=f6e2c4172c98a8225b45424dd87ab43224df878cfb4148d679e2c98a7a7dd24d"
python -m pip install "$FLASH_ATTN_WHEEL" \
  || { echo "install.sh: prebuilt flash-attn wheel failed; building flash-attn 2.8.3 from source (needs nvcc)" >&2
       python -m pip install --no-build-isolation --no-cache-dir flash-attn==2.8.3; }
python -m pip install --no-deps -e "$ROOT_DIR"
python -m pip check
python "$ROOT_DIR/scripts/check_install.py"
echo "Done. Activate with: conda activate $ENV_NAME"
