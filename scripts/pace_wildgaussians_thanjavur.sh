#!/usr/bin/env bash
# WildGaussians on PACE — one-shot setup + train.
# Replays the install we just got working in WSL, on a machine with enough RAM/VRAM
# to actually finish.
#
# Prerequisites on PACE before running this:
#   1. Interactive GPU node allocated:
#        salloc -N 1 -n 8 --gres=gpu:L40S:1 --mem=64G -t 04:00:00
#      (substitute your group's actual partition/QOS conventions)
#   2. ~/thanjavur/ exists with the COLMAP-format reconstruction:
#        ~/thanjavur/sparse/0/{cameras,images,points3D}.txt
#        ~/thanjavur/images/                                   (508 source images)
#      Transfer from your WSL desktop:
#        rsync -avz <wsl>:~/thanjavur/sparse ~/thanjavur/
#        rsync -avz <wsl>:~/thanjavur/images ~/thanjavur/
#
# Then on the interactive node:
#        bash pace_wildgaussians_thanjavur.sh

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-$HOME/thanjavur}
ENV_NAME=${ENV_NAME:-wg}
WG_REPO=${WG_REPO:-$HOME/wild-gaussians}

# ── Step 1: conda env (one-time) ─────────────────────────────────────────
module load anaconda3 2>/dev/null || true   # adjust per PACE module names
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "[setup] creating conda env ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" python=3.11
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ── Step 2: CUDA 11.8 toolkit + gcc-11 (compatible compiler) ─────────────
if ! conda list 2>/dev/null | grep -q "cuda-toolkit"; then
    echo "[setup] installing CUDA 11.8 toolkit + gcc-11 (compatible with nvcc 11.8)"
    conda install -y --override-channels -c nvidia/label/cuda-11.8.0 cuda-toolkit
    conda install -y -c conda-forge gxx=11 gcc=11
fi

# Build env vars for the CUDA submodule compiles
export CUDA_HOME=$CONDA_PREFIX
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export MAX_JOBS=4   # PACE node has plenty of cores; can go higher than WSL

conda env config vars set NERFBASELINES_BACKEND=python 2>/dev/null || true

# ── Step 3: clone wild-gaussians + nerfbaselines + submodules ────────────
if [ ! -d "${WG_REPO}" ]; then
    echo "[setup] cloning wild-gaussians repo"
    git clone https://github.com/jkulhanek/wild-gaussians.git "${WG_REPO}"
fi

cd "${WG_REPO}"

if ! python -c "import nerfbaselines" 2>/dev/null; then
    echo "[setup] installing nerfbaselines + repo requirements"
    pip install --upgrade pip
    pip install nerfbaselines>=1.2.0
    pip install -r requirements.txt
fi

if ! python -c "from simple_knn._C import distCUDA2" 2>/dev/null; then
    echo "[setup] installing simple-knn (CUDA build, NOT editable)"
    pip install --no-build-isolation ./submodules/simple-knn
fi

if ! python -c "from diff_gaussian_rasterization import _C" 2>/dev/null; then
    echo "[setup] installing diff-gaussian-rasterization (CUDA build, NOT editable)"
    pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
fi

if ! python -c "import wildgaussians" 2>/dev/null; then
    echo "[setup] installing wildgaussians package"
    pip install --no-build-isolation .
fi

echo "[verify] all imports work"
python -c "
import nerfbaselines
import wildgaussians
from simple_knn._C import distCUDA2
from diff_gaussian_rasterization import _C
print('all imports successful')
"

# ── Step 4: clean Windows ADS files if present ───────────────────────────
find "${DATA_ROOT}" -name "*:Zone.Identifier" -delete 2>/dev/null || true

# ── Step 5: fire training ────────────────────────────────────────────────
echo ""
echo "[run] firing wild-gaussians training on ${DATA_ROOT}"
echo "[run] L40S has 48 GB VRAM, node has 64+ GB RAM — no OOM concerns"
echo "[run] expect ~45-60 min for full training, with DINO feature extraction (~2 min) up front"
echo ""

nerfbaselines train --method wild-gaussians --data "${DATA_ROOT}"

# After training finishes:
#   - checkpoint goes to outputs/<run-name>/checkpoint
#   - Render a flythrough:
#       nerfbaselines render --checkpoint <output>/checkpoint --output thanjavur.mp4 --trajectory <traj>
#   - Or interactive viewer (port-forward 6006 if remote):
#       nerfbaselines viewer --checkpoint <output>/checkpoint --data ${DATA_ROOT}
