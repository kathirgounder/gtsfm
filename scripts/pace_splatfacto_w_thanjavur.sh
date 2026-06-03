#!/usr/bin/env bash
# One-shot setup for splatfacto-w on PACE (L40S or A100 interactive node).
# Replays every patch we hit during the local WSL2 attempt.
#
# Prerequisites on PACE before running this:
#   1. Allocated an interactive node with GPU (e.g.):
#        salloc -N 1 -n 1 --gres=gpu:L40S:1 --mem=64G -t 04:00:00
#      (substitute your actual partition/QOS conventions)
#   2. ~/thanjavur/ exists with these subdirs:
#        ~/thanjavur/colmap/sparse/0/{cameras,images,points3D}.txt   (from GTSFM)
#        ~/thanjavur/images/                                          (the 508+ source images)
#      rsync from your local box:
#        rsync -avz <local>:~/dg/thanjavur/colmap/ ~/thanjavur/colmap/
#        rsync -avz <local>:~/dg/thanjavur/images/ ~/thanjavur/images/
#
# Then on the interactive node:
#        bash pace_splatfacto_w_thanjavur.sh

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-$HOME/thanjavur}
ENV_NAME=${ENV_NAME:-ns}

# ── Step 1: conda env (one-time on PACE) ─────────────────────────────────
module load anaconda3 2>/dev/null || true   # adjust per PACE module names
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "[setup] creating conda env ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" python=3.11
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ── Step 2: install pytorch + nerfstudio + splatfacto-w ──────────────────
if ! python -c "import torch, nerfstudio, splatfactow, pycolmap" 2>/dev/null; then
    echo "[setup] installing pytorch + nerfstudio + splatfacto-w"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install nerfstudio
    pip install git+https://github.com/KevinXu02/splatfacto-w
    pip install pycolmap
    ns-install-cli
fi

# ── Step 3: replay patches to nerfw_dataparser.py + splatfactow_datamanager.py ──
SITE_PACK=$(python -c "import splatfactow, os; print(os.path.dirname(splatfactow.__file__))")
DATAPARSER="${SITE_PACK}/nerfw_dataparser.py"
DATAMGR="${SITE_PACK}/splatfactow_datamanager.py"

echo "[patch] relaxing data_name Literal type"
sed -i 's|data_name: Literal\["brandenburg", "trevi", "sacre"\] = "brandenburg"|data_name: str = "brandenburg"|' "${DATAPARSER}"

echo "[patch] disabling PINHOLE-only assertion (we convert upstream)"
python3 << EOF
import re
fp = "${DATAPARSER}"
src = open(fp).read()
new = re.sub(
    r'assert \(\s*\n?\s*cam\.model == "PINHOLE"\s*\n?\s*\),\s*"Only pinhole \(perspective\) camera model is supported at the moment"',
    '# patched: bypass — cameras converted to PINHOLE upstream\n            pass',
    src,
    flags=re.DOTALL,
)
open(fp, 'w').write(new)
EOF

echo "[patch] disabling pin_memory for cached images (PACE has plenty of RAM but stays consistent)"
sed -i 's|cache\["image"\] = cache\["image"\].pin_memory()|cache["image"] = cache["image"]  # pin_memory disabled|' "${DATAMGR}"

# ── Step 4: convert RADIAL/SIMPLE_PINHOLE → PINHOLE in cameras.txt ───────
echo "[data] converting cameras to PINHOLE model"
python3 << EOF
import shutil
src = "${DATA_ROOT}/colmap/sparse/0/cameras.txt"
shutil.copy(src, src + ".bak")
out, n = [], 0
with open(src) as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            out.append(line); continue
        parts = line.split()
        if parts[1] in ('RADIAL', 'SIMPLE_PINHOLE'):
            f_, cx, cy = parts[4], parts[5], parts[6]
            out.append(f'{parts[0]} PINHOLE {parts[2]} {parts[3]} {f_} {f_} {cx} {cy}\n')
            n += 1
        else:
            out.append(line)
open(src, 'w').writelines(out)
print(f'  converted {n} cameras → PINHOLE')
EOF

# ── Step 5: export to binary at NerfW's expected path ────────────────────
echo "[data] exporting cameras/images/points to binary at dense/sparse/"
python3 << EOF
import pycolmap, os
src = "${DATA_ROOT}/colmap/sparse/0"
dst = "${DATA_ROOT}/dense/sparse"
os.makedirs(dst, exist_ok=True)
recon = pycolmap.Reconstruction(src)
print(f'  loaded: {recon.num_cameras()} cams, {recon.num_images()} imgs, {recon.num_points3D()} pts')
print(f'  models: {set(c.model.name for c in recon.cameras.values())}')
try:
    recon.write_binary(dst)
except AttributeError:
    recon.write(dst)
print(f'  wrote binary → {dst}')
EOF

# ── Step 6: symlink images to dense/images ───────────────────────────────
echo "[data] symlinking images to dense/images"
mkdir -p "${DATA_ROOT}/dense"
[ ! -e "${DATA_ROOT}/dense/images" ] && ln -s "${DATA_ROOT}/images" "${DATA_ROOT}/dense/images"

# ── Step 7: generate brandenburg.tsv from images.bin (the keys NerfW uses) ──
echo "[data] generating brandenburg.tsv with 90/10 train/test split"
python3 << EOF
import pycolmap
recon = pycolmap.Reconstruction("${DATA_ROOT}/dense/sparse")
images = sorted(img.name for img in recon.images.values())
out_tsv = "${DATA_ROOT}/brandenburg.tsv"
with open(out_tsv, 'w') as f:
    f.write('id\tfilename\tsplit\tdataset\n')
    for i, name in enumerate(images):
        split = 'test' if i % 10 == 0 else 'train'
        f.write(f'{i}\t{name}\t{split}\tthanjavur\n')
print(f'  wrote {len(images)} entries → {out_tsv}')
EOF

# ── Step 8: cleanup any leftover Zone.Identifier files ───────────────────
find "${DATA_ROOT}/images" -name "*:Zone.Identifier" -delete 2>/dev/null || true

# ── Step 9: fire training ────────────────────────────────────────────────
echo "[run] firing splatfacto-w training (L40S has 48 GB VRAM, no OOM concerns)"
echo "[run] expect ~30 min for 30K iters, much faster than the 2080 Ti would be"
echo ""
cd "${DATA_ROOT}"
ns-train splatfacto-w --vis tensorboard --data thanjavur

# After training finishes, render the flythrough:
#   ns-viewer --load-config outputs/thanjavur/splatfacto-w/<latest>/config.yml
# Or export the splat .ply for SuperSplat:
#   ns-export gaussian-splat --load-config <config.yml> --output-dir exports/
