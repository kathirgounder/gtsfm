# Doppelgangers++ integration plan (Thanjavur, Windows desktop / 2080 Ti)

## TL;DR

1. SSH/WSL2 into the Windows desktop, set up a conda env with PyTorch+CUDA+MASt3R deps (~30 min, one-time)
2. Copy `benchmarks/thanjavur/colmap_shared.db` and `benchmarks/thanjavur/images_532/` to the desktop
3. Run a small driver (10 lines, sketched below) that calls `doppelgangers_classifier()` + `remove_doppelgangers()` against the existing DB — **no SIFT re-extraction, no re-matching**
4. Output: `colmap_shared_threshold_0.8.db` — a copy of the original DB with ghost pairs deleted from `two_view_geometries` and `matches` tables
5. scp the filtered DB back, point the yaml configs at it, re-run trace pipeline

Estimated total runtime on 2080 Ti: **~30 min one-time setup + ~75 min classifier inference for 22K pairs**.

---

## Why this avoids re-running COLMAP

The stock `colmap_usage.py` flow is:

```
colmap_runner()              # SIFT extract + vocab-tree matching from scratch — SLOW
  └─ doppelgangers_classifier()   # the actual MASt3R classifier
       └─ remove_doppelgangers()  # writes filtered DB
```

We already paid for the SIFT + matching cost when we built `colmap_shared.db` (hours of exhaustive matching). The two functions we actually need are:

- `utils.process_database.create_image_pair_list(db_path, output_dir)` — dumps a `.npy` of (img_i, img_j) pair IDs by reading `two_view_geometries` from the existing DB
- `colmap_usage.doppelgangers_classifier(args)` — for each pair, loads the two RGB images, runs them through MASt3R + the classifier head, writes `pair_probability_list_dust3r.npy`
- `utils.process_database.remove_doppelgangers(db_path, prob_file, pair_path, threshold)` — copies the original DB to `<db>_threshold_0.8.db`, deletes rows where p > threshold from `matches` and `two_view_geometries`

So a 10-line driver bypasses `colmap_runner()` entirely.

---

## Step 1: WSL2 + CUDA prerequisites (Windows desktop)

If WSL2 isn't installed yet, in PowerShell as admin:

```powershell
wsl --install -d Ubuntu-22.04
```

After reboot + Ubuntu first-launch (set username/password), confirm CUDA is visible inside WSL2:

```bash
nvidia-smi
```

Should show the 2080 Ti and a CUDA version. **Required: NVIDIA Windows driver ≥ 535** (CUDA 12.1 in the conda env needs that or newer). If `nvidia-smi` errors or the CUDA version is old, update the Windows-side driver from the GeForce Experience app first.

Install miniconda inside WSL2:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# accept defaults, restart shell after
```

---

## Step 2: Clone Doppelgangers++ and set up the conda env

```bash
mkdir -p ~/dg && cd ~/dg
git clone --recursive https://github.com/doppelgangers25/doppelgangers-plusplus.git
cd doppelgangers-plusplus

# Conda env per their README (one-time, ~5 min)
conda create -y -n doppelgangers_pp python=3.11 cmake=3.14.0
conda activate doppelgangers_pp
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -r dust3r/requirements.txt
pip install -r dust3r/requirements_optional.txt

# Optional but recommended (~30% speedup on attention):
cd dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```

---

## Step 3: Download checkpoints (~3 GB total, one-time)

```bash
mkdir -p checkpoints

# Doppelgangers++ classifier head (trained on Doppelganger + Visym datasets)
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download(repo_id='doppelgangers25/doppelgangers_plusplus', \
                  filename='checkpoint-dg+visym.pth', local_dir='checkpoints')"

# MASt3R backbone weights
wget -P checkpoints/ \
  https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
```

---

## Step 4: Copy the Thanjavur inputs from the Mac

From the Mac:

```bash
# Replace <user>@<host> with your Windows-WSL SSH endpoint (or use rsync over SMB)
scp -r benchmarks/thanjavur/images_532 <user>@<host>:~/dg/thanjavur/
scp benchmarks/thanjavur/colmap_shared.db <user>@<host>:~/dg/thanjavur/
```

Result: `~/dg/thanjavur/images_532/` and `~/dg/thanjavur/colmap_shared.db` on the desktop.

---

## Step 5: Driver script (the 10 lines that skip COLMAP re-extraction)

Save as `~/dg/doppelgangers-plusplus/run_filter.py`:

```python
"""Run Doppelgangers++ classifier on pairs from an EXISTING COLMAP database.

Bypasses colmap_runner() (which would re-extract SIFT and re-do matching).
We already have a fully-populated colmap_shared.db, so we just need to
classify its existing pairs and write a filtered copy.
"""
import argparse, os, types
from colmap_usage import doppelgangers_classifier
from utils.process_database import create_image_pair_list, remove_doppelgangers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--database_path", required=True, help="Path to existing colmap_shared.db")
    p.add_argument("--input_image_path", required=True, help="Source images dir")
    p.add_argument("--output_path", required=True, help="Where to write probabilities + filtered DB")
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--pretrained", default="checkpoints/checkpoint-dg+visym.pth")
    args = p.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    # The classifier reads `args.database_path` + `args.input_image_path`; mimic
    # the namespace the stock script builds:
    pair_path = create_image_pair_list(args.database_path, args.output_path)
    doppelgangers_classifier(args)
    filtered_db = remove_doppelgangers(
        args.database_path,
        f"{args.output_path}/pair_probability_list_dust3r.npy",
        pair_path,
        args.threshold,
    )
    print(f"[done] filtered DB: {filtered_db}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
cd ~/dg/doppelgangers-plusplus
conda activate doppelgangers_pp

python run_filter.py \
    --database_path ~/dg/thanjavur/colmap_shared.db \
    --input_image_path ~/dg/thanjavur/images_532 \
    --output_path ~/dg/thanjavur/dg_output \
    --threshold 0.8 \
    --pretrained checkpoints/checkpoint-dg+visym.pth
```

**Expected runtime on 2080 Ti**: ~75 min for 22 K pairs (MASt3R inference at fp16, ~200 ms/pair). VRAM usage ~5 GB at 512 px input — comfortably under the 11 GB budget.

**Expected output**: `~/dg/thanjavur/colmap_shared_threshold_0.8.db` — same schema as the input, just with ghost pairs removed from `matches` and `two_view_geometries`.

---

## Step 6: Sanity-check before re-running the pipeline

Doppelgangers++ was trained on a Western-architecture-heavy mix (cathedrals, government buildings, Doppelgangers + Visym datasets). South Indian temple architecture is OOD. **Don't trust the classifier blindly** — eyeball ~30 of its highest-confidence "ghost" calls first:

```python
import numpy as np
probs = np.load("~/dg/thanjavur/dg_output/pair_probability_list_dust3r.npy")
# probs is shape (n_pairs, 2) where col 1 is P(doppelganger)
order = np.argsort(-probs[:, 1])[:30]
# Print the top-30 highest-probability ghost pairs and visually open them
```

Cross-reference with `pair_probability_list_dust3r.npy`'s pair IDs → the COLMAP `images` table → the actual filenames. If most of the top-30 are obvious front↔back symmetry confusions on the Brihadisvara, ship it. If most are legitimate matches that the classifier mis-fired on, we'd need to either lower the threshold or fine-tune.

Predicted drop count: based on 95/257 cameras being doppelganger victims in our last run, and pair-level bad-edge density typically 2-3× camera-level, I'd guess **2–5 K of the 22 K pairs (~10-20%) get filtered**. If the number comes back drastically different (>50% or <2%) it's a sign something's miscalibrated.

---

## Step 7: scp filtered DB back + wire into yaml configs

From the Mac:

```bash
scp <user>@<host>:~/dg/thanjavur/colmap_shared_threshold_0.8.db \
    benchmarks/thanjavur/colmap_shared_no_ghosts.db
```

Update both yaml configs to point at the new DB. In each file there are TWO references — `image_pairs_generator.retriever.database_path` and `cluster_optimizer.correspondence_generator.database_path`:

- [gtsfm/configs/megaloc_sift_gp_single_pt_thanjavur_peak.yaml:38](gtsfm/configs/megaloc_sift_gp_single_pt_thanjavur_peak.yaml#L38) and [:53](gtsfm/configs/megaloc_sift_gp_single_pt_thanjavur_peak.yaml#L53)
- [gtsfm/configs/megaloc_sift_gp_single_pt_thanjavur_trace.yaml:38](gtsfm/configs/megaloc_sift_gp_single_pt_thanjavur_trace.yaml#L38) and [:53](gtsfm/configs/megaloc_sift_gp_single_pt_thanjavur_trace.yaml#L53)

Change `colmap_shared.db` → `colmap_shared_no_ghosts.db` in all 4 spots.

Then re-run the trace pipeline as before. Compare the new audit's "collapsed cluster" count against the previous 95 — that's the headline metric.

---

## Risks / unknowns to flag

1. **OOD generalization**: as discussed, classifier was trained mostly on Western architecture. Sanity-check before trusting (Step 6).
2. **2080 Ti MASt3R memory**: should be fine at 512 px input but if you OOM, drop to 384 px via the dust3r config — there's a `--image_size 384` style flag in the dust3r utils.
3. **WSL2 CUDA driver**: needs Windows NVIDIA driver ≥ 535 for CUDA 12.1. Older drivers cause silent slow-fallback to CPU, which would make the run take days instead of hours. Confirm with `nvidia-smi` showing CUDA 12.1+ inside WSL2 before kicking off the long run.
4. **DB schema compatibility**: `remove_doppelgangers` assumes the standard COLMAP DB schema. Our `colmap_shared.db` was built by COLMAP exhaustive matching, so this should be safe — but worth confirming the row counts in `two_view_geometries` match what the classifier saw before/after.

---

## What this DOESN'T solve

- Doppelgangers++ filters image *pairs*, not individual images. If a camera is a doppelganger victim because all 30 of its matches are ghosts, all 30 edges drop and the camera disconnects from the LCC — which means it gets pruned by GTSFM's existing connectivity filter. Net effect: bad cameras get removed implicitly.
- The classifier doesn't help with intra-image symmetry (e.g. matching one tier of the gopuram to the tier above it within the same scene). That's a different failure mode and may need a separate intervention later.
- It also doesn't help if the BACKBONE (MASt3R) itself confuses front/back — MASt3R's training data is also Western-biased. The classifier head was tuned to compensate, but there's residual risk.
