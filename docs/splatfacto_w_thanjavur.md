# Splatfacto-W on Thanjavur (WSL2 / RTX 2080 Ti)

Goal: produce a slick Gaussian splat of the gopuram + prakara from the
dg-filtered GTSFM output, suitable for a 30-60s flythrough video that
demonstrates the full pipeline (SfM → dense reconstruction).

Splatfacto-W is the [Wild-GS](https://arxiv.org/abs/2407.08447) variant of
Nerfstudio's `splatfacto`, designed specifically for in-the-wild crowdsourced
phototourism. It uses per-image appearance embeddings to absorb the exposure /
white-balance variance across photographers, and handles sky + transient
occluders gracefully. Produces noticeably better results than vanilla 3DGS on
this exact regime (proven on Brussels, Notre Dame, Trevi).

---

## TL;DR

1. WSL2 setup: install Nerfstudio + tinycudann + gsplat (~30 min one-time)
2. Stage data: copy our COLMAP-format reconstruction + source images to desktop
3. `ns-process-data` to wrap our COLMAP output in nerfstudio's data format
4. `ns-train splatfacto-w` for ~30-60 min training on the 2080 Ti
5. `ns-viewer` to record a flythrough, or export `.ply` for browser viewers

---

## Step 1: Nerfstudio install (one-time, on the WSL2/desktop)

Activate a fresh conda env (don't mix with the doppelgangers_pp env):

```bash
conda create -n ns python=3.11 -y
conda activate ns

# PyTorch + CUDA (your 2080 Ti supports up to 12.x driver-side; pick a
# pytorch wheel that matches what your driver supports — same logic as the
# doppelgangers setup)
pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu124

# tinycudann (custom CUDA kernels for the appearance MLP)
pip install ninja
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# gsplat (the rasterizer)
pip install git+https://github.com/nerfstudio-project/gsplat.git

# Nerfstudio itself
pip install nerfstudio

# Smoke test
ns-train --help | head -10
```

**Common gotcha:** tiny-cuda-nn build needs CUDA toolkit (nvcc), not just the
driver. If it fails, install via `conda install -c nvidia cuda-toolkit=12.4`
inside the env, then retry.

---

## Step 2: Stage data on the desktop

You need:
- `~/ns/thanjavur/colmap/` — our dg-filtered reconstruction in COLMAP txt format
  (cameras.txt, images.txt, points3D.txt)
- `~/ns/thanjavur/images/` — the 529 source images

From the Mac:

```bash
# Reconstruction (the dg-filtered output you already verified visually)
rsync -avz --progress \
  benchmark_results/GLOMAP/thanjavur/dg_filtered_0.8/full/0_txt/ \
  kathirgounder@<desktop-ip>:/home/kathirgounder/ns/thanjavur/colmap/

# Source images
rsync -avz --progress benchmarks/thanjavur/images_532/ \
  kathirgounder@<desktop-ip>:/home/kathirgounder/ns/thanjavur/images/
```

(Or use whichever GTSFM-output COLMAP folder gave the cleanest single-component
reconstruction — both GLOMAP and GTSFM versions should work; GLOMAP-version
is fewer files and was visually verified on the temple side.)

---

## Step 3: Wrap in nerfstudio's data format

Nerfstudio uses its own JSON-based dataset spec. `ns-process-data` reads our
COLMAP output and emits a `transforms.json` it understands, while leaving the
images alone:

```bash
cd ~/ns
ns-process-data images \
  --data thanjavur/images \
  --output-dir thanjavur/processed \
  --skip-colmap \
  --colmap-model-path thanjavur/colmap
```

Result: `~/ns/thanjavur/processed/transforms.json` + symlinks to the images
in `~/ns/thanjavur/processed/images/`.

---

## Step 4: Train splatfacto-w

```bash
ns-train splatfacto-w \
  --data ~/ns/thanjavur/processed \
  --output-dir ~/ns/thanjavur/runs \
  --pipeline.model.appearance-embed-dim 32 \
  --pipeline.model.use-scale-regularization True \
  --max-num-iterations 30000
```

What to expect on a 2080 Ti:
- **VRAM**: ~7-9 GB at default settings; should fit comfortably in 11 GB
- **Training time**: 30-60 min for 30K iters
- **Iteration speed**: ~10-15 it/sec early, slowing to ~5 it/sec as the splat
  count grows (typical ends with 500K-1M Gaussians)
- **Live viewer**: nerfstudio prints a URL like `http://0.0.0.0:7007` — open
  in browser to watch the splat refine in real time

If you OOM during training (the gopuram has lots of fine detail and the splat
count can balloon):
```bash
# Cap Gaussian count and/or downscale images
--pipeline.model.cull-alpha-thresh 0.1 \
--pipeline.datamanager.train-num-images-to-sample-from 200
```

If quality on the back-of-temple faces is poor (low coverage), training longer
won't help — that's a data limitation, not a hyperparameter one.

---

## Step 5: Render the flythrough

Two paths:

**(A) Interactive viewer + screen-recorded flythrough** (easiest, best for
social posts):

```bash
ns-viewer --load-config ~/ns/thanjavur/runs/<latest-run>/config.yml
```

Open the printed URL, orbit/dolly to a starting view, hit "Render" → "Add
Camera Path" → drag waypoints around the gopuram → set duration (30-60s) →
"Generate Camera Path" → download MP4.

**(B) Headless render** (scriptable, for recipe paper-figure shots):

```bash
ns-render camera-path \
  --load-config ~/ns/thanjavur/runs/<latest-run>/config.yml \
  --camera-path-filename ~/ns/thanjavur/runs/<latest-run>/camera_paths/<your-path>.json \
  --output-path ~/ns/thanjavur/flythrough.mp4
```

---

## Step 6: Web sharing (optional)

For an interactive splat anyone can orbit in a browser (way more compelling
than a video for evangelism), export the splat as `.ply`:

```bash
ns-export gaussian-splat \
  --load-config ~/ns/thanjavur/runs/<latest-run>/config.yml \
  --output-dir ~/ns/thanjavur/exports/
```

The `.ply` will be 50-300 MB. Hosting options:
- **[SuperSplat](https://playcanvas.com/superplat)** — drag-drop your .ply, get
  a hosted viewer URL, share the link. Zero engineering.
- **[gsplat.js](https://github.com/huggingface/gsplat.js) / [Babylon.js](https://doc.babylonjs.com/features/featuresDeepDive/mesh/gaussianSplatting)**
  — drop the renderer into our existing `./viz` for a self-hosted splat
  viewer. ~1 day of frontend work.
- **[Polycam](https://poly.cam/) / [Niantic Scaniverse](https://scaniverse.com/)**
  — upload, get a hosted viewer with mobile/AR support.

For the social-media-evangelism use case, **SuperSplat is the path of least
resistance**: drag-drop, copy link, post.

---

## Expected quality on Thanjavur

What will look great:
- **Gopuram surface** — well-covered by tourist photos, multiple angles, lots
  of carved detail. Splatfacto-w should render it crisp with the carved tiers
  legible.
- **Front courtyard / entrance** — popular tourist viewpoint cluster, dense
  coverage.
- **Pradakshina path around the prakara** — moderate coverage, should look ok.

What will look meh:
- **Back of gopuram** — only a handful of cameras, will be visibly blurrier.
- **Sky** — splatfacto-w models it but you'll see some haze.
- **Distant city background** — mostly garbage, that's expected.

What will look bad:
- **Tourists in foreground** — moving people get baked as ghostly floaters.
  Splatfacto-w's appearance embedding helps but doesn't fully suppress.
- **Interior shots** (if any in the dataset) — completely different lighting
  regime, will float weirdly.

For the evangelism artifact, just orbit around the gopuram + courtyard. Avoid
flying behind the back face or down close to the ground where coverage drops.

---

## Validation: try Brussels first (optional but recommended)

Brussels is the canonical phototourism splatfacto-w demo. Running it first
gets you familiar with the tooling on a known-good case before the headline
Thanjavur run. Same recipe, just swap the data dir to point at the
`F-megaloc_sift_gp_single_pt/brussels/...` reconstruction. Should produce
results visually identical to the Brussels splatfacto-w demos online.

---

## Risks / unknowns

1. **tiny-cuda-nn build failures** are the #1 install pain point for
   nerfstudio. Workarounds: install CUDA toolkit in conda env, downgrade
   gcc if needed (same fix as the curope build for doppelgangers++).
2. **Splatfacto-w may not be in current `pip install nerfstudio`** — if not,
   install nerfstudio from source (`pip install -e .` from a git clone), it's
   in the main branch as of late 2024.
3. **2080 Ti memory** is the binding constraint. If the gopuram has too many
   Gaussians, training will OOM in the second half. Cap Gaussian count via
   `cull-alpha-thresh` higher (e.g., 0.2) or downscale images.
