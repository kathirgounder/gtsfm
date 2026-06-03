#!/usr/bin/env bash
# Run GLOMAP against the Doppelgangers++-filtered DB and compare against the
# canonical Thanjavur GLOMAP baseline.
#
# Output: benchmark_results/GLOMAP/thanjavur/dg_filtered_0.8/full/0/{cameras,images,points3D}.bin
# Existing baseline: benchmark_results/GLOMAP/thanjavur/full/0/...
#
# Headline metric: does GLOMAP still produce a single-component, single-gopuram
# reconstruction on the filtered pair set? If yes, doppelganger filtering is
# the right intervention. If no, threshold sweep down.

set -euo pipefail

THRESH="${THRESH:-0.8}"
DB="benchmarks/thanjavur/colmap_shared_threshold_0.800.db"
IMAGES="benchmarks/thanjavur/images_532"
OUT_DIR="benchmark_results/GLOMAP/thanjavur/dg_filtered_${THRESH}/full"
GLOMAP="/Users/kathir/Desktop/glomap/build/glomap/glomap"

if [ ! -f "$DB" ]; then
    echo "ERROR: $DB not found — wait for rsync to finish"
    exit 1
fi

# ── 1. Connectivity sanity: how many images survive in the filtered pair set?
echo "── Pre-flight: pair-set connectivity ──────────────────────────────"
python3 -c "
import sqlite3
c = sqlite3.connect('$DB').cursor()
n_pairs = c.execute('SELECT COUNT(*) FROM two_view_geometries WHERE config != 0').fetchone()[0]
c.execute('''
SELECT COUNT(DISTINCT img) FROM (
  SELECT (pair_id / 2147483647) AS img FROM two_view_geometries WHERE config != 0
  UNION
  SELECT (pair_id % 2147483647) AS img FROM two_view_geometries WHERE config != 0
)
''')
n_imgs = c.fetchone()[0]
print(f'  filtered DB:  {n_pairs:,} verified pairs, {n_imgs}/529 images participate')
"

# ── 2. Fire GLOMAP (full mode for fairest comparison vs baseline)
echo ""
echo "── Running GLOMAP (full mode) on filtered DB ──────────────────────"
mkdir -p "$OUT_DIR"
echo "  out: $OUT_DIR"
echo ""

START=$(date +%s)
"$GLOMAP" mapper \
    --database_path "$DB" \
    --image_path "$IMAGES" \
    --output_path "$OUT_DIR"
END=$(date +%s)
echo ""
echo "GLOMAP finished in $((END - START))s"

# ── 3. Reconstruction summary
echo ""
echo "── Reconstruction summary ─────────────────────────────────────────"
ls -la "$OUT_DIR/" 2>/dev/null
for COMPONENT in "$OUT_DIR"/*/; do
    if [ -f "$COMPONENT/cameras.bin" ]; then
        echo "  component: $(basename $COMPONENT)"
        python3 -c "
import struct
with open('$COMPONENT/images.bin', 'rb') as f:
    n = struct.unpack('Q', f.read(8))[0]
    print(f'    images registered: {n}')
with open('$COMPONENT/points3D.bin', 'rb') as f:
    n = struct.unpack('Q', f.read(8))[0]
    print(f'    3D points: {n:,}')
"
    fi
done

echo ""
echo "── Compare against baseline ──────────────────────────────────────"
echo "  baseline: benchmark_results/GLOMAP/thanjavur/full/0/"
ls benchmark_results/GLOMAP/thanjavur/full/ 2>/dev/null
echo ""
echo "Eval (if GT exists):"
echo "  python scripts/eval_glomap_reconstruction.py \\"
echo "      --recon $OUT_DIR/0 \\"
echo "      --gt benchmarks/thanjavur/sfm_updated \\"
echo "      --output $OUT_DIR/glomap_metrics.json"
