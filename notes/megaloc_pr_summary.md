# MegaLoc PR analysis — cross-dataset summary

Score thresholds at fixed precision floors (lower = better — means MegaLoc is more discriminative on this scene):

| Dataset | N | Verified pairs | T@99% | T@95% | T@90% | T@80% | T@70% |
|---|---:|---:|---:|---:|---:|---:|---:|
| brussels | 236 | 11,371 | 0.75 | 0.75 | 0.68 | 0.45 | 0.23 |
| sacre_coeur | 281 | 28,396 | — | — | — | 0.55 | 0.00 |
| pantheon | 321 | 38,932 | — | — | — | 0.73 | 0.00 |

## Reading the table

- `T@95%` = lowest score threshold where ≥95% of kept pairs are GLOMAP-verified.
- A LOW `T@95%` means MegaLoc scores are well-calibrated on this scene (score acts as a reliable verifiability oracle even at moderate values).
- A HIGH `T@95%` means even high MegaLoc scores have many unverifiable pairs — the scene has visual aliases / repetitive structure.
- `—` means precision never reaches the target at any threshold.

## Pairs/recall in current production configs (reference)

| Dataset | Current (nm, ms) | recall | precision | #kept | %exhaustive |
|---|---|---:|---:|---:|---:|
| brussels | (100, 0.15) | 83.3% | 67.8% | 13,983 | 50% |
| sacre_coeur | (100, 0.25) | 55.3% | 77.1% | 20,395 | 52% |
| pantheon | (100, 0.45) | 23.5% | 79.3% | 11,525 | 22% |
