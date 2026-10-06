# Validation: one A549 plate (phase contrast, single scan)

Setup: 45 wells and 92 fields of A549, imaged at 10x on an Incucyte, with plate-map labels.

- **alive**: untreated, 6 wells
- **senescent** reference: hippuristanol ≥ 0.63 µM, 8 wells
- **dead** reference: antisense-oligonucleotide transfection, 16 wells
- **predicted only**: hippuristanol 0.01 to 0.16 µM, and glutamine-free medium

Segmentation used Cellpose-SAM. Features were DINOv2 ViT-B/14 on 96 px crops plus morphology, from up to 300 cells per field.

| test | per-cell balanced accuracy | per-well |
|---|---|---|
| held-out plate row (6 folds) | 0.98 | 1.00 (30 wells) |
| baseline: cell area only | 0.70 | 0.94 |

Held-out confusion (cells; rows are the label, columns the prediction):

| | alive | senescent | dead |
|---|---|---|---|
| alive | 3295 | 4 | 1 |
| senescent | 7 | 4087 | 43 |
| dead | 44 | 242 | 7905 |

Wells that were not used for training come out as graded mixtures:

| condition | % alive | % senescent | % dead | cells vs untreated |
|---|---|---|---|---|
| hippuristanol 0.16 µM | 50 | 48 | 2 | 35% |
| hippuristanol 0.039 µM | 70 | 27 | 2 | 45% |
| hippuristanol 0.0098 µM | 97 | 2 | 1 | 60% |
| glutamine-free medium | 59 | 40 | 1 | 36% |

Each dose sat in its own plate row, but held-out rows still classify correctly, and untreated wells show no row
trend in cell count.

**Limits.** These are plate-map labels, so "senescent" here means "looks like high-dose hippuristanol" and "dead"
means "looks like transfected cells". No senescence or death marker was imaged. Only one cell line has been tested,
so performance on other cell types is still unmeasured. Use marker labels and several cell lines (see the README)
to establish both.
