# Incucyte Classifier

Per-cell **alive / senescent / dead** classification of Incucyte live-cell images, for any adherent cell line.
It reports the percentage of cells in each state for every well, alongside the cell count relative to untreated controls.

```
Incucyte export ─▶ Cellpose-SAM segmentation (phase) ─▶ per-cell morphology + DINOv2 embedding
               ─▶ normalise to each cell line's own untreated controls ─▶ classifier ─▶ % per well
```

## Quick start (one file, no config)

```bash
git clone https://github.com/dehrlich2/Incucyte_Classifier && cd Incucyte_Classifier
python -m venv .venv && source .venv/bin/activate && pip install -e ".[deep]"   # once
python classify_incucyte.py
```

The script asks for four things:

```
Folder of Incucyte images (drag it here): /Users/me/Downloads/Incucyte_Images
Cell line [cells]: A549
Untreated control wells (e.g. B2-G2): B2-G2
Trained model file to reuse (press Enter to train on this plate):
Senescence reference wells (e.g. B3-D5, Enter for none): B3-D5
Death reference wells (e.g. B8-G10, Enter for none): B8-G10
```

Results are written to `<image folder>/incucyte_classifier_results/`:

- `percent_per_well.csv`
- `percent_per_condition.png`
- `model.joblib`, which you can reuse on later plates by giving its path at the "Trained model file" prompt. Then only the untreated wells are needed.

Well lists accept ranges (`B3-D5` is rows B to D × columns 3 to 5) and commas (`B2, C2, E4`). Add `--fast` to skip
deep learning, which runs on any laptop with lower accuracy. For multi-plate, multi-cell-line or marker-based training,
use the full workflow below.

## What the three classes mean (read this first)

Phase contrast shows size, shape, spreading, granularity and detachment. It does not show senescence or apoptosis
directly. What a call means depends on where the training labels came from:

| `labels.mode` | Training label comes from | A "senescent" call means | A "dead" call means |
|---|---|---|---|
| `platemap` | Reference wells you label in the plate map (e.g. untreated = alive, a senescence inducer = senescent, a killing treatment = dead) | the cell looks like the cells in your senescence reference wells | the cell looks like the cells in your death reference wells |
| `marker` | Per-cell fluorescence in the same scan (e.g. SPiDER-βGal / p21 reporter for senescence, Cytotox or caspase-3/7 dye for death) | the phase image predicts marker-positive senescence | the phase image predicts the death marker |

`platemap` mode is quick but circular: the reference wells come out near 100% by construction, so only wells
that were *not* used for training are informative. `marker` mode trains on biology rather than treatment, and is
the way to get numbers you can call "% senescent" and "% apoptotic". Once trained, a marker-mode model runs on
phase-only scans.

Cells that have already detached are not imaged. Always read `pct_dead` together with `cells_vs_control_pct`.

## Install

```bash
git clone https://github.com/dehrlich2/Incucyte_Classifier
cd Incucyte_Classifier
python -m venv .venv && source .venv/bin/activate
pip install -e ".[deep]"        # Cellpose-SAM + DINOv2 (PyTorch; CUDA, Apple MPS or CPU)
# pip install -e .              # lightweight: threshold segmentation + morphology features only
```

## Use

1. **Export images** from Incucyte as TIFFs. The default names (`<label>_<well>_<image#>_<DDdHHhMMm>.tif`)
   are parsed automatically. Put each channel in its own sub-folder (`Phase/`, `Green/`, `Red/`), or keep
   the channel name in the file name.
2. **Write a plate map CSV** for each plate ([example](examples/platemap_example.csv)):

   | column | required | meaning |
   |---|---|---|
   | `well` | yes | `B2` or `B02` |
   | `cell_line` | yes for more than one line | used for normalisation and cross-cell-line validation |
   | `control` | yes | `TRUE` for untreated wells. Every plate × cell line × time point needs some |
   | `label` | platemap mode | `alive`, `senescent`, `dead`, or blank (predict only) |
   | `treatment`, `dose_*`, anything else | no | carried into the output tables |

3. **Copy [`configs/example.yaml`](configs/example.yaml)** and point it at your plates.
4. **Run**:

```bash
incucyte-classifier run my_config.yaml                    # index → extract → train → predict
incucyte-classifier predict new_plate.yaml --model outputs/model.joblib   # apply a trained model to new plates
```

Outputs in `work_dir`:

- `percent_per_well.csv`: `pct_alive`, `pct_senescent`, `pct_dead`, `cells_per_field`, `cells_vs_control_pct`, `confluence`, and whether the cell line was in the training data
- `percent_per_condition.png`: stacked bars per cell line × treatment × dose
- `cell_calls.parquet`: every cell's call and class probabilities
- `training_report.json`: held-out validation (see below)
- `model.joblib`: the trained model, reusable with `predict`

## Working across cell types

Cell lines differ in baseline size and shape. Before classification, every feature is robust z-scored against the
untreated control cells of the same plate, cell line and time point, so the model learns *change from that line's
own untreated state*. That is why every cell line on every plate needs control wells.

The model is only as general as its training data. To use it on many cell types, train on plates that include
those cell types, ideally with marker labels. `training_report.json` then reports:

- **`cv_row`** (or your `cv_group`): accuracy on held-out plate rows. This catches position effects.
- **`cv_cell_line`** and **`cv_cell_line_each`**: accuracy on a cell line the model never saw, which is the honest
  estimate for a new cell type.
- **`baseline_area_only`**: accuracy from cell area alone. If this is close to the full model, the classifier is
  mostly measuring size.

Predictions for a cell line that was not in training are flagged (`cell_line_in_training = False`).

## Validation so far

One A549 plate (phase only, single scan, plate-map labels): untreated (alive), hippuristanol ≥0.63 µM
(senescent reference), antisense oligonucleotide transfection (dead reference). See
[docs/validation_A549.md](docs/validation_A549.md). No other cell line has been tested yet, and no marker-labelled
data yet, so cross-cell-type performance is still unmeasured.

## Plate design tips

- Put each treatment in at least two different rows **and** columns, so treatment is not confounded with position.
- Include untreated controls for every cell line on every plate.
- For marker mode, add a live-cell senescence or death dye to a subset of wells. The model trained on them then
  classifies the phase-only wells.
- Image several time points. Death, arrest and senescence diverge over time, while a single scan only shows a snapshot.

## Tests

```bash
pip install -e ".[dev]" && pytest -q
```

The tests generate synthetic two-cell-line plates (with and without marker channels) and run the full pipeline with
the lightweight backends.
