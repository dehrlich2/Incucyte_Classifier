"""Labels, per-cell-line normalisation, training, validation and prediction."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import CLASSES

LOG_FEATS = ["area", "perimeter", "major_axis_length", "minor_axis_length"]
SHAPE_FEATS = ["eccentricity", "solidity", "extent", "phase_mean", "phase_std", "phase_edge"]
NORM_KEYS = ["plate", "cell_line", "elapsed_h"]


# ---------------------------------------------------------------- labels
def assign_labels(cells: pd.DataFrame, cfg: dict) -> pd.Series:
    """Training label per cell: from the plate map (`label` column of the well), or from fluorescence markers.

    Marker mode thresholds each marker at a percentile of the untreated control cells of the same plate,
    cell line and time point, so the same settings carry across cell lines and imaging runs. A cell positive
    for the death marker is `dead`; otherwise positive for the senescence marker is `senescent`; negative for
    both is `alive`."""
    lab = cfg["labels"]
    if lab["mode"] == "platemap":
        return cells["label"]
    if lab["mode"] != "marker":
        raise ValueError("labels.mode must be 'platemap' or 'marker'")
    out = pd.Series(np.nan, index=cells.index, dtype=object)
    pos = {}
    for cls in ("dead", "senescent"):
        spec = lab.get(cls)
        if not spec:
            continue
        col = f"{spec['channel']}_mean"
        if col not in cells:
            raise ValueError(f"marker channel {spec['channel']} not found; available: {[c for c in cells if c.endswith('_mean')]}")
        if "above" in spec:
            thr = pd.Series(float(spec["above"]), index=cells.index)
        else:
            q = spec.get("control_percentile", 99) / 100
            ref = cells[cells["control"]].groupby(NORM_KEYS)[col].quantile(q).rename("thr")
            thr = cells[NORM_KEYS].join(ref, on=NORM_KEYS)["thr"]
        pos[cls] = cells[col] > thr
    has = cells[[f"{lab[c]['channel']}_mean" for c in pos]].notna().all(1)
    out[has] = "alive"
    if "senescent" in pos:
        out[has & pos["senescent"]] = "senescent"
    if "dead" in pos:
        out[has & pos["dead"]] = "dead"
    return out


# ---------------------------------------------------------------- features
def feature_matrix(cells: pd.DataFrame, use_embedding=True) -> tuple[np.ndarray, list[str]]:
    """Morphology (log sizes, shape, phase texture) plus the image embedding, each robust z-scored against the
    untreated control cells of the same plate, cell line and time point. Normalising every cell line to its own
    controls is what lets one model work across cell lines with different baseline size and shape."""
    cols = [c for c in LOG_FEATS + SHAPE_FEATS if c in cells]
    if use_embedding:
        cols += [c for c in cells.columns if c.startswith("emb_")]
    X = cells[cols].to_numpy(np.float64)
    li = [cols.index(c) for c in LOG_FEATS if c in cols]
    X[:, li] = np.log(np.clip(X[:, li], 1e-3, None))
    ctrl = cells["control"].to_numpy(bool)
    if not ctrl.any():
        raise ValueError("no control wells: mark untreated wells with control=TRUE in the plate map")
    groups = cells[NORM_KEYS].astype(str).agg("|".join, axis=1).to_numpy()
    for g in np.unique(groups):
        m = groups == g
        ref = m & ctrl
        if not ref.any():
            raise ValueError(f"no control cells for {g.replace('|', ' / ')}; every plate x cell line x time point needs untreated controls")
        med = np.median(X[ref], 0)
        mad = np.median(np.abs(X[ref] - med), 0) * 1.4826 + 1e-6
        X[m] = (X[m] - med) / mad
    return np.clip(X, -50, 50), cols


def make_model(cfg, n_features):
    c = cfg["classifier"]
    k = min(c["n_pcs"], n_features)
    steps = [StandardScaler()] + ([PCA(k, random_state=0)] if n_features > k else [])
    return make_pipeline(*steps, LogisticRegression(max_iter=5000, C=c["C"], class_weight="balanced"))


# ---------------------------------------------------------------- validation
def grouped_cv(X, y, groups, cfg, wells=None):
    """Leave-one-group-out predictions; returns per-cell and per-well balanced accuracy and the predictions."""
    pred = np.empty(len(y), dtype=object)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        pred[te] = make_model(cfg, X.shape[1]).fit(X[tr], y[tr]).predict(X[te])
    ok = pred != None  # noqa: E711
    res = {"cell_balanced_accuracy": float(balanced_accuracy_score(y[ok], pred[ok].astype(str))),
           "n_cells": int(ok.sum()), "n_groups": int(len(np.unique(groups)))}
    if wells is not None:
        d = pd.DataFrame({"w": wells[ok], "y": y[ok], "p": pred[ok].astype(str)})
        wv = d.groupby("w").agg(y=("y", "first"), p=("p", lambda s: s.value_counts().idxmax()))
        res["well_balanced_accuracy"] = float(balanced_accuracy_score(wv["y"], wv["p"]))
        res["n_wells"] = int(len(wv))
    labs = [c for c in CLASSES if c in set(y)]
    res["confusion"] = {"labels": labs, "matrix": confusion_matrix(y[ok], pred[ok].astype(str), labels=labs).tolist()}
    return res, pred


def train(cells: pd.DataFrame, cfg: dict, log=print) -> dict:
    out = Path(cfg["work_dir"])
    y_all = assign_labels(cells, cfg)
    X_all, cols = feature_matrix(cells, use_embedding=True)
    m = y_all.notna().to_numpy()
    X, y, C = X_all[m], y_all[m].to_numpy().astype(str), cells[m].reset_index(drop=True)
    counts = pd.Series(y).value_counts()
    log(f"training cells per class: {counts.to_dict()}")
    if len(counts) < 2:
        raise ValueError("need at least two labelled classes to train")
    wells = (C["plate"].astype(str) + ":" + C["well"]).to_numpy()
    group_col = cfg["classifier"]["cv_group"]
    groups = (C["plate"].astype(str) + ":" + C[group_col].astype(str)).to_numpy() if group_col != "plate" else C["plate"].astype(str).to_numpy()

    report = {"labels_mode": cfg["labels"]["mode"], "class_counts": counts.to_dict(), "features": len(cols)}
    report["cv_" + group_col], pred = grouped_cv(X, y, groups, cfg, wells)
    log(f"held-out {group_col}: per-cell balanced accuracy {report['cv_' + group_col]['cell_balanced_accuracy']:.2f}")
    # how much is cell size alone? (the most common confound: treatments that kill or arrest change size and number)
    ai = [cols.index("area")]
    report["baseline_area_only"], _ = grouped_cv(X[:, ai], y, groups, cfg, wells)
    log(f"  baseline, cell area only: {report['baseline_area_only']['cell_balanced_accuracy']:.2f}")
    # does a model trained on other cell lines work on an unseen one?
    if C["cell_line"].nunique() > 1:
        report["cv_cell_line"], _ = grouped_cv(X, y, C["cell_line"].astype(str).to_numpy(), cfg, wells)
        log(f"held-out cell line: per-cell balanced accuracy {report['cv_cell_line']['cell_balanced_accuracy']:.2f}")
        per = {}
        for cl in C["cell_line"].unique():
            mm = (C["cell_line"] == cl).to_numpy()
            r, _ = grouped_cv(X, y, np.where(mm, "test", "train"), cfg)
            per[str(cl)] = r["cell_balanced_accuracy"]
        report["cv_cell_line_each"] = per
    else:
        report["note"] = "one cell line only: this model is not validated on other cell lines"

    model = make_model(cfg, X.shape[1]).fit(X, y)
    joblib.dump({"model": model, "columns": cols, "classes": list(model.classes_),
                 "cell_lines": sorted(C["cell_line"].astype(str).unique()), "labels_mode": cfg["labels"]["mode"],
                 "embedding": cfg["embedding"]}, out / "model.joblib")
    (out / "training_report.json").write_text(json.dumps(report, indent=2))
    C.assign(heldout_call=pred.astype(str), training_label=y).drop(columns=[c for c in C if c.startswith("emb_")]) \
        .to_parquet(out / "training_heldout_calls.parquet", index=False)
    return report


# ---------------------------------------------------------------- prediction
def predict(cells: pd.DataFrame, fields: pd.DataFrame, model_path: str | Path, cfg: dict) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    if bundle["embedding"].get("backend") != cfg["embedding"].get("backend") or bundle["embedding"].get("model") != cfg["embedding"].get("model"):
        raise ValueError(f"model was trained with embedding {bundle['embedding']}, data extracted with {cfg['embedding']}")
    X, cols = feature_matrix(cells, use_embedding=True)
    if cols != bundle["columns"]:
        raise ValueError("feature columns differ from the model's; extract with the same settings as training")
    prob = bundle["model"].predict_proba(X)
    classes = list(bundle["classes"])
    calls = cells[["plate", "well", "row", "column", "field", "elapsed_h", "cell_line", "treatment", "control"]].copy()
    calls["call"] = np.array(classes)[prob.argmax(1)]
    for k, c in enumerate(classes):
        calls[f"p_{c}"] = prob[:, k]
    out = Path(cfg["work_dir"])
    calls.to_parquet(out / "cell_calls.parquet", index=False)

    keys = ["plate", "well", "elapsed_h"]
    pct = pd.crosstab([calls[k] for k in keys], calls["call"], normalize="index").mul(100).round(1)
    pct = pct.reindex(columns=[c for c in CLASSES if c in classes], fill_value=0).add_prefix("pct_")
    extra = [c for c in fields.columns if c not in keys + ["field", "n_cells", "confluence", "label"]]
    meta = fields.groupby(keys)[extra].first()
    f = fields.groupby(keys).agg(cells_per_field=("n_cells", "mean"), confluence=("confluence", "mean"))
    ctrl = fields[fields["control"]].groupby(NORM_KEYS)["n_cells"].mean().rename("ctrl_cells_per_field")
    T = meta.join(pct, how="inner").join(f).reset_index()
    T = T.join(ctrl, on=NORM_KEYS)
    T["cells_vs_control_pct"] = (100 * T["cells_per_field"] / T["ctrl_cells_per_field"]).round(1)
    T["n_cells_classified"] = calls.groupby(keys).size().reindex(pd.MultiIndex.from_frame(T[keys])).to_numpy()
    T["cell_line_in_training"] = T["cell_line"].astype(str).isin(bundle["cell_lines"])
    T = T.drop(columns="ctrl_cells_per_field").sort_values(["plate", "elapsed_h", "column", "row"])
    T.to_csv(out / "percent_per_well.csv", index=False)
    return T
