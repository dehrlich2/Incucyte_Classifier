"""Configuration, Incucyte file-name parsing, plate maps and image loading."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import yaml
from PIL import Image

from . import CLASSES

IMG_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
# Incucyte exports: <label>_<well>_<image#>_<DD>d<HH>h<MM>m.tif (elapsed) or
# <label>_<well>_<image#>_<YYYY>y<MM>m<DD>d_<HH>h<MM>m.tif (timestamp). Channel is a folder or file-name token.
WELL_RE = re.compile(r"(?:^|[_\-\s])(?P<row>[A-P])0?(?P<col>\d{1,2})_(?P<field>\d{1,3})(?=_|\.|$)")
DATE_RE = re.compile(r"(?P<Y>\d{4})y(?P<M>\d{2})m(?P<D>\d{2})d_(?P<h>\d{2})h(?P<m>\d{2})m")
ELAPSED_RE = re.compile(r"(?<!\d)(?P<d>\d{1,3})d(?P<h>\d{2})h(?P<m>\d{2})m")
CHANNEL_WORDS = ("phase", "green", "red", "nir", "orange", "blue", "brightfield")

DEFAULTS = {
    "phase_channel": "Phase",
    "segmentation": {"backend": "cellpose", "model": "cpsam", "diameter_px": None, "min_area_px": 40},
    "embedding": {"backend": "dinov2", "model": "facebook/dinov2-base", "crop_px": 96, "input_px": 224,
                  "batch_size": 128, "max_cells_per_field": 300},
    "labels": {"mode": "platemap"},
    "classifier": {"n_pcs": 50, "C": 0.1, "cv_group": "row"},
    "timepoints": "all",
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as fh:
        cfg = _merge(DEFAULTS, yaml.safe_load(fh) or {})
    base = path.parent
    for p in cfg["plates"]:
        p["images"] = str((base / p["images"]).resolve())
        p["platemap"] = str((base / p["platemap"]).resolve())
    cfg["work_dir"] = str((base / cfg.get("work_dir", "outputs")).resolve())
    Path(cfg["work_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def parse_path(p: Path) -> dict | None:
    """Well, row, column, field, elapsed hours and channel from an Incucyte export path; None if no well."""
    m = WELL_RE.search(p.name)
    if not m:
        return None
    out = {"well": f"{m['row']}{int(m['col']):02d}", "row": m["row"], "column": int(m["col"]), "field": int(m["field"])}
    if d := DATE_RE.search(p.stem):
        out["timestamp"] = pd.Timestamp(int(d["Y"]), int(d["M"]), int(d["D"]), int(d["h"]), int(d["m"]))
    elif e := ELAPSED_RE.search(p.stem):
        out["elapsed_h"] = int(e["d"]) * 24 + int(e["h"]) + int(e["m"]) / 60
    stem_tokens = set(re.split(r"[_\-\s.]", p.stem.lower()))
    folders = [s.lower() for s in p.parts[-3:-1]]
    out["channel"] = next((w.capitalize() for w in CHANNEL_WORDS
                           if w in stem_tokens or any(w in f for f in folders)), "Phase")
    return out


def read_platemap(path: str | Path) -> pd.DataFrame:
    """CSV with at least `well`; optional cell_line, treatment, dose, label (alive/senescent/dead or blank)."""
    pm = pd.read_csv(path, dtype={"well": str})
    pm["well"] = pm["well"].str.strip().str.upper().map(lambda w: f"{w[0]}{int(w[1:]):02d}")
    for col, default in [("cell_line", "unknown"), ("treatment", ""), ("label", np.nan), ("control", False)]:
        if col not in pm:
            pm[col] = default
    pm["label"] = pm["label"].where(pm["label"].isna(), pm["label"].astype(str).str.strip().str.lower())
    pm.loc[pm["label"].isin(["", "nan", "none"]), "label"] = np.nan
    bad = set(pm["label"].dropna()) - set(CLASSES)
    if bad:
        raise ValueError(f"{path}: unknown label(s) {sorted(bad)}; use {CLASSES} or leave blank")
    pm["control"] = pm["control"].astype(str).str.lower().isin(["true", "1", "yes"])
    return pm


def build_samplesheet(cfg: dict) -> pd.DataFrame:
    """One row per plate x well x field x time point, with a Path_<channel> column per channel."""
    rows = []
    for plate in cfg["plates"]:
        for p in sorted(Path(plate["images"]).rglob("*")):
            if p.suffix.lower() in IMG_EXT and (info := parse_path(p)):
                rows.append({"plate": plate["name"], **info, "path": str(p)})
    if not rows:
        raise FileNotFoundError("no Incucyte images with a <well>_<field> token found under the configured folders")
    df = pd.DataFrame(rows)
    if "elapsed_h" not in df:
        df["elapsed_h"] = np.nan
    if "timestamp" in df:
        t0 = df.groupby("plate")["timestamp"].transform("min")
        df["elapsed_h"] = df["elapsed_h"].fillna((df["timestamp"] - t0).dt.total_seconds() / 3600)
    df["elapsed_h"] = df["elapsed_h"].fillna(0.0).round(2)
    keys = ["plate", "well", "row", "column", "field", "elapsed_h"]
    sheet = df.pivot_table(index=keys, columns="channel", values="path", aggfunc="first")
    sheet.columns = [f"Path_{c}" for c in sheet.columns]
    sheet = sheet.reset_index()
    maps = pd.concat([read_platemap(p["platemap"]).assign(plate=p["name"]) for p in cfg["plates"]])
    sheet = sheet.merge(maps, on=["plate", "well"], how="inner")
    if f"Path_{cfg['phase_channel']}" not in sheet:
        raise ValueError(f"no {cfg['phase_channel']} images found; set phase_channel in the config")
    tp = cfg["timepoints"]
    if tp == "last":
        sheet = sheet[sheet["elapsed_h"] == sheet.groupby("plate")["elapsed_h"].transform("max")]
    elif isinstance(tp, list):
        sheet = sheet[sheet["elapsed_h"].isin(tp)]
    return sheet.reset_index(drop=True)


def load_image(path: str) -> np.ndarray:
    im = tifffile.imread(path) if path.lower().endswith((".tif", ".tiff")) else np.asarray(Image.open(path))
    im = np.asarray(im, dtype=np.float32)
    if im.ndim == 3:  # RGB export, or a stack of pages: use the first plane
        im = im.mean(-1) if im.shape[-1] in (3, 4) else im[0]
    return im


def normalise(im: np.ndarray, lo_pct=1.0, hi_pct=99.8) -> np.ndarray:
    lo, hi = np.percentile(im, [lo_pct, hi_pct])
    return np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
