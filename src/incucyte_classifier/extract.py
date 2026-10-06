"""Segment cells on phase contrast, measure them and embed them.

For every field: segmentation (Cellpose-SAM, or a threshold fallback), per-cell morphology (area, shape,
phase texture), mean intensity in each fluorescence channel when present (used for marker labels), and
an image embedding of a crop around each cell (DINOv2, or the morphology alone with backend "none").
Writes <work_dir>/cells.parquet (one row per measured cell) and fields.parquet (cell count and confluence
per field, from the full segmentation, before any subsampling)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation

from .io import load_image, normalise

META = ["plate", "well", "row", "column", "field", "elapsed_h", "cell_line", "treatment", "label", "control"]
MORPH = ["area", "perimeter", "eccentricity", "solidity", "extent", "major_axis_length", "minor_axis_length",
         "phase_mean", "phase_std", "phase_edge"]


# ---------------------------------------------------------------- segmentation
def _drop_small(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Zero out labelled objects smaller than min_area (version-stable replacement for remove_small_objects)."""
    labels = np.array(labels, copy=True)
    small = np.bincount(labels.ravel()) < min_area
    small[0] = False
    labels[small[labels]] = 0
    return labels


class ThresholdSegmenter:
    """Dependency-free fallback: local-contrast threshold + watershed. Good enough for tests and sparse wells."""

    def __init__(self, min_area_px=40, **_):
        self.min_area = min_area_px

    def __call__(self, img: np.ndarray) -> np.ndarray:
        contrast = ndi.gaussian_filter(np.abs(img - ndi.gaussian_filter(img, 15)), 2)
        fg = contrast > filters.threshold_otsu(contrast)
        fg = ndi.binary_fill_holes(ndi.binary_closing(fg, morphology.disk(2)))
        fg = _drop_small(measure.label(fg), self.min_area) > 0
        dist = ndi.distance_transform_edt(fg)
        peaks = measure.label(morphology.h_maxima(ndi.gaussian_filter(dist, 2), 2))
        return segmentation.watershed(-dist, peaks, mask=fg)


class CellposeSegmenter:
    def __init__(self, model="cpsam", diameter_px=None, min_area_px=40, device=None, **_):
        import torch
        from cellpose import models, version

        self.cp4 = int(version.split(".")[0]) >= 4
        dev = torch.device(device or _device())
        kw = {"pretrained_model": model} if self.cp4 else {"model_type": model}
        self.model = models.CellposeModel(gpu=dev.type != "cpu", device=dev, **kw)
        self.diameter, self.min_area = diameter_px, min_area_px

    def __call__(self, img: np.ndarray) -> np.ndarray:
        kw = dict(diameter=self.diameter, flow_threshold=0.4, cellprob_threshold=0.0)
        if not self.cp4:
            kw["channels"] = [0, 0]
        m = self.model.eval(img.copy(), **kw)[0]
        return _drop_small(m, self.min_area)


# ---------------------------------------------------------------- embedding
class DinoEmbedder:
    def __init__(self, model="facebook/dinov2-base", input_px=224, batch_size=128, device=None, **_):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device or _device()
        self.dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.net = AutoModel.from_pretrained(model).to(self.device).eval().to(self.dtype)
        self.size, self.bs = input_px, batch_size
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def __call__(self, crops: np.ndarray) -> np.ndarray:
        torch, F = self.torch, self.torch.nn.functional
        x = torch.from_numpy(np.ascontiguousarray(crops[:, None])).to(self.device)
        out = []
        with torch.no_grad():
            for i in range(0, len(x), self.bs):
                b = F.interpolate(x[i : i + self.bs], size=self.size, mode="bilinear", align_corners=False)
                b = ((b.expand(-1, 3, -1, -1) - self.mean) / self.std).to(self.dtype)
                out.append(self.net(pixel_values=b).last_hidden_state[:, 0].float().cpu().numpy())
        return np.concatenate(out)


def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def make_segmenter(cfg):
    s = dict(cfg["segmentation"])
    return (CellposeSegmenter if s.pop("backend") == "cellpose" else ThresholdSegmenter)(**s)


def make_embedder(cfg):
    e = dict(cfg["embedding"])
    return DinoEmbedder(**e) if e.pop("backend") == "dinov2" else None


# ---------------------------------------------------------------- per field
def measure_cells(masks: np.ndarray, phase: np.ndarray, fluor: dict[str, np.ndarray]) -> pd.DataFrame:
    if masks.max() == 0:
        return pd.DataFrame(columns=["label_id", "y", "x", *MORPH])
    edge = filters.sobel(phase)
    props = measure.regionprops_table(
        masks, intensity_image=np.stack([phase, edge, *fluor.values()], -1),
        properties=["label", "centroid", "area", "perimeter", "eccentricity", "solidity", "extent",
                    "major_axis_length", "minor_axis_length", "intensity_mean", "intensity_std"])
    df = pd.DataFrame(props).rename(columns={"label": "label_id", "centroid-0": "y", "centroid-1": "x",
                                             "intensity_mean-0": "phase_mean", "intensity_std-0": "phase_std",
                                             "intensity_mean-1": "phase_edge"})
    for k, name in enumerate(fluor, start=2):
        df = df.rename(columns={f"intensity_mean-{k}": f"{name}_mean"})
    return df.drop(columns=[c for c in df.columns if c.startswith("intensity_")])


def run_extract(cfg: dict, sheet: pd.DataFrame, log=lambda s: print(s, flush=True)) -> tuple[pd.DataFrame, pd.DataFrame]:
    seg, emb = make_segmenter(cfg), make_embedder(cfg)
    phase_col = f"Path_{cfg['phase_channel']}"
    fluor_cols = [c for c in sheet.columns if c.startswith("Path_") and c != phase_col]
    half = cfg["embedding"]["crop_px"] // 2
    cap = cfg["embedding"].get("max_cells_per_field")
    meta_cols = META + [c for c in sheet.columns if c not in META and not c.startswith("Path_")]  # extra plate-map columns
    cells, fields = [], []
    for i, row in enumerate(sheet.to_dict("records")):
        phase = normalise(load_image(row[phase_col]))
        fluor = {c.removeprefix("Path_"): load_image(row[c]) for c in fluor_cols if isinstance(row.get(c), str)}
        masks = seg(phase)
        df = measure_cells(masks, phase, fluor)
        meta = {k: row.get(k) for k in meta_cols}
        fields.append({**meta, "n_cells": len(df), "confluence": float((masks > 0).mean())})
        H, W = phase.shape
        df = df[(df.y >= half) & (df.y < H - half) & (df.x >= half) & (df.x < W - half)]
        if cap and len(df) > cap:
            df = df.sample(cap, random_state=i).sort_index()
        if len(df):
            if emb is not None:
                crops = np.stack([phase[int(y) - half : int(y) + half, int(x) - half : int(x) + half] for y, x in zip(df.y, df.x)])
                f = emb(crops)
                df = pd.concat([df.reset_index(drop=True), pd.DataFrame(f.astype(np.float32), columns=[f"emb_{j}" for j in range(f.shape[1])])], axis=1)
            for k, v in meta.items():
                df[k] = v
            cells.append(df)
        log(f"  field {i + 1}/{len(sheet)} {row['plate']} {row['well']} t={row['elapsed_h']}h: {fields[-1]['n_cells']} cells")
    cells = pd.concat(cells, ignore_index=True) if cells else pd.DataFrame()
    fields = pd.DataFrame(fields)
    out = Path(cfg["work_dir"])
    cells.to_parquet(out / "cells.parquet", index=False)
    fields.to_parquet(out / "fields.parquet", index=False)
    return cells, fields
