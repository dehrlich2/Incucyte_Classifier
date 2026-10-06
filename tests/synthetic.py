"""Synthetic Incucyte-style plates for tests: phase-like fields where alive cells are medium and dense,
senescent cells large and flat, dead cells small, round and bright. Optional Green (senescence) and
Red (death) marker images. Two cell lines differ in baseline size, as real cell lines do."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

SIZE = (256, 320)
PHENO = {  # radius, count per field, brightness
    "alive": (7, 60, 0.35),
    "senescent": (13, 18, 0.25),
    "dead": (4, 25, 0.8),
}
LINES = {"LineA": 1.0, "LineB": 1.3}  # radius scale per cell line


def _field(rng, mix, scale):
    yy, xx = np.mgrid[: SIZE[0], : SIZE[1]]
    img = rng.normal(0.5, 0.02, SIZE)
    green, red = rng.normal(100, 5, SIZE), rng.normal(100, 5, SIZE)
    for cls, frac in mix.items():
        r0, n, b = PHENO[cls]
        for _ in range(int(n * frac)):
            r = r0 * scale * rng.uniform(0.85, 1.15)
            cy, cx = rng.uniform(r, SIZE[0] - r), rng.uniform(r, SIZE[1] - r)
            d = np.hypot((yy - cy) / rng.uniform(0.8, 1.2), xx - cx)
            ring = np.exp(-((d - r) ** 2) / 2)
            img += b * np.exp(-(d / r) ** 4) * (0.6 if cls != "dead" else 1.0) + 0.3 * ring
            if cls == "senescent":
                green[d < r] += 300
            if cls == "dead":
                red[d < r] += 300
    return np.clip(img * 120, 0, 255).astype(np.uint8), green.astype(np.uint16), red.astype(np.uint16)


def make_plate(root: Path, name: str, markers=False, seed=0) -> Path:
    """Writes images/ and platemap.csv under root/name; returns that folder."""
    rng = np.random.default_rng(seed)
    d = root / name
    img_dir = d / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    layout = []
    for line, scale in LINES.items():
        rows = "BCD" if line == "LineA" else "EFG"
        for r in rows:
            layout += [(f"{r}2", line, "Untreated", "alive", True, {"alive": 1.0}),
                       (f"{r}3", line, "DrugS", "senescent", False, {"senescent": 0.9, "alive": 0.1}),
                       (f"{r}4", line, "DrugD", "dead", False, {"dead": 0.9, "alive": 0.1}),
                       (f"{r}5", line, "Unknown", "", False, {"alive": 0.5, "senescent": 0.5})]
    for well, line, treat, label, ctrl, mix in layout:
        for field in (1, 2):
            ph, g, rd = _field(rng, mix, LINES[line])
            stem = f"{name}_{well}_{field}_00d12h00m"
            if markers:
                for ch, im in (("Phase", ph), ("Green", g), ("Red", rd)):
                    (img_dir / ch).mkdir(exist_ok=True)
                    tifffile.imwrite(img_dir / ch / f"{stem}.tif", im)
            else:
                tifffile.imwrite(img_dir / f"{stem}.tif", ph)
    pd.DataFrame([dict(well=w, cell_line=l, treatment=t, label=lab, control=c) for w, l, t, lab, c, _ in layout]) \
        .to_csv(d / "platemap.csv", index=False)
    return d
