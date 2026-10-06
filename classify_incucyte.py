#!/usr/bin/env python
"""One-file entry point: classify an Incucyte image folder into % alive / senescent / dead per well.

    python classify_incucyte.py

It asks for:
  1. the folder of exported Incucyte images (drag it from Finder into the terminal),
  2. which wells are untreated controls, e.g.  B2-G2
  3. either a trained model file to reuse, or the wells to learn from on this plate:
     senescence reference wells, e.g. B3-D5 , and death reference wells, e.g. B8-G10

Well lists accept ranges (B3-D5 is the rectangle B..D x 3..5) and commas (B2,C2,D2).
Results go to <image folder>/incucyte_classifier_results: percent_per_well.csv, percent_per_condition.png,
and model.joblib (reusable on later plates). Everything can also be passed as options, see --help."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from incucyte_classifier.cli import cmd_extract, cmd_index, cmd_predict, cmd_train  # noqa: E402
from incucyte_classifier.io import load_config  # noqa: E402

RANGE = re.compile(r"^([A-Pa-p])0?(\d{1,2})(?:\s*[-:]\s*([A-Pa-p])0?(\d{1,2}))?$")


def parse_wells(text: str) -> list[str]:
    wells = []
    for part in filter(None, (p.strip() for p in (text or "").replace(";", ",").split(","))):
        m = RANGE.match(part)
        if not m:
            raise ValueError(f"cannot read well '{part}'; use e.g. B2, B2-G2 or B3-D5")
        r0, c0 = m[1].upper(), int(m[2])
        r1, c1 = (m[3].upper(), int(m[4])) if m[3] else (r0, c0)
        for r in map(chr, range(min(ord(r0), ord(r1)), max(ord(r0), ord(r1)) + 1)):
            wells += [f"{r}{c:02d}" for c in range(min(c0, c1), max(c0, c1) + 1)]
    return wells


def clean_path(text: str) -> Path:
    """Accepts paths dragged from Finder (quoted or with backslash-escaped spaces)."""
    text = text.strip()
    try:
        text = shlex.split(text)[0] if text else text
    except ValueError:
        text = text.strip("'\"")
    return Path(text).expanduser()


def ask(prompt, default=None):
    v = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    return v or (default or "")


def deep_available() -> bool:
    try:
        import cellpose  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", help="folder of Incucyte images")
    ap.add_argument("--untreated", help="untreated control wells, e.g. B2-G2")
    ap.add_argument("--model", help="trained model.joblib to reuse (skips training)")
    ap.add_argument("--senescent", help="senescence reference wells, e.g. B3-D5")
    ap.add_argument("--dead", help="death reference wells, e.g. B8-G10")
    ap.add_argument("--cell-line", default=None, help="cell line name (default: asks, or 'cells')")
    ap.add_argument("--out", help="results folder (default: <images>/incucyte_classifier_results)")
    ap.add_argument("--fast", action="store_true", help="no deep learning: threshold segmentation + morphology only")
    a = ap.parse_args(argv)
    interactive = sys.stdin.isatty()

    images = clean_path(a.images or ask("Folder of Incucyte images (drag it here)"))
    if not images.is_dir():
        sys.exit(f"not a folder: {images}")
    cell_line = a.cell_line or (ask("Cell line", "cells") if interactive else "cells")
    untreated = parse_wells(a.untreated or ask("Untreated control wells (e.g. B2-G2)"))
    if not untreated:
        sys.exit("untreated control wells are required: every result is relative to them")
    model = a.model
    if model is None and interactive and not (a.senescent or a.dead):
        model = ask("Trained model file to reuse (press Enter to train on this plate)")
    model = clean_path(model) if model else None
    senescent = dead = []
    if not model:
        senescent = parse_wells(a.senescent if a.senescent is not None else ask("Senescence reference wells (e.g. B3-D5, Enter for none)"))
        dead = parse_wells(a.dead if a.dead is not None else ask("Death reference wells (e.g. B8-G10, Enter for none)"))
        if len([x for x in (untreated, senescent, dead) if x]) < 2:
            sys.exit("to train, give at least one reference besides untreated (senescence and/or death wells)")

    out = Path(a.out).expanduser() if a.out else images / "incucyte_classifier_results"
    out.mkdir(parents=True, exist_ok=True)
    overlap = (set(untreated) & set(senescent)) | (set(untreated) & set(dead)) | (set(senescent) & set(dead))
    if overlap:
        sys.exit(f"wells listed twice: {sorted(overlap)}")

    # plate map: every imaged well, with labels on the reference wells
    from incucyte_classifier.io import IMG_EXT, parse_path

    found = sorted({i["well"] for p in images.rglob("*") if p.suffix.lower() in IMG_EXT and (i := parse_path(p))})
    if not found:
        sys.exit(f"no Incucyte images (names like X_B2_1_00d00h00m.tif) under {images}")
    missing = sorted(set(untreated + senescent + dead) - set(found))
    if missing:
        print(f"note: no images for {missing}")
    lab = {**{w: "alive" for w in untreated}, **{w: "senescent" for w in senescent}, **{w: "dead" for w in dead}}
    role = {**{w: "untreated" for w in untreated}, **{w: "senescence reference" for w in senescent}, **{w: "death reference" for w in dead}}
    pm = pd.DataFrame({"well": found, "cell_line": cell_line,
                       "treatment": [role.get(w, f"well {w}") for w in found],
                       "label": [lab.get(w, "") for w in found],
                       "control": [w in untreated for w in found]})
    pm.to_csv(out / "platemap.csv", index=False)

    fast = a.fast or not deep_available()
    if fast and not a.fast:
        print("Cellpose/DINOv2 not installed: using the fast morphology-only mode (pip install -e '.[deep]' for full accuracy)")
    cfg_d = {"plates": [{"name": images.name, "images": str(images), "platemap": str(out / "platemap.csv")}], "work_dir": str(out)}
    if fast:
        cfg_d["segmentation"] = {"backend": "threshold"}
        cfg_d["embedding"] = {"backend": "none"}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg_d))
    cfg = load_config(out / "config.yaml")
    ns = argparse.Namespace(model=str(model) if model else None)

    print(f"\n{len(found)} wells found; untreated {len(untreated)}, senescence ref {len(senescent)}, death ref {len(dead)}")
    cmd_index(cfg, ns)
    print("\nSegmenting and measuring cells (one line per field)...")
    cmd_extract(cfg, ns)
    if not model:
        print("\nTraining on the reference wells...")
        cmd_train(cfg, ns)
    print("\nClassifying every cell...")
    cmd_predict(cfg, ns)
    print(f"\nDone. Results: {out}\n  percent_per_well.csv, percent_per_condition.png"
          + ("" if model else ", model.joblib (reuse with --model), training_report.json"))
    if sys.platform == "darwin" and interactive:
        subprocess.run(["open", str(out)], check=False)


if __name__ == "__main__":
    main()
