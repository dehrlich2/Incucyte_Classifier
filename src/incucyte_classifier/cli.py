"""Command line: incucyte-classifier {index,extract,train,predict,run} CONFIG"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .io import build_samplesheet, load_config


def _paths(cfg):
    return Path(cfg["work_dir"])


def cmd_index(cfg, args):
    sheet = build_samplesheet(cfg)
    sheet.to_parquet(_paths(cfg) / "samplesheet.parquet", index=False)
    print(f"{len(sheet)} fields: {sheet['plate'].nunique()} plate(s), {sheet['well'].nunique()} wells, "
          f"channels {[c[5:] for c in sheet if c.startswith('Path_')]}, time points {sorted(sheet['elapsed_h'].unique())}")
    print("cell lines:", sheet.groupby("cell_line")["well"].nunique().to_dict())
    print("labelled wells:", sheet.dropna(subset=["label"]).groupby("label")["well"].nunique().to_dict())
    print("control wells:", sheet[sheet["control"]].groupby(["plate", "cell_line"])["well"].nunique().to_dict())
    return sheet


def cmd_extract(cfg, args):
    from .extract import run_extract

    p = _paths(cfg) / "samplesheet.parquet"
    sheet = pd.read_parquet(p) if p.exists() else cmd_index(cfg, args)
    cells, fields = run_extract(cfg, sheet)
    print(f"{len(cells)} cells measured in {len(fields)} fields")


def cmd_train(cfg, args):
    from .model import train

    rep = train(pd.read_parquet(_paths(cfg) / "cells.parquet"), cfg)
    print(f"model and report written to {_paths(cfg)}")
    return rep


def cmd_predict(cfg, args):
    from .model import predict
    from .plots import plot_percentages

    model = args.model or _paths(cfg) / "model.joblib"
    T = predict(pd.read_parquet(_paths(cfg) / "cells.parquet"), pd.read_parquet(_paths(cfg) / "fields.parquet"), model, cfg)
    plot_percentages(T, _paths(cfg) / "percent_per_condition.png")
    print(T.to_string(index=False, max_rows=60))
    if not T["cell_line_in_training"].all():
        print("\nWARNING: some cell lines were not in the training data; their calls are untested extrapolation.")


def cmd_run(cfg, args):
    cmd_index(cfg, args)
    cmd_extract(cfg, args)
    rep = cmd_train(cfg, args)
    print(json.dumps({k: v for k, v in rep.items() if not isinstance(v, dict)}, indent=1))
    cmd_predict(cfg, args)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="incucyte-classifier", description=__doc__)
    ap.add_argument("command", choices=["index", "extract", "train", "predict", "run"])
    ap.add_argument("config", help="YAML config (see configs/example.yaml)")
    ap.add_argument("--model", help="model.joblib to predict with (default: the one in work_dir)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    {"index": cmd_index, "extract": cmd_extract, "train": cmd_train, "predict": cmd_predict, "run": cmd_run}[args.command](cfg, args)


if __name__ == "__main__":
    main()
