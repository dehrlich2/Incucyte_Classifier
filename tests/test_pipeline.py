"""End to end on synthetic plates with the dependency-free backends (threshold segmentation, morphology only)."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from incucyte_classifier.cli import main
from incucyte_classifier.io import parse_path

from synthetic import make_plate


def test_parse_incucyte_names():
    p = parse_path(Path("exp/Phase/DE_A549_B10_6_00d00h00m.tif"))
    assert p == {"well": "B10", "row": "B", "column": 10, "field": 6, "elapsed_h": 0.0, "channel": "Phase"}
    p = parse_path(Path("VID123_C04_2_2026y10m05d_14h30m.tif"))
    assert (p["well"], p["field"], p["channel"]) == ("C04", 2, "Phase")
    assert parse_path(Path("exp/Red/P1_G7_1_01d02h30m.tif"))["channel"] == "Red"
    assert parse_path(Path("notes.tif")) is None


def _config(tmp_path, markers):
    plate = make_plate(tmp_path, "P1", markers=markers)
    cfg = {"plates": [{"name": "P1", "images": str(plate / "images"), "platemap": str(plate / "platemap.csv")}],
           "work_dir": str(tmp_path / "out"),
           "segmentation": {"backend": "threshold", "min_area_px": 15},
           "embedding": {"backend": "none", "crop_px": 32},
           "classifier": {"n_pcs": 8, "cv_group": "row"}}
    if markers:
        cfg["labels"] = {"mode": "marker", "senescent": {"channel": "Green", "control_percentile": 99},
                         "dead": {"channel": "Red", "control_percentile": 99}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.mark.parametrize("markers", [False, True])
def test_run_end_to_end(tmp_path, markers):
    main(["run", str(_config(tmp_path, markers))])
    out = tmp_path / "out"
    rep = pd.read_json(out / "training_report.json", typ="series")
    assert rep["cv_row"]["cell_balanced_accuracy"] > 0.7
    assert "cv_cell_line" in rep  # two cell lines -> held-out cell line test
    T = pd.read_csv(out / "percent_per_well.csv")
    by = T.groupby("treatment")[["pct_alive", "pct_senescent", "pct_dead"]].mean()
    assert by.loc["Untreated", "pct_alive"] > 70
    assert by.loc["DrugS", "pct_senescent"] > 50
    assert by.loc["DrugD", "pct_dead"] > 50
    assert 15 < by.loc["Unknown", "pct_senescent"] < 85  # unlabelled 50/50 wells come out mixed
    assert (out / "percent_per_condition.png").exists()


def test_missing_controls_is_an_error(tmp_path):
    cfg = _config(tmp_path, False)
    pm = Path(yaml.safe_load(cfg.read_text())["plates"][0]["platemap"])
    df = pd.read_csv(pm)
    df["control"] = False
    df.to_csv(pm, index=False)
    with pytest.raises(ValueError, match="control"):
        main(["run", str(cfg)])
