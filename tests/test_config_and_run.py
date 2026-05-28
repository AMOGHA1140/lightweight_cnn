"""Checks for the YAML config loader and run-dir management (no torch/DOTA).

Run:  python -m tests.test_config_and_run
"""

import json
import tempfile
from pathlib import Path

from common.config import Config, load_config
from common.run import append_results_row, create_run_dir


def test_base_merge():
    cfg = load_config("configs/exp/gaconv_neck.yaml")
    # override applied
    assert cfg.model.neck.smooth_conv == "gaconv", cfg.model.neck.smooth_conv
    # base keys preserved through inheritance
    assert cfg.model.backbone.name == "resnet50"
    assert cfg.model.neck.out_channels == 256
    assert cfg.data.img_size == 1024
    assert cfg.experiment.name == "gaconv_neck"
    print("base_merge: ok (override wins, base inherited)")


def test_attr_and_dict():
    cfg = load_config("configs/base.yaml")
    d = cfg.to_dict()
    assert d["model"]["neck"]["smooth_conv"] == "standard"
    assert cfg["data"]["root"] == cfg.data.root
    assert cfg.get("nope", 7) == 7
    print("attr_and_dict: ok")


def _tmp_cfg(runs_dir):
    return Config({
        "data": {"root": "./data/dota_dataset"},
        "paths": {"runs_dir": str(runs_dir)},
        "experiment": {"name": "unit_exp", "why": "testing", "method": "X / Y"},
    })


def test_create_run_dir():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _tmp_cfg(Path(tmp) / "runs")
        run_dir = create_run_dir(cfg, device="cpu")
        assert run_dir.name.startswith("unit_exp_")
        for rel in ("config.yaml", "meta.json", "NOTES.md", "checkpoints", "tb"):
            assert (run_dir / rel).exists(), rel
        meta = json.loads((run_dir / "meta.json").read_text())
        assert "git_commit" in meta and "command" in meta
        assert "testing" in (run_dir / "NOTES.md").read_text()
        # snapshot round-trips
        snap = load_config(run_dir / "config.yaml")
        assert snap.experiment.name == "unit_exp"
    print("create_run_dir: ok (files + meta + snapshot)")


def test_results_index():
    with tempfile.TemporaryDirectory() as tmp:
        append_results_row(tmp, "run_a", "0.50 mAP", "dota", "A")
        append_results_row(tmp, "run_b", "0.55 mAP", "dota", "B")
        text = (Path(tmp) / "README.md").read_text()
        assert text.count("| Run | Accuracy | Dataset | Method |") == 1
        assert "run_a" in text and "run_b" in text
        assert text.count("\n|") >= 4  # header + separator + 2 rows
    print("results_index: ok (header once, rows appended)")


def main():
    test_base_merge()
    test_attr_and_dict()
    test_create_run_dir()
    test_results_index()
    print("\nall config/run tests passed")


if __name__ == "__main__":
    main()
