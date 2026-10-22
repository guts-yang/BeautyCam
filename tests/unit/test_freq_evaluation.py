"""M4 的 RMS 协议、不可评估规则与诊断产物测试。"""
import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("evaluate_m4", Path(__file__).resolve().parents[2] / "tools/evaluate_m4.py")
_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tool)


def test_half_contrast_has_half_rms_and_epi():
    gray = np.broadcast_to(np.arange(32, dtype=np.uint8) * 4 + 40, (32, 32)).copy()
    before = np.repeat(gray[:, :, None], 3, axis=2)
    full = np.full((32, 32), 255, np.uint8)
    assert _tool.high_frequency_retention(before, before // 2, full)["value"] == pytest.approx(0.5)
    assert _tool.high_frequency_retention(before, before, full)["value"] == pytest.approx(1)
    assert _tool.edge_preservation(before, before // 2, full)["value"] == pytest.approx(0.5)


def test_empty_or_zero_energy_regions_produce_no_score():
    image = np.full((24, 24, 3), 128, np.uint8)
    empty = np.zeros((24, 24), np.uint8)
    full = np.full((24, 24), 255, np.uint8)
    assert _tool.high_frequency_retention(image, image, empty)["value"] is None
    assert _tool.high_frequency_retention(image, image, full)["reason"] == "zero_input_hf_energy"
    assert _tool.edge_preservation(image, image, empty)["value"] is None
    assert _tool.edge_preservation(image, image, full)["value"] is None
    assert _tool._aggregate([0.8, None, 1.2])["macro_mean"] == pytest.approx(1)


def test_evaluator_saves_two_layers_and_does_not_invent_review_scores(tmp_path):
    root = tmp_path / "data"
    for folder in ("images", "skin_masks", "masks"):
        (root / folder).mkdir(parents=True)
    gray = np.broadcast_to(np.arange(32, dtype=np.uint8) * 4 + 40, (32, 32)).copy()
    image = np.repeat(gray[:, :, None], 3, axis=2)
    skin = np.zeros((16, 16), np.uint8)
    skin[3:13, 3:13] = 255
    labels = np.full((16, 16), 4, np.uint8)
    labels[skin != 0] = 1
    for folder, suffix, data in (("images", ".jpg", image), ("skin_masks", ".png", skin), ("masks", ".png", labels)):
        ok, encoded = cv2.imencode(suffix, data)
        assert ok
        (root / folder / ("1" + suffix)).write_bytes(encoded.tobytes())
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sample_id,scene\n1,normal\n", encoding="utf-8")
    out = tmp_path / "out"
    summary = _tool.evaluate(root, manifest, {"highpass_gain": 1}, out)
    assert all(summary["quantitative_acceptance"].values())
    assert summary["diagnostic_invariance_samples"] == 1
    assert (out / "diagnostics/1/low_frequency.png").is_file()
    assert (out / "diagnostics/1/high_frequency.png").is_file()
    assert (out / "spectra/1.png").is_file()
    assert json.loads((out / "summary.json").read_text(encoding="utf-8")) == summary
    with (out / "visual_review.csv").open(encoding="utf-8", newline="") as fp:
        review = list(csv.DictReader(fp))[0]
    assert all(value == "" for key, value in review.items() if key.startswith("reviewer_"))
    with pytest.raises(ValueError, match="输出目录"):
        _tool.evaluate(root, manifest, {}, out)
    (root / "masks/1.png").unlink()
    with pytest.raises(FileNotFoundError):
        _tool.evaluate(root, manifest, {}, tmp_path / "missing")
    assert not (tmp_path / "missing").exists()
