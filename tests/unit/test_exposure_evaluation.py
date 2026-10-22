"""M2 统计口径、真实欠曝筛选与直方图材料验证。"""
import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("evaluate_m2", Path(__file__).resolve().parents[2] / "tools/evaluate_m2.py")
_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tool)


def test_statistics_use_same_gt_and_strict_thresholds():
    image = np.full((4, 4, 3), 60, np.uint8)
    image[:, 2:] = 255
    after = image.copy()
    after[:, :2] = 90
    skin = np.zeros((4, 4), np.uint8)
    skin[:, :2] = 255
    result = _tool.exposure_statistics(image, after, skin)
    assert result["mean_before_y"] == 60
    assert result["mean_after_y"] == 90
    assert result["relative_gain"] == 0.5
    assert result["underexposed"] and result["overexposure_after_ratio"] == 0
    for value, over, under in ((89, 0, True), (90, 0, False), (250, 0, False), (251, 1, False)):
        uniform = np.full_like(image, value)
        result = _tool.exposure_statistics(uniform, uniform, skin)
        assert result["underexposed"] is under
        assert result["overexposure_after_ratio"] == over
    empty = _tool.exposure_statistics(image, after, np.zeros_like(skin))
    assert not empty["valid"] and empty["relative_gain"] is None
    assert empty["reason"] == "empty_skin_gt"


def _dataset(tmp_path, value):
    root = tmp_path / "data"
    for folder in ("images", "skin_masks"):
        (root / folder).mkdir(parents=True)
    image = np.full((16, 24, 3), value, np.uint8)
    skin = np.zeros((8, 12), np.uint8)
    skin[2:6, 3:9] = 255
    for folder, suffix, data in (("images", ".jpg", image), ("skin_masks", ".png", skin)):
        ok, encoded = cv2.imencode(suffix, data)
        assert ok
        (root / folder / ("1" + suffix)).write_bytes(encoded.tobytes())
    manifest = tmp_path / "manifest.csv"
    # scene 故意写 normal；实际欠曝筛选只按 GT 平均 Y。
    manifest.write_text("sample_id,scene\n1,normal\n", encoding="utf-8")
    return root, manifest


def test_evaluation_resizes_gt_saves_histograms_and_keeps_clahe_pending(tmp_path):
    root, manifest = _dataset(tmp_path, 60)
    out = tmp_path / "out"
    summary = _tool.evaluate(root, manifest, {}, out)
    assert all(summary["quantitative_acceptance"].values())
    assert summary["underexposed_sample_ids"] == ["1"]
    assert summary["full_issue_19_acceptance"] is False
    assert not summary["clahe_enabled"] and not summary["local_acceptance"]
    assert (out / "outputs/1.png").is_file() and (out / "figures/1.png").is_file()
    assert json.loads((out / "summary.json").read_text(encoding="utf-8")) == summary
    histogram = list(csv.DictReader((out / "histograms.csv").read_text().splitlines()))
    assert len(histogram) == 256
    assert sum(int(row["before_count"]) for row in histogram) == 96
    assert sum(int(row["after_count"]) for row in histogram) == 96
    samples = list(csv.DictReader((out / "samples.csv").read_text().splitlines()))
    assert float(samples[0]["relative_gain"]) > 0.15
    clahe = _tool.evaluate(root, manifest, {"clahe": {"enabled": True}}, tmp_path / "clahe")
    assert clahe["clahe_enabled"] and clahe["algorithm_requirements_met"]
    assert clahe["local_acceptance"]
    assert clahe["full_issue_19_acceptance"] is False
    assert "CI pending" in clahe["full_issue_19_pending_reason"]
    with pytest.raises(ValueError, match="输出目录"):
        _tool.evaluate(root, manifest, {}, out)
    (root / "skin_masks/1.png").unlink()
    with pytest.raises(FileNotFoundError):
        _tool.evaluate(root, manifest, {}, tmp_path / "missing")
    assert not (tmp_path / "missing").exists()


def test_no_underexposed_sample_must_not_vacuously_pass(tmp_path):
    root, manifest = _dataset(tmp_path, 160)
    summary = _tool.evaluate(root, manifest, {}, tmp_path / "out")
    assert summary["underexposed_relative_gain"]["macro_mean"] is None
    assert summary["underexposed_sample_ids"] == []
    assert not any(summary["quantitative_acceptance"].values())


def test_empty_gt_must_remain_invalid(tmp_path):
    root, manifest = _dataset(tmp_path, 60)
    ok, encoded = cv2.imencode(".png", np.zeros((8, 12), np.uint8))
    assert ok
    (root / "skin_masks/1.png").write_bytes(encoded.tobytes())
    summary = _tool.evaluate(root, manifest, {}, tmp_path / "out")
    assert summary["invalid_samples"] == 1
    assert not any(summary["quantitative_acceptance"].values())
