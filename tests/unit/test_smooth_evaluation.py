"""M3 测量协议的解析样例和离线工具契约。"""
import importlib.util
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("evaluate_m3", Path(__file__).resolve().parents[2] / "tools/evaluate_m3.py")
_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tool)


def test_variance_and_epi_for_known_contrast_scaling():
    gray = np.broadcast_to(np.arange(32, dtype=np.uint8) * 4 + 40, (32, 32)).copy()
    before = np.repeat(gray[:, :, None], 3, axis=2)
    after = before // 2
    full = np.full((32, 32), 255, np.uint8)
    assert _tool.variance_drop(before, after, full)["value"] == pytest.approx(0.75)
    assert _tool.edge_preservation(before, after, full)["value"] == pytest.approx(0.5)
    assert _tool.edge_preservation(before, before, full)["value"] == pytest.approx(1)


def test_empty_regions_and_flat_images_are_not_fabricated_scores():
    image = np.full((24, 24, 3), 128, np.uint8)
    empty = np.zeros((24, 24), np.uint8)
    full = np.full((24, 24), 255, np.uint8)
    assert _tool.variance_drop(image, image, empty)["value"] is None
    assert _tool.variance_drop(image, image, full)["reason"] == "zero_input_variance"
    assert _tool.edge_preservation(image, image, empty)["value"] is None
    assert _tool.edge_preservation(image, image, full)["value"] is None
    assert _tool._aggregate([0.2, None, 0.4])["macro_mean"] == pytest.approx(0.3)
    assert _tool._aggregate([None])["valid_samples"] == 0


def test_evaluator_records_mask_source_and_pending_human_review(tmp_path):
    root = tmp_path / "data"
    for folder in ("images", "skin_masks", "masks"):
        (root / folder).mkdir(parents=True)
    gray = np.broadcast_to(np.arange(32, dtype=np.uint8) * 4 + 40, (32, 32)).copy()
    image = np.repeat(gray[:, :, None], 3, axis=2)
    skin = np.zeros((32, 32), np.uint8)
    skin[4:28, 4:28] = 255
    labels = np.full((32, 32), 4, np.uint8)
    labels[skin != 0] = 1
    for folder, suffix, data in (("images", ".jpg", image), ("skin_masks", ".png", skin), ("masks", ".png", labels)):
        ok, encoded = cv2.imencode(suffix, data)
        assert ok
        (root / folder / ("1" + suffix)).write_bytes(encoded.tobytes())
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sample_id,scene\n1,normal\n", encoding="utf-8")
    summary = _tool.evaluate(root, manifest, {"strength": 0}, tmp_path / "out")
    assert summary["mask_source"] == "skin_gt"
    assert summary["variance_drop"]["macro_mean"] == 0
    assert "pending" in summary["subjective_review"]
    with (tmp_path / "out/visual_review.csv").open(encoding="utf-8", newline="") as fp:
        review = list(csv.DictReader(fp))[0]
    assert review["sample_id"] == "1"
    assert all(value == "" for key, value in review.items() if key != "sample_id")
    assert json.loads((tmp_path / "out/summary.json").read_text(encoding="utf-8")) == summary
    with pytest.raises(ValueError, match="输出目录"):
        _tool.evaluate(root, manifest, {}, tmp_path / "out")
    (root / "masks/1.png").unlink()
    with pytest.raises(FileNotFoundError):
        _tool.evaluate(root, manifest, {}, tmp_path / "missing")
    assert not (tmp_path / "missing").exists()
