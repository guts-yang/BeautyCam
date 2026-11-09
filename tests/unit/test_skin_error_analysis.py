"""M1 离线归因与连通域上界：用穷举验证优化口径，禁止冒充预测。"""
import csv
import importlib.util
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("analyze_m1_errors", Path(__file__).resolve().parents[2] / "tools/analyze_m1_errors.py")
_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tool)


def test_component_ceiling_matches_exhaustive_selection_and_is_not_baseline_score():
    pred = np.zeros((9, 12), np.uint8)
    pred[1, 1:4] = 255
    pred[3, 1:5] = 255
    pred[5, 1:6] = 255
    pred[7, 10] = 255
    gt = np.zeros_like(pred)
    gt[1, 1:3] = gt[3, 1:3] = 255
    gt[7, 10] = gt[0, 10] = 255
    count, labels = cv2.connectedComponents(pred, connectivity=8)
    possibilities = []
    for selected in itertools.product((False, True), repeat=count - 1):
        keep = np.array([False, *selected])
        candidate = keep[labels]
        possibilities.append(np.count_nonzero(candidate & (gt != 0)) /
                             np.count_nonzero(candidate | (gt != 0)))
    result = _tool.component_selection_ceiling(pred, gt)
    assert result["component_count"] == 4
    assert result["iou_ceiling"] == pytest.approx(max(possibilities))
    assert result["iou_ceiling"] > np.count_nonzero((pred != 0) & (gt != 0)) / np.count_nonzero((pred != 0) | (gt != 0))


def test_component_empty_and_diagonal_connectivity():
    empty = np.zeros((3, 3), np.uint8)
    diagonal = np.eye(3, dtype=np.uint8) * 255
    assert _tool.component_selection_ceiling(diagonal, diagonal)["component_count"] == 1
    assert _tool.component_selection_ceiling(empty, diagonal)["iou_ceiling"] == 0
    assert _tool.component_selection_ceiling(diagonal, empty)["iou_ceiling"] is None
    with pytest.raises(ValueError):
        _tool.component_selection_ceiling(empty, np.zeros((1, 1), np.uint8))


def test_semantic_fp_accounting_and_skin_consistency():
    labels = np.array([[1, 1, 13], [0, 2, 17]], np.uint8)
    gt = (labels == 1).astype(np.uint8) * 255
    pred = np.array([[255, 0, 255], [0, 255, 255]], np.uint8)
    result = _tool.label_error_counts(pred, labels, gt)
    assert result["tp"] == 1 and result["fp"] == 3 and result["fn"] == 1
    assert result["fp_pixels"][13] == result["fp_pixels"][2] == result["fp_pixels"][17] == 1
    assert result["fp_pixels"][1] == 0
    assert result["iou"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="不一致"):
        _tool.label_error_counts(pred, labels, np.zeros_like(gt))
    with pytest.raises(ValueError, match="0..18"):
        _tool.label_error_counts(pred, labels + 20, gt)


def test_audit_uses_complete_manifest_and_resizes_gt_without_changing_labels(tmp_path):
    root = tmp_path / "data"
    for folder in ("images", "skin_masks", "masks"):
        (root / folder).mkdir(parents=True)
    image = np.full((16, 16, 3), (80, 120, 180), np.uint8)
    labels = np.ones((8, 8), np.uint8)
    labels[4:] = 17
    skin = (labels == 1).astype(np.uint8) * 255
    for folder, suffix, data in (("images", ".jpg", image), ("skin_masks", ".png", skin), ("masks", ".png", labels)):
        ok, encoded = cv2.imencode(suffix, data)
        assert ok
        (root / folder / ("1" + suffix)).write_bytes(encoded.tobytes())
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sample_id,scene\n1,normal\n", encoding="utf-8")
    out = tmp_path / "audit"
    summary = _tool.analyze(root, manifest, {}, out)
    assert summary["baseline"]["iou_macro_mean"] == 0.5
    assert summary["component_selection_oracle_iou_macro_mean"] == 0.5
    assert summary["fp_pixels"] == 128
    assert summary["false_positive_sources_pixel_weighted"][0]["label"] == "neck"
    assert summary["false_positive_sources_pixel_weighted"][0]["share_of_all_fp"] == 1
    assert json.loads((out / "summary.json").read_text(encoding="utf-8")) == summary
    records = list(csv.DictReader((out / "label_errors.csv").read_text().splitlines()))
    assert len(records) == 19
    assert (out / "figures/1.png").is_file()
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    ok, encoded = cv2.imencode(".png", np.zeros((16, 16), np.uint8))
    assert ok
    (predictions / "1.png").write_bytes(encoded.tobytes())
    candidate = _tool.analyze(root, manifest, {}, tmp_path / "candidate", predictions)
    assert candidate["baseline"] is None and candidate["algorithm_sha256"] is None
    assert candidate["prediction_metrics"]["iou_macro_mean"] == 0
    assert candidate["predicted_arrays_sha256"] != summary["predicted_arrays_sha256"]
    with pytest.raises(ValueError, match="输出目录"):
        _tool.analyze(root, manifest, {}, out)
    (root / "masks/1.png").unlink()
    with pytest.raises(FileNotFoundError):
        _tool.analyze(root, manifest, {}, tmp_path / "missing")
    assert not (tmp_path / "missing").exists()
