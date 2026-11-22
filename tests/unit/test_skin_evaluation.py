"""独立验证 B 的离线评测口径，避免空 GT 和像素加权掩盖失败样本。"""
import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

# tools/ 是独立脚本目录；不要求 pytest 启动器将仓库根目录加入 sys.path。
_spec = importlib.util.spec_from_file_location(
    "evaluate_m1", Path(__file__).resolve().parents[2] / "tools/evaluate_m1.py")
_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tool)
evaluate = _tool.evaluate
mask_scores = _tool.mask_scores
summarize_scores = _tool.summarize_scores


def test_scores_known_intersection_and_missed_skin():
    gt = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    pred = np.array([[255, 0], [255, 0]], dtype=np.uint8)
    scores = mask_scores(pred, gt)
    assert scores["iou"] == pytest.approx(1 / 3)
    assert scores["recall"] == 0.5 and scores["valid"]
    zero = mask_scores(np.zeros_like(gt), gt)
    assert zero["valid"] and zero["iou"] == zero["recall"] == 0


def test_empty_gt_excluded_and_per_image_macro_average():
    empty = mask_scores(np.ones((1, 1), np.uint8), np.zeros((1, 1), np.uint8))
    perfect_large = mask_scores(np.ones((100, 100), np.uint8), np.ones((100, 100), np.uint8))
    missed_small = mask_scores(np.zeros((1, 1), np.uint8), np.ones((1, 1), np.uint8))
    result = summarize_scores([empty, perfect_large, missed_small])
    assert result["valid_samples"] == 2 and result["empty_gt_samples"] == 1
    assert result["iou_macro_mean"] == result["recall_macro_mean"] == 0.5
    assert summarize_scores([empty])["iou_macro_mean"] is None


def test_score_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        mask_scores(np.zeros((2, 3), np.uint8), np.zeros((3, 2), np.uint8))


def _fixture(tmp_path: Path):
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "skin_masks").mkdir()
    image = np.full((32, 32, 3), (80, 120, 180), np.uint8)
    gt = np.full((16, 16), 255, np.uint8)
    for path, img, ext in [(data / "images/1.jpg", image, ".jpg"),
                           (data / "skin_masks/1.png", gt, ".png")]:
        ok, encoded = cv2.imencode(ext, img)
        assert ok
        path.write_bytes(encoded.tobytes())
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sample_id,scene\n1,normal\n", encoding="utf-8")
    return data, manifest


def test_evaluate_aligns_gt_and_writes_finite_reproducible_evidence(tmp_path):
    data, manifest = _fixture(tmp_path)
    output = tmp_path / "out"
    summary = evaluate(data, manifest, {}, output)
    assert all(summary["acceptance"].values())
    assert summary["post_on"]["iou_macro_mean"] == 1.0
    assert (output / "contact_sheet.png").is_file()
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    with (output / "samples.csv").open(encoding="utf-8", newline="") as fp:
        assert list(csv.DictReader(fp))[0]["gt_pixels"] == "1024"
    repeated = evaluate(data, manifest, {}, tmp_path / "repeat")
    assert repeated == summary


def test_missing_sample_fails_instead_of_silently_dropping_it(tmp_path):
    data, manifest = _fixture(tmp_path)
    manifest.write_text("sample_id,scene\n1,normal\n2,normal\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        evaluate(data, manifest, {}, tmp_path / "out")
    assert not (tmp_path / "out").exists()
