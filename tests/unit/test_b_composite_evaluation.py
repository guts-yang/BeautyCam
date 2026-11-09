"""新增亮度边界测量不能把原图的真实边缘误判为算法产生的接缝。"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("evaluate_b_composite", Path(__file__).resolve().parents[2]/"tools/evaluate_b_composite.py")
_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tool)


def test_added_boundary_measure_ignores_existing_image_edge():
    image = np.full((12, 16, 3), 30, np.uint8)
    image[:, 8:] = 180
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[2:10, 8:14] = 255
    result = _tool.boundary_change(image, image.copy(), mask)
    assert result["pixels"] > 0
    assert result["mean"] == result["p95"] == result["max"] == 0


def test_known_uniform_lift_creates_expected_boundary_step():
    before = np.full((12, 16, 3), 30, np.uint8)
    mask = np.zeros(before.shape[:2], np.uint8)
    mask[2:10, 4:12] = 255
    after = before.copy()
    after[mask != 0] += 20
    result = _tool.boundary_change(before, after, mask)
    assert result["mean"] == result["p95"] == result["max"] == pytest.approx(20)


@pytest.mark.parametrize("value", [0, 255])
def test_no_internal_mask_boundary_is_not_fabricated_zero_score(value):
    image = np.full((12, 16, 3), 30, np.uint8)
    mask = np.full(image.shape[:2], value, np.uint8)
    assert _tool.boundary_change(image, image, mask) == {"pixels": 0, "mean": None, "p95": None, "max": None}


def test_skin_and_nose_secondary_protocol_counts_nose_without_relabeling_gt():
    semantic = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [4, 4, 11, 11]], np.uint8)
    semantic.setflags(write=False)
    mask = np.zeros_like(semantic)
    mask[:2] = 255
    mask[0, 0] = 0
    result = _tool.semantic_coverage(mask, semantic)
    assert result["skin_plus_nose_iou"] == pytest.approx(7/8)
    assert result["skin_plus_nose_recall"] == pytest.approx(7/8)
    assert result["nose_recall"] == 1
    assert result["eye_selected_fraction"] == result["lip_selected_fraction"] == 0
    mask[2, 0] = 255
    leaked = _tool.semantic_coverage(mask, semantic)
    assert leaked["skin_plus_nose_iou"] == pytest.approx(7/9)
    assert leaked["eye_selected_fraction"] == .5


def test_missing_semantic_regions_do_not_fabricate_coverage_scores():
    result = _tool.semantic_coverage(np.zeros((3, 4), np.uint8), np.zeros((3, 4), np.uint8))
    assert all(value is None for value in result.values())
