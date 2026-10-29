# -*- coding: utf-8 -*-
"""M7 指标与匿名问卷汇总单元测试（owner: C）。"""

from __future__ import annotations

import cv2
import numpy as np

from beauty.metrics.metrics import (
    _delta_e_ciede2000,
    compute_image_metrics,
    summarize_survey,
)


def _image(size: int = 48) -> np.ndarray:
    y, x = np.mgrid[:size, :size].astype(np.float32)
    return np.stack([
        25.0 + 120.0 * x / (size - 1),
        30.0 + 110.0 * y / (size - 1),
        35.0 + 100.0 * (x + y) / (2 * (size - 1)),
    ], axis=-1).astype(np.uint8)


def _cfg() -> dict:
    return {"protocol": {
        "skin_erode_kernel": 5, "variance_window": 7,
        "epsilon": 1.0e-6, "hf_sigma": 3.0,
    }}


def test_metric_records_obey_valid_invalid_contract():
    records = compute_image_metrics(None, None, {}, None, None, _cfg())
    assert records
    for record in records.values():
        assert set(record) == {"value", "unit", "valid", "not_applicable_reason"}
        assert record["valid"] is False
        assert record["value"] is None
        assert record["not_applicable_reason"]


def test_ciede2000_matches_published_reference_pairs():
    """用 Sharma CIEDE2000 标准样例守护公式实现。"""
    first = np.array([
        [[50.0000, 2.6772, -79.7751]],
        [[50.0000, 3.1571, -77.2803]],
    ])
    second = np.array([
        [[50.0000, 0.0000, -82.7485]],
        [[50.0000, 0.0000, -82.7485]],
    ])
    actual = _delta_e_ciede2000(first, second).ravel()
    assert np.allclose(actual, [2.0425, 2.8615], atol=1.0e-4)


def test_reference_metrics_report_finite_improvement_and_delta_e():
    reference = _image(64)
    rng = np.random.default_rng(42)
    noisy = np.clip(reference.astype(np.int16) +
                    rng.normal(0, 18, reference.shape), 0, 255).astype(np.uint8)
    output = np.clip(reference.astype(np.int16) +
                     rng.normal(0, 5, reference.shape), 0, 255).astype(np.uint8)
    skin = np.full(reference.shape[:2], 255, dtype=np.uint8)
    records = compute_image_metrics(
        noisy, output, {"skin_gt": skin}, reference, skin, _cfg())

    assert records["psnr_delta_db"]["valid"]
    assert records["psnr_delta_db"]["value"] > 0.5
    assert records["ssim_delta"]["valid"]
    assert records["ssim_delta"]["value"] > 0.02
    assert records["delta_e2000"]["valid"]
    assert np.isfinite(records["delta_e2000"]["value"])
    assert records["skin_iou"]["value"] == 1.0
    assert records["skin_recall"]["value"] == 1.0


def test_region_metrics_and_mask_resize_are_supported():
    before = _image(64)
    # 增加确定性棋盘细节，使方差/高频/EPI 可评估。
    checker = ((np.indices((64, 64)).sum(axis=0) % 2) * 16).astype(np.uint8)
    before = np.clip(before.astype(np.int16) + checker[..., None], 0, 255).astype(np.uint8)
    after = cv2.GaussianBlur(before, (5, 5), 0, borderType=cv2.BORDER_REFLECT_101)
    small_mask = np.full((32, 32), 255, dtype=np.uint8)
    records = compute_image_metrics(
        before, after,
        {"skin_gt": small_mask, "face_feature_gt": small_mask},
        None, None, _cfg())

    assert records["variance_drop"]["valid"]
    assert records["variance_drop"]["value"] > 0
    assert records["hf_ratio"]["valid"]
    assert records["epi"]["valid"]
    assert records["overexposure_ratio"]["valid"]


def test_non_underexposed_input_marks_exposure_gain_not_applicable():
    before = np.full((32, 32, 3), 150, dtype=np.uint8)
    after = np.full_like(before, 160)
    skin = np.full((32, 32), 255, dtype=np.uint8)
    records = compute_image_metrics(before, after, {"skin_gt": skin}, None, None, _cfg())
    assert records["exposure_gain"]["valid"] is False
    assert records["exposure_gain"]["value"] is None
    assert "Y<90" in records["exposure_gain"]["not_applicable_reason"]


def test_survey_summary_contains_only_anonymous_aggregates():
    questions = ["naturalness", "texture_preservation", "facial_detail", "skin_tone"]
    responses = []
    for index in range(20):
        responses.append({question: 3 + (index % 3) for question in questions})
    # 身份类字段即使存在也必须被汇总器忽略。
    responses[0]["name"] = "should-not-appear"
    summary = summarize_survey(responses, {"survey": {
        "minimum_valid_responses": 20, "score_min": 1, "score_max": 5,
        "questions": questions,
    }})

    assert summary["valid_responses"] == 20
    assert summary["meets_minimum"] is True
    assert "name" not in repr(summary)
    for question in questions:
        item = summary["questions"][question]
        assert item["valid_sample_count"] == 20
        assert sum(item["distribution"].values()) == 20


def test_survey_partial_or_invalid_answers_do_not_count_as_valid_response():
    summary = summarize_survey([
        {"naturalness": 5, "texture_preservation": 4,
         "facial_detail": 3, "skin_tone": 2},
        {"naturalness": 6, "texture_preservation": 4},
        {"scores": {"naturalness": [4, 5], "texture_preservation": [4],
                    "facial_detail": [3], "skin_tone": [4]}},
    ], {"survey": {"minimum_valid_responses": 3}})
    assert summary["valid_responses"] == 2
    assert summary["meets_minimum"] is False
