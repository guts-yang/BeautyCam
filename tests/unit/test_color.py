# -*- coding: utf-8 -*-
"""M6 白平衡与色彩增强单元测试（owner: C）。"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from beauty.modules.color import process


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:96, :96].astype(np.float32)
    neutral = 70.0 + 90.0 * (0.55 * x + 0.45 * y) / 95.0
    image = np.repeat(neutral[..., None], 3, axis=2)
    image[28:72, 30:68] = (112, 146, 190)
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[28:72, 30:68] = 255
    return np.clip(image, 0, 255).astype(np.uint8), mask


def _lab_error(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    a = cv2.cvtColor(first.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    b = cv2.cvtColor(second.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    distance = np.linalg.norm(a - b, axis=2)
    return float(np.mean(distance[mask != 0]))


def test_gray_world_reduces_fixed_synthetic_color_cast_and_keeps_outside():
    clean, mask = _fixture()
    # 记录的合成 RGB 增益：R=1.25, G=0.90, B=0.75。
    rgb_gains = (1.25, 0.90, 0.75)
    bgr_gains = np.array(rgb_gains[::-1], dtype=np.float32)
    cast = np.clip(np.rint(clean.astype(np.float32) * bgr_gains), 0, 255).astype(np.uint8)
    output = process(cast, mask, {
        "wb_method": "gray_world", "skin_only": True,
        "saturation_gain": 1.0, "balance_strength": 1.0,
        "gain_min": 0.5, "gain_max": 2.0,
    })

    assert _lab_error(output, clean, mask) < _lab_error(cast, clean, mask)
    assert np.array_equal(output[mask == 0], cast[mask == 0])


def test_color_preserves_contract_and_input():
    image, mask = _fixture()
    original = image.copy()
    output = process(image, mask, {
        "wb_method": "max_rgb", "max_rgb_percentile": 95.0,
        "skin_only": True, "saturation_gain": 1.05,
    })
    assert output.shape == image.shape and output.dtype == np.uint8
    assert output is not image
    assert np.array_equal(image, original)


@pytest.mark.parametrize("mask", [None, np.zeros((96, 96), dtype=np.uint8)])
def test_color_missing_or_empty_mask_safely_skips(mask):
    image, _ = _fixture()
    output = process(image, mask, {"skin_only": False})
    assert output is not image
    assert np.array_equal(output, image)


@pytest.mark.parametrize("cfg", [
    {"wb_method": "unknown"},
    {"balance_strength": 1.1},
    {"saturation_gain": -1.0},
    {"shade_power": 0.0},
    {"valid_luma_min": 250, "valid_luma_max": 5},
    {"gain_min": 2.0, "gain_max": 1.0},
])
def test_color_rejects_invalid_config(cfg):
    image, mask = _fixture()
    with pytest.raises(ValueError):
        process(image, mask, cfg)


def test_color_rejects_invalid_mask_contract():
    image, mask = _fixture()
    invalid = mask.copy()
    invalid[0, 0] = 1
    with pytest.raises(ValueError):
        process(image, invalid, {})
