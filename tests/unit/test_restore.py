# -*- coding: utf-8 -*-
"""M5 运动模糊复原单元测试（owner: C）。"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from beauty.modules.restore import restore_motion_blur
from beauty.metrics.metrics import compute_image_metrics


def _clean_image(size: int = 128) -> np.ndarray:
    """构造同时包含渐变、边缘和细纹的确定性彩色图。"""
    y, x = np.mgrid[:size, :size].astype(np.float32)
    image = np.stack([
        30.0 + 150.0 * x / (size - 1),
        40.0 + 130.0 * y / (size - 1),
        35.0 + 100.0 * (x + y) / (2 * (size - 1)),
    ], axis=-1)
    texture = (20.0 * np.sin(2.0 * np.pi * x / 5.0)
               + 14.0 * np.cos(2.0 * np.pi * y / 7.0))
    image += texture[..., None]
    image[24:58, 18:52] = (220, 65, 40)
    cv2.circle(image, (91, 39), 18, (25, 225, 210), thickness=-1)
    cv2.line(image, (12, 99), (116, 82), (245, 245, 245), thickness=3)
    return np.clip(image, 0, 255).astype(np.uint8)


def _horizontal_psf(length: int) -> np.ndarray:
    psf = np.zeros((length, length), dtype=np.float32)
    psf[length // 2, :] = 1.0 / length
    return psf


def _psnr(first: np.ndarray, second: np.ndarray) -> float:
    mse = float(np.mean((first.astype(np.float64) - second.astype(np.float64)) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * np.log10(255.0 ** 2 / mse)


def test_restore_improves_fixed_synthetic_motion_blur():
    """固定 PSF/噪声/种子的配对复原指标达到 R5 预设门槛。"""
    clean = _clean_image()
    psf_length = 15
    noise_sigma = 0.01
    seed = 42
    blurred = cv2.filter2D(clean, -1, _horizontal_psf(psf_length),
                           borderType=cv2.BORDER_REFLECT_101)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_sigma * 255.0, size=blurred.shape)
    blurred = np.clip(np.rint(blurred.astype(np.float32) + noise), 0, 255).astype(np.uint8)

    output = restore_motion_blur(blurred, {
        "psf_length": psf_length,
        "psf_angle": 0.0,
        "noise_sigma": 0.01,
        "wiener_k": 0.04,
        "edge_pad": 48,
        "seed": seed,
    })

    records = compute_image_metrics(blurred, output, {}, clean, None, {})
    assert _psnr(output, clean) - _psnr(blurred, clean) >= 0.5
    assert records["ssim_delta"]["valid"]
    assert records["ssim_delta"]["value"] >= 0.02


def test_restore_preserves_contract_and_input():
    image = _clean_image(48)
    original = image.copy()
    output = restore_motion_blur(image, {
        "psf_length": 7, "psf_angle": 20.0, "noise_sigma": 0.02, "edge_pad": 8,
    })
    assert output.shape == image.shape and output.dtype == np.uint8
    assert output is not image
    assert np.array_equal(image, original)


@pytest.mark.parametrize("cfg", [
    {"psf_length": 0},
    {"psf_length": 7.5},
    {"noise_sigma": -0.1},
    {"wiener_k": 0.0},
    {"edge_pad": -1},
])
def test_restore_rejects_invalid_config(cfg):
    with pytest.raises(ValueError):
        restore_motion_blur(_clean_image(32), cfg)


def test_restore_rejects_non_uint8_input():
    with pytest.raises(ValueError):
        restore_motion_blur(_clean_image(32).astype(np.float32), {})
