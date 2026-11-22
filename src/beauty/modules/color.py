# -*- coding: utf-8 -*-
"""M6 色彩/白平衡模块（owner: C / Shiqi0502）。

契约（对齐 docs/baseline/arch.md 5.1）：
    process(image_bgr: np.ndarray, skin_mask: np.ndarray, cfg: Mapping) -> np.ndarray

约束：只依赖 mask 格式契约，不 import M1；不原地修改输入；掩码为空时安全跳过。

白平衡增益由全图有效像素估计，处理结果再按皮肤掩码限定作用域，避免把皮肤
本身的暖色均值误当作光源色偏。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np


def _validate_inputs(image_bgr: np.ndarray, skin_mask: np.ndarray | None) -> None:
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8
            or image_bgr.size == 0):
        raise ValueError("color.process 需要非空 BGR uint8 H×W×3 输入")
    if skin_mask is None:
        return
    if (not isinstance(skin_mask, np.ndarray) or skin_mask.ndim != 2
            or skin_mask.dtype != np.uint8
            or skin_mask.shape != image_bgr.shape[:2]):
        raise ValueError("skin_mask 必须是与 image_bgr 同尺寸的 uint8 H×W 数组")
    if not set(np.unique(skin_mask)).issubset({0, 255}):
        raise ValueError("skin_mask 取值必须仅为 0/255")


def _estimation_pixels(image: np.ndarray, low: float, high: float) -> np.ndarray:
    """排除近黑与近饱和像素，降低极端区域对光源估计的影响。"""
    luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    valid = (luminance >= low) & (luminance <= high)
    if int(np.count_nonzero(valid)) < 16:
        return image.reshape(-1, 3).astype(np.float32)
    return image[valid].astype(np.float32)


def _estimate_gains(image_bgr: np.ndarray, method: str, low: float, high: float,
                    percentile: float, shade_power: float) -> np.ndarray:
    pixels = _estimation_pixels(image_bgr, low, high)
    if method == "none":
        return np.ones(3, dtype=np.float32)
    if method == "gray_world":
        statistic = np.mean(pixels, axis=0, dtype=np.float64)
    elif method == "shades_of_gray":
        normalized = np.maximum(pixels.astype(np.float64) / 255.0, 1.0e-12)
        statistic = np.mean(normalized ** shade_power, axis=0) ** (1.0 / shade_power)
    elif method == "max_rgb":
        statistic = np.percentile(pixels, percentile, axis=0)
    else:
        raise ValueError(
            "color.wb_method 必须是 shades_of_gray / gray_world / max_rgb / none")
    statistic = np.maximum(statistic, 1.0e-6)
    target = float(np.cbrt(np.prod(statistic)))
    return (target / statistic).astype(np.float32)


def _enhance_saturation(image_bgr: np.ndarray, gain: float) -> np.ndarray:
    if abs(gain - 1.0) <= 1.0e-8:
        return image_bgr
    hsv = cv2.cvtColor(np.clip(np.rint(image_bgr), 0, 255).astype(np.uint8),
                       cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * gain, 0, 255)
    return cv2.cvtColor(np.rint(hsv).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)


def process(image_bgr: np.ndarray, skin_mask: np.ndarray,
            cfg: Mapping[str, Any]) -> np.ndarray:
    """返回新的 BGR uint8 图像，不原地修改输入。

    缺失或全零掩码时始终安全跳过，不会把整图误当皮肤区。
    """
    _validate_inputs(image_bgr, skin_mask)
    if not isinstance(cfg, Mapping):
        raise ValueError("color.cfg 必须是映射")
    if skin_mask is None or not bool(np.any(skin_mask)):
        return image_bgr.copy()

    method = str(cfg.get("wb_method", "shades_of_gray")).lower()
    temperature = float(cfg.get("temperature", 0.0))
    saturation_gain = float(cfg.get("saturation_gain", 1.05))
    balance_strength = float(cfg.get("balance_strength", 1.0))
    low = float(cfg.get("valid_luma_min", 5.0))
    high = float(cfg.get("valid_luma_max", 250.0))
    percentile = float(cfg.get("max_rgb_percentile", 95.0))
    shade_power = float(cfg.get("shade_power", 6.0))
    gain_min = float(cfg.get("gain_min", 0.5))
    gain_max = float(cfg.get("gain_max", 2.0))
    skin_only = cfg.get("skin_only", True)

    values = (temperature, saturation_gain, balance_strength, low, high,
              percentile, shade_power, gain_min, gain_max)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("color 数值配置必须为有限数")
    if not isinstance(skin_only, bool):
        raise ValueError("color.skin_only 必须是布尔值")
    if not (0.0 <= balance_strength <= 1.0):
        raise ValueError("color.balance_strength 必须在 [0, 1]")
    if saturation_gain < 0.0:
        raise ValueError("color.saturation_gain 必须 >= 0")
    if not (0.0 <= low < high <= 255.0):
        raise ValueError("color 有效亮度范围必须满足 0 <= min < max <= 255")
    if not (0.0 < percentile <= 100.0):
        raise ValueError("color.max_rgb_percentile 必须在 (0, 100]")
    if shade_power <= 0.0:
        raise ValueError("color.shade_power 必须 > 0")
    if not (0.0 < gain_min <= gain_max):
        raise ValueError("color 增益范围必须满足 0 < gain_min <= gain_max")

    gains = _estimate_gains(image_bgr, method, low, high, percentile, shade_power)
    # temperature 使用 [-1, 1] 归一化尺度：正值增暖，负值降温。
    temperature = float(np.clip(temperature, -1.0, 1.0))
    gains *= np.array([1.0 - 0.15 * temperature, 1.0,
                       1.0 + 0.15 * temperature], dtype=np.float32)
    gains = np.clip(gains, gain_min, gain_max)
    gains = 1.0 + balance_strength * (gains - 1.0)

    corrected = image_bgr.astype(np.float32) * gains.reshape(1, 1, 3)
    corrected = _enhance_saturation(corrected, saturation_gain)
    corrected_u8 = np.clip(np.rint(corrected), 0, 255).astype(np.uint8)

    if not skin_only:
        return corrected_u8
    output = image_bgr.copy()
    selected = skin_mask != 0
    output[selected] = corrected_u8[selected]
    return output


__all__ = ["process"]
