# -*- coding: utf-8 -*-
"""M3 空间域平滑模块（owner: B / qixing1227）。

契约（对齐 docs/baseline/arch.md 5.1）：
    process(image_bgr: np.ndarray, skin_mask: np.ndarray, cfg: Mapping) -> np.ndarray

约束：只吃 mask、不 import M1；不原地修改输入；掩码为空时安全跳过。

双边滤波降低皮肤纹理噪声，局部中值参考抑制小型暗色斑点。
仅在传入皮肤掩码内合成，并按掩码内距离衰减边界处的处理强度。
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import cv2
import numpy as np


def _configuration(cfg: Mapping[str, Any]) -> tuple[float, int, float, float]:
    if not isinstance(cfg, Mapping):
        raise ValueError("smooth.cfg 必须为映射")
    values = [cfg.get("strength", 0.5), cfg.get("sigma_color", 75),
              cfg.get("sigma_space", 75)]
    for key, value in zip(("strength", "sigma_color", "sigma_space"), values):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value)):
            raise ValueError(f"smooth.{key} 必须为有限数值")
    strength, color, space = map(float, values)
    if not 0 <= strength <= 1 or color <= 0 or space <= 0:
        raise ValueError("smooth.strength 须在 0..1，sigma_color/sigma_space 须为正数")
    diameter = cfg.get("bilateral_d", 9)
    if (isinstance(diameter, (bool, np.bool_)) or not isinstance(diameter, Integral)
            or diameter < 1 or diameter % 2 == 0):
        raise ValueError("smooth.bilateral_d 必须为正奇数")
    if cfg.get("border_type", "BORDER_REFLECT_101") != "BORDER_REFLECT_101":
        raise ValueError("smooth.border_type 须为 BORDER_REFLECT_101")
    return strength, int(diameter), color, space


def _repair_small_dark_spots(image: np.ndarray, smoothed: np.ndarray,
                             interior: np.ndarray, diameter: int) -> np.ndarray:
    """由 Otsu 自动决定暗残差阈值，只修补滤波窗口内的小型局部结构。

    面积与包围盒上限取现有 bilateral_d 的窗口尺度，不引入新配置键。
    无法从颜色图判定一处暗点是痘印还是雀斑；这一路径仍需人工检查。
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    radius = diameter // 2
    padded = cv2.copyMakeBorder(y, radius, radius, radius, radius, cv2.BORDER_REFLECT_101)
    median = cv2.medianBlur(padded, diameter)[radius:-radius, radius:-radius]
    residual = cv2.subtract(median, y)
    values = residual[interior]
    if not values.size or not np.any(values):
        return smoothed
    threshold, _ = cv2.threshold(values.reshape(-1, 1), 0, 255,
                                  cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    candidates = ((residual > threshold) & interior).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    keep = ((stats[:, cv2.CC_STAT_AREA] <= diameter * diameter)
            & (stats[:, cv2.CC_STAT_WIDTH] <= diameter)
            & (stats[:, cv2.CC_STAT_HEIGHT] <= diameter))
    keep[0] = False
    spots = keep[labels]
    if not np.any(spots):
        return smoothed
    filtered_ycrcb = cv2.cvtColor(smoothed, cv2.COLOR_BGR2YCrCb)
    filtered_ycrcb[:, :, 0][spots] = np.maximum(
        filtered_ycrcb[:, :, 0][spots], median[spots])
    repaired = cv2.cvtColor(filtered_ycrcb, cv2.COLOR_YCrCb2BGR)
    # 色彩空间往返的取整误差只保留在实际修补位置。
    result = smoothed.copy()
    result[spots] = repaired[spots]
    return result


def process(image_bgr: np.ndarray, skin_mask: np.ndarray,
            cfg: Mapping[str, Any]) -> np.ndarray:
    """返回新的 BGR uint8 图像；掩码外严格保持输入像素。"""
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8
            or min(image_bgr.shape[:2]) < 1):
        raise ValueError("smooth.process 需要非空 BGR uint8 H×W×3 输入")
    strength, diameter, color, space = _configuration(cfg)
    if skin_mask is None:
        return image_bgr.copy()
    if (not isinstance(skin_mask, np.ndarray) or skin_mask.ndim != 2
            or skin_mask.shape != image_bgr.shape[:2] or skin_mask.dtype != np.uint8
            or not np.all((skin_mask == 0) | (skin_mask == 255))):
        raise ValueError("smooth.skin_mask 必须为同尺寸 uint8 0/255 二维掩码")
    if not np.any(skin_mask) or strength == 0 or diameter == 1:
        return image_bgr.copy()
    candidate = cv2.bilateralFilter(image_bgr, diameter, color, space,
                                    borderType=cv2.BORDER_REFLECT_101)
    radius = diameter // 2
    distance = cv2.distanceTransform(skin_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    interior = distance > radius
    candidate = _repair_small_dark_spots(image_bgr, candidate, interior, diameter)
    alpha = strength * np.clip(distance / radius, 0, 1)
    mixed = image_bgr.astype(np.float32) + alpha[:, :, None] * (
        candidate.astype(np.float32) - image_bgr.astype(np.float32))
    result = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    result[skin_mask == 0] = image_bgr[skin_mask == 0]
    return result


__all__ = ["process"]
