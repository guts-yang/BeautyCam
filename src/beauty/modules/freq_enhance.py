# -*- coding: utf-8 -*-
"""M4 频域增强模块（owner: B / qixing1227）。

契约（对齐 docs/baseline/arch.md 5.1）：
    process(image_bgr: np.ndarray, skin_mask: np.ndarray, cfg: Mapping) -> np.ndarray
    get_frequency_diagnostics(image_bgr: np.ndarray, cfg: Mapping)
        -> Mapping[str, np.ndarray]  # low_frequency / high_frequency 同尺寸 BGR uint8

约束：诊断接口不改主输出、不修改输入；主处理返回新的 BGR uint8 图像。

当前状态：地基签名桩（占位实现），process 返回输入副本；
get_frequency_diagnostics 返回两张与输入同尺寸的全零 BGR 图。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def process(image_bgr: np.ndarray, skin_mask: np.ndarray,
            cfg: Mapping[str, Any]) -> np.ndarray:
    """返回新的 BGR uint8 图像。"""
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("freq_enhance.process 需要 BGR H×W×3 输入")
    if skin_mask is not None and skin_mask.shape[:2] != image_bgr.shape[:2]:
        raise ValueError("skin_mask 尺寸必须与 image_bgr 一致")
    return image_bgr.copy()


def get_frequency_diagnostics(image_bgr: np.ndarray,
                              cfg: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    """返回低频图与高频可视化图（同尺寸 BGR uint8），不修改输入。"""
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("get_frequency_diagnostics 需要 BGR H×W×3 输入")
    h, w = image_bgr.shape[:2]
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    return {"low_frequency": blank.copy(), "high_frequency": blank.copy()}


__all__ = ["process", "get_frequency_diagnostics"]