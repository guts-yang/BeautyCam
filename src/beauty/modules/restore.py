# -*- coding: utf-8 -*-
"""M5 运动模糊复原模块（owner: C / Shiqi0502）。

契约（对齐 docs/baseline/arch.md 5.1）：
    restore_motion_blur(image_bgr: np.ndarray, cfg: Mapping) -> np.ndarray
    BGR uint8 图像 -> 新的 BGR uint8 图像；不依赖 M1。

当前状态：地基签名桩（占位实现），返回输入副本。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def restore_motion_blur(image_bgr: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """返回新的 BGR uint8 图像。"""
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("restore_motion_blur 需要 BGR H×W×3 输入")
    return image_bgr.copy()


__all__ = ["restore_motion_blur"]