# -*- coding: utf-8 -*-
"""M2 曝光改善模块（owner: B / qixing1227）。

契约（对齐 docs/baseline/arch.md 5.1）：
    process(image_bgr: np.ndarray, skin_mask: np.ndarray, cfg: Mapping) -> np.ndarray
    BGR uint8 H×W×3 -> 新的 BGR uint8 图像。

约束：不原地修改输入；掩码为空时不对整图执行肤色限定处理（安全跳过）。

当前状态：地基签名桩（占位实现），返回输入副本，满足 shape/dtype 与
输入不变性契约；真实算法由 B 在本文件内实现替换。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def process(image_bgr: np.ndarray, skin_mask: np.ndarray,
            cfg: Mapping[str, Any]) -> np.ndarray:
    """返回新的 BGR uint8 图像。"""
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("exposure.process 需要 BGR H×W×3 输入")
    if skin_mask is not None and skin_mask.shape[:2] != image_bgr.shape[:2]:
        raise ValueError("skin_mask 尺寸必须与 image_bgr 一致")
    # 占位：返回输入副本，保证不原地修改输入。
    return image_bgr.copy()


__all__ = ["process"]