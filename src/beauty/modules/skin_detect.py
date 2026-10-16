# -*- coding: utf-8 -*-
"""M1 肤色检测模块（owner: B / qixing1227）。

契约（对齐 docs/baseline/arch.md 5.1）：
    detect_skin(image_bgr: np.ndarray, cfg: Mapping) -> np.ndarray
    BGR uint8 H×W×3 -> 同尺寸 uint8 H×W 掩码，取值 0/255。

约束：只吃 image_bgr 与 cfg，不读写文件、不维护全局状态、不原地修改输入。

当前状态：地基签名桩（占位实现）。返回全零掩码，满足 shape/dtype 契约，
供 core.pipeline 与集成测试先行跑通；真实算法由 B 在本文件内实现替换。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def detect_skin(image_bgr: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """返回与输入同尺寸的 uint8 皮肤掩码（0/255）。"""
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("detect_skin 需要 BGR H×W×3 输入")
    h, w = image_bgr.shape[:2]
    # 占位：全零掩码是合法结果（arch 5.1：M1 全零掩码由 pipeline 给状态）。
    return np.zeros((h, w), dtype=np.uint8)


__all__ = ["detect_skin"]