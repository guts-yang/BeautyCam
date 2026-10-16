# -*- coding: utf-8 -*-
"""M7 评价与指标计算模块（owner: C / Shiqi0502）。

契约（对齐 docs/baseline/arch.md 5.1）：
    compute_image_metrics(stage_input_bgr, stage_output_bgr, regions,
                          reference_bgr, predicted_skin_mask, cfg)
        -> Mapping[str, Mapping[str, Any]]
    summarize_survey(responses, cfg) -> Mapping[str, Any]

约束：纯计算，不读写文件、不 import 图像处理模块。
指标记录字段：value / unit / valid / not_applicable_reason；
valid=false 时 value 必须为空，不得写 0/NaN。

当前状态：地基签名桩（占位实现），返回空记录；真实实现由 C 替换。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def compute_image_metrics(stage_input_bgr: np.ndarray | None,
                          stage_output_bgr: np.ndarray | None,
                          regions: Mapping[str, np.ndarray],
                          reference_bgr: np.ndarray | None,
                          predicted_skin_mask: np.ndarray | None,
                          cfg: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    """返回纯数据指标记录（metric_name -> 记录映射），不做文件 I/O。"""
    return {}


def summarize_survey(responses: Sequence[Mapping[str, Any]],
                     cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """匿名问卷记录 -> 汇总统计；不读写文件。"""
    return {}


__all__ = ["compute_image_metrics", "summarize_survey"]