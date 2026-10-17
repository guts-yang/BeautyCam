# -*- coding: utf-8 -*-
"""M1 肤色检测模块（owner: B / qixing1227）。

契约（对齐 docs/baseline/arch.md 5.1）：
    detect_skin(image_bgr: np.ndarray, cfg: Mapping) -> np.ndarray
    BGR uint8 H×W×3 -> 同尺寸 uint8 H×W 掩码，取值 0/255。

约束：只吃 image_bgr 与 cfg，不读写文件、不维护全局状态、不原地修改输入。

使用 YCrCb 色度阈值生成候选掩码；可选椭圆核开闭运算及小连通域清理。
颜色判定不等于语义人脸分割：肤色背景、偏色和遮挡的误差须离线评测。
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import cv2
import numpy as np


def _integer(value: Any, key: str, minimum: int, maximum: int | None = None) -> int:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or value < minimum or (maximum is not None and value > maximum)):
        limit = f"{minimum}..{maximum}" if maximum is not None else f">={minimum}"
        raise ValueError(f"skin_detect.{key} 必须为整数（{limit}）")
    return int(value)


def _configuration(cfg: Mapping[str, Any]) -> tuple[int, int, int, int, bool, int, float]:
    """读取既有 YAML schema；空配置使用与 YAML 一致的默认值。"""
    if not isinstance(cfg, Mapping):
        raise ValueError("skin_detect.cfg 必须为映射")
    color = cfg.get("ycbcr", {})
    post = cfg.get("postprocess", {})
    if not isinstance(color, Mapping) or not isinstance(post, Mapping):
        raise ValueError("skin_detect.ycbcr/postprocess 必须为映射")
    cr_min = _integer(color.get("cr_min", 133), "ycbcr.cr_min", 0, 255)
    cr_max = _integer(color.get("cr_max", 173), "ycbcr.cr_max", 0, 255)
    cb_min = _integer(color.get("cb_min", 77), "ycbcr.cb_min", 0, 255)
    cb_max = _integer(color.get("cb_max", 127), "ycbcr.cb_max", 0, 255)
    if cr_min > cr_max or cb_min > cb_max:
        raise ValueError("skin_detect.ycbcr 的 min 不得大于 max")
    enabled = post.get("enabled", True)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("skin_detect.postprocess.enabled 必须为布尔值")
    kernel = _integer(post.get("morph_kernel", 5), "postprocess.morph_kernel", 1)
    if kernel % 2 == 0:
        raise ValueError("skin_detect.postprocess.morph_kernel 必须为正奇数")
    ratio = post.get("min_area_ratio", 0.001)
    if (isinstance(ratio, (bool, np.bool_)) or not isinstance(ratio, Real)
            or not np.isfinite(ratio) or not 0 <= ratio <= 1):
        raise ValueError("skin_detect.postprocess.min_area_ratio 必须为 0..1 有限数值")
    return cr_min, cr_max, cb_min, cb_max, bool(enabled), kernel, float(ratio)


def detect_skin(image_bgr: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """返回新建的同尺寸 uint8 皮肤掩码（0/255），不改变输入或配置。

    OpenCV 通道顺序为 Y、Cr、Cb。色度区间含两端点，不对亮度 Y
    另加阈值。后处理开启时先开运算去孤点，再闭运算填小缝隙，
    最后删除面积小于整图像素数乘 min_area_ratio 的 8 连通区域。
    """
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8
            or min(image_bgr.shape[:2]) < 1):
        raise ValueError("detect_skin 需要非空 BGR uint8 H×W×3 输入")
    cr_min, cr_max, cb_min, cb_max, enabled, kernel_size, area_ratio = _configuration(cfg)
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, cr_min, cb_min), (255, cr_max, cb_max))
    if not enabled:
        return mask
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                borderType=cv2.BORDER_REFLECT_101)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                borderType=cv2.BORDER_REFLECT_101)
    if area_ratio > 0 and np.any(mask):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= mask.size * area_ratio
        keep[0] = False  # 背景永远不是皮肤连通域。
        mask = np.where(keep[labels], 255, 0).astype(np.uint8)
    return mask


__all__ = ["detect_skin"]
