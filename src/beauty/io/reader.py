# -*- coding: utf-8 -*-
"""io.reader —— 图像与 GT 掩码读取与规范化（A 独占）。

约束（对齐 docs/baseline/arch.md 5.2、7.2）：
- 彩色图统一读为 OpenCV BGR、uint8、H×W×3；
- 掩码统一读为灰度、uint8、H×W、取值 0/255；
- GT 标签图按最近邻插值对齐到目标尺寸（尺寸不一致时）。
"""

from __future__ import annotations

import os

import cv2
import numpy as np


class ImageReadError(IOError):
    """图像读取失败，消息须可读且不含机器绝对路径以外的敏感信息。"""


def read_image(path: str) -> np.ndarray:
    """读取彩色图为 BGR uint8 H×W×3。"""
    if not os.path.isfile(path):
        raise ImageReadError(f"文件不存在：{path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageReadError(f"无法解码图像：{path}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def read_mask(path: str, target_hw: tuple[int, int] | None = None) -> np.ndarray:
    """读取掩码为 uint8 H×W、取值 0/255。

    target_hw 给定时按最近邻插值对齐到 (H, W)。
    """
    if not os.path.isfile(path):
        raise ImageReadError(f"文件不存在：{path}")
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ImageReadError(f"无法解码掩码：{path}")
    if target_hw is not None and mask.shape[:2] != tuple(target_hw):
        mask = cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    return mask


__all__ = ["ImageReadError", "read_image", "read_mask"]