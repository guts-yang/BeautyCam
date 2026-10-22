# -*- coding: utf-8 -*-
"""M2：冻结配置内的自适应 gamma / linear 曝光改善（owner: B）。

只在传入皮肤掩码内提升 Y，保留色度；不修改输入、不读写文件。
CLAHE 尚未接入：其参数不在冻结 schema 内，变更建议见 report/。
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import cv2
import numpy as np


def _configuration(cfg: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, Mapping):
        raise ValueError("exposure.cfg 必须为映射")
    values = {}
    for key, default, lower, upper in (
        ("target_y", 150, 1, 255), ("gain", 0.3, 0, 1),
        ("max_gain", 1.5, 1, 255), ("overexposure_y", 250, 0, 255),
        ("overexposure_ratio_limit", 0.01, 0, 1),
    ):
        value = cfg.get(key, default)
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or not lower <= value <= upper):
            raise ValueError(f"exposure.{key} 必须为 [{lower}, {upper}] 内的有限数值")
        values[key] = float(value)
    gamma = cfg.get("gamma", 1.0)
    if (isinstance(gamma, (bool, np.bool_)) or not isinstance(gamma, Real)
            or not np.isfinite(gamma) or gamma <= 0):
        raise ValueError("exposure.gamma 必须为有限正数")
    values["gamma"] = float(gamma)
    method = cfg.get("method", "gamma")
    if not isinstance(method, str) or method not in ("gamma", "linear"):
        raise ValueError("exposure.method 仅支持 gamma / linear")
    values["method"] = method
    return values


def _render(original: np.ndarray, ycrcb: np.ndarray,
            mapped_y: np.ndarray, amount: float) -> np.ndarray:
    """只回写亮度实际改变的像素，避免未处理处发生色彩往返漂移。"""
    new_y = np.rint(ycrcb[:, 0] + amount * (mapped_y - ycrcb[:, 0])).astype(np.uint8)
    changed = new_y != ycrcb[:, 0]
    result = original.copy()
    if np.any(changed):
        color = ycrcb[changed].copy()
        color[:, 0] = new_y[changed]
        result[changed] = cv2.cvtColor(color[:, None, :], cv2.COLOR_YCrCb2BGR)[:, 0, :]
    return result


def process(image_bgr: np.ndarray, skin_mask: np.ndarray,
            cfg: Mapping[str, Any]) -> np.ndarray:
    """返回新的同尺寸 BGR uint8；空掩码和已足够明亮的皮肤安全跳过。

    desired = min(mean + gain*(target_y-mean), mean*max_gain)。
    gamma 曲线为 255*(Y/255)**exponent，其中 exponent 自适应目标均值，
    再乘 cfg.gamma（1 表示无额外修正）；linear 使用 desired/mean。
    逐像素提升也不超过 max_gain，纯黑不凭空生成细节。

    在实际输出 BGR 重算 Y 后检查过曝预算；超预算时二分减弱整个提升。
    若输入已超预算则原样返回，本模块不把既有过曝伪称为已修复。
    """
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8
            or min(image_bgr.shape[:2]) < 1):
        raise ValueError("exposure.image_bgr 必须为非空 BGR uint8 H×W×3")
    settings = _configuration(cfg)
    if skin_mask is None:
        return image_bgr.copy()
    if (not isinstance(skin_mask, np.ndarray) or skin_mask.ndim != 2
            or skin_mask.shape != image_bgr.shape[:2] or skin_mask.dtype != np.uint8
            or not np.all((skin_mask == 0) | (skin_mask == 255))):
        raise ValueError("exposure.skin_mask 必须为同尺寸 uint8 0/255 二维掩码")
    result = image_bgr.copy()
    region = skin_mask != 0
    if not np.any(region) or settings["gain"] == 0 or settings["max_gain"] == 1:
        return result
    colors = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)[region]
    y = colors[:, 0].astype(np.float64)
    mean = float(np.mean(y))
    threshold, budget = settings["overexposure_y"], settings["overexposure_ratio_limit"]
    if (mean == 0 or mean >= settings["target_y"]
            or float(np.mean(y > threshold)) > budget):
        return result
    desired = min(mean + settings["gain"] * (settings["target_y"] - mean),
                  mean * settings["max_gain"])
    if settings["method"] == "linear":
        mapped = y * (desired / mean)
    else:
        exponent = min(1.0, settings["gamma"] * np.log(desired / 255) / np.log(mean / 255))
        # 正浮点数下溢时仍保留 0**positive == 0 的纯黑语义。
        exponent = max(exponent, np.finfo(np.float64).tiny)
        mapped = 255 * (y / 255)**exponent
    mapped = np.minimum(np.maximum(mapped, y), np.minimum(y * settings["max_gain"], 255))
    original = image_bgr[region]

    def within_budget(pixels):
        actual_y = cv2.cvtColor(pixels[:, None, :], cv2.COLOR_BGR2YCrCb)[:, 0, 0]
        return float(np.mean(actual_y > threshold)) <= budget

    pixels = _render(original, colors, mapped, 1.0)
    if not within_budget(pixels):
        low, high = 0.0, 1.0
        pixels = original.copy()
        # 8-bit 输出的 1/256 强度分辨率；这是数值求解精度，不是算法调参。
        while high - low > 1 / 256:
            middle = (low + high) / 2
            candidate = _render(original, colors, mapped, middle)
            if within_budget(candidate):
                low, pixels = middle, candidate
            else:
                high = middle
    result[region] = pixels
    return result


__all__ = ["process"]
