# -*- coding: utf-8 -*-
"""M2：CLAHE + 自适应 gamma / linear 曝光改善（owner: B）。

只在传入皮肤掩码内提升 Y，保留色度；不修改输入、不读写文件。
CLAHE 参数来自 cfg.clahe；缺少该块或 enabled=false 时兼容旧曲线。
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import cv2
import numpy as np


def _number(value: Any, key: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"exposure.{key} 必须为有限数值")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"exposure.{key} 必须为有限数值") from None
    if not np.isfinite(result):
        raise ValueError(f"exposure.{key} 必须为有限数值")
    return result


def _configuration(cfg: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, Mapping):
        raise ValueError("exposure.cfg 必须为映射")
    values = {}
    for key, default, lower, upper in (
        ("target_y", 150, 1, 255), ("gain", 0.3, 0, 1),
        ("max_gain", 1.5, 1, 255), ("overexposure_y", 250, 0, 255),
        ("overexposure_ratio_limit", 0.01, 0, 1),
    ):
        value = _number(cfg.get(key, default), key)
        if not lower <= value <= upper:
            raise ValueError(f"exposure.{key} 必须为 [{lower}, {upper}] 内的有限数值")
        values[key] = float(value)
    gamma = _number(cfg.get("gamma", 1.0), "gamma")
    if gamma <= 0:
        raise ValueError("exposure.gamma 必须为有限正数")
    values["gamma"] = float(gamma)
    method = cfg.get("method", "gamma")
    if not isinstance(method, str) or method not in ("gamma", "linear"):
        raise ValueError("exposure.method 仅支持 gamma / linear")
    values["method"] = method
    clahe = cfg.get("clahe", {})
    if not isinstance(clahe, Mapping):
        raise ValueError("exposure.clahe 必须为映射")
    enabled = clahe.get("enabled", False)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("exposure.clahe.enabled 必须为布尔值")
    clip = _number(clahe.get("clip_limit", 2.0), "clahe.clip_limit")
    if clip <= 0:
        raise ValueError("exposure.clahe.clip_limit 必须为正数")
    grid = clahe.get("tile_grid_size", [8, 8])
    if (not isinstance(grid, (list, tuple)) or len(grid) != 2
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 1 for v in grid)):
        raise ValueError("exposure.clahe.tile_grid_size 必须含两个正整数")
    values["clahe"] = {"enabled": bool(enabled), "clip_limit": clip,
                       "tile_grid_size": tuple(map(int, grid))}
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
    可选 CLAHE 先在亮度上增强，以既有 gain 混合，再进入曲线。
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
    ycrcb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    colors = ycrcb_image[region]
    y = colors[:, 0].astype(np.float64)
    mean = float(np.mean(y))
    threshold, budget = settings["overexposure_y"], settings["overexposure_ratio_limit"]
    if (mean == 0 or mean >= settings["target_y"]
            or float(np.mean(y > threshold)) > budget):
        return result
    desired = min(mean + settings["gain"] * (settings["target_y"] - mean),
                  mean * settings["max_gain"])
    curve_y, curve_mean = y, mean
    clahe = settings["clahe"]
    luminance = ycrcb_image[:, :, 0]
    if clahe["enabled"] and np.max(luminance) != np.min(luminance):
        # 网格多于像素没有意义，按当前图尺寸收敛；避免极小图过度填充。
        grid = (min(clahe["tile_grid_size"][0], image_bgr.shape[1]),
                min(clahe["tile_grid_size"][1], image_bgr.shape[0]))
        # 8-bit 共 256 个桶，clip_limit>=256 已等价于不裁剪，避免数值溢出。
        clip = min(clahe["clip_limit"], np.iinfo(np.uint8).max + 1)
        enhanced = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(luminance)
        curve_y = np.maximum(y, y + settings["gain"] * (enhanced[region].astype(np.float64) - y))
        curve_mean = float(np.mean(curve_y))
    if settings["method"] == "linear":
        mapped = curve_y * max(1.0, desired / curve_mean)
    elif curve_mean >= 255:
        mapped = curve_y
    else:
        adaptive_exponent = float(np.log(desired / 255) / np.log(curve_mean / 255))
        exponent = min(1.0, settings["gamma"] * adaptive_exponent)
        # 正浮点数下溢时仍保留 0**positive == 0 的纯黑语义。
        exponent = max(exponent, np.finfo(np.float64).tiny)
        mapped = 255 * (curve_y / 255)**exponent
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
