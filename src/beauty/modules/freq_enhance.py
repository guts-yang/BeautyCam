# -*- coding: utf-8 -*-
"""M4 频域增强模块（owner: B / qixing1227）。

契约（对齐 docs/baseline/arch.md 5.1）：
    process(image_bgr: np.ndarray, skin_mask: np.ndarray, cfg: Mapping) -> np.ndarray
    get_frequency_diagnostics(image_bgr: np.ndarray, cfg: Mapping)
        -> Mapping[str, np.ndarray]  # low_frequency / high_frequency 同尺寸 BGR uint8

约束：诊断接口不改主输出、不修改输入；主处理返回新的 BGR uint8 图像。

在 Y 亮度通道做实数二维 FFT 与高斯频率分离，保留原图色度。
lowpass_sigma 表示空间等效高斯标准差（像素），频率单位为 cycles/pixel。
诊断图独立计算：低频亮度图、高频有符号残差加 128 的可视化。
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import cv2
import numpy as np


def _validate_image(image: np.ndarray) -> None:
    if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
            or image.dtype != np.uint8 or min(image.shape[:2]) < 1):
        raise ValueError("freq_enhance 需要非空 BGR uint8 H×W×3 输入")


def _configuration(cfg: Mapping[str, Any]) -> tuple[float, float, float]:
    if not isinstance(cfg, Mapping):
        raise ValueError("freq_enhance.cfg 必须为映射")
    values = []
    for key, default in (("lowpass_sigma", 3.0), ("lowpass_gain", 1.0), ("highpass_gain", 0.5)):
        value = cfg.get(key, default)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"freq_enhance.{key} 必须为有限数值")
        value = float(value)
        if not np.isfinite(value) or value < 0 or (key == "lowpass_sigma" and value == 0):
            raise ValueError(f"freq_enhance.{key} 必须为有限非负数，sigma 须大于 0")
        if key != "lowpass_sigma" and value > np.finfo(np.float64).max / 1024:
            raise ValueError(f"freq_enhance.{key} 超出安全浮点运算范围")
        values.append(value)
    diagnostics = cfg.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping) or any(
            not isinstance(diagnostics.get(key, True), (bool, np.bool_))
            for key in ("low_frequency", "high_frequency")):
        raise ValueError("freq_enhance.diagnostics 必须为布尔显示选项映射")
    return tuple(values)


def _layers(image: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """完整偶反射延拓对应 REFLECT_101，避免 FFT 周期边界连接左右两端。

    H(fx,fy)=exp(-2*pi²*sigma²*(fx²+fy²))，其 DC 增益为 1。
    low + high 精确重构原亮度；高频保留负值直到最终合成。
    """
    y = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    height, width = y.shape
    extended = np.pad(y, ((0, max(height - 2, 0)), (0, max(width - 2, 0))), mode="reflect")
    fy = np.fft.fftfreq(extended.shape[0])[:, None]
    fx = np.fft.rfftfreq(extended.shape[1])[None, :]
    with np.errstate(over="ignore"):
        transfer = np.exp(-2 * np.pi**2 * ((sigma * fx)**2 + (sigma * fy)**2))
    spectrum = np.fft.rfft2(extended)
    low = np.fft.irfft2(spectrum * transfer, s=extended.shape)[:height, :width]
    return y, low, y - low


def process(image_bgr: np.ndarray, skin_mask: np.ndarray,
            cfg: Mapping[str, Any]) -> np.ndarray:
    """只在传入皮肤掩码内合成低/高频，返回新 BGR uint8 图。

    以 sigma 尺度衰减掩码边缘强度。保持未改变亮度的像素原值，
    避免 YCrCb 往返转换的取整误差扩散到无需处理的区域。
    """
    _validate_image(image_bgr)
    sigma, low_gain, high_gain = _configuration(cfg)
    if skin_mask is None:
        return image_bgr.copy()
    if (not isinstance(skin_mask, np.ndarray) or skin_mask.ndim != 2
            or skin_mask.shape != image_bgr.shape[:2] or skin_mask.dtype != np.uint8
            or not np.all((skin_mask == 0) | (skin_mask == 255))):
        raise ValueError("freq_enhance.skin_mask 必须为同尺寸 uint8 0/255 二维掩码")
    if not np.any(skin_mask) or (low_gain == 1 and high_gain == 1):
        return image_bgr.copy()
    y, low, high = _layers(image_bgr, sigma)
    if np.all(skin_mask):
        alpha = np.ones(skin_mask.shape, dtype=np.float64)
    else:
        distance = cv2.distanceTransform(skin_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        alpha = np.clip(distance.astype(np.float64) / sigma, 0, 1)
    candidate = low_gain * low + high_gain * high
    new_y = np.clip(np.rint(y + alpha * (candidate - y)), 0, 255).astype(np.uint8)
    changed = (skin_mask != 0) & (new_y != y)
    result = image_bgr.copy()
    if np.any(changed):
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = new_y
        converted = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        result[changed] = converted[changed]
    return result


def get_frequency_diagnostics(image_bgr: np.ndarray,
                              cfg: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    """显式调用时始终返回冻结的两张图；中灰 128 表示零高频残差。

    diagnostics 显示偏好不改变返回键，也不影响 process；调用与落盘开关
    由公共流水线负责。本接口只分层输入图，不使用合成强度或皮肤掩码。
    """
    _validate_image(image_bgr)
    sigma, _, _ = _configuration(cfg)
    _, low, high = _layers(image_bgr, sigma)
    low_vis = np.clip(np.rint(low), 0, 255).astype(np.uint8)
    high_vis = np.clip(np.rint(high + 128), 0, 255).astype(np.uint8)
    return {"low_frequency": cv2.cvtColor(low_vis, cv2.COLOR_GRAY2BGR),
            "high_frequency": cv2.cvtColor(high_vis, cv2.COLOR_GRAY2BGR)}


__all__ = ["process", "get_frequency_diagnostics"]
