# -*- coding: utf-8 -*-
"""M5 运动模糊复原模块（owner: C / Shiqi0502）。

契约（对齐 docs/baseline/arch.md 5.1）：
    restore_motion_blur(image_bgr: np.ndarray, cfg: Mapping) -> np.ndarray
    BGR uint8 图像 -> 新的 BGR uint8 图像；不依赖 M1。

实现采用线性运动 PSF + 维纳反卷积。为降低 FFT 循环卷积在图像边界造成的
振铃，复原前使用反射填充，复原后再裁回原尺寸。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np


def _validate_image(image_bgr: np.ndarray) -> None:
    """校验冻结的 BGR uint8 图像契约。"""
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8
            or image_bgr.size == 0):
        raise ValueError("restore_motion_blur 需要非空 BGR uint8 H×W×3 输入")


def _motion_psf(length: int, angle_degrees: float) -> np.ndarray:
    """生成归一化线性运动 PSF（仅在内存中计算）。"""
    if isinstance(length, bool) or not isinstance(length, (int, np.integer)):
        raise ValueError("restore.psf_length 必须是整数")
    if length < 1:
        raise ValueError("restore.psf_length 必须 >= 1")
    if not np.isfinite(angle_degrees):
        raise ValueError("restore.psf_angle 必须是有限数")

    # 奇数核具有唯一中心；偶数 length 仍表示线段长度，核尺寸向上取奇数。
    size = max(3, length if length % 2 == 1 else length + 1)
    center = (size - 1) / 2.0
    horizontal = np.zeros((size, size), dtype=np.float32)
    half = (length - 1) / 2.0
    cv2.line(horizontal,
             (int(round(center - half)), int(round(center))),
             (int(round(center + half)), int(round(center))),
             color=1.0, thickness=1, lineType=cv2.LINE_AA)
    matrix = cv2.getRotationMatrix2D((center, center), float(angle_degrees), 1.0)
    psf = cv2.warpAffine(horizontal, matrix, (size, size), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    psf = np.maximum(psf, 0.0).astype(np.float32, copy=False)
    total = float(psf.sum())
    if total <= 0.0:
        raise ValueError("无法构造有效运动 PSF")
    return psf / total


def _psf_to_otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """将以中心为原点的 PSF 转为指定尺寸的 OTF。"""
    if psf.shape[0] > shape[0] or psf.shape[1] > shape[1]:
        raise ValueError("PSF 尺寸不能大于图像")
    padded = np.zeros(shape, dtype=np.float32)
    padded[:psf.shape[0], :psf.shape[1]] = psf
    padded = np.roll(padded, -(psf.shape[0] // 2), axis=0)
    padded = np.roll(padded, -(psf.shape[1] // 2), axis=1)
    return np.fft.fft2(padded)


def _wiener_channel(channel: np.ndarray, otf: np.ndarray,
                    regularization: float) -> np.ndarray:
    spectrum = np.fft.fft2(channel)
    denominator = np.abs(otf) ** 2 + regularization
    restored = np.fft.ifft2(spectrum * np.conj(otf) / denominator).real
    return restored.astype(np.float32, copy=False)


def restore_motion_blur(image_bgr: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """使用维纳反卷积复原轻微线性运动模糊。

    ``noise_sigma`` 按 0..1 归一化强度记录合成噪声；``wiener_k`` 是独立可调的
    维纳正则项。``edge_pad`` 是 FFT 复原前的反射填充宽度。
    """
    _validate_image(image_bgr)
    if not isinstance(cfg, Mapping):
        raise ValueError("restore.cfg 必须是映射")

    length = cfg.get("psf_length", 15)
    angle = float(cfg.get("psf_angle", 0.0))
    noise_sigma = float(cfg.get("noise_sigma", 0.01))
    wiener_k = float(cfg.get("wiener_k", max(noise_sigma, 1.0e-8)))
    edge_pad = cfg.get("edge_pad", max(8, int(length)))
    if not np.isfinite(noise_sigma) or noise_sigma < 0.0:
        raise ValueError("restore.noise_sigma 必须是非负有限数")
    if not np.isfinite(wiener_k) or wiener_k <= 0.0:
        raise ValueError("restore.wiener_k 必须是正有限数")
    if isinstance(edge_pad, bool) or not isinstance(edge_pad, (int, np.integer)):
        raise ValueError("restore.edge_pad 必须是整数")
    if edge_pad < 0:
        raise ValueError("restore.edge_pad 必须 >= 0")

    psf = _motion_psf(length, angle)
    h, w = image_bgr.shape[:2]
    if psf.shape[0] > h + 2 * edge_pad or psf.shape[1] > w + 2 * edge_pad:
        raise ValueError("restore.psf_length 相对输入图像过大")

    pad = min(int(edge_pad), max(h - 1, 0), max(w - 1, 0))
    normalized = image_bgr.astype(np.float32) / 255.0
    if pad:
        working = cv2.copyMakeBorder(normalized, pad, pad, pad, pad,
                                     cv2.BORDER_REFLECT_101)
    else:
        working = normalized

    otf = _psf_to_otf(psf, working.shape[:2])
    regularization = wiener_k
    restored = np.empty_like(working, dtype=np.float32)
    for channel_index in range(3):
        restored[..., channel_index] = _wiener_channel(
            working[..., channel_index], otf, regularization)

    if pad:
        restored = restored[pad:pad + h, pad:pad + w]
    return np.clip(np.rint(restored * 255.0), 0, 255).astype(np.uint8)


__all__ = ["restore_motion_blur"]
