# -*- coding: utf-8 -*-
"""M1 肤色检测模块（owner: B / qixing1227）。

契约（对齐 docs/baseline/arch.md 5.1）：
    detect_skin(image_bgr: np.ndarray, cfg: Mapping) -> np.ndarray
    BGR uint8 H×W×3 -> 同尺寸 uint8 H×W 掩码，取值 0/255。

约束：只吃 image_bgr 与 cfg，不读写文件、不维护全局状态、不原地修改输入。

使用 YCrCb 色度候选、可选的确定性图割共识与面部构图先验，再作掩码清理。
空间参数来自 cfg.spatial；缺失或关闭时兼容原颜色判定。GT 仅用于离线评测。
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
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


def _spatial_configuration(cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    spatial = cfg.get("spatial", {})
    if not isinstance(spatial, Mapping):
        raise ValueError("skin_detect.spatial 必须为映射")
    enabled = spatial.get("enabled", False)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("skin_detect.spatial.enabled 必须为布尔值")
    if not enabled:
        return None

    def vector(value, key, length, radius=False):
        if not isinstance(value, (list, tuple)) or len(value) != length:
            raise ValueError(f"skin_detect.spatial.{key} 长度须为 {length}")
        result = []
        for index, item in enumerate(value):
            try:
                number = float(item)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"skin_detect.spatial.{key} 必须为有限数值") from None
            lower_open = radius or (length == 4 and index >= 2)
            if (isinstance(item, (bool, np.bool_)) or not isinstance(item, Real)
                    or not np.isfinite(number) or not 0 <= number <= 1
                    or (lower_open and number == 0)):
                raise ValueError(f"skin_detect.spatial.{key} 坐标须为 0..1，半径须大于 0")
            result.append(number)
        return tuple(result)

    center = vector(spatial.get("center", [0.5, 0.48]), "center", 2)
    roi = vector(spatial.get("roi_radius", [0.4, 0.43]), "roi_radius", 2, True)
    seed = vector(spatial.get("seed_radius", [0.15, 0.2]), "seed_radius", 2, True)
    if any(a > b for a, b in zip(seed, roi)):
        raise ValueError("skin_detect.spatial.seed_radius 不能大于 roi_radius")
    iterations = _integer(spatial.get("iterations", 3), "spatial.iterations", 1, 1000)
    gmm_iterations = _integer(spatial.get("gmm_iterations", 10), "spatial.gmm_iterations", 1, 1000)
    workers = _integer(spatial.get("workers", 1), "spatial.workers", 1, 32)
    profiles = spatial.get("profiles", [
        {"space": "gray", "working_size": 256, "trusted": False, "bright_seed": False},
        {"space": "bgr", "working_size": 512, "trusted": True, "bright_seed": False},
        {"space": "bgr", "working_size": 512, "trusted": True, "bright_seed": True},
    ])
    if not isinstance(profiles, (list, tuple)) or not profiles:
        raise ValueError("skin_detect.spatial.profiles 必须为非空列表")
    parsed = []
    for item in profiles:
        if not isinstance(item, Mapping) or item.get("space") not in ("bgr", "gray"):
            raise ValueError("skin_detect.spatial.profiles.space 仅支持 bgr / gray")
        size = _integer(item.get("working_size", 512), "spatial.profiles.working_size", 1, 2048)
        trusted, bright = item.get("trusted", True), item.get("bright_seed", False)
        if any(not isinstance(flag, (bool, np.bool_)) for flag in (trusted, bright)):
            raise ValueError("skin_detect.spatial.profiles 的 trusted/bright_seed 必须为布尔值")
        parsed.append({"space": item["space"], "working_size": size,
                       "trusted": bool(trusted), "bright_seed": bool(bright)})
    votes = _integer(spatial.get("vote_threshold", len(parsed) // 2 + 1),
                     "spatial.vote_threshold", 1, len(parsed))
    face = spatial.get("face_ellipse")
    if face is not None:
        face = vector(face, "face_ellipse", 4)
    exclusions = spatial.get("exclude_ellipses", [])
    if not isinstance(exclusions, (list, tuple)):
        raise ValueError("skin_detect.spatial.exclude_ellipses 必须为列表")
    exclusions = [vector(value, "exclude_ellipses", 4) for value in exclusions]
    return {"center": center, "roi_radius": roi, "seed_radius": seed,
            "iterations": iterations, "gmm_iterations": gmm_iterations, "workers": workers,
            "profiles": parsed, "vote_threshold": votes, "face_ellipse": face,
            "exclude_ellipses": exclusions}


def _ellipse(shape, values):
    cx, cy, rx, ry = values
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (((xx / max(shape[1] - 1, 1) - cx) / rx)**2
            + ((yy / max(shape[0] - 1, 1) - cy) / ry)**2 <= 1)


def _gmm_model(pixels: np.ndarray, iterations: int) -> np.ndarray:
    """确定性 Lloyd 初始化；模型的 5/65 布局是 OpenCV GrabCut 的固定格式。

    不调用 kmeans++ 或全局 RNG。1/12 为单位量化误差的方差，保证协方差正定。
    模型完全由当前输入像素估计，不加载任何预训练参数。
    """
    data = pixels.astype(np.float64)
    order = np.argsort(data.mean(axis=1), kind="stable")
    indices = np.rint(np.linspace(0, len(data) - 1, 5)).astype(int)
    means = data[order[indices]].copy()
    previous = None
    for _ in range(iterations):
        labels = np.argmin(np.sum((data[:, None, :] - means[None, :, :])**2, axis=2), axis=1)
        if previous is not None and np.array_equal(labels, previous):
            break
        previous = labels.copy()
        for index in range(5):
            if np.any(labels == index):
                means[index] = data[labels == index].mean(axis=0)
    model = np.zeros(65, np.float64)
    for index in range(5):
        group = data[labels == index]
        if not len(group):
            continue
        center = group.mean(axis=0)
        delta = group - center
        covariance = delta.T @ delta / len(group) + np.eye(3) / 12
        model[index] = len(group) / len(data)
        model[5 + 3 * index:8 + 3 * index] = center
        model[20 + 9 * index:29 + 9 * index] = covariance.ravel()
    return model[None, :]


def _spatial_profile(image, bounds, settings, profile):
    cr_min, cr_max, cb_min, cb_max = bounds
    scale = min(1.0, profile["working_size"] / max(image.shape[:2]))
    working = cv2.resize(image, (max(1, round(image.shape[1] * scale)),
                                max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    ycrcb = cv2.cvtColor(working, cv2.COLOR_BGR2YCrCb)
    color = cv2.inRange(ycrcb, (0, cr_min, cb_min), (255, cr_max, cb_max)) != 0
    cx, cy = settings["center"]
    broad = _ellipse(color.shape, (cx, cy, *settings["roi_radius"]))
    seed = _ellipse(color.shape, (cx, cy, *settings["seed_radius"])) & color
    mark = np.full(color.shape, cv2.GC_PR_BGD, np.uint8)
    mark[color & broad] = cv2.GC_PR_FGD
    mark[~broad] = cv2.GC_BGD
    mark[seed] = cv2.GC_FGD
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    if profile["trusted"]:
        foreground, background = seed.copy(), (~broad) | (~color)
        if profile["bright_seed"] and np.any(foreground):
            threshold = cv2.threshold(gray[foreground].reshape(-1, 1), 0, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
            foreground &= gray >= threshold
    else:
        foreground = (mark == cv2.GC_FGD) | (mark == cv2.GC_PR_FGD)
        background = ~foreground
    if not np.any(foreground) or not np.any(background):
        prediction = color
    else:
        features = (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    if profile["space"] == "gray" else working)
        fgd = _gmm_model(features[foreground], settings["gmm_iterations"])
        bgd = _gmm_model(features[background], settings["gmm_iterations"])
        if profile["trusted"]:
            cv2.grabCut(features, mark, None, bgd, fgd, 1, cv2.GC_EVAL_FREEZE_MODEL)
        cv2.grabCut(features, mark, None, bgd, fgd, settings["iterations"], cv2.GC_EVAL)
        prediction = color & ((mark == cv2.GC_FGD) | (mark == cv2.GC_PR_FGD))
    return cv2.resize(prediction.astype(np.uint8), (image.shape[1], image.shape[0]),
                      interpolation=cv2.INTER_NEAREST)


def _spatial_mask(image, color_mask, bounds, settings):
    if not np.any(color_mask) or not np.any(image != image[0, 0]):
        return color_mask.copy()
    votes = np.zeros(image.shape[:2], np.uint32)
    profiles = settings["profiles"]
    if settings["workers"] == 1 or len(profiles) == 1:
        for profile in profiles:
            votes += _spatial_profile(image, bounds, settings, profile)
    else:
        # 每路只读同一输入，持有自己的模型和标记；投票在主调用线程按序汇总。
        with ThreadPoolExecutor(max_workers=min(settings["workers"], len(profiles))) as executor:
            for prediction in executor.map(lambda p: _spatial_profile(image, bounds, settings, p), profiles):
                votes += prediction
    selected = votes >= settings["vote_threshold"]
    if settings["face_ellipse"] is not None:
        selected &= _ellipse(selected.shape, settings["face_ellipse"])
    for values in settings["exclude_ellipses"]:
        selected &= ~_ellipse(selected.shape, values)
    return selected.astype(np.uint8) * 255


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
    spatial = _spatial_configuration(cfg)
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, cr_min, cb_min), (255, cr_max, cb_max))
    if spatial is not None:
        mask = _spatial_mask(image_bgr, mask, (cr_min, cr_max, cb_min, cb_max), spatial)
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
