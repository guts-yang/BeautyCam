# -*- coding: utf-8 -*-
"""M1 肤色检测模块（owner: B / qixing1227）。

契约（对齐 docs/baseline/arch.md 5.1）：
    detect_skin(image_bgr: np.ndarray, cfg: Mapping) -> np.ndarray
    BGR uint8 H×W×3 -> 同尺寸 uint8 H×W 掩码，取值 0/255。

输入图像通过数组传入，不维护全局状态、不原地修改输入。

YCrCb 候选和可选图割分割后保护五官，再作掩码清理。spatial 控制显式开启的
固定构图模式，face_locator 独立控制传统预训练 Haar 定位及包资源加载。
五官参数来自 cfg.features。GT 仅用于离线评测。
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from numbers import Integral, Real
import re
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
    recovery = spatial.get("coverage_recovery", {})
    if not isinstance(recovery, Mapping):
        raise ValueError("skin_detect.spatial.coverage_recovery 必须为映射")
    recover_enabled = recovery.get("enabled", False)
    if not isinstance(recover_enabled, (bool, np.bool_)):
        raise ValueError("skin_detect.spatial.coverage_recovery.enabled 必须为布尔值")
    recover_size = _integer(recovery.get("working_size", 512),
                            "spatial.coverage_recovery.working_size", 1, 2048)
    recover_limits = {}
    for key, default in (("lower_quantile", 0.55), ("horizontal_padding", 0.25)):
        value = recovery.get(key, default)
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"skin_detect.spatial.coverage_recovery.{key} 必须为0..1有限值")
        recover_limits[key] = float(value)
    if recover_enabled and not any(p["space"] == "bgr" and p["trusted"] for p in parsed):
        raise ValueError("skin_detect.spatial.coverage_recovery 需要可信BGR配置")
    surface = spatial.get("surface", {})
    if not isinstance(surface, Mapping):
        raise ValueError("skin_detect.spatial.surface 必须为映射")
    surface_values = {}
    for key in ("enabled", "recover_highlights", "protect_features"):
        value = surface.get(key, key != "enabled")
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"skin_detect.spatial.surface.{key} 必须为布尔值")
        surface_values[key] = bool(value)
    for key, default in (("horizontal_radius", 0.1), ("vertical_radius", 0.3), ("upper_radius", .03)):
        value = surface.get(key, default)
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"skin_detect.spatial.surface.{key} 必须为0..1有限值") from None
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(number) or not 0 <= number <= 1):
            raise ValueError(f"skin_detect.spatial.surface.{key} 必须为0..1有限值")
        surface_values[key] = number
    return {"center": center, "roi_radius": roi, "seed_radius": seed,
            "iterations": iterations, "gmm_iterations": gmm_iterations, "workers": workers,
            "profiles": parsed, "vote_threshold": votes, "face_ellipse": face,
            "exclude_ellipses": exclusions,
            "surface": surface_values,
            "coverage_recovery": {"enabled": bool(recover_enabled), "working_size": recover_size,
                                  **recover_limits}}


def _ellipse(shape, values):
    cx, cy, rx, ry = values
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (((xx / max(shape[1] - 1, 1) - cx) / rx)**2
            + ((yy / max(shape[0] - 1, 1) - cy) / ry)**2 <= 1)


def _feature_configuration(cfg):
    """解析局部五官保护参数；旧配置缺项时沿用原默认值。"""
    defaults = {
        "dark": {"blackhat_width": .20, "blackhat_height": .12, "residual_min": 8.,
                 "residual_percentile": 75., "local_radius": .03, "local_difference": 6.,
                 "max_luminance_ratio": .55, "search_bands": [[.20, .38], [.38, .56]],
                 "close_kernel": [3, 5], "open_width": .03, "min_width": .045,
                 "max_width": .50, "min_aspect": 1.2, "min_area": .0003, "max_height": .15},
        "lips": {"local_radius": .06, "residual_min": .015, "residual_percentile": 85.,
                 "redness_margin": .025, "search_top": .68, "search_bottom": .97,
                 "close_kernel": [5, 9], "min_width": .10, "max_width": .55,
                 "min_height": .02, "max_height": .20, "min_aspect": 1.2,
                 "center_min_y": .72, "group_y": .07, "group_x": .3},
        "eyes": {"radius_width": .45, "radius_height": .12,
                 "max_radius_x_distance": .24, "max_radius_y_distance": .07},
        "brows": {"local_radius": .18, "difference_min": 6., "left_margin": .20,
                  "right_extent": 1.20, "top_offset": .95, "bottom_offset": .25,
                  "close_kernel": [3, 5], "min_width": .25, "min_aspect": 1.3,
                  "min_area": .0005},
        "mouth_holes": {"search_top": .66, "eye_offset": .34, "center_x_radius": .24,
                        "center_max_y": .92, "bbox_top": .61, "bbox_bottom": .99,
                        "min_width": .055, "max_width": .59, "max_height": .20,
                        "min_aspect": 1.1, "min_area": .00065, "red_radius": .055,
                        "dark_radius": .040, "red_percentile": 80., "red_min": .010,
                        "dark_percentile": 60., "dark_min": 8.},
    }
    supplied = cfg.get("features", {})
    if not isinstance(supplied, Mapping) or set(supplied) - (set(defaults) | {"dilate_kernel"}):
        raise ValueError("skin_detect.features 必须为包含已知五官参数的映射")
    result = {}
    for group, template in defaults.items():
        patch = supplied.get(group, {})
        if not isinstance(patch, Mapping) or set(patch) - set(template):
            raise ValueError(f"skin_detect.features.{group} 包含未知参数或不是映射")
        parsed = dict(template)
        for key, default in template.items():
            value = patch.get(key, default)
            path = f"skin_detect.features.{group}.{key}"
            if key == "search_bands":
                if (not isinstance(value, (list, tuple)) or not value
                        or any(not isinstance(pair, (list, tuple)) or len(pair) != 2 for pair in value)):
                    raise ValueError(f"{path} 必须为非空的起止比例列表")
                bands = []
                for pair in value:
                    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
                           or not np.isfinite(v) or not 0 <= v <= 1 for v in pair) or pair[0] >= pair[1]:
                        raise ValueError(f"{path} 起止比例须在0..1且递增")
                    bands.append(tuple(map(float, pair)))
                parsed[key] = bands
            elif key == "close_kernel":
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    raise ValueError(f"{path} 必须为两个正奇数")
                sizes = [_integer(v, f"features.{group}.{key}", 1, 101) for v in value]
                if any(v % 2 == 0 for v in sizes):
                    raise ValueError(f"{path} 必须为两个正奇数")
                parsed[key] = tuple(sizes)
            else:
                limit = 100 if key.endswith("percentile") else (255 if key in {
                    "residual_min", "local_difference", "difference_min", "dark_min"} and group != "lips"
                    else (4 if key in {"min_aspect", "right_extent"} else 1))
                try:
                    number = float(value)
                except (ValueError, TypeError, OverflowError):
                    raise ValueError(f"{path} 必须为有限数值") from None
                if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                        or not np.isfinite(number) or not 0 <= number <= limit):
                    raise ValueError(f"{path} 必须为0..{limit}的有限数值")
                if ("radius" in key or key.startswith("max_") or key == "min_aspect") and value <= 0:
                    raise ValueError(f"{path} 必须大于0")
                parsed[key] = number
        for lower, upper in (("min_width", "max_width"), ("min_height", "max_height"),
                             ("search_top", "search_bottom"), ("bbox_top", "bbox_bottom")):
            if lower in parsed and upper in parsed and parsed[lower] >= parsed[upper]:
                raise ValueError(f"skin_detect.features.{group}.{lower} 必须小于 {upper}")
        result[group] = parsed
    dilation = _integer(supplied.get("dilate_kernel", 3), "features.dilate_kernel", 1, 101)
    if dilation % 2 == 0:
        raise ValueError("skin_detect.features.dilate_kernel 必须为正奇数")
    result["dilate_kernel"] = dilation
    return result


def _gmm_model(pixels: np.ndarray, iterations: int) -> np.ndarray:
    """确定性 Lloyd 初始化；模型的 5/65 布局是 OpenCV GrabCut 的固定格式。

    重复的 BGR 值按出现次数加权，不降低颜色精度；保留原像素顺序的初始
    亮度分位点。不调用 kmeans++ 或全局 RNG。1/12 为单位量化误差的方差，
    保证协方差正定。OpenCV 4.10 的 GMM 构造器从 model[20:65] 读取协方差，
    inverseCovs / covDeterms 是另外计算的内部缓存，GC_EVAL_FREEZE_MODEL 也如此。
    依据 modules/imgproc/src/grabcut.cpp 的 GMM::GMM / calcInverseCovAndDeterm。
    模型完全由当前输入像素估计。
    """
    order = np.argsort(pixels.sum(axis=1, dtype=np.uint16), kind="stable")
    indices = np.rint(np.linspace(0, len(pixels) - 1, 5)).astype(int)
    means = pixels[order[indices]].astype(np.float64)
    packed = ((pixels[:, 0].astype(np.uint32) << 16)
              | (pixels[:, 1].astype(np.uint32) << 8) | pixels[:, 2])
    colors, counts = np.unique(packed, return_counts=True)
    data = np.column_stack((colors >> 16, (colors >> 8) & 255, colors & 255)).astype(np.float64)
    weights = counts.astype(np.float64)
    previous = None
    for _ in range(iterations):
        # 分通道累加，避免 N×5×3 的距离临时数组及重复扫描各簇像素。
        distance = np.square(data[:, None, 0] - means[None, :, 0])
        distance += np.square(data[:, None, 1] - means[None, :, 1])
        distance += np.square(data[:, None, 2] - means[None, :, 2])
        labels = np.argmin(distance, axis=1)
        if previous is not None and np.array_equal(labels, previous):
            break
        previous = labels.copy()
        totals = np.bincount(labels, weights=weights, minlength=5)
        active = totals != 0
        for channel in range(3):
            sums = np.bincount(labels, weights=weights * data[:, channel], minlength=5)
            means[active, channel] = sums[active] / totals[active]
    model = np.zeros(65, np.float64)
    for index in range(5):
        selected = labels == index
        if not np.any(selected):
            continue
        group, weight = data[selected], weights[selected]
        total = weight.sum()
        center = (group * weight[:, None]).sum(axis=0) / total
        delta = group - center
        covariance = np.einsum("ni,nj,n->ij", delta, delta, weight, optimize=False) / total + np.eye(3) / 12
        model[index] = total / len(pixels)
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
        if settings["surface"]["enabled"]:
            # 连续皮肤模式的首轮也不能把ROI外的同色皮肤混入初始背景GMM。
            background = ~color
        if profile.get("non_skin_background", False):
            # ROI之外仍可能是真实面颊/下巴；恢复分支不把这些肤色当成背景模型样本。
            background = ~color
            if "recovery_core" in profile:
                # 以首轮实际检测结果为种子，在主体附近重新判断侧脸与下颌。
                # 不再用固定整图椭圆锁死背景，也不向整幅肤色背景无界扩张。
                core = cv2.resize(profile["recovery_core"].astype(np.uint8),
                                  (working.shape[1], working.shape[0]), interpolation=cv2.INTER_NEAREST) != 0
                foreground = core & color
                support = _surface_support(_largest_component(core), settings["surface"])
                mark[:] = cv2.GC_PR_BGD
                mark[color & support] = cv2.GC_PR_FGD
                mark[~support | ~color] = cv2.GC_BGD
                mark[foreground] = cv2.GC_FGD
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


def _spatial_mask(image, color_mask, bounds, settings, features=None):
    if not np.any(color_mask) or not np.any(image != image[0, 0]):
        return color_mask.copy()
    profiles = settings["profiles"]
    recovery = settings["coverage_recovery"]
    replacements = [i for i, p in enumerate(profiles) if p["space"] == "bgr" and p["trusted"]]
    def infer(tasks):
        if settings["workers"] == 1 or len(tasks) == 1:
            return [_spatial_profile(image, bounds, settings, profile) for profile in tasks]
        # 每路只读输入与种子，持有自己的模型；投票按序汇总。
        with ThreadPoolExecutor(max_workers=min(settings["workers"], len(tasks))) as executor:
            return list(executor.map(lambda p: _spatial_profile(image, bounds, settings, p), tasks))

    predictions = infer(profiles)
    votes = np.sum(predictions, axis=0, dtype=np.uint32)
    selected = votes >= settings["vote_threshold"]
    if recovery["enabled"]:
        tasks = [dict(profiles[i], non_skin_background=True) for i in replacements]
        if settings["surface"]["enabled"] and np.any(selected):
            tasks = [dict(profile, recovery_core=selected) for profile in tasks]
        if tasks:
            predictions += infer(tasks)
    if settings["surface"]["enabled"] and np.any(selected):
        candidate = selected.copy()
        if len(predictions) > len(profiles):
            candidate |= np.any(predictions[len(profiles):], axis=0)
        selected = _skin_surface(image, selected, candidate, settings, features)
    elif recovery["enabled"] and np.any(selected):
        candidate = np.any(predictions[len(profiles):], axis=0) | selected
        # 恢复域由已检测主体的实际位置与大小确定，优先补回下半脸，
        # 不让亮度投票或阴影边缘再次裁掉这些区域，也不向整幅背景无界扩张。
        scale = min(1.0, recovery["working_size"] / max(image.shape[:2]))
        size = (max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale)))
        core = cv2.resize(selected.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) != 0
        uncertain = cv2.resize(candidate.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) != 0
        ys, xs = np.where(core)
        if len(xs):
            x0, x1 = np.percentile(xs, [1, 99])
            y0, y1 = np.percentile(ys, [1, 99])
            padding = recovery["horizontal_padding"] * max(1.0, x1-x0)
            yy, xx = np.ogrid[:core.shape[0], :core.shape[1]]
            allowed = ((yy >= np.quantile(ys, recovery["lower_quantile"]))
                       & (yy <= y1 + max(1.0, y1-y0))
                       & (xx >= x0-padding) & (xx <= x1+padding))
            recovered = core | (uncertain & allowed)
            selected = cv2.resize(recovered.astype(np.uint8), (image.shape[1], image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST) != 0
    if settings["face_ellipse"] is not None:
        selected &= _ellipse(selected.shape, settings["face_ellipse"])
    for values in settings["exclude_ellipses"]:
        selected &= ~_ellipse(selected.shape, values)
    return selected.astype(np.uint8) * 255


def _largest_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask.copy()
    return labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


def _content_features(image, surface, reference=None, protect_dark=True, parameters=None):
    """按当前皮肤轮廓定位搜索范围，只保护实际暗结构/唇色，不挖固定鼻子洞。"""
    ys, xs = np.where(surface if reference is None else reference)
    if not len(xs) or not np.any(surface):
        return surface
    parameters = _feature_configuration({}) if parameters is None else parameters
    dark_cfg, lips_cfg = parameters["dark"], parameters["lips"]
    x0, x1 = np.percentile(xs, [1, 99])
    y0, y1 = np.percentile(ys, [1, 99])
    width, height = max(1.0, x1-x0), max(1.0, y1-y0)
    y = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (max(3, round(width*dark_cfg["blackhat_width"]) | 1),
                                      max(3, round(height*dark_cfg["blackhat_height"]) | 1)))
    dark_residual = cv2.morphologyEx(y, cv2.MORPH_BLACKHAT, kernel)
    threshold = max(dark_cfg["residual_min"], float(np.percentile(
        dark_residual[surface], dark_cfg["residual_percentile"])))
    local_y = cv2.GaussianBlur(y.astype(np.float32), (0, 0), max(1.0, height*dark_cfg["local_radius"]),
                               borderType=cv2.BORDER_REFLECT_101)
    # 眼下阴影仍然是皮肤；仅用明显更暗的实际结构作五官保护种子。
    dark = ((dark_residual > threshold) & (local_y-y > dark_cfg["local_difference"])
            & (y < dark_cfg["max_luminance_ratio"]*float(np.median(y[surface]))) & surface).astype(np.uint8)
    protected = np.zeros_like(surface)
    yy, xx = np.ogrid[:surface.shape[0], :surface.shape[1]]
    # 分开检查两侧眼眉，避免鼻梁连接使整个暗连通域过大而被漏掉。
    # 只保护真实横向暗结构；不把轮廓凸包扩大成整块眼周皮肤缺口。
    for lower, upper in (dark_cfg["search_bands"] if protect_dark else ()):
        for left in (True, False):
            side = xx < (x0+x1)/2 if left else xx >= (x0+x1)/2
            region = (dark != 0) & side & (yy > y0+lower*height) & (yy < y0+upper*height)
            region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE,
                                      np.ones(dark_cfg["close_kernel"], np.uint8))
            region = cv2.morphologyEx(region, cv2.MORPH_OPEN,
                                      np.ones((1, max(3, round(width*dark_cfg["open_width"]))), np.uint8))
            count, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
            for index in range(1, count):
                _, _, w, h, area = stats[index]
                if (w >= dark_cfg["min_width"]*width and w > dark_cfg["min_aspect"]*h
                        and area > dark_cfg["min_area"]*width*height
                        and w < dark_cfg["max_width"]*width and h < dark_cfg["max_height"]*height):
                    feature = (labels == index).astype(np.uint8)
                    contours, _ = cv2.findContours(feature, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(feature, contours, -1, 1, -1)
                    protected |= feature != 0
    r, g = image[:, :, 2].astype(np.float32), image[:, :, 1].astype(np.float32)
    redness = (r-g)/(r+g+1)
    local_red = redness-cv2.GaussianBlur(redness, (0, 0), max(1.0, width*lips_cfg["local_radius"]),
                                        borderType=cv2.BORDER_REFLECT_101)
    red_threshold = max(lips_cfg["residual_min"], float(np.percentile(
        local_red[surface], lips_cfg["residual_percentile"])))
    lips = (surface & (local_red > red_threshold)
            & (redness > np.median(redness[surface])+lips_cfg["redness_margin"])
            & (yy > y0+lips_cfg["search_top"]*height)
            & (yy < y0+lips_cfg["search_bottom"]*height)).astype(np.uint8)
    lips = cv2.morphologyEx(lips, cv2.MORPH_CLOSE, np.ones(lips_cfg["close_kernel"], np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(lips, connectivity=8)
    mouth_candidates = []
    for index in range(1, count):
        _, _, w, h, _ = stats[index]
        if (lips_cfg["min_width"]*width < w < lips_cfg["max_width"]*width
                and w > lips_cfg["min_aspect"]*h
                and lips_cfg["min_height"]*height < h < lips_cfg["max_height"]*height
                and centers[index, 1] > y0+lips_cfg["center_min_y"]*height):
            region = labels == index
            score = float(np.sum(np.maximum(local_red[region]-red_threshold, 0)))
            score *= float(np.mean(redness[region]))**2
            mouth_candidates.append((score, index))
    if mouth_candidates:
        _, best = max(mouth_candidates)
        for _, index in mouth_candidates:
            if (abs(centers[index, 1]-centers[best, 1]) > lips_cfg["group_y"]*height
                    or abs(centers[index, 0]-centers[best, 0]) > lips_cfg["group_x"]*width):
                continue
            feature = (labels == index).astype(np.uint8)
            contours, _ = cv2.findContours(feature, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(feature, contours, -1, 1, -1)
            protected |= feature != 0
    size = parameters["dilate_kernel"]
    protected = cv2.dilate(protected.astype(np.uint8), np.ones((size, size), np.uint8)) != 0
    return surface & ~protected


def _surface_support(core, options):
    """依据检测主体形成圆滑恢复邻域，向下可补下颌，向上仅小幅扩展。"""
    ys, xs = np.where(core)
    if not len(xs):
        return core.copy()
    x0, x1 = np.percentile(xs, [1, 99])
    y0, y1 = np.percentile(ys, [1, 99])
    rx = max(1, round(options["horizontal_radius"]*max(1.0, x1-x0)))
    ry = max(1, round(options["vertical_radius"]*max(1.0, y1-y0)))
    upper = max(1, round(options["upper_radius"]*max(1.0, y1-y0)))
    kernel_y, kernel_x = np.ogrid[-ry:ry+1, -rx:rx+1]
    vertical_scale = np.where(kernel_y < 0, ry, upper)
    kernel = ((kernel_x/rx)**2+(kernel_y/vertical_scale)**2 <= 1).astype(np.uint8)
    return cv2.dilate(core.astype(np.uint8), kernel) != 0


def _skin_surface(image, selected, candidate, settings, features=None):
    """连续恢复主体附近的肤色及暴露高光边缘；运行时不使用任何语义GT。"""
    options = settings["surface"]
    scale = min(1.0, settings["coverage_recovery"]["working_size"] / max(image.shape[:2]))
    size = (max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale)))
    working = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    core = _largest_component(cv2.resize(selected.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) != 0)
    if not np.any(core):
        return selected.copy()
    uncertain = cv2.resize(candidate.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) != 0
    support = _surface_support(core, options)
    surface = _largest_component(core | (uncertain & support))
    if options["recover_highlights"]:
        contours, _ = cv2.findContours(surface.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull, filled = np.zeros_like(core, np.uint8), np.zeros_like(core, np.uint8)
        cv2.drawContours(hull, [cv2.convexHull(max(contours, key=cv2.contourArea))], -1, 1, -1)
        cv2.drawContours(filled, contours, -1, 1, -1)
        y = cv2.cvtColor(working, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        saturation = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)[:, :, 1]
        # 仅恢复外露亮边，保留原本封闭的眼部/五官孔洞。
        surface |= ((hull != 0) & (filled == 0) & (y >= np.percentile(y[surface], 60))
                    & (saturation <= np.percentile(saturation[surface], 75)))
    surface = cv2.morphologyEx(surface.astype(np.uint8), cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) != 0
    if options["protect_features"]:
        surface = _content_features(working, surface, reference=core, parameters=features)
    return cv2.resize(surface.astype(np.uint8), (image.shape[1], image.shape[0]),
                      interpolation=cv2.INTER_NEAREST) != 0


def _haar_configuration(cfg):
    """Haar独立开关，模型使用现有opencv-python-headless包内的资源名称。"""
    options = cfg.get("face_locator", {})
    if not isinstance(options, Mapping):
        raise ValueError("skin_detect.face_locator 必须为映射")
    enabled = options.get("enabled", False)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("skin_detect.face_locator.enabled 必须为布尔值")
    if not enabled:
        return None
    size = _integer(options.get("working_size", 512), "face_locator.working_size", 64, 1024)
    models = options.get("models")
    if not isinstance(models, Mapping) or set(models) != {"face_default", "face_alt2", "eyes"}:
        raise ValueError("skin_detect.face_locator.models 需要 face_default/face_alt2/eyes")
    for name in models.values():
        if not isinstance(name, str) or not re.fullmatch(r"haarcascade_[a-z0-9_]+[.]xml", name):
            raise ValueError("skin_detect.face_locator.models 必须是包内haarcascade_*.xml资源名")
    refinement = options.get("segmentation", {})
    if not isinstance(refinement, Mapping):
        raise ValueError("skin_detect.face_locator.segmentation 必须为映射")
    try:
        segmentation = _spatial_configuration({"spatial": refinement})
    except ValueError as error:
        raise ValueError(str(error).replace("skin_detect.spatial", "skin_detect.face_locator.segmentation")) from None
    scales = {}
    for key, default in (("center_offset", [.5, .37]), ("roi_scale", [.64, .68]),
                         ("seed_scale", [.24, .32])):
        value = refinement.get(key, default)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"skin_detect.face_locator.segmentation.{key} 需要两个比例")
        parsed = []
        for item in value:
            if (isinstance(item, (bool, np.bool_)) or not isinstance(item, Real)
                    or not np.isfinite(item) or not 0 < item <= 1):
                raise ValueError(f"skin_detect.face_locator.segmentation.{key} 需要0..1正比例")
            parsed.append(float(item))
        scales[key] = tuple(parsed)
    if any(a > b for a, b in zip(scales["seed_scale"], scales["roi_scale"])):
        raise ValueError("skin_detect.face_locator.segmentation.seed_scale 不能大于roi_scale")
    return {"working_size": size, "models": dict(models), "segmentation": segmentation, **scales}


def _haar_classifier(resource_name):
    """读取cv2.data包资源并内存加载，支持Windows中文安装路径。"""
    storage = None
    try:
        xml = files("cv2.data").joinpath(resource_name).read_text(encoding="utf-8")
        storage = cv2.FileStorage(xml, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        model = cv2.CascadeClassifier()
        if not model.read(storage.getFirstTopLevelNode()) or model.empty():
            raise ValueError(f"包内级联资源不可用：{resource_name}")
        return model
    except (OSError, AttributeError, ValueError, cv2.error) as error:
        raise ValueError(f"skin_detect.face_locator 模型加载失败：{error}") from None
    finally:
        if storage is not None:
            storage.release()


def _box_overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    area = max(0, min(ax+aw, bx+bw)-max(ax, bx)) * max(0, min(ay+ah, by+bh)-max(ay, by))
    return area / max(1, aw*ah+bw*bh-area)


def _haar_localize(image, models):
    """选择主要人脸；两种级联一致或有效双眼才信任定位，否则返回None。"""
    gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    candidates = []
    for index, key in enumerate(("face_default", "face_alt2")):
        boxes = _haar_classifier(models[key]).detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=3, minSize=(60, 60))
        candidates.extend((tuple(map(int, box)), index) for box in boxes)
    if not candidates:
        return None

    def rank(item):
        (x, y, w, h), _ = item
        return w*h/(1+((x+w/2-gray.shape[1]/2)/gray.shape[1])**2
                    + ((y+h/2-gray.shape[0]/2)/gray.shape[0])**2)

    best = max(candidates, key=rank)[0]
    agreeing = [(box, index) for box, index in candidates if _box_overlap(box, best) > .5]
    face = np.rint(np.mean([box for box, _ in agreeing], axis=0)).astype(int)
    x, y, w, h = face
    roi = gray[y:min(y+round(.65*h), gray.shape[0]), x:min(x+w, gray.shape[1])]
    boxes = _haar_classifier(models["eyes"]).detectMultiScale(
        roi, scaleFactor=1.05, minNeighbors=3,
        minSize=(max(10, round(w*.1)), max(10, round(h*.1))),
        maxSize=(max(11, round(w*.30)), max(11, round(h*.30))))
    eyes = [(int(ex+x), int(ey+y), int(ew), int(eh)) for ex, ey, ew, eh in boxes]
    pairs = []
    for index, first in enumerate(eyes):
        for second in eyes[index+1:]:
            left, right = sorted((first, second), key=lambda box: box[0])
            lx, ly, lw, lh = left
            rx, ry, rw, rh = right
            dx, dy = rx+rw/2-lx-lw/2, ry+rh/2-ly-lh/2
            if .2*w < dx < .75*w and abs(dy) < .15*h and max(lw, rw)/min(lw, rw) < 1.8:
                score = (lw*lh+rw*rh)/(1+5*abs(dy)/h+abs((lx+lw/2+rx+rw/2)/2-(x+w/2))/w)
                pairs.append((score, left, right))
    # 无成对眼睛时不凭单个误检框挖洞，双人脸级联仍可约束外轮廓。
    pair = list(max(pairs, key=lambda item: item[0])[1:]) if pairs else []
    if len({index for _, index in agreeing}) < 2 and not pair:
        return None
    return face.astype(float), [np.asarray(box, float) for box in pair]


def _contour_path(cost, step=3, penalty=.15):
    """动态规划追踪连续边缘；代价只来自图像和Haar局部几何先验。"""
    height, width = cost.shape
    previous = cost[:, 0].copy()
    back = np.zeros((height, width), np.int8)
    offsets = np.arange(-step, step+1)
    for col in range(1, width):
        padded = np.pad(previous, (step, step), constant_values=np.inf)
        alternatives = np.stack([padded[step+d:step+d+height]+penalty*d*d for d in offsets])
        choice = np.argmin(alternatives, axis=0)
        back[:, col] = offsets[choice]
        previous = cost[:, col] + alternatives[choice, np.arange(height)]
    row = int(np.argmin(previous))
    path = np.empty(width, int)
    path[-1] = row
    for col in range(width-1, 0, -1):
        row += int(back[row, col])
        path[col-1] = row
    return path


def _haar_contour(image, mask, face, eyes, parameters=None):
    """宽松人脸域内，分别用皮肤转变和下颌边缘排除头发、颈部。"""
    x, y, w, h = face
    distance = .35*w
    if len(eyes) == 2:
        centers = [eye[:2]+eye[2:]/2 for eye in eyes]
        distance = abs(centers[1][0]-centers[0][0])
        factor = np.clip(distance/(.45*w), 1, 1.18)
        x -= w*(factor-1)/2
        w, h = w*factor, h*factor
    cx = x+w/2
    yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]]
    top, bottom = max(0, y-.35*h), min(mask.shape[0]-1, y+1.23*h)
    cy = (top+bottom)/2
    pred = mask & ((np.abs((xx-cx)/(.50*w))**3
                    + np.abs((yy-cy)/max(1, (bottom-top)/2))**3) <= 1)
    ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    gray = ycc[:, :, 0]
    dy = cv2.Sobel(cv2.GaussianBlur(gray, (0, 0), 2), cv2.CV_32F, 0, 1, ksize=3)/8
    a, b = max(0, round(y+.03*h)), min(mask.shape[0], round(y+.18*h))
    left, right = max(0, round(cx-.2*w)), min(mask.shape[1], round(cx+.2*w))
    seed = mask[a:b, left:right]
    if np.any(seed):
        samples = ycc[a:b, left:right][seed]
        median = np.median(samples, axis=0)
        mad = np.median(np.abs(samples-median), axis=0)
        high = gray-cv2.GaussianBlur(gray, (0, 0), 1)
        rms = np.sqrt(cv2.GaussianBlur(high*high, (0, 0), 2))
        texture = max(6, 3*float(np.median(rms[a:b, left:right][seed])))
        skin = (mask & (np.abs(ycc[:, :, 1]-median[1]) < max(8, 3*mad[1]))
                & (np.abs(ycc[:, :, 2]-median[2]) < max(8, 3*mad[2]))
                & (gray > median[0]*.6) & (rms < texture)).astype(np.float32)
    else:
        skin = mask.astype(np.float32)
    edge = cv2.Sobel(cv2.GaussianBlur(skin, (0, 0), 2), cv2.CV_32F, 0, 1, ksize=3)/8
    x0, x1 = max(0, round(cx-.50*w)), min(mask.shape[1], round(cx+.50*w)+1)
    norm = (np.arange(x0, x1)-cx)/(.5*w)
    lo, hi = max(0, round(y-.35*h)), min(mask.shape[0], round(y+.25*h))
    if hi > lo and x1 > x0:
        strength = np.maximum(edge[lo:hi, x0:x1], 0)
        strength = np.clip(strength/max(.01, float(np.percentile(strength, 95))), 0, 3)
        prior = y-.10*h+.13*h*norm**2
        rows = np.arange(lo, hi)[:, None]
        path = _contour_path(-strength+.25*((rows-prior[None, :])/max(5, .15*h))**2)+lo
        boundary = np.full(mask.shape[1], lo)
        boundary[x0:x1] = path
        pred &= yy >= boundary[None, :]
    lo, hi = max(0, round(y+.72*h)), min(mask.shape[0], round(y+1.23*h))
    if hi > lo and x1 > x0:
        strength = np.maximum(-dy[lo:hi, x0:x1], 0)
        strength /= max(.5, float(np.percentile(strength, 95)))
        drop = cv2.Sobel(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2), cv2.CV_32F, 0, 1, ksize=3)/8
        strength = np.clip(strength+3*np.maximum(-drop[lo:hi, x0:x1], 0), 0, 3)
        prior = y+1.04*h-.18*h*norm**2
        rows = np.arange(lo, hi)[:, None]
        path = _contour_path(-strength+.8*((rows-prior[None, :])/max(5, .11*h))**2)+lo
        boundary = np.full(mask.shape[1], hi)
        boundary[x0:x1] = path
        pred &= yy <= boundary[None, :]
    eye_cfg = (_feature_configuration({}) if parameters is None else parameters)["eyes"]
    for ex, ey, ew, eh in eyes:
        rx = min(eye_cfg["radius_width"]*ew, eye_cfg["max_radius_x_distance"]*distance)
        ry = min(eye_cfg["radius_height"]*eh, eye_cfg["max_radius_y_distance"]*distance)
        pred &= ((xx-ex-ew/2)/max(1, rx))**2 + ((yy-ey-eh/2)/max(1, ry))**2 > 1
    return pred


def _haar_brows(image, pred, face, eyes, parameters=None):
    """眼睛上方只剔除实际横向眉毛暗结构，不把整块阴影挖掉。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    yy, xx = np.ogrid[:pred.shape[0], :pred.shape[1]]
    fw, fh = face[2:]
    parameters = _feature_configuration({}) if parameters is None else parameters
    cfg = parameters["brows"]
    protected = np.zeros_like(pred, np.uint8)
    for ex, ey, ew, eh in eyes:
        cy = ey+eh/2
        residual = cv2.GaussianBlur(gray, (0, 0), max(1, cfg["local_radius"]*ew))-gray
        band = ((xx > ex-cfg["left_margin"]*ew) & (xx < ex+cfg["right_extent"]*ew)
                & (yy > cy-cfg["top_offset"]*eh) & (yy < cy-cfg["bottom_offset"]*eh))
        dark = cv2.morphologyEx((band & (residual > cfg["difference_min"]) & pred).astype(np.uint8),
                                cv2.MORPH_CLOSE, np.ones(cfg["close_kernel"], np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        options = []
        for index in range(1, count):
            _, _, width, height, area = stats[index]
            if (width > cfg["min_width"]*ew and width > cfg["min_aspect"]*height
                    and area > cfg["min_area"]*fw*fh):
                options.append((area*float(residual[labels == index].mean()), index))
        if options:
            index = max(options)[1]
            contours, _ = cv2.findContours((labels == index).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(protected, contours, -1, 1, -1)
    size = parameters["dilate_kernel"]
    return pred & ~(cv2.dilate(protected, np.ones((size, size), np.uint8)) != 0)


def _mouth_holes(image, selected, filled, face, eyes, parameters=None):
    """保留原先已分出的嘴部空洞，用位置、横向形状和局部反差验证。"""
    holes = filled & ~selected
    if not np.any(holes):
        return holes
    parameters = _feature_configuration({}) if parameters is None else parameters
    cfg = parameters["mouth_holes"]
    x, y, width, height = face
    mouth_top = y + cfg["search_top"] * height
    if eyes:
        eye_y = np.mean([ey + eh / 2 for _, ey, _, eh in eyes])
        mouth_top = max(mouth_top, eye_y + cfg["eye_offset"] * height)

    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        holes.astype(np.uint8), connectivity=8)
    candidates = []
    for index in range(1, count):
        _, top, span, depth, area = stats[index]
        cx, cy = centers[index]
        centered = abs(cx - (x + width / 2)) < cfg["center_x_radius"] * width
        in_mouth_band = (mouth_top < cy < y + cfg["center_max_y"] * height
                         and top > y + cfg["bbox_top"] * height
                         and top + depth < y + cfg["bbox_bottom"] * height)
        shape_matches = (cfg["min_width"] * width < span < cfg["max_width"] * width
                         and depth < cfg["max_height"] * height
                         and span >= cfg["min_aspect"] * depth
                         and area > cfg["min_area"] * width * height)
        if centered and in_mouth_band and shape_matches:
            candidates.append(index)
    if not candidates:
        return np.zeros_like(holes)

    _, green, red = image.astype(np.float32).transpose(2, 0, 1)
    redness = (red - green) / (red + green + 1)
    red_residual = redness - cv2.GaussianBlur(redness, (0, 0), max(1, cfg["red_radius"] * width))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dark_residual = cv2.GaussianBlur(gray, (0, 0), max(1, cfg["dark_radius"] * width)) - gray
    keep = np.zeros(count, bool)
    size = parameters["dilate_kernel"]
    kernel = np.ones((size, size), np.uint8)
    for index in candidates:
        region = labels == index
        border = cv2.dilate(region.astype(np.uint8), kernel) != 0
        keep[index] = (np.percentile(red_residual[border], cfg["red_percentile"]) > cfg["red_min"]
                       or np.percentile(dark_residual[region], cfg["dark_percentile"]) > cfg["dark_min"])
    return keep[labels]


def _locate_image(image, options):
    """在工作尺度定位一次，供脸框初始化与最终五官保护共享。"""
    scale = min(1., options["working_size"]/max(image.shape[:2]))
    size = (max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale)))
    working = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return working, _haar_localize(working, options["models"])


def _face_segmentation(located, options):
    """将局部种子和搜索域锚定实际人脸框，随人脸平移、缩放。"""
    working, (face, _) = located
    x, y, width, height = face
    settings = dict(options["segmentation"])
    size = np.array([max(1, working.shape[1]-1), max(1, working.shape[0]-1)])
    settings["center"] = tuple((np.array([x, y])+np.array(options["center_offset"])*face[2:])/size)
    settings["roi_radius"] = tuple(np.minimum(1, np.array(options["roi_scale"])*face[2:]/size))
    settings["seed_radius"] = tuple(np.array(options["seed_scale"])*face[2:]/size)
    return settings


def _haar_refine(image, mask, options, located=None):
    if options is None or min(image.shape[:2]) < 60 or not np.any(mask):
        return mask
    working, location = _locate_image(image, options) if located is None else located
    if location is None:
        return mask
    face, eyes = location
    features = options.get("features")
    selected = cv2.resize(mask, (working.shape[1], working.shape[0]), interpolation=cv2.INTER_NEAREST) != 0
    # 先恢复被高光/阴影造成的闭合空洞，再依据真实检测位置重新保护眼眉、唇。
    filled = selected.astype(np.uint8)
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 1, -1)
    filled = filled != 0
    result = _haar_contour(working, filled, face, eyes, features)
    result = _haar_brows(working, result, face, eyes, features)
    reference = np.zeros_like(selected)
    x, y, width, height = np.rint(face).astype(int)
    reference[max(0, y):y+height, max(0, x):x+width] = True
    result = _content_features(working, result, reference=reference,
                               protect_dark=not bool(eyes), parameters=features)
    result &= ~_mouth_holes(working, selected, filled, face, eyes, features)
    return cv2.resize(result.astype(np.uint8)*255, (image.shape[1], image.shape[0]),
                      interpolation=cv2.INTER_NEAREST)


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
    locator = _haar_configuration(cfg)
    features = _feature_configuration(cfg)
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, cr_min, cb_min), (255, cr_max, cb_max))
    located = None
    if locator is not None and min(image_bgr.shape[:2]) >= 60 and np.any(mask):
        locator = dict(locator, features=features)
        located = _locate_image(image_bgr, locator)
        if spatial is None and locator["segmentation"] is not None and located[1] is not None:
            spatial = _face_segmentation(located, locator)
    if spatial is not None:
        mask = _spatial_mask(image_bgr, mask, (cr_min, cr_max, cb_min, cb_max), spatial, features)
    mask = _haar_refine(image_bgr, mask, locator, located)
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
