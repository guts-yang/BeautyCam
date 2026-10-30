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


def _gmm_model(pixels: np.ndarray, iterations: int) -> np.ndarray:
    """确定性 Lloyd 初始化；模型的 5/65 布局是 OpenCV GrabCut 的固定格式。

    重复的 BGR 值按出现次数加权，不降低颜色精度；保留原像素顺序的初始
    亮度分位点。不调用 kmeans++ 或全局 RNG。1/12 为单位量化误差的方差，
    保证协方差正定。模型完全由当前输入像素估计，不加载任何预训练参数。
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


def _spatial_mask(image, color_mask, bounds, settings):
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
        selected = _skin_surface(image, selected, candidate, settings)
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


def _content_features(image, surface, reference=None):
    """按当前皮肤轮廓定位搜索范围，只保护实际暗结构/唇色，不挖固定鼻子洞。"""
    ys, xs = np.where(surface if reference is None else reference)
    if not len(xs):
        return surface
    x0, x1 = np.percentile(xs, [1, 99])
    y0, y1 = np.percentile(ys, [1, 99])
    width, height = max(1.0, x1-x0), max(1.0, y1-y0)
    y = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (max(3, round(width*.20) | 1), max(3, round(height*.12) | 1)))
    dark_residual = cv2.morphologyEx(y, cv2.MORPH_BLACKHAT, kernel)
    threshold = max(8.0, float(np.percentile(dark_residual[surface], 75)))
    local_y = cv2.GaussianBlur(y.astype(np.float32), (0, 0), max(1.0, height*.03),
                               borderType=cv2.BORDER_REFLECT_101)
    # 眼下阴影仍然是皮肤；仅用明显更暗的实际结构作五官保护种子。
    dark = ((dark_residual > threshold) & (local_y-y > 6)
            & (y < .55*float(np.median(y[surface]))) & surface).astype(np.uint8)
    protected = np.zeros_like(surface)
    yy, xx = np.ogrid[:surface.shape[0], :surface.shape[1]]
    # 分开检查两侧眼眉，避免鼻梁连接使整个暗连通域过大而被漏掉。
    # 只保护真实横向暗结构；不把轮廓凸包扩大成整块眼周皮肤缺口。
    for lower, upper in ((.20, .38), (.38, .56)):
        for left in (True, False):
            side = xx < (x0+x1)/2 if left else xx >= (x0+x1)/2
            region = (dark != 0) & side & (yy > y0+lower*height) & (yy < y0+upper*height)
            region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
            region = cv2.morphologyEx(region, cv2.MORPH_OPEN,
                                      np.ones((1, max(3, round(width*.03))), np.uint8))
            count, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
            for index in range(1, count):
                _, _, w, h, area = stats[index]
                if (w >= .045*width and w > 1.2*h and area > .0003*width*height
                        and w < .50*width and h < .15*height):
                    feature = (labels == index).astype(np.uint8)
                    contours, _ = cv2.findContours(feature, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(feature, contours, -1, 1, -1)
                    protected |= feature != 0
    r, g = image[:, :, 2].astype(np.float32), image[:, :, 1].astype(np.float32)
    redness = (r-g)/(r+g+1)
    local_red = redness-cv2.GaussianBlur(redness, (0, 0), max(1.0, width*.06),
                                        borderType=cv2.BORDER_REFLECT_101)
    red_threshold = max(.015, float(np.percentile(local_red[surface], 85)))
    lips = (surface & (local_red > red_threshold) & (redness > np.median(redness[surface])+.025)
            & (yy > y0+.68*height) & (yy < y0+.97*height)).astype(np.uint8)
    lips = cv2.morphologyEx(lips, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(lips, connectivity=8)
    mouth_candidates = []
    for index in range(1, count):
        _, _, w, h, _ = stats[index]
        if (.10*width < w < .55*width and w > 1.2*h and .02*height < h < .20*height
                and centers[index, 1] > y0+.72*height):
            region = labels == index
            score = float(np.sum(np.maximum(local_red[region]-red_threshold, 0)))
            score *= float(np.mean(redness[region]))**2
            mouth_candidates.append((score, index))
    if mouth_candidates:
        _, best = max(mouth_candidates)
        for _, index in mouth_candidates:
            if (abs(centers[index, 1]-centers[best, 1]) > .07*height
                    or abs(centers[index, 0]-centers[best, 0]) > .3*width):
                continue
            feature = (labels == index).astype(np.uint8)
            contours, _ = cv2.findContours(feature, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(feature, contours, -1, 1, -1)
            protected |= feature != 0
    protected = cv2.dilate(protected.astype(np.uint8), np.ones((3, 3), np.uint8)) != 0
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


def _skin_surface(image, selected, candidate, settings):
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
        surface = _content_features(working, surface, reference=core)
    return cv2.resize(surface.astype(np.uint8), (image.shape[1], image.shape[0]),
                      interpolation=cv2.INTER_NEAREST) != 0


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
