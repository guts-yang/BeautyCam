# -*- coding: utf-8 -*-
"""M7 评价与指标计算模块（owner: C / Shiqi0502）。

契约（对齐 docs/baseline/arch.md 5.1）：
    compute_image_metrics(stage_input_bgr, stage_output_bgr, regions,
                          reference_bgr, predicted_skin_mask, cfg)
        -> Mapping[str, Mapping[str, Any]]
    summarize_survey(responses, cfg) -> Mapping[str, Any]

本模块只做内存中的纯计算，不读写文件、不 import 任何图像处理模块。
无效指标的 ``value`` 固定为 ``None``，不使用 0/NaN/Infinity 伪装有效结果。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np


def _invalid(unit: str, reason: str) -> dict[str, Any]:
    return {"value": None, "unit": unit, "valid": False,
            "not_applicable_reason": reason}


def _valid(value: float, unit: str) -> dict[str, Any]:
    number = float(value)
    if not np.isfinite(number):
        return _invalid(unit, "计算结果非有限数")
    return {"value": number, "unit": unit, "valid": True,
            "not_applicable_reason": ""}


def _is_bgr(image: np.ndarray | None) -> bool:
    return (isinstance(image, np.ndarray) and image.ndim == 3
            and image.shape[2] == 3 and image.dtype == np.uint8
            and image.size > 0)


def _binary_mask(mask: np.ndarray | None,
                 shape: tuple[int, int]) -> np.ndarray | None:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size == 0:
        return None
    if mask.shape != shape:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    return mask != 0


def _erode(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    size = int(kernel_size)
    if size < 1:
        size = 1
    if size % 2 == 0:
        size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.erode(mask.astype(np.uint8), kernel,
                     borderType=cv2.BORDER_CONSTANT) != 0


def _luma(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)


def _psnr(image: np.ndarray, reference: np.ndarray) -> float | None:
    difference = image.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(difference * difference))
    if mse <= 0.0:
        return None
    return 10.0 * np.log10((255.0 ** 2) / mse)


def _ssim_fallback(image: np.ndarray, reference: np.ndarray,
                   win_size: int = 7) -> float | None:
    """按通道计算 7×7 均匀窗 SSIM，语义对齐 data_range=255。"""
    if min(image.shape[:2]) < win_size:
        return None
    first = image.astype(np.float32)
    second = reference.astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    values = []
    for index in range(3):
        x = first[..., index]
        y = second[..., index]
        mean_x = cv2.boxFilter(x, cv2.CV_32F, (win_size, win_size),
                               borderType=cv2.BORDER_REFLECT_101)
        mean_y = cv2.boxFilter(y, cv2.CV_32F, (win_size, win_size),
                               borderType=cv2.BORDER_REFLECT_101)
        mean_x2 = cv2.boxFilter(x * x, cv2.CV_32F, (win_size, win_size),
                                borderType=cv2.BORDER_REFLECT_101)
        mean_y2 = cv2.boxFilter(y * y, cv2.CV_32F, (win_size, win_size),
                                borderType=cv2.BORDER_REFLECT_101)
        mean_xy = cv2.boxFilter(x * y, cv2.CV_32F, (win_size, win_size),
                                borderType=cv2.BORDER_REFLECT_101)
        variance_x = np.maximum(mean_x2 - mean_x * mean_x, 0.0)
        variance_y = np.maximum(mean_y2 - mean_y * mean_y, 0.0)
        covariance = mean_xy - mean_x * mean_y
        numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2)
        denominator = ((mean_x * mean_x + mean_y * mean_y + c1)
                       * (variance_x + variance_y + c2))
        score = numerator / np.maximum(denominator, 1.0e-12)
        pad = win_size // 2
        if score.shape[0] > 2 * pad and score.shape[1] > 2 * pad:
            score = score[pad:-pad, pad:-pad]
        values.append(float(np.mean(score)))
    return float(np.mean(values))


def _ssim(image: np.ndarray, reference: np.ndarray) -> float | None:
    # CI 环境按 requirements.txt 安装 scikit-image 时使用基线指定 API；
    # 本地缺少可选 wheel 时使用等价的 OpenCV/NumPy 后备实现。
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return _ssim_fallback(image, reference, win_size=7)
    if min(image.shape[:2]) < 7:
        return None
    return float(structural_similarity(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(reference, cv2.COLOR_BGR2RGB),
        channel_axis=-1, data_range=255, win_size=7,
    ))


def _delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """向量化 CIEDE2000，输入为 OpenCV float Lab（D65）。"""
    l1, a1, b1 = np.moveaxis(lab1.astype(np.float64), -1, 0)
    l2, a2, b2 = np.moveaxis(lab2.astype(np.float64), -1, 0)
    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_bar = (c1 + c2) / 2.0
    c_bar7 = c_bar ** 7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + 25.0 ** 7 + 1.0e-30)))
    ap1, ap2 = (1.0 + g) * a1, (1.0 + g) * a2
    cp1, cp2 = np.hypot(ap1, b1), np.hypot(ap2, b2)
    hp1 = np.mod(np.degrees(np.arctan2(b1, ap1)), 360.0)
    hp2 = np.mod(np.degrees(np.arctan2(b2, ap2)), 360.0)

    delta_l = l2 - l1
    delta_c = cp2 - cp1
    delta_h_angle = hp2 - hp1
    delta_h_angle = np.where(delta_h_angle > 180.0, delta_h_angle - 360.0,
                             delta_h_angle)
    delta_h_angle = np.where(delta_h_angle < -180.0, delta_h_angle + 360.0,
                             delta_h_angle)
    delta_h_angle = np.where(cp1 * cp2 == 0.0, 0.0, delta_h_angle)
    delta_h = 2.0 * np.sqrt(cp1 * cp2) * np.sin(np.radians(delta_h_angle) / 2.0)

    mean_l = (l1 + l2) / 2.0
    mean_c = (cp1 + cp2) / 2.0
    hue_sum = hp1 + hp2
    hue_diff = np.abs(hp1 - hp2)
    mean_h = np.where(cp1 * cp2 == 0.0, hue_sum,
                      np.where(hue_diff <= 180.0, hue_sum / 2.0,
                               np.where(hue_sum < 360.0,
                                        (hue_sum + 360.0) / 2.0,
                                        (hue_sum - 360.0) / 2.0)))
    t = (1.0 - 0.17 * np.cos(np.radians(mean_h - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * mean_h))
         + 0.32 * np.cos(np.radians(3.0 * mean_h + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * mean_h - 63.0)))
    delta_theta = 30.0 * np.exp(-((mean_h - 275.0) / 25.0) ** 2)
    rc = 2.0 * np.sqrt(mean_c ** 7 / (mean_c ** 7 + 25.0 ** 7 + 1.0e-30))
    sl = 1.0 + 0.015 * (mean_l - 50.0) ** 2 / np.sqrt(
        20.0 + (mean_l - 50.0) ** 2)
    sc = 1.0 + 0.045 * mean_c
    sh = 1.0 + 0.015 * mean_c * t
    rt = -np.sin(np.radians(2.0 * delta_theta)) * rc
    dl, dc, dh = delta_l / sl, delta_c / sc, delta_h / sh
    return np.sqrt(np.maximum(dl * dl + dc * dc + dh * dh + rt * dc * dh, 0.0))


def _delta_e(image: np.ndarray, reference: np.ndarray,
             region: np.ndarray) -> float | None:
    if not bool(np.any(region)):
        return None
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    lab_reference = cv2.cvtColor(rgb_reference, cv2.COLOR_RGB2LAB)
    delta = _delta_e_ciede2000(lab_image, lab_reference)
    return float(np.mean(delta[region]))


def _base_records() -> dict[str, dict[str, Any]]:
    unavailable = "缺少计算该指标所需的图像、GT 区域或参考图"
    units = {
        "skin_iou": "ratio", "skin_recall": "ratio",
        "variance_drop": "ratio", "epi": "ratio", "hf_ratio": "ratio",
        "exposure_gain": "ratio", "overexposure_ratio": "ratio",
        "psnr_input_db": "dB", "psnr_output_db": "dB", "psnr_delta_db": "dB",
        "ssim_input": "ratio", "ssim_output": "ratio", "ssim_delta": "ratio",
        "delta_e2000": "DeltaE2000",
    }
    return {name: _invalid(unit, unavailable) for name, unit in units.items()}


def compute_image_metrics(stage_input_bgr: np.ndarray | None,
                          stage_output_bgr: np.ndarray | None,
                          regions: Mapping[str, np.ndarray],
                          reference_bgr: np.ndarray | None,
                          predicted_skin_mask: np.ndarray | None,
                          cfg: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    """计算所有当前输入可支持的指标，不做文件 I/O。"""
    if not isinstance(regions, Mapping) or not isinstance(cfg, Mapping):
        raise ValueError("regions 与 cfg 必须是映射")
    records = _base_records()
    input_ok = _is_bgr(stage_input_bgr)
    output_ok = _is_bgr(stage_output_bgr)
    if input_ok and output_ok and stage_input_bgr.shape != stage_output_bgr.shape:
        input_ok = output_ok = False
    image = stage_output_bgr if output_ok else stage_input_bgr if input_ok else None
    if image is None:
        return records
    shape = image.shape[:2]
    skin_gt = _binary_mask(regions.get("skin_gt"), shape)
    feature_gt = _binary_mask(regions.get("face_feature_gt"), shape)
    predicted = _binary_mask(predicted_skin_mask, shape)

    if skin_gt is not None and predicted is not None and bool(np.any(skin_gt)):
        intersection = int(np.count_nonzero(skin_gt & predicted))
        union = int(np.count_nonzero(skin_gt | predicted))
        gt_count = int(np.count_nonzero(skin_gt))
        if union:
            records["skin_iou"] = _valid(intersection / union, "ratio")
        records["skin_recall"] = _valid(intersection / gt_count, "ratio")
    elif skin_gt is not None and not bool(np.any(skin_gt)):
        records["skin_iou"] = _invalid("ratio", "皮肤 GT 为空")
        records["skin_recall"] = _invalid("ratio", "皮肤 GT 为空")

    protocol = cfg.get("protocol", {}) or {}
    epsilon = float(protocol.get("epsilon", 1.0e-6))
    erode_kernel = int(protocol.get("skin_erode_kernel", 11))
    variance_window = int(protocol.get("variance_window", 7))
    hf_sigma = float(protocol.get("hf_sigma", 3.0))
    if (not np.isfinite(epsilon) or epsilon <= 0.0 or variance_window < 1
            or not np.isfinite(hf_sigma) or hf_sigma <= 0.0):
        raise ValueError("metrics.protocol 包含非法参数")

    if input_ok and output_ok:
        before_y = _luma(stage_input_bgr)
        after_y = _luma(stage_output_bgr)
        if skin_gt is not None and bool(np.any(skin_gt)):
            inner = _erode(skin_gt, erode_kernel)
            if bool(np.any(inner)):
                window = variance_window if variance_window % 2 == 1 else variance_window + 1
                mean_before = cv2.boxFilter(before_y, cv2.CV_32F, (window, window),
                                             borderType=cv2.BORDER_REFLECT_101)
                mean_after = cv2.boxFilter(after_y, cv2.CV_32F, (window, window),
                                            borderType=cv2.BORDER_REFLECT_101)
                var_before = cv2.boxFilter(before_y * before_y, cv2.CV_32F,
                                            (window, window),
                                            borderType=cv2.BORDER_REFLECT_101) - mean_before ** 2
                var_after = cv2.boxFilter(after_y * after_y, cv2.CV_32F,
                                           (window, window),
                                           borderType=cv2.BORDER_REFLECT_101) - mean_after ** 2
                vb = float(np.mean(np.maximum(var_before[inner], 0.0)))
                va = float(np.mean(np.maximum(var_after[inner], 0.0)))
                if vb >= epsilon:
                    records["variance_drop"] = _valid((vb - va) / max(vb, epsilon),
                                                       "ratio")
                else:
                    records["variance_drop"] = _invalid("ratio", "阶段输入皮肤方差过小")

                hp_before = before_y - cv2.GaussianBlur(
                    before_y, (0, 0), hf_sigma, hf_sigma,
                    borderType=cv2.BORDER_REFLECT_101)
                hp_after = after_y - cv2.GaussianBlur(
                    after_y, (0, 0), hf_sigma, hf_sigma,
                    borderType=cv2.BORDER_REFLECT_101)
                rms_before = float(np.sqrt(np.mean(hp_before[inner] ** 2)))
                rms_after = float(np.sqrt(np.mean(hp_after[inner] ** 2)))
                if rms_before >= epsilon:
                    records["hf_ratio"] = _valid(rms_after / max(rms_before, epsilon),
                                                  "ratio")
                else:
                    records["hf_ratio"] = _invalid("ratio", "阶段输入皮肤高频能量过小")

            skin_before = float(np.mean(before_y[skin_gt]))
            skin_after = float(np.mean(after_y[skin_gt]))
            if skin_before < 90.0:
                records["exposure_gain"] = _valid(
                    (skin_after - skin_before) / max(skin_before, 1.0), "ratio")
            else:
                records["exposure_gain"] = _invalid("ratio", "输入不属于平均 Y<90 的欠曝样本")
            records["overexposure_ratio"] = _valid(
                float(np.mean(after_y[skin_gt] > 250.0)), "ratio")

        if feature_gt is not None and bool(np.any(feature_gt)):
            gx_before = cv2.Sobel(before_y, cv2.CV_32F, 1, 0, ksize=3,
                                  borderType=cv2.BORDER_REFLECT_101)
            gy_before = cv2.Sobel(before_y, cv2.CV_32F, 0, 1, ksize=3,
                                  borderType=cv2.BORDER_REFLECT_101)
            gx_after = cv2.Sobel(after_y, cv2.CV_32F, 1, 0, ksize=3,
                                 borderType=cv2.BORDER_REFLECT_101)
            gy_after = cv2.Sobel(after_y, cv2.CV_32F, 0, 1, ksize=3,
                                 borderType=cv2.BORDER_REFLECT_101)
            gradient_before = cv2.magnitude(gx_before, gy_before)[feature_gt]
            gradient_after = cv2.magnitude(gx_after, gy_after)[feature_gt]
            denominator = float(np.sum(np.maximum(gradient_before, gradient_after)))
            if denominator >= 1.0e-8:
                records["epi"] = _valid(
                    float(np.sum(np.minimum(gradient_before, gradient_after))) /
                    (denominator + 1.0e-8), "ratio")
            else:
                records["epi"] = _invalid("ratio", "五官区域梯度能量过小")

    reference_ok = (_is_bgr(reference_bgr) and reference_bgr.shape == image.shape)
    if reference_ok and input_ok and output_ok:
        psnr_input = _psnr(stage_input_bgr, reference_bgr)
        psnr_output = _psnr(stage_output_bgr, reference_bgr)
        if psnr_input is not None:
            records["psnr_input_db"] = _valid(psnr_input, "dB")
        else:
            records["psnr_input_db"] = _invalid("dB", "输入与参考完全相同，PSNR 为无穷大")
        if psnr_output is not None:
            records["psnr_output_db"] = _valid(psnr_output, "dB")
        else:
            records["psnr_output_db"] = _invalid("dB", "输出与参考完全相同，PSNR 为无穷大")
        if psnr_input is not None and psnr_output is not None:
            records["psnr_delta_db"] = _valid(psnr_output - psnr_input, "dB")

        ssim_input = _ssim(stage_input_bgr, reference_bgr)
        ssim_output = _ssim(stage_output_bgr, reference_bgr)
        if ssim_input is not None:
            records["ssim_input"] = _valid(ssim_input, "ratio")
        if ssim_output is not None:
            records["ssim_output"] = _valid(ssim_output, "ratio")
        if ssim_input is not None and ssim_output is not None:
            records["ssim_delta"] = _valid(ssim_output - ssim_input, "ratio")

        if skin_gt is not None and bool(np.any(skin_gt)):
            delta_e = _delta_e(stage_output_bgr, reference_bgr, skin_gt)
            if delta_e is not None:
                records["delta_e2000"] = _valid(delta_e, "DeltaE2000")
    return records


def _score_values(value: Any, minimum: int, maximum: int) -> list[int]:
    if isinstance(value, np.ndarray):
        candidates = value.ravel().tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        candidates = list(value)
    else:
        candidates = [value]
    result = []
    for candidate in candidates:
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float, np.number)):
            continue
        number = float(candidate)
        if np.isfinite(number) and number.is_integer() and minimum <= number <= maximum:
            result.append(int(number))
    return result


def summarize_survey(responses: Sequence[Mapping[str, Any]],
                     cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """对匿名问卷做去身份化汇总，不保留或返回原始答卷。"""
    if (not isinstance(responses, Sequence)
            or isinstance(responses, (str, bytes, bytearray))
            or not isinstance(cfg, Mapping)):
        raise ValueError("responses 必须是答卷映射序列，cfg 必须是映射")
    survey_cfg = cfg.get("survey", {}) or {}
    questions = survey_cfg.get("questions", [
        "naturalness", "texture_preservation", "facial_detail", "skin_tone",
    ])
    if (not isinstance(questions, Sequence) or isinstance(questions, (str, bytes))
            or not questions or any(not isinstance(item, str) or not item for item in questions)):
        raise ValueError("metrics.survey.questions 必须是非空字符串列表")
    minimum = int(survey_cfg.get("score_min", 1))
    maximum = int(survey_cfg.get("score_max", 5))
    required = int(survey_cfg.get("minimum_valid_responses", 20))
    if minimum >= maximum or required < 1:
        raise ValueError("metrics.survey 评分范围或有效问卷门槛非法")

    collected: dict[str, list[int]] = {question: [] for question in questions}
    present_by_question = {question: 0 for question in questions}
    valid_responses = 0
    for response in responses:
        if not isinstance(response, Mapping):
            continue
        nested = response.get("scores", {})
        nested = nested if isinstance(nested, Mapping) else {}
        complete = True
        for question in questions:
            raw = response.get(question, nested.get(question))
            scores = _score_values(raw, minimum, maximum)
            if scores:
                collected[question].extend(scores)
                present_by_question[question] += 1
            else:
                complete = False
        if complete:
            valid_responses += 1

    question_summary: dict[str, Any] = {}
    total = len(responses)
    for question in questions:
        values = collected[question]
        distribution = {str(score): values.count(score)
                        for score in range(minimum, maximum + 1)}
        question_summary[question] = {
            "valid_sample_count": len(values),
            "missing_response_count": total - present_by_question[question],
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "distribution": distribution,
        }
    return {
        "valid_responses": valid_responses,
        "minimum_required": required,
        "meets_minimum": valid_responses >= required,
        "questions": question_summary,
    }


__all__ = ["compute_image_metrics", "summarize_survey"]
