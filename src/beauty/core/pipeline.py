# -*- coding: utf-8 -*-
"""core.pipeline —— 唯一流水线编排点（A 独占）。

固定顺序（对齐 docs/baseline/arch.md 第 6 节）：
    输入 BGR 图
      → M5 运动模糊复原（restore_enabled=true 时；默认 false）
      → M1 肤色检测
      → M2 曝光改善
      → M3 空间域平滑
      → M4 频域增强（+ 可选 get_frequency_diagnostics）
      → M6 色彩/白平衡
      → M7 阶段评价（evaluation 模式）
      → io 落盘（process/evaluation 模式）

约束：算法模块彼此不互相 import，只有本文件按顺序调用；数组在进程内显式传递。
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import numpy as np

from beauty.core.config import compute_config_id, load_pipeline_config
from beauty.core.contracts import PipelineResult, StageRecord, StageStatus
from beauty.io.writer import write_failures_csv, write_image, write_metrics_csv
from beauty.metrics import metrics as metrics_mod
from beauty.modules import color, exposure, freq_enhance, restore, skin_detect, smooth

# 固定执行顺序（M5 受开关控制）。
_STAGE_ORDER = ["restore", "skin_detect", "exposure", "smooth", "freq_enhance", "color"]


def _new_run_id() -> str:
    """UTC 时间戳 + 随机短后缀。"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{uuid.uuid4().hex[:6]}"


def _stage_enabled(pipeline_cfg: Mapping[str, Any], stage: str) -> bool:
    """读取某阶段开关；M5（restore）默认关闭。"""
    stages = pipeline_cfg.get("stages", {}) or {}
    if stage == "restore":
        return bool(stages.get("restore_enabled", False))
    return bool(stages.get(f"{stage}_enabled", True))


def run_pipeline(image_bgr: np.ndarray,
                 pipeline_cfg: Mapping[str, Any],
                 module_cfgs: Mapping[str, Mapping[str, Any]],
                 mode: str = "preview",
                 output_root: str | None = None,
                 sample_id: str | None = None,
                 regions: Mapping[str, np.ndarray] | None = None,
                 reference_bgr: np.ndarray | None = None) -> PipelineResult:
    """按固定顺序执行流水线，返回 PipelineResult。

    mode 取 preview / process / evaluation：
    - preview：只在内存处理，不创建持久化目录；
    - process / evaluation：写结果图与运行记录。
    """
    if mode not in {"preview", "process", "evaluation"}:
        raise ValueError(f"未知运行模式：{mode}")

    resolved = {"pipeline": dict(pipeline_cfg), "modules": {k: dict(v) for k, v in module_cfgs.items()}}
    config_id = compute_config_id(resolved)
    run_id = _new_run_id() if mode in {"process", "evaluation"} else None
    result = PipelineResult(run_id=run_id, config_id=config_id, mode=mode)

    current = image_bgr.copy()
    mask: np.ndarray | None = None
    diagnostics: dict[str, np.ndarray] = {}
    failures: list[dict] = []

    def _cfg(name: str) -> Mapping[str, Any]:
        return module_cfgs.get(name, {}) or {}

    for stage in _STAGE_ORDER:
        if not _stage_enabled(pipeline_cfg, stage):
            result.stages.append(StageRecord(stage, StageStatus.SKIPPED, "阶段开关关闭"))
            continue
        try:
            if stage == "restore":
                current = restore.restore_motion_blur(current, _cfg("restore"))
            elif stage == "skin_detect":
                mask = skin_detect.detect_skin(current, _cfg("skin_detect"))
                if mask is not None and not bool(np.any(mask)):
                    result.stages.append(StageRecord(
                        stage, StageStatus.OK, "肤色掩码为空，后续模块将安全跳过肤色限定处理"))
                    continue
            elif stage == "exposure":
                current = exposure.process(current, mask, _cfg("exposure"))
            elif stage == "smooth":
                current = smooth.process(current, mask, _cfg("smooth"))
            elif stage == "freq_enhance":
                current = freq_enhance.process(current, mask, _cfg("freq_enhance"))
                diag_cfg = (pipeline_cfg.get("diagnostics", {}) or {})
                if bool(diag_cfg.get("frequency_enabled", False)):
                    diagnostics = dict(freq_enhance.get_frequency_diagnostics(current, _cfg("freq_enhance")))
            elif stage == "color":
                current = color.process(current, mask, _cfg("color"))
            result.stages.append(StageRecord(stage, StageStatus.OK, "完成"))
        except Exception as exc:  # 单阶段失败不伪造最终结果
            result.stages.append(StageRecord(stage, StageStatus.FAILED, str(exc)))
            failures.append({
                "run_id": run_id or "", "input_relative_path": sample_id or "",
                "stage": stage, "error_type": type(exc).__name__, "error_message": str(exc),
            })
            result.error = f"阶段 {stage} 失败：{exc}"
            _persist(result, current, diagnostics, failures, output_root, sample_id, regions, reference_bgr)
            return result

    result.final_bgr = current
    result.skin_mask = mask
    result.succeeded = True

    if mode == "evaluation":
        result.metrics = dict(metrics_mod.compute_image_metrics(
            image_bgr, current, regions or {}, reference_bgr, mask, _cfg("metrics")))

    _persist(result, current, diagnostics, failures, output_root, sample_id, regions, reference_bgr)
    return result


def _persist(result: PipelineResult, final_bgr: np.ndarray,
             diagnostics: Mapping[str, np.ndarray], failures: list[dict],
             output_root: str | None, sample_id: str | None,
             regions: Mapping[str, np.ndarray] | None,
             reference_bgr: np.ndarray | None) -> None:
    """按运行模式落盘。preview 不创建任何持久化目录。"""
    if result.mode == "preview" or not output_root or not result.run_id:
        return
    sample = sample_id or "sample"
    if result.succeeded and final_bgr is not None:
        out_path = os.path.join(output_root, "data", "output", result.run_id,
                                "samples", sample, "final.png")
        write_image(out_path, final_bgr)
        for name, img in diagnostics.items():
            write_image(os.path.join(output_root, "data", "output", result.run_id,
                                     "samples", sample, "diagnostics", "frequency", f"{name}.png"), img)
    if failures:
        write_failures_csv(os.path.join(output_root, "results", result.run_id, "failures.csv"), failures)
    if result.mode == "evaluation" and result.metrics:
        rows = _metrics_to_rows(result)
        write_metrics_csv(os.path.join(output_root, "results", result.run_id, "metrics.csv"), rows)


def _metrics_to_rows(result: PipelineResult) -> list[dict]:
    """把指标记录展开为 metrics.csv 长表行。"""
    rows: list[dict] = []
    for stage, record in _iter_metric_records(result.metrics):
        for metric_name, metric in record.items():
            rows.append({
                "run_id": result.run_id or "",
                "sample_id": "",
                "stage": stage,
                "metric_name": metric_name,
                "value": metric.get("value") if isinstance(metric, Mapping) else metric,
                "unit": metric.get("unit", "") if isinstance(metric, Mapping) else "",
                "valid": metric.get("valid", True) if isinstance(metric, Mapping) else True,
                "not_applicable_reason": metric.get("not_applicable_reason", "") if isinstance(metric, Mapping) else "",
                "config_id": result.config_id,
            })
    return rows


def _iter_metric_records(records: Mapping[str, Any]):
    """兼容 metrics 返回 {metric_name: record} 或 {stage: {metric_name: record}} 两种形状。"""
    for key, value in records.items():
        if isinstance(value, Mapping) and any(
                isinstance(v, Mapping) and "valid" in v for v in value.values()):
            yield key, value
        else:
            yield "", {key: value}


__all__ = ["run_pipeline"]