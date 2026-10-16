# -*- coding: utf-8 -*-
"""io.writer —— 图像、CSV 与结果持久化（A 独占）。

约束（对齐 docs/baseline/arch.md 7.3）：
- metrics.csv 固定字段：run_id,sample_id,stage,metric_name,value,unit,valid,not_applicable_reason,config_id
- failures.csv 固定字段：run_id,input_relative_path,stage,error_type,error_message
- CSV 使用 UTF-8、LF 换行，字段顺序固定。
"""

from __future__ import annotations

import csv
import os

import cv2
import numpy as np

# 冻结的 CSV 字段顺序（契约基线，测试会断言）。
METRICS_FIELDS = [
    "run_id", "sample_id", "stage", "metric_name",
    "value", "unit", "valid", "not_applicable_reason", "config_id",
]
FAILURES_FIELDS = [
    "run_id", "input_relative_path", "stage", "error_type", "error_message",
]


class ImageWriteError(IOError):
    """图像写入失败。"""


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def write_image(path: str, image_bgr: np.ndarray) -> None:
    """写入图像（PNG 等由扩展名决定）。"""
    if image_bgr is None or image_bgr.size == 0:
        raise ImageWriteError(f"待写入图像为空：{path}")
    _ensure_dir(path)
    ok = cv2.imwrite(path, image_bgr)
    if not ok:
        raise ImageWriteError(f"图像写入失败：{path}")


def write_csv(path: str, rows: list[dict], fields: list[str]) -> None:
    """按固定字段顺序写入 CSV，使用 UTF-8 + LF。"""
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, lineterminator="\n",
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _format_cell(row.get(k, "")) for k in fields})


def _format_cell(value) -> str:
    """有效浮点值以十进制格式保留 6 位；None 写为空串。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        return f"{value:.6f}"
    return str(value)


def write_metrics_csv(path: str, rows: list[dict]) -> None:
    """写 metrics.csv（固定九字段）。"""
    write_csv(path, rows, METRICS_FIELDS)


def write_failures_csv(path: str, rows: list[dict]) -> None:
    """写 failures.csv（固定五字段）。"""
    write_csv(path, rows, FAILURES_FIELDS)


__all__ = [
    "METRICS_FIELDS", "FAILURES_FIELDS",
    "ImageWriteError", "write_image", "write_csv",
    "write_metrics_csv", "write_failures_csv",
]