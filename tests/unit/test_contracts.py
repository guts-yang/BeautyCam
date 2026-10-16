# -*- coding: utf-8 -*-
"""契约守卫测试（A 维护）——把冻结项变成可执行断言。

覆盖（对齐 docs/baseline/arch.md 第 5、8 节与设计方案第 6 章）：
- 7 个模块签名可 import 且调用形状/类型契约成立；
- 掩码格式为 uint8 0/255；
- metrics.csv / failures.csv 字段顺序固定；
- 配置 schema 关键键存在。

约束：只用内存合成数据，不依赖 CelebAMask-HQ 或本机文件。
"""

from __future__ import annotations

import inspect

import numpy as np

from beauty.core.contracts import PipelineResult, StageStatus  # noqa: F401
from beauty.core.config import MODULE_CONFIG_FILES, compute_config_id
from beauty.io.writer import FAILURES_FIELDS, METRICS_FIELDS
from beauty.metrics.metrics import compute_image_metrics, summarize_survey
from beauty.modules import color, exposure, freq_enhance, restore, skin_detect, smooth


def _synthetic_bgr(h: int = 32, w: int = 24) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# --- 签名契约 -------------------------------------------------------------

def test_seven_module_signatures_importable():
    """7 个冻结签名可从冻结模块导入（含 M7 两个接口）。"""
    assert callable(skin_detect.detect_skin)
    assert callable(exposure.process)
    assert callable(smooth.process)
    assert callable(freq_enhance.process)
    assert callable(color.process)
    assert callable(restore.restore_motion_blur)
    assert callable(compute_image_metrics)
    assert callable(summarize_survey)


def test_detect_skin_signature_params():
    """M1 签名参数名为 (image_bgr, cfg)，冻结后不得私改。"""
    params = list(inspect.signature(skin_detect.detect_skin).parameters)
    assert params == ["image_bgr", "cfg"]


def test_restore_signature_params():
    """M5 签名参数名为 (image_bgr, cfg)，且不依赖 mask。"""
    params = list(inspect.signature(restore.restore_motion_blur).parameters)
    assert params == ["image_bgr", "cfg"]


# --- 掩码与图像契约 -------------------------------------------------------

def test_detect_skin_mask_contract():
    """M1 输出同尺寸 uint8 掩码，取值仅 0/255。"""
    img = _synthetic_bgr()
    mask = skin_detect.detect_skin(img, {})
    assert mask.shape == img.shape[:2]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_transform_modules_return_new_bgr():
    """M2/M3/M4/M6 返回新的 BGR uint8 图像且不原地修改输入。"""
    img = _synthetic_bgr()
    mask = np.full(img.shape[:2], 255, dtype=np.uint8)
    for mod in (exposure, smooth, freq_enhance, color):
        original = img.copy()
        out = mod.process(img, mask, {})
        assert out.shape == img.shape and out.dtype == np.uint8
        assert out is not img
        assert np.array_equal(img, original), f"{mod.__name__} 原地修改了输入"


def test_frequency_diagnostics_contract():
    """M4 诊断接口返回 two 张同尺寸 BGR uint8 图且不修改输入。"""
    img = _synthetic_bgr()
    original = img.copy()
    diag = freq_enhance.get_frequency_diagnostics(img, {})
    assert set(diag) >= {"low_frequency", "high_frequency"}
    for vis in diag.values():
        assert vis.shape == img.shape and vis.dtype == np.uint8
    assert np.array_equal(img, original)


def test_restore_returns_new_bgr():
    """M5 返回新的 BGR uint8 图像，不原地修改输入。"""
    img = _synthetic_bgr()
    original = img.copy()
    out = restore.restore_motion_blur(img, {})
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(img, original)


# --- 空掩码安全跳过 -------------------------------------------------------

def test_empty_mask_does_not_raise():
    """空掩码（全零）时图像变换模块不崩溃。"""
    img = _synthetic_bgr()
    empty = np.zeros(img.shape[:2], dtype=np.uint8)
    for mod in (exposure, smooth, freq_enhance, color):
        out = mod.process(img, empty, {})
        assert out.shape == img.shape


# --- CSV 字段顺序契约 -----------------------------------------------------

def test_metrics_csv_field_order():
    """metrics.csv 九字段顺序冻结。"""
    assert METRICS_FIELDS == [
        "run_id", "sample_id", "stage", "metric_name",
        "value", "unit", "valid", "not_applicable_reason", "config_id",
    ]


def test_failures_csv_field_order():
    """failures.csv 五字段顺序冻结。"""
    assert FAILURES_FIELDS == [
        "run_id", "input_relative_path", "stage", "error_type", "error_message",
    ]


# --- 配置契约 -------------------------------------------------------------

def test_module_config_schema_keys():
    """7 个模块配置文件均已登记（schema 契约）。"""
    assert set(MODULE_CONFIG_FILES) == {
        "skin_detect", "exposure", "smooth", "freq_enhance",
        "restore", "color", "metrics",
    }


def test_config_id_is_12_hex():
    """config_id 为 12 位十六进制摘要。"""
    cid = compute_config_id({"pipeline": {"mode": "process"}, "modules": {}})
    assert len(cid) == 12 and all(c in "0123456789abcdef" for c in cid)


def test_config_id_ignores_paths():
    """config_id 排除输入/输出路径，避免泄露个人路径。"""
    a = compute_config_id({"pipeline": {"mode": "process", "input_path": "/home/x"}, "modules": {}})
    b = compute_config_id({"pipeline": {"mode": "process", "input_path": "/tmp/y"}, "modules": {}})
    assert a == b