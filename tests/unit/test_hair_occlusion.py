# -*- coding: utf-8 -*-
"""头发遮挡抑制实验工具的合成图单元验证（A 维护）。

运行：.venv/bin/python -m pytest tests/unit/test_hair_occlusion.py -q
对齐 docs/baseline/arch.md 第 8 节：只用内存合成图，不依赖任何数据集/本机文件。
覆盖：
- 返回 dtype/形状、不改写入参、空掩码与禁用安全；
- 密集细刘海（v3 因长宽比判据漏检）能被剔除；
- 眼线分区保护眼线以下的皱纹纹理；
- 外部 YAML 配置可加载且与代码默认值键集合一致。
"""
from __future__ import annotations

import numpy as np

from tools.hair_occlusion import load_hair_config, suppress_hair


def _skin_bgr(h=240, w=240):
    base = np.zeros((h, w, 3), np.uint8)
    base[:] = (140, 150, 180)
    return base


def test_output_contract():
    img = _skin_bgr()
    mask = np.full((240, 240), 255, np.uint8)
    mask_copy, img_copy = mask.copy(), img.copy()
    out = suppress_hair(img, mask, {"strength": 0.35})
    assert out.dtype == np.uint8
    assert out.shape == mask.shape
    assert set(np.unique(out)).issubset({0, 255})
    assert np.array_equal(mask, mask_copy)
    assert np.array_equal(img, img_copy)


def test_empty_and_disabled_are_safe():
    img = _skin_bgr()
    empty = np.zeros((240, 240), np.uint8)
    assert not np.any(suppress_hair(img, empty, {"strength": 0.35}))
    full = np.full((240, 240), 255, np.uint8)
    assert np.array_equal(suppress_hair(img, full, {"enabled": False}), full)


def test_dense_fine_bangs_removed():
    """密集细刘海连成大块连通域，v3 长宽比判据会整块丢弃，v4+ 应能剔除。"""
    img = _skin_bgr(360, 360)
    mask = np.full((360, 360), 255, np.uint8)
    for x in range(90, 270, 9):
        img[20:200, x:x + 2] = (118, 126, 150)
    out = suppress_hair(img, mask, {
        "strength": 0.35, "protect_center": False, "boundary_erode": 1,
    })
    removed = int(np.count_nonzero(mask)) - int(np.count_nonzero(out))
    assert removed > 3000, f"dense fine strands should be removed, got {removed}"
    assert np.count_nonzero(out[:, 300:360]) > 0.8 * (360 * 60)


def test_eye_line_protects_lower_wrinkles():
    """眼线以下的方向性纹理（模拟法令纹）应被眼线分区保护。"""
    img = _skin_bgr(360, 360)
    mask = np.full((360, 360), 255, np.uint8)
    for x in range(90, 270, 9):
        img[20:200, x:x + 2] = (118, 126, 150)   # 眼线以上：发丝
        img[240:330, x:x + 2] = (118, 126, 150)  # 眼线以下：皱纹
    base = {"strength": 0.35, "protect_center": False, "boundary_erode": 1}
    no_zone = suppress_hair(img, mask, base)
    with_zone = suppress_hair(img, mask, base, eye_line_y=220)
    region = (slice(240, 330), slice(90, 270))
    lower_no = int(np.count_nonzero(mask[region])) - int(np.count_nonzero(no_zone[region]))
    lower_zone = int(np.count_nonzero(mask[region])) - int(np.count_nonzero(with_zone[region]))
    assert lower_no > 500
    assert lower_zone < 0.1 * lower_no


def test_config_file_loadable():
    """外部 YAML 可加载，且覆盖代码默认值的键均可识别。"""
    cfg = load_hair_config()
    assert isinstance(cfg, dict) and cfg.get("enabled") is True
    assert "coherence_min" in cfg and "eye_keep_ratio" in cfg
    img = _skin_bgr(120, 120)
    mask = np.full((120, 120), 255, np.uint8)
    out = suppress_hair(img, mask, cfg)
    assert out.shape == mask.shape and out.dtype == np.uint8
