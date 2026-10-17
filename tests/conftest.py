# -*- coding: utf-8 -*-
"""tests 共享 fixture（A 维护）。"""

from __future__ import annotations

import pytest

from beauty.core.config import load_module_config

# 七个模块的注册顺序（与 config/*.yaml 一一对应）。
MODULES = [
    "skin_detect", "exposure", "smooth", "freq_enhance",
    "restore", "color", "metrics",
]


@pytest.fixture
def module_cfgs() -> dict:
    """全部模块的默认配置（读取 config/*.yaml），供各层级测试复用。"""
    return {name: load_module_config(name) for name in MODULES}
