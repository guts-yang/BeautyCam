# -*- coding: utf-8 -*-
"""core.config —— 配置加载、合并与校验（A 独占）。

职责（对齐 docs/baseline/arch.md 7.1）：
- 从 config/ 加载 pipeline.yaml 与各模块 YAML；
- 校验必需键、类型与范围，报错须指出配置文件与键路径；
- 计算 config_id（合并后有效配置规范化序列化的 SHA-256 前 12 位十六进制）。

配置优先级：代码默认值 < YAML 配置 < 本次 CLI/UI 显式覆盖。
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from typing import Any

import yaml

# 仓库根目录下的配置目录（以本文件位置回溯，不依赖当前工作目录）。
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CONFIG_DIR = os.path.join(_REPO_ROOT, "config")

# 模块名到配置文件的映射（顺序即冻结接口中出现的模块顺序）。
MODULE_CONFIG_FILES = {
    "skin_detect": "skin_detect.yaml",
    "exposure": "exposure.yaml",
    "smooth": "smooth.yaml",
    "freq_enhance": "freq_enhance.yaml",
    "restore": "restore.yaml",
    "color": "color.yaml",
    "metrics": "metrics.yaml",
}


class ConfigError(ValueError):
    """配置校验失败，消息须包含配置文件与键路径。"""


def _load_yaml(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise ConfigError(f"配置文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件根节点必须是映射：{path}")
    return data


def load_pipeline_config(path: str | None = None) -> dict[str, Any]:
    """加载 pipeline.yaml（固定顺序、阶段开关、运行模式、路径与 I/O 选项）。"""
    return _load_yaml(path or os.path.join(CONFIG_DIR, "pipeline.yaml"))


def load_module_config(module_name: str) -> dict[str, Any]:
    """按模块名加载其专属 YAML。"""
    if module_name not in MODULE_CONFIG_FILES:
        raise ConfigError(f"未知模块：{module_name}")
    return _load_yaml(os.path.join(CONFIG_DIR, MODULE_CONFIG_FILES[module_name]))


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """浅层合并，override 覆盖 base；值为 None 时保留 base。"""
    merged = dict(base)
    for key, value in override.items():
        if value is not None:
            merged[key] = value
    return merged


def resolve_config(pipeline_cfg: Mapping[str, Any],
                   module_cfgs: Mapping[str, Mapping[str, Any]],
                   overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """合并流水线配置、模块配置与本次运行的显式覆盖。"""
    resolved: dict[str, Any] = {"pipeline": dict(pipeline_cfg),
                                "modules": {k: dict(v) for k, v in module_cfgs.items()},
                                "overrides": dict(overrides or {})}
    resolved["pipeline"] = _merge(resolved["pipeline"], resolved["overrides"].get("pipeline", {}))
    for name, patch in resolved["overrides"].get("modules", {}).items():
        if name in resolved["modules"]:
            resolved["modules"][name] = _merge(resolved["modules"][name], patch)
    return resolved


def compute_config_id(resolved: Mapping[str, Any]) -> str:
    """计算合并后有效配置的 12 位十六进制摘要。

    排除输入/输出路径、用户目录与凭据，避免泄露个人路径（arch 7.2）。
    """
    safe = {
        "pipeline": {k: v for k, v in resolved.get("pipeline", {}).items()
                     if k not in {"input_path", "output_path", "input_dir", "output_dir"}},
        "modules": resolved.get("modules", {}),
    }
    canonical = json.dumps(safe, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


__all__ = [
    "CONFIG_DIR",
    "MODULE_CONFIG_FILES",
    "ConfigError",
    "load_pipeline_config",
    "load_module_config",
    "resolve_config",
    "compute_config_id",
]