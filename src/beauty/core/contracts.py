# -*- coding: utf-8 -*-
"""core.contracts —— core 与入口（CLI/UI）之间的结果类型（A 独占）。

契约基线（对齐 docs/baseline/arch.md 5.1）：
- 本模块只定义 core 与入口之间的数据结构；
- 算法模块（modules/metrics）禁止 import 本模块或用它互相传递状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class StageStatus(str, Enum):
    """单个流水线阶段的运行状态。"""

    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class StageRecord:
    """单个阶段的运行记录。"""

    stage: str
    status: StageStatus
    message: str = ""
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """一次运行的完整结果，供 CLI/UI 读取。

    字段含义：
    - run_id：本次运行标识；preview 模式下为 None（不创建持久化目录）。
    - config_id：合并后有效配置的 12 位十进制摘要（SHA-256 前 12 位十六进制）。
    - mode：运行模式，取值 preview / process / evaluation。
    - final_bgr：最终结果图；失败时为 None。
    - stages：各阶段记录（顺序与执行顺序一致）。
    - skin_mask：运行时皮肤掩码（M1 产出）；未产出时为 None。
    - metrics：M7 指标记录；preview 模式为空。
    - succeeded：整条流水线是否成功产出最终图。
    - error：失败时的可读原因。
"""

    run_id: str | None
    config_id: str
    mode: str
    final_bgr: np.ndarray | None = None
    stages: list[StageRecord] = field(default_factory=list)
    skin_mask: np.ndarray | None = None
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    succeeded: bool = False
    error: str = ""

    def status_text(self) -> str:
        """返回可直接展示给用户的运行状态文本。"""
        if self.succeeded:
            return f"处理成功（{self.mode}，run_id={self.run_id or 'preview'}）"
        return f"处理失败：{self.error or '未知错误'}"


@dataclass
class BatchOutcome:
    """一次批处理的汇总结果，供 CLI/UI 展示。

    字段含义：
    - run_id：本次批处理共享的运行标识（对齐 arch 7.3：一次批处理共享一个 run_id）。
    - total / succeeded / failed：输入总数、成功数、失败数。
    - failures：失败项记录，字段与 failures.csv 一致（已剔除本机绝对路径）。
    - messages：每个样本一行可展示文本。
    """

    run_id: str | None
    total: int
    succeeded: int
    failed: int
    failures: list[dict[str, Any]] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def status_text(self) -> str:
        """返回可直接展示给用户的批处理状态文本。"""
        return (f"批处理完成：成功 {self.succeeded}/{self.total}"
                f"（run_id={self.run_id or 'preview'}）")


__all__ = ["StageStatus", "StageRecord", "PipelineResult", "BatchOutcome"]