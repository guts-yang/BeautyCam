"""core 层：唯一流程编排与配置合并层（A 独占）。

- config.py：配置加载/合并/校验
- contracts.py：core 与入口之间的结果类型（PipelineResult）
- pipeline.py：唯一流水线编排点
"""

from beauty.core.contracts import PipelineResult, StageStatus

__all__ = ["PipelineResult", "StageStatus"]