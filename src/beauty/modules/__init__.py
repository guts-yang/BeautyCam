"""modules 包：M1-M6 纯计算算法模块。

模块之间禁止互相 import；只依赖 core.pipeline 传入的数组与各自配置。
文件所有权：M1-M4 归 B，M5/M6 归 C（见 docs/baseline/arch.md 4.2）。
本包 __init__ 不聚合导入具体模块，避免形成隐式模块间耦合。
"""

__all__ = [
    "skin_detect",
    "exposure",
    "smooth",
    "freq_enhance",
    "restore",
    "color",
]