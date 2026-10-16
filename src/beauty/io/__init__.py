"""io 层：图像/GT 读取与产物落盘（A 独占）。

只做输入/输出，不承载算法；不被算法模块调用。
"""

from beauty.io.reader import read_image, read_mask
from beauty.io.writer import write_csv, write_image

__all__ = ["read_image", "read_mask", "write_image", "write_csv"]