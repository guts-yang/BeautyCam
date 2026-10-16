"""焕颜相机 BeautyCam 主包。

分层：core（编排/配置/契约）→ modules（M1-M6）/ metrics（M7）/ io。
依赖方向与文件所有权见 docs/baseline/arch.md 第 3、4 节。
"""

__all__ = ["core", "io", "modules", "metrics"]
__version__ = "0.1.0"