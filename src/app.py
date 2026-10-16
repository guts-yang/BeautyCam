# -*- coding: utf-8 -*-
"""app.py —— Streamlit 薄入口（A 独占）。

本文件在 PR#1（E0 地基）阶段仅提供可加载的入口骨架；完整上传/处理/导出交互
在 PR#2（E1，I-07）中接入 core.pipeline。

约束（对齐 arch 第 6 节、design 2.5）：
- 上传原图仅在本地进程内存处理，不自动持久化，须用户显式导出；
- 无 GT 时不显示参考型客观质量分数。
"""

from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="焕颜相机 BeautyCam", layout="wide")
    st.title("焕颜相机 BeautyCam")
    st.caption("上传人脸照片，调整参数查看美颜效果。原图仅在本地内存处理，不会自动保存。")
    st.info("演示界面骨架已就位，处理交互将于 E1 阶段接入 core.pipeline。")


if __name__ == "__main__":
    main()