# -*- coding: utf-8 -*-
"""app.py —— Streamlit 薄入口（A 独占）。

只调用 core 层。约束（对齐 arch 第 6 节、design 2.5）：
- 上传原图仅在本地进程内存处理，不自动持久化，须用户显式导出；
- 无 GT 时不显示参考型客观质量分数，明确说明哪些指标不可评估。
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import streamlit as st

from beauty.core.config import load_module_config, load_pipeline_config
from beauty.core.pipeline import run_pipeline

_MODULES = ["skin_detect", "exposure", "smooth", "freq_enhance", "restore", "color", "metrics"]


def _decode(uploaded) -> np.ndarray | None:
    data = np.frombuffer(uploaded.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main() -> None:
    st.set_page_config(page_title="焕颜相机 BeautyCam", layout="wide")
    st.title("焕颜相机 BeautyCam")
    st.caption("上传人脸照片，调整参数查看美颜效果。原图仅在本地内存处理，不会自动保存。")

    uploaded = st.file_uploader("上传图片（JPG / JPEG / PNG / BMP）",
                                type=["jpg", "jpeg", "png", "bmp"])
    smooth_strength = st.slider("磨皮强度", 0.0, 1.0, 0.5, 0.05)
    exposure_gain = st.slider("曝光改善", 0.0, 1.0, 0.3, 0.05)
    color_temp = st.slider("色温调整", -1.0, 1.0, 0.0, 0.1)

    if uploaded is None:
        st.info("请先上传一张图片。")
        return

    image = _decode(uploaded)
    if image is None:
        st.error("无法解码该文件，请确认是有效的 JPG/JPEG/PNG/BMP 图片。")
        return

    pipeline_cfg = load_pipeline_config()
    module_cfgs = {n: load_module_config(n) for n in _MODULES}
    # UI 覆盖只作用于当前 run，不回写 YAML（arch 第 6 节）。
    module_cfgs["smooth"] = {**module_cfgs["smooth"], "strength": smooth_strength}
    module_cfgs["exposure"] = {**module_cfgs["exposure"], "gain": exposure_gain}
    module_cfgs["color"] = {**module_cfgs["color"], "temperature": color_temp}

    if st.button("开始处理"):
        with st.spinner("正在处理…"):
            result = run_pipeline(image, pipeline_cfg, module_cfgs, mode="preview")
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("原图")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col_right:
            st.subheader("处理后")
            if result.succeeded and result.final_bgr is not None:
                st.image(cv2.cvtColor(result.final_bgr, cv2.COLOR_BGR2RGB),
                         use_container_width=True)
            else:
                st.error(result.status_text())

        st.caption("注意：当前无 GT 参考，IoU/ΔE2000/PSNR/SSIM 等参考型客观指标不可评估，"
                   "界面不显示无参考依据的质量分数。")
        st.caption(f"处理状态：{result.status_text()}｜当前参数：磨皮 {smooth_strength}、"
                   f"曝光 {exposure_gain}、色温 {color_temp}")

        if result.succeeded and result.final_bgr is not None:
            ok, buf = cv2.imencode(".png", result.final_bgr)
            if ok:
                st.download_button("显式导出结果图", data=io.BytesIO(buf.tobytes()),
                                   file_name="beautycam_result.png", mime="image/png")


if __name__ == "__main__":
    main()