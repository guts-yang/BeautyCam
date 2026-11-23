# -*- coding: utf-8 -*-
"""app.py —— Streamlit 薄入口（A 独占）。

桌面修图软件式布局：左侧工具箱 / 中央预览画布 / 右侧参数面板。
参数调节即实时重算预览（结果以 file_id + 参数组合缓存，避免重复处理）。

只调用 core 层。约束（对齐 arch 第 6 节、design 2.5）：
- 上传原图仅在本地进程内存处理，不自动持久化，须用户显式导出；
- UI 覆盖只作用于当前 run，不回写 YAML；
- 无 GT 时不显示参考型客观质量分数。
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import streamlit as st

from beauty.core.config import load_module_config, load_pipeline_config
from beauty.core.pipeline import run_pipeline

_MODULES = ["skin_detect", "exposure", "smooth", "freq_enhance", "restore", "color", "metrics"]
_MAX_UPLOAD_MB = 20

_STAGE_LABELS = {
    "restore": "运动模糊复原",
    "skin_detect": "肤色检测",
    "exposure": "曝光改善",
    "smooth": "磨皮平滑",
    "freq_enhance": "频域增强",
    "color": "色彩调整",
    "metrics": "质量指标",
}


def _decode(uploaded) -> np.ndarray | None:
    data = np.frombuffer(uploaded.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


@st.cache_resource(show_spinner=False)
def _load_configs() -> tuple[dict, dict[str, dict]]:
    pipeline_cfg = load_pipeline_config()
    module_cfgs = {n: load_module_config(n) for n in _MODULES}
    return pipeline_cfg, module_cfgs


def _inject_css() -> None:
    """局部样式：仅作用于自定义 bc-* 类与稳定属性选择器，全部引用 design Token。"""
    st.markdown(
        """
<style>
:root{
  --bc-bg-base:#FAF7F4;
  --bc-bg-surface:#FFFFFF;
  --bc-text-primary:#2B2622;
  --bc-text-secondary:#6B6259;
  --bc-accent:#E8A598;
  --bc-accent-hover:#D98E80;
  --bc-accent-active:#C57A6C;
  --bc-text-on-accent:#2B2622;
  --bc-border:#E6DFD8;
  --bc-border-strong:#8A8178;
  --bc-success:#3E7857;
  --bc-danger:#B23A48;
}
.bc-brand{display:flex;align-items:center;gap:12px;padding:4px 4px 14px 4px;
  border-bottom:1px solid var(--bc-border);margin-bottom:14px;}
.bc-logo{width:46px;height:46px;flex:0 0 46px;border-radius:10px;background:var(--bc-accent);
  color:var(--bc-text-on-accent);display:flex;align-items:center;justify-content:center;
  font-size:24px;line-height:32px;font-weight:700;}
.bc-brand h1{font-size:24px;line-height:32px;font-weight:700;color:var(--bc-text-primary);margin:0;}
.bc-brand p{font-size:14px;line-height:22px;font-weight:400;color:var(--bc-text-secondary);margin:2px 0 0;}
.bc-panel-title{font-size:20px;line-height:28px;font-weight:600;color:var(--bc-text-primary);margin:0 0 4px;}
.bc-tool-title{font-size:16px;line-height:24px;font-weight:500;color:var(--bc-text-primary);margin:0 0 8px;}
.bc-cap{font-size:12px;line-height:18px;font-weight:400;color:var(--bc-text-secondary);margin:0;}
.bc-img-label{font-size:16px;line-height:24px;font-weight:500;color:var(--bc-text-primary);
  text-align:center;margin:0 0 8px;}
.bc-step{display:flex;align-items:center;justify-content:space-between;gap:8px;
  font-size:14px;line-height:22px;font-weight:400;color:var(--bc-text-primary);padding:2px 0;}
.bc-badge{display:inline-block;padding:1px 8px;border-radius:6px;font-size:12px;line-height:18px;
  font-weight:600;white-space:nowrap;}
.bc-badge-ok{background:var(--bc-success);color:#FFFFFF;}
.bc-badge-skip{background:var(--bc-bg-surface);color:var(--bc-text-secondary);
  border:1px solid var(--bc-border-strong);}
.bc-badge-fail{background:var(--bc-danger);color:#FFFFFF;}
.bc-empty{padding:40px 24px;text-align:center;border:1px dashed var(--bc-border-strong);
  border-radius:10px;background:var(--bc-bg-surface);}
.bc-empty p{font-size:16px;line-height:24px;font-weight:500;color:var(--bc-text-primary);margin:0 0 6px;}
.bc-empty span{font-size:14px;line-height:22px;font-weight:400;color:var(--bc-text-secondary);}
[data-testid="stButton"] button[kind="primary"],
[data-testid="stButton"] button[kind="primary"]:hover,
[data-testid="stButton"] button[kind="primary"]:active{
  background-color:var(--bc-accent);color:var(--bc-text-on-accent);
  border-color:var(--bc-border-strong);}
[data-testid="stButton"] button[kind="primary"]:hover{background-color:var(--bc-accent-hover);}
[data-testid="stButton"] button[kind="primary"]:active{background-color:var(--bc-accent-active);}
</style>
""",
        unsafe_allow_html=True,
    )


def _status_badge(status) -> str:
    value = getattr(status, "value", str(status))
    if value == "ok":
        return '<span class="bc-badge bc-badge-ok">完成</span>'
    if value == "skipped":
        return '<span class="bc-badge bc-badge-skip">跳过</span>'
    return '<span class="bc-badge bc-badge-fail">失败</span>'


def main() -> None:
    st.set_page_config(page_title="焕颜相机 BeautyCam", layout="wide")
    _inject_css()

    st.markdown(
        '<div class="bc-brand">'
        '<div class="bc-logo">焕</div>'
        '<div><h1>焕颜相机 BeautyCam</h1>'
        '<p>人像美颜工作台 · 原图仅在本地内存处理，导出需手动触发</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    col_tool, col_canvas, col_param = st.columns([1.2, 3.0, 1.8], gap="medium")

    with col_tool:
        st.markdown('<p class="bc-tool-title">工具箱</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "导入人像照片",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=False,
            help="JPG / JPEG / PNG / BMP，最大 20MB。仅在本机内存处理。",
        )
        if uploaded is not None:
            st.markdown(
                f'<p class="bc-cap">{uploaded.name} · {uploaded.size / 1024:.0f} KB</p>',
                unsafe_allow_html=True,
            )

    with col_param:
        st.markdown('<p class="bc-panel-title">美颜参数</p>', unsafe_allow_html=True)
        smooth_strength = st.slider("磨皮强度", 0.0, 1.0, 0.5, 0.05,
                                    disabled=(uploaded is None))
        exposure_gain = st.slider("曝光改善", 0.0, 1.0, 0.3, 0.05,
                                  disabled=(uploaded is None))
        color_temp = st.slider("色温调整", -1.0, 1.0, 0.0, 0.1,
                               disabled=(uploaded is None))
        st.divider()
        st.markdown('<p class="bc-cap">调节任意参数，中央画布实时更新处理结果。</p>',
                    unsafe_allow_html=True)

    with col_canvas:
        if "bc_cmp" not in st.session_state:
            st.session_state["bc_cmp"] = "side"
        _cmp_btns = st.columns(3)
        for _c, (_label, _key) in zip(_cmp_btns, [
            ("并排对比", "side"), ("仅原图", "orig"), ("仅结果", "result")
        ]):
            with _c:
                _selected = st.session_state["bc_cmp"] == _key
                if st.button(_label, key=f"bc_cmp_{_key}",
                             type="primary" if _selected else "secondary",
                             use_container_width=True):
                    st.session_state["bc_cmp"] = _key
        cmp_mode = st.session_state["bc_cmp"]

    result = None
    image = None
    if uploaded is not None:
        if uploaded.size > _MAX_UPLOAD_MB * 1024 * 1024:
            with col_canvas:
                st.error(f"图片过大（{uploaded.size / 1024 / 1024:.1f}MB），请上传不超过 {_MAX_UPLOAD_MB}MB 的图片。")
            return
        image = _decode(uploaded)
        if image is None:
            with col_canvas:
                st.error("无法解码该文件，请确认是有效的 JPG/JPEG/PNG/BMP 图片。")
            return

        params = (smooth_strength, exposure_gain, color_temp)
        file_key = getattr(uploaded, "file_id", None) or (uploaded.name, uploaded.size)
        cache_key = (file_key, params)
        cached = st.session_state.get("bc_result")
        if cached is None or cached["key"] != cache_key:
            pipeline_cfg, module_cfgs = _load_configs()
            module_cfgs = {k: dict(v) for k, v in module_cfgs.items()}
            module_cfgs["smooth"] = {**module_cfgs["smooth"], "strength": smooth_strength}
            module_cfgs["exposure"] = {**module_cfgs["exposure"], "gain": exposure_gain}
            module_cfgs["color"] = {**module_cfgs["color"], "temperature": color_temp}
            with st.spinner("正在美颜处理…"):
                result = run_pipeline(image, pipeline_cfg, module_cfgs, mode="preview")
            st.session_state["bc_result"] = {"key": cache_key, "result": result}
        else:
            result = cached["result"]

    with col_canvas:
        if uploaded is None or image is None:
            st.markdown(
                '<div class="bc-empty"><p>请从左侧工具箱导入一张人像照片</p>'
                '<span>导入后可调节磨皮、曝光与色温，实时查看美颜效果</span></div>',
                unsafe_allow_html=True,
            )
        elif result is not None and not result.succeeded:
            st.error(result.status_text())
        elif result is not None and result.final_bgr is not None:
            original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            beautified = cv2.cvtColor(result.final_bgr, cv2.COLOR_BGR2RGB)
            if cmp_mode == "side":
                a, b = st.columns(2)
                with a:
                    st.markdown('<p class="bc-img-label">原图</p>', unsafe_allow_html=True)
                    st.image(original, use_column_width=True)
                with b:
                    st.markdown('<p class="bc-img-label">处理后</p>', unsafe_allow_html=True)
                    st.image(beautified, use_column_width=True)
            elif cmp_mode == "orig":
                st.markdown('<p class="bc-img-label">原图</p>', unsafe_allow_html=True)
                st.image(original, use_column_width=True)
            else:
                st.markdown('<p class="bc-img-label">处理后</p>', unsafe_allow_html=True)
                st.image(beautified, use_column_width=True)

    with col_tool:
        st.divider()
        st.markdown('<p class="bc-tool-title">处理流程</p>', unsafe_allow_html=True)
        if result is None:
            st.markdown('<p class="bc-cap">等待导入图片…</p>', unsafe_allow_html=True)
        else:
            for rec in result.stages:
                st.markdown(
                    f'<div class="bc-step"><span>{_STAGE_LABELS.get(rec.stage, rec.stage)}</span>'
                    f'{_status_badge(rec.status)}</div>',
                    unsafe_allow_html=True,
                )
            st.divider()
            if result.succeeded and result.final_bgr is not None:
                ok, buf = cv2.imencode(".png", result.final_bgr)
                if ok:
                    st.download_button("导出结果图", data=io.BytesIO(buf.tobytes()),
                                       file_name="beautycam_result.png", mime="image/png",
                                       use_container_width=True)

    with col_param:
        if result is not None:
            st.caption(f"处理状态：{result.status_text()}")
        st.caption("当前无 GT 参考，IoU / ΔE2000 / PSNR / SSIM 等参考型客观指标不可评估，"
                   "界面不显示无参考依据的质量分数。")


if __name__ == "__main__":
    main()
