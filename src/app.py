# -*- coding: utf-8 -*-
"""app.py —— Streamlit 薄入口（A 独占）。

桌面三栏（工具 / 画布 / 参数），窄屏（iPhone 14 Pro 393×852pt）自动改为单列。
参数调节即实时重算预览（结果以图片内容哈希 + 参数组合缓存）。

只调用 core 层。约束（对齐 arch 第 6 节、design 2.5）：
- 上传原图仅在本地进程内存处理，不自动持久化，须用户显式导出；
- UI 覆盖只作用于当前 run，不回写 YAML；
- 无 GT 时不显示参考型客观质量分数。
界面文案面向最终用户，不展示内部算法/指标术语。
"""

from __future__ import annotations

import base64
import hashlib
import io

import cv2
import numpy as np
import streamlit as st
from streamlit.components.v1 import html as st_component_html

from beauty.core.config import load_module_config, load_pipeline_config
from beauty.core.pipeline import run_pipeline

_MODULES = ["skin_detect", "exposure", "smooth", "freq_enhance", "restore", "color", "metrics"]
_MAX_UPLOAD_MB = 20


def _decode(uploaded) -> np.ndarray | None:
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
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
.bc-empty{padding:40px 16px;text-align:center;border:1px dashed var(--bc-border-strong);
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
[data-testid="stDownloadButton"] button{
  background-color:var(--bc-accent);color:var(--bc-text-on-accent);
  border-color:var(--bc-border-strong);}

/* 窄屏（iPhone 14 Pro 393×852pt）：三栏改单列，顺序为 上传 → 参数 → 结果 */
@media (max-width: 768px){
  [data-testid="stHorizontalBlock"]{
    flex-direction:column !important; flex-wrap:nowrap !important; gap:0.5rem !important;
  }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{
    width:100% !important; flex:1 1 100% !important; min-width:0 !important;
  }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(1){ order:1; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2){ order:3; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(3){ order:2; }
  .bc-brand{ gap:8px; padding-bottom:12px; margin-bottom:10px; }
  .bc-brand h1{ font-size:20px; line-height:28px; }
  .bc-logo{ width:38px; height:38px; flex-basis:38px; font-size:20px; border-radius:8px; }
  .block-container{ padding-top:1rem; padding-left:0.8rem !important; padding-right:0.8rem !important; }
}
</style>
""",
        unsafe_allow_html=True,
    )


def _compare_view(original_bgr: np.ndarray, result_bgr: np.ndarray) -> None:
    """默认按原比例展示结果图；长按按钮或图片显示原图，松手返回结果图。

    纯前端 pointer/touch/keyboard 事件切换，不触发 Streamlit 重算。
    两张图以绝对定位完全重叠，任一时刻仅一张可见，杜绝图片片段露出；
    iframe 高度按图片真实宽高比计算，任何比例都完整显示、不裁剪。
    """
    def _data_uri(bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            return ""
        return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

    orig_uri = _data_uri(original_bgr)
    result_uri = _data_uri(result_bgr)
    height, width = result_bgr.shape[:2]
    display_width = 420
    image_height = display_width * height / width
    frame_height = int(image_height + 100)
    component = f"""
<style>
.hc-wrap{{font-family:system-ui,-apple-system,"PingFang SC",sans-serif;}}
.hc-btn{{
  display:block;width:100%;margin:0 0 10px;padding:12px 16px;font-size:16px;font-weight:600;
  color:#2B2622;background:#E8A598;border:1px solid #8A8178;border-radius:6px;
  cursor:pointer;user-select:none;touch-action:none;-webkit-user-select:none;
}}
.hc-btn:active,.hc-btn.hc-pressed{{background:#C57A6C;}}
.hc-btn:focus-visible{{outline:3px solid #0B2E59;outline-offset:2px;}}
.hc-stage{{
  position:relative;width:100%;max-width:420px;margin:0 auto;
  cursor:pointer;touch-action:none;user-select:none;-webkit-user-select:none;
}}
.hc-stage img{{
  position:absolute;top:0;left:0;display:block;width:100%;height:auto;max-width:100%;
  border-radius:10px;pointer-events:none;
}}
.hc-stage{{height:auto;}}
.hc-stage::before{{
  content:"";display:block;width:100%;padding-top:{image_height / display_width * 100:.4f}%;
}}
.hc-orig{{visibility:hidden;}}
.hc-wrap.hc-show-orig .hc-orig{{visibility:visible;}}
.hc-wrap.hc-show-orig .hc-result{{visibility:hidden;}}
.hc-tip{{font-size:12px;color:#6B6259;margin:8px 0 0;text-align:center;}}
</style>
<div class="hc-wrap" id="hc_root">
  <button type="button" class="hc-btn" id="hc_btn"
          aria-label="长按查看原图，松开查看美颜结果">长按看原图 · 松开看效果</button>
  <div class="hc-stage" id="hc_stage" role="button" tabindex="0"
       aria-label="长按查看原图，松手返回美颜效果">
    <img class="hc-result" src="{result_uri}" alt="美颜结果">
    <img class="hc-orig" src="{orig_uri}" alt="原图">
  </div>
  <p class="hc-tip">长按按钮或图片看原图，松手返回美颜效果</p>
</div>
<script>
(function(){{
  var root=document.getElementById('hc_root');
  var btn=document.getElementById('hc_btn');
  var stage=document.getElementById('hc_stage');
  function show(v){{
    root.classList.toggle('hc-show-orig',v);
    btn.classList.toggle('hc-pressed',v);
  }}
  [btn,stage].forEach(function(el){{
    ['pointerdown','touchstart'].forEach(function(ev){{
      el.addEventListener(ev,function(e){{e.preventDefault();show(true);}});
    }});
    ['pointerup','pointercancel','pointerleave','touchend','touchcancel','mouseleave','blur'].forEach(function(ev){{
      el.addEventListener(ev,function(){{show(false);}});
    }});
  }});
  window.addEventListener('pointerup',function(){{show(false);}});
  btn.addEventListener('keydown',function(e){{if(e.code==='Space'||e.code==='Enter'){{e.preventDefault();show(true);}}}});
  btn.addEventListener('keyup',function(e){{if(e.code==='Space'||e.code==='Enter'){{show(false);}}}});
  stage.addEventListener('keydown',function(e){{if(e.code==='Space'||e.code==='Enter'){{e.preventDefault();show(true);}}}});
  stage.addEventListener('keyup',function(e){{if(e.code==='Space'||e.code==='Enter'){{show(false);}}}});
}})();
</script>
"""
    st_component_html(component, height=frame_height, scrolling=False)


def main() -> None:
    st.set_page_config(page_title="焕颜相机 BeautyCam", layout="wide")
    _inject_css()

    st.markdown(
        '<div class="bc-brand">'
        '<div class="bc-logo">焕</div>'
        '<div><h1>焕颜相机 BeautyCam</h1>'
        '<p>上传照片，轻松美颜，效果实时可见</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    col_tool, col_canvas, col_param = st.columns([1.2, 3.0, 1.8], gap="medium")

    with col_tool:
        st.markdown('<p class="bc-tool-title">选择照片</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "上传照片",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=False,
            help="支持 JPG / PNG 等常见格式，最大 20MB。",
        )
        if uploaded is not None:
            st.markdown(
                f'<p class="bc-cap">{uploaded.name} · {uploaded.size / 1024:.0f} KB</p>',
                unsafe_allow_html=True,
            )

    with col_param:
        st.markdown('<p class="bc-panel-title">美颜调节</p>', unsafe_allow_html=True)
        smooth_strength = st.slider(
            "磨皮", 0.0, 1.0, 0.7, 0.05, disabled=(uploaded is None),
            help="数值越大，皮肤越光滑细腻。")
        exposure_gain = st.slider(
            "亮度", 0.0, 1.0, 0.25, 0.05, disabled=(uploaded is None),
            help="数值越大，照片越明亮；建议保持默认，避免过亮。")
        color_temp = st.slider(
            "冷暖", -1.0, 1.0, 0.0, 0.1, disabled=(uploaded is None),
            help="向左偏冷，向右偏暖。")
        skin_warmth = st.slider(
            "红润", 0.0, 1.0, 0.3, 0.05, disabled=(uploaded is None),
            help="数值越大，肤色越红润自然；脸色偏白可调大。")
        st.divider()
        st.markdown('<p class="bc-cap">调节后效果实时更新，满意即可保存。</p>',
                    unsafe_allow_html=True)

    result = None
    image = None
    if uploaded is not None:
        if uploaded.size > _MAX_UPLOAD_MB * 1024 * 1024:
            with col_canvas:
                st.error(f"图片过大（{uploaded.size / 1024 / 1024:.1f}MB），请上传不超过 {_MAX_UPLOAD_MB}MB 的照片。")
            return
        image = _decode(uploaded)
        if image is None:
            with col_canvas:
                st.error("无法识别这张照片，请换一张 JPG / PNG 图片试试。")
            return

        params = (smooth_strength, exposure_gain, color_temp, skin_warmth)
        # 以图片内容哈希为签名：换图必重新处理，内容与参数都不变才复用结果。
        file_key = hashlib.sha256(uploaded.getvalue()).hexdigest()
        cache_key = (file_key, params)
        cached = st.session_state.get("bc_result")
        if cached is None or cached["key"] != cache_key:
            pipeline_cfg, module_cfgs = _load_configs()
            module_cfgs = {k: dict(v) for k, v in module_cfgs.items()}
            module_cfgs["smooth"] = {**module_cfgs["smooth"], "strength": smooth_strength}
            # 磨皮滑块联动高频回注：0→1.1（细节最全），0.7→0.75（默认，基线内最强），
            # 1→0.7。低于 0.70 会越 PRD 高频保留下限，故不到 1 时取 0.7 兜底。
            hp = round(1.1 - 0.5 * smooth_strength, 3)
            module_cfgs["freq_enhance"] = {
                **module_cfgs["freq_enhance"],
                "highpass_gain": max(0.7, hp),
            }
            module_cfgs["exposure"] = {**module_cfgs["exposure"], "gain": exposure_gain}
            module_cfgs["color"] = {**module_cfgs["color"],
                                    "temperature": color_temp,
                                    "skin_balance_strength": skin_warmth}
            with st.spinner("正在美颜…"):
                result = run_pipeline(image, pipeline_cfg, module_cfgs, mode="preview")
            st.session_state["bc_result"] = {"key": cache_key, "result": result}
        else:
            result = cached["result"]

    with col_canvas:
        if uploaded is None or image is None:
            st.markdown(
                '<div class="bc-empty"><p>请先上传一张人像照片</p>'
                '<span>上传后可调节磨皮、亮度、肤色，实时查看效果</span></div>',
                unsafe_allow_html=True,
            )
        elif result is not None and not result.succeeded:
            st.error("处理未成功，请换一张照片再试。")
        elif result is not None and result.final_bgr is not None:
            _compare_view(image, result.final_bgr)
            ok, buf = cv2.imencode(".png", result.final_bgr)
            if ok:
                st.download_button("保存美颜照片", data=io.BytesIO(buf.tobytes()),
                                   file_name="beautycam_result.png", mime="image/png",
                                   use_container_width=True)


if __name__ == "__main__":
    main()
