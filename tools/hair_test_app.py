# -*- coding: utf-8 -*-
"""hair_test_app.py —— 头发遮挡抑制本地对照测试页（A 维护的实验工具，非核心交付）。

用途：上传一张人像照片，同屏对比：
  - 原始 M1 掩码 / 头发抑制后掩码；
  - 原始美颜结果 / 抑制后美颜结果。
只在本机内存处理，不回写任何 YAML，不持久化用户图片。
算法参数来自 config/hair_occlusion.yaml，UI 滑块仅作本次运行覆盖。

启动：
  .venv/bin/python -m streamlit run tools/hair_test_app.py \
      --server.port 51285 --server.headless true --browser.gatherUsageStats false
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import streamlit as st

# 允许以 `-m streamlit run tools/hair_test_app.py` 方式导入仓库内的 beauty 与 tools。
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _path in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from beauty.core.config import load_module_config, load_pipeline_config
from beauty.core.pipeline import run_pipeline
from tools.hair_occlusion import load_hair_config, suppress_hair

_MAX_UPLOAD_MB = 20


def _decode(uploaded) -> np.ndarray | None:
    data = np.frombuffer(uploaded.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """掩码以半透明红色叠加到原图，便于观察误检区域。"""
    base = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay = base.copy()
    tint = np.zeros_like(base)
    tint[:] = (220, 40, 40)
    alpha = 0.45
    sel = mask > 0
    overlay[sel] = (overlay[sel] * (1 - alpha) + tint[sel] * alpha).astype(np.uint8)
    return overlay


def _mask_rgb(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def main() -> None:
    st.set_page_config(page_title="头发遮挡抑制 · 本地对照测试", layout="wide")
    st.markdown("## 头发遮挡抑制 · 本地对照测试")
    st.caption("上传一张人像照片，对比 M1 原始掩码与头发抑制后的掩码及美颜效果。图片仅在本机内存处理。")

    hair_defaults = load_hair_config()
    strength_default = float(hair_defaults.get("strength") or 0.35)
    with st.sidebar:
        st.markdown("### 抑制强度")
        strength = st.slider(
            "强度（越大剔除越多）", 0.0, 1.0, strength_default, 0.05,
            help="自动调节方向相干性/梯度幅值/主方向三个阈值。基于方向性纹理识别发丝，与颜色无关。")
        st.markdown("### 高级阈值（可选）")
        coh = st.slider("方向相干性阈值", 0.50, 0.90,
                        float(hair_defaults["coherence_min"]), 0.01,
                        help="皮肤约 0.55，发丝约 0.86；降低会剔除更多。")
        mag_fac = st.slider("梯度幅值倍数", 0.8, 3.0,
                            float(hair_defaults["mag_factor"]), 0.1,
                            help="相对掩码内皮肤梯度中值的倍数。")
        ori = st.slider("纵向主方向阈值", -0.2, 0.6,
                        float(hair_defaults["orientation_min"]), 0.05,
                        help="越大越偏向纵向发丝、越保护横向眉唇。")
        erode = st.slider("轮廓保护(腐蚀半径)", 1, 11,
                          int(hair_defaults["boundary_erode"]), 2)
        st.markdown("---")
        fine_mode = st.checkbox(
            "细刘海模式", value=False,
            help="针对很细、密集的刘海发丝：缩小边界腐蚀、降低面积阈值、放宽中心保护。")
        wrinkle_safe = st.checkbox(
            "区分皱纹与头发（眼线分区）", value=True,
            help="用 Haar 眼线做分界：只剔除眼线以上的额头刘海，眼线以下的法令纹/鱼尾纹等皱纹整体保护。")
        use_advanced = st.checkbox("使用高级阈值（否则按强度自动）", value=False)
        show_beauty = st.checkbox("同时计算美颜结果（较慢）", value=True)
        st.caption("说明：本页调用现有 M1 与流水线，不改动任何源码与配置。")

    uploaded = st.file_uploader("导入人像照片", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is None:
        st.info("请从上方上传一张人像照片开始测试。")
        return
    if uploaded.size > _MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"图片过大（{uploaded.size / 1024 / 1024:.1f}MB），请上传不超过 {_MAX_UPLOAD_MB}MB 的图片。")
        return

    image = _decode(uploaded)
    if image is None:
        st.error("无法解码该文件，请确认是有效的 JPG/JPEG/PNG/BMP 图片。")
        return

    pipeline_cfg = load_pipeline_config()
    skin_cfg = load_module_config("skin_detect")
    with st.spinner("正在运行 M1 肤色检测…"):
        base_mask = skin_detect_mask(image, skin_cfg)

    # 配置优先级：YAML 默认值 < 本次 UI 覆盖（arch 7.1，UI 不回写 YAML）。
    hair_cfg = dict(hair_defaults)
    if use_advanced:
        hair_cfg.update({
            "coherence_min": coh,
            "mag_factor": mag_fac,
            "orientation_min": ori,
            "boundary_erode": erode,
        })
    else:
        hair_cfg.update({"strength": strength, "boundary_erode": erode})
    if fine_mode:
        # 细刘海预设：缩小边界腐蚀、降低面积阈值以保留细而短的发丝，
        # 放宽中心保护半宽以覆盖靠近额头的细刘海。仅作用于本次运行。
        hair_cfg.update({
            "boundary_erode": min(erode, 3),
            "min_area_ratio": 0.00005,
            "protect_half_x": 0.24,
            "protect_top_y": 0.46,
        })
    eye_y = _eye_line_y(image, skin_cfg) if wrinkle_safe else None
    fixed_mask = suppress_hair(image, base_mask, hair_cfg, eye_line_y=eye_y)
    eye_tip = (f"眼线分区生效（眼线 y≈{eye_y:.0f}）" if eye_y is not None
               else ("眼线分区失效（未定位到人脸）" if wrinkle_safe else "眼线分区关闭"))

    orig_count = int(np.count_nonzero(base_mask))
    fixed_count = int(np.count_nonzero(fixed_mask))
    removed = orig_count - fixed_count
    removed_pct = (100.0 * removed / orig_count) if orig_count else 0.0

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("#### 原始 M1 掩码")
        st.image(_overlay(image, base_mask), use_column_width=True)
        st.caption(f"皮肤像素 {orig_count}")
    with m2:
        st.markdown("#### 抑制后掩码")
        st.image(_overlay(image, fixed_mask), use_column_width=True)
        st.caption(f"皮肤像素 {fixed_count} · 剔除 {removed} ({removed_pct:.1f}%) · {eye_tip}")
    with m3:
        st.markdown("#### 掩码差异（红=被剔除）")
        diff = ((base_mask > 0) & (fixed_mask == 0)).astype(np.uint8) * 255
        st.image(_mask_rgb(diff), use_column_width=True)
        st.caption("高亮部分为算法判定为头发遮挡的像素")

    if show_beauty:
        module_cfgs = {n: load_module_config(n) for n in
                       ["skin_detect", "exposure", "smooth", "freq_enhance", "color"]}
        with st.spinner("正在计算美颜结果（原始掩码）…"):
            res_before = run_pipeline(image, pipeline_cfg, module_cfgs, mode="preview")
        # 通过临时替换 detect_skin 实现注入（仅本进程 run）。
        # 用保存的原始函数构建 patch，避免递归调用自身。
        import beauty.core.pipeline as pipeline_mod
        original_detect = pipeline_mod.skin_detect.detect_skin

        def _patched_detect(img, cfg):
            return suppress_hair(img, original_detect(img, cfg), hair_cfg,
                                 eye_line_y=_eye_line_y(img, cfg) if wrinkle_safe else None)

        pipeline_mod.skin_detect.detect_skin = _patched_detect
        try:
            with st.spinner("正在计算美颜结果（抑制后掩码）…"):
                res_after = run_pipeline(image, pipeline_cfg, module_cfgs, mode="preview")
        finally:
            pipeline_mod.skin_detect.detect_skin = original_detect

        b1, b2 = st.columns(2)
        with b1:
            st.markdown("#### 美颜结果（原始）")
            if res_before.final_bgr is not None:
                st.image(cv2.cvtColor(res_before.final_bgr, cv2.COLOR_BGR2RGB),
                         use_column_width=True)
            else:
                st.error(res_before.status_text())
        with b2:
            st.markdown("#### 美颜结果（抑制头发后）")
            if res_after.final_bgr is not None:
                st.image(cv2.cvtColor(res_after.final_bgr, cv2.COLOR_BGR2RGB),
                         use_column_width=True)
            else:
                st.error(res_after.status_text())


def skin_detect_mask(image: np.ndarray, cfg) -> np.ndarray:
    from beauty.modules import skin_detect
    return skin_detect.detect_skin(image, cfg)


def _eye_line_y(image: np.ndarray, cfg):
    """用 skin_detect 自带的 Haar 定位眼线 y；失败时返回 None。"""
    from beauty.modules import skin_detect
    locator = skin_detect._haar_configuration(cfg)
    if locator is None:
        return None
    try:
        working, located = skin_detect._locate_image(image, locator)
    except Exception:
        return None
    if located is None:
        return None
    face, eyes = located
    sy = image.shape[0] / working.shape[0]
    if eyes:
        return float(np.mean([(ey + eh / 2) * sy for _, ey, _, eh in eyes]))
    return float((face[1] + 0.4 * face[3]) * sy)


if __name__ == "__main__":
    main()
