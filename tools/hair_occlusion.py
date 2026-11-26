# -*- coding: utf-8 -*-
"""tools/hair_occlusion.py —— 头发遮挡抑制（实验性独立工具）。

不属于 A/B/C 任何板块的核心交付，也不被主流水线 import。对 M1
`detect_skin` 产出的 uint8 0/255 掩码做后处理：剔除被误判为皮肤的头发
遮挡像素（刘海、鬓角、垂发），返回同尺寸 uint8 0/255 掩码。

算法演进：
- v1：暗色 + 色度偏离 + 长宽比。对与肤色相近的浅色/同色垂发失效。
- v2：结构张量方向相干性 + 梯度幅值 + 主方向（纵向发丝梯度水平），与颜色无关。
- v3：增加脸部中心保护 + 边界腐蚀，避免误伤五官与下颌轮廓。
- v4：去掉连通域长宽比约束。真实细刘海密集交织，MORPH_CLOSE 后连成近似
  方块的大片连通域，长宽比判据（h>=k·w）会把整块丢弃，导致细刘海完全无法
  剔除。改为仅按面积过滤零散噪声，保留中心/边界保护。
- v5（本次）：区分皱纹与发丝。实测方向/曲率/线宽/边界连通四类局部纹理
  判据在两者上数值分布几乎完全重叠（发丝与皱纹同为脸上的细线结构），无法
  区分。改用空间位置先验：刘海/垂发来自额头（眼线以上），而法令纹/鱼尾纹/
  鼻唇沟均在眼线以下。调用方传入 Haar 眼线 y（eye_line_y），只在其以上区域
  应用纹理剔除，眼线以下整体保护，从根上避免把皱纹当头发。

核心判据（仅作用于 M1 掩码内部）：
  1. coherence = (l1-l2)/(l1+l2)：方向性强弱（皮肤~0.55，发丝~0.86+）；
  2. magnitude = sqrt(l1)：局部梯度能量（相对皮肤中值取阈值）；
  3. orientation = (Jxx-Jyy)/(Jxx+Jyy)：+1 表示水平梯度=纵向发丝，
     -1 表示垂直梯度=横向眉/唇，据此保护五官；
  4. 眼线几何分区：仅眼线以上剔除（额头刘海），眼线以下保护（皱纹）；
  5. 掩码腐蚀得到 interior，保护下颌/脸颊轮廓；脸部中心矩形兜底保护；
  6. 面积过滤去除零散噪声。

公开接口：
    load_hair_config() -> dict                     # 读取 config/hair_occlusion.yaml
    suppress_hair(image_bgr, skin_mask, cfg=None, eye_line_y=None) -> np.ndarray

合规说明（对齐 docs/baseline）：
- 本工具是 A 维护的实验性独立工具，不属于 M1-M7，也不被主流水线 import；
- 不引入任何新模型或新依赖：眼线定位所需的 Haar 能力复用 M1（skin_detect）
  已基线化的人脸/眼睛定位，例外授权见 report/lht_haar_20261031.md；
- 可调参数外置于 config/hair_occlusion.yaml，源码仅保留代码默认值兜底。
"""
from __future__ import annotations

import os
from collections.abc import Mapping

import cv2
import numpy as np
import yaml

_DEFAULTS = {
    "enabled": True,
    "strength": None,          # 0..1，None 时用下面绝对阈值；给定则自动映射阈值
    "coherence_min": 0.70,     # 方向相干性阈值（皮肤~0.55，发丝~0.86）
    "mag_factor": 1.8,         # 梯度幅值阈值 = factor * 掩码内幅值中值
    "mag_min": 3.0,            # 幅值下限，防止整体偏暗的图误伤
    "orientation_min": 0.25,   # >0 偏向纵向发丝；保护横向五官
    "boundary_erode": 5,       # 掩码腐蚀半径，保护轮廓
    "min_area_ratio": 0.0003,  # 连通域最小面积占整图比例（细发丝取较小值）
    "dilate_kernel": 3,
    "smooth_sigma": 1.0,
    "tensor_sigma": 3.0,
    # 脸部中心保护：bbox 归一化坐标下，中心矩形内不剔除（保护鼻/法令纹/嘴），
    # 遮挡头发多来自额头与两侧。
    "protect_center": True,
    "protect_half_x": 0.30,    # x 相对 bbox 中心 ±0.30
    "protect_top_y": 0.40,     # y 相对 bbox 顶部 0.40 以下开始保护
    "protect_bottom_y": 0.92,  # y 相对 bbox 顶部 0.92 以上不保护（保留下巴尖可剔除）
    # 眼线几何分区：eye_line_y 由调用方传入（Haar 眼框中心 y，像素坐标）。
    # 仅眼线以上剔除发丝；eye_keep_ratio 放宽眼线附近的剔除带。
    # eye_line_y=None 时退化为原行为（不分区）。
    "eye_keep_ratio": 1.02,
}


def load_hair_config(path: str | None = None) -> dict:
    """加载头发抑制实验工具配置（config/hair_occlusion.yaml）。

    以本文件位置回溯仓库根，不依赖当前工作目录（arch 7.2）。
    """
    if path is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path = os.path.join(repo_root, "config", "hair_occlusion.yaml")
    with open(path, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError("hair_occlusion.yaml 根节点必须是映射")
    return data


def _resolve(cfg):
    resolved = dict(_DEFAULTS)
    if cfg:
        if not isinstance(cfg, Mapping):
            raise ValueError("hair cfg 必须是映射")
        for key in _DEFAULTS:
            if key in cfg and cfg[key] is not None:
                resolved[key] = cfg[key]
    return resolved


def _structure_tensor(gray: np.ndarray, smooth_sigma: float, tensor_sigma: float):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), tensor_sigma)
    Jyy = cv2.GaussianBlur(gy * gy, (0, 0), tensor_sigma)
    Jxy = cv2.GaussianBlur(gx * gy, (0, 0), tensor_sigma)
    tr = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0))
    l1 = tr / 2.0 + disc
    l2 = tr / 2.0 - disc
    coh = (l1 - l2) / (l1 + l2 + 1e-6)
    mag = np.sqrt(l1 + 1e-6)
    orient = (Jxx - Jyy) / (tr + 1e-6)
    return coh, mag, orient


def suppress_hair(image_bgr: np.ndarray, skin_mask: np.ndarray,
                  cfg: Mapping | None = None,
                  eye_line_y: float | None = None) -> np.ndarray:
    """剔除皮肤掩码中的头发遮挡像素，返回同尺寸 uint8 0/255 掩码。

    eye_line_y 为可选的眼线 y 坐标（像素）。给定时只在其以上区域剔除发丝，
    眼线以下整体保护，避免把法令纹/鱼尾纹等皱纹误判为头发；不给定时不分区。

    不改写入参。空掩码或禁用时原样返回新数组。
    """
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8):
        raise ValueError("suppress_hair 需要非空 BGR uint8 H×W×3 输入")
    if (not isinstance(skin_mask, np.ndarray) or skin_mask.ndim != 2
            or skin_mask.shape[:2] != image_bgr.shape[:2]):
        raise ValueError("skin_mask 须为与图像同尺寸的二维掩码")

    p = _resolve(cfg)
    out = np.zeros_like(skin_mask, dtype=np.uint8)
    if not p["enabled"] or not np.any(skin_mask):
        out[skin_mask > 0] = 255
        return out

    strength = p["strength"]
    if strength is not None:
        s = float(np.clip(strength, 0.0, 1.0))
        coh_cut = float(p["coherence_min"]) - 0.22 * s
        mag_fac = max(0.8, float(p["mag_factor"]) - 1.0 * s)
        ori_cut = float(p["orientation_min"]) - 0.30 * s
    else:
        coh_cut, mag_fac, ori_cut = (float(p["coherence_min"]),
                                     float(p["mag_factor"]),
                                     float(p["orientation_min"]))

    mask = (skin_mask > 0).astype(np.uint8)
    gray = cv2.cvtColor(cv2.GaussianBlur(image_bgr, (0, 0),
                                         float(p["smooth_sigma"])), cv2.COLOR_BGR2GRAY)
    coh, mag, orient = _structure_tensor(gray, float(p["smooth_sigma"]),
                                         float(p["tensor_sigma"]))

    mag_med = float(np.median(mag[mask > 0])) if np.any(mask) else 0.0
    mag_cut = max(float(p["mag_min"]), mag_fac * mag_med)

    er = int(p["boundary_erode"])
    interior = (cv2.erode(mask, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(1, er), max(1, er)))) > 0) if er > 1 else mask > 0

    cand = ((mask > 0) & interior & (coh > coh_cut) & (mag > mag_cut)
            & (orient > ori_cut)).astype(np.uint8)
    # 小核闭运算仅连接断裂发丝；避免大核把密集刘海与皮肤进一步粘连。
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # 眼线几何分区：区分皱纹与发丝的关键。局部纹理无法区分两者（同为脸上
    # 的细线），但刘海/垂发来自额头（眼线以上），法令纹/鱼尾纹/鼻唇沟都在
    # 眼线以下。给定眼线时只保留其以上的候选，从根上保护下半脸的皱纹。
    if eye_line_y is not None:
        cut = float(eye_line_y) * float(p["eye_keep_ratio"])
        rows_below = np.arange(mask.shape[0]) >= cut
        cand[rows_below, :] = 0

    # 脸部中心保护：无眼线时的兜底方案。遮挡头发来自额头与两侧，中央五官
    # 与皮肤纹理（法令纹/鼻唇沟/下巴褶皱）与发丝纹理相似，须按位置保护。
    # 眼线分区已覆盖下半脸保护，启用时跳过以免重复过度保护额头。
    if p["protect_center"] and eye_line_y is None:
        yy_idx, xx_idx = np.where(mask > 0)
        if len(xx_idx):
            bx0, bx1 = np.percentile(xx_idx, [2, 98])
            by0, by1 = np.percentile(yy_idx, [2, 98])
            bw, bh = max(1.0, bx1 - bx0), max(1.0, by1 - by0)
            cx = (bx0 + bx1) / 2.0
            px = bw * float(p["protect_half_x"])
            pt = by0 + bh * float(p["protect_top_y"])
            pb = by0 + bh * float(p["protect_bottom_y"])
            yy_grid, xx_grid = np.ogrid[:mask.shape[0], :mask.shape[1]]
            protect = ((np.abs(xx_grid - cx) < px) & (yy_grid > pt) & (yy_grid < pb))
            cand[protect] = 0

    # 仅按面积过滤零散噪声。刻意不使用长宽比：密集细刘海会连成近似方块，
    # h>=k·w 会将其整块判为非发丝而丢弃（v3 细刘海 removed=0 的根因）。
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    hair = np.zeros_like(mask)
    min_area = float(p["min_area_ratio"]) * skin_mask.size
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= min_area:
            hair[labels == index] = 1

    dk = int(p["dilate_kernel"]) | 1
    if dk > 1:
        hair = cv2.dilate(hair, np.ones((dk, dk), np.uint8))

    kept = mask & (hair == 0)
    out[kept > 0] = 255
    return out


__all__ = ["load_hair_config", "suppress_hair"]
