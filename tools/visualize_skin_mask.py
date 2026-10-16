#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""皮肤掩码可视化：用真实 GT 掩码产出可直接展示的效果图。

为什么只做皮肤掩码：
    M1 之外的模块（M2-M7）当前仍是地基桩，直接跑 pipeline 只会得到
    与原图逐像素一致的结果图。真正有信息量、可展示的是数据子集里
    已经由 CelebAMask-HQ 标注合并出的真实皮肤掩码。

输入（只读，不修改）：data/input/subset/{images,masks,skin_masks}/
输出：report/figures/
    - skin_mask_panel.png    五场景代表样本 × 四联面板
    - skin_mask_contact.png  全部样本的「原图 / 皮肤叠加」contact sheet
    - skin_mask_stats.csv    每张样本的皮肤占比统计

依赖：numpy、opencv-python-headless、matplotlib（见 requirements.txt）。
"""

from __future__ import annotations

import csv
import os
import sys

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSET = os.path.join(REPO_ROOT, "data", "input", "subset")
MANIFEST = os.path.join(REPO_ROOT, "assets", "evaluation_manifest.csv")
OUT_DIR = os.path.join(REPO_ROOT, "report", "figures")

# CelebAMask-HQ 18 个组件 + 背景，用于给合并索引掩码上色（仅供辨识，非皮肤部分不上色）
LABEL_LIST = [
    "skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow",
    "l_ear", "r_ear", "mouth", "u_lip", "l_lip", "hair", "hat",
    "ear_r", "neck_l", "neck", "cloth",
]
SKIN_INDEX = LABEL_LIST.index("skin") + 1  # = 1

SCENES = ["normal", "backlight", "lowlight", "blemish", "blur"]
SCENE_CN = {
    "normal": "normal 正常光",
    "backlight": "backlight 逆光",
    "lowlight": "lowlight 暗光",
    "blemish": "blemish 瑕疵",
    "blur": "blur 模糊",
}

# 一个稳定的调色板（背景用黑色，皮肤用醒目绿色，其余组件用低饱和色）
_PALETTE = [
    (0, 0, 0),        # 0 背景
    (0, 200, 0),      # 1 skin 绿色高亮
    (230, 90, 60), (230, 160, 60), (230, 220, 60),
    (160, 230, 60), (60, 230, 120), (60, 230, 200),
    (60, 180, 230), (60, 120, 230), (120, 60, 230),
    (200, 60, 230), (230, 60, 160), (150, 150, 150),
    (200, 200, 200), (110, 110, 110), (170, 120, 90), (90, 120, 170),
]


def load_manifest(path):
    """读取 sample_id -> scene 映射。"""
    with open(path, encoding="utf-8") as fp:
        return {r["sample_id"]: r["scene"] for r in csv.DictReader(fp)}


def colorize_index_mask(mask):
    """把 0..18 的索引掩码上色为 RGB。"""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, color in enumerate(_PALETTE):
        rgb[mask == idx] = color
    return rgb


def overlay_skin(image_bgr, skin_mask, alpha=0.45, color=(0, 255, 0)):
    """在原图上叠加半透明皮肤色块，用于直观查看掩码贴合度。"""
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    h, w = img.shape[:2]
    mask = skin_mask
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    m = (mask > 0)
    layer = np.zeros_like(img)
    layer[m] = color
    out = np.where(m[..., None], img * (1 - alpha) + layer * alpha, img)
    return out.astype(np.uint8)


def _configure_font():
    """尽量启用中文字体，失败则回退英文标签（不影响图像本身）。"""
    for name in ("PingFang SC", "Arial Unicode MS", "Heiti TC", "STHeiti"):
        try:
            matplotlib.font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def build_panel(manifest, out_path):
    """五场景代表样本 × 四联面板：原图 / 索引掩码 / 皮肤掩码 / 皮肤叠加。"""
    zh = _configure_font()
    labels = ["原图", "合并索引掩码", "皮肤掩码 (GT)", "皮肤叠加 (GT)"] if zh else \
             ["Image", "Index mask", "Skin mask (GT)", "Skin overlay (GT)"]

    # 每个场景挑第一个样本，保证可复现（按 sample_id 排序）
    picked = {}
    for sid, scene in sorted(manifest.items(), key=lambda kv: int(kv[0])):
        picked.setdefault(scene, sid)

    scenes = [s for s in SCENES if s in picked]
    fig, axes = plt.subplots(len(scenes), 4, figsize=(13, 3.1 * len(scenes)))
    if len(scenes) == 1:
        axes = axes[None, :]

    for r, scene in enumerate(scenes):
        sid = picked[scene]
        img = cv2.imread(os.path.join(SUBSET, "images", f"{sid}.jpg"), cv2.IMREAD_COLOR)
        idx = cv2.imread(os.path.join(SUBSET, "masks", f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        skin = cv2.imread(os.path.join(SUBSET, "skin_masks", f"{sid}.png"),
                          cv2.IMREAD_GRAYSCALE)
        if img is None or idx is None or skin is None:
            raise SystemExit(f"样本 {sid} 读取失败，请确认 data/input/subset 存在。")

        ratio = float((skin > 0).mean() * 100)
        panels = [
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            colorize_index_mask(idx),
            skin,
            overlay_skin(img, skin),
        ]
        titles = list(labels)
        titles[0] = f"{labels[0]}  id={sid}  ({SCENE_CN.get(scene, scene)})"
        titles[2] = f"{labels[2]}  {ratio:.1f}%"

        for c, (ax, panel, title) in enumerate(zip(axes[r], panels, titles)):
            if c == 2:
                ax.imshow(panel, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(panel)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

    fig.suptitle(
        "BeautyCam M1 皮肤掩码（真实 GT，CelebAMask-HQ 子集）" if zh else
        "BeautyCam M1 Skin Mask (real GT, CelebAMask-HQ subset)",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[panel] 已输出 {out_path}")


def build_contact(manifest, out_path, cols=8):
    """全部样本的「原图 / 皮肤叠加」contact sheet。"""
    zh = _configure_font()
    sids = sorted(manifest, key=int)
    rows = (len(sids) + cols - 1) // cols
    fig, axes = plt.subplots(rows * 2, cols, figsize=(1.7 * cols, 1.9 * rows * 2))
    axes = np.atleast_2d(axes)

    for i, sid in enumerate(sids):
        r, c = divmod(i, cols)
        img = cv2.imread(os.path.join(SUBSET, "images", f"{sid}.jpg"), cv2.IMREAD_COLOR)
        skin = cv2.imread(os.path.join(SUBSET, "skin_masks", f"{sid}.png"),
                          cv2.IMREAD_GRAYSCALE)
        axes[r * 2][c].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[r * 2][c].set_title(str(sid), fontsize=7)
        axes[r * 2][c].axis("off")
        axes[r * 2 + 1][c].imshow(overlay_skin(img, skin))
        axes[r * 2 + 1][c].axis("off")

    for j in range(len(sids), rows * cols):
        r, c = divmod(j, cols)
        axes[r * 2][c].axis("off")
        axes[r * 2 + 1][c].axis("off")

    fig.suptitle(
        "全部样本：上=原图，下=皮肤掩码叠加（GT）" if zh else
        "All samples: top=image, bottom=skin overlay (GT)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[contact] 已输出 {out_path}")


def write_stats(manifest, out_path):
    """写出每张样本的皮肤占比，便于快速核对掩码质量。"""
    rows = []
    for sid in sorted(manifest, key=int):
        skin = cv2.imread(os.path.join(SUBSET, "skin_masks", f"{sid}.png"),
                          cv2.IMREAD_GRAYSCALE)
        ratio = float((skin > 0).mean() * 100)
        rows.append({
            "sample_id": sid,
            "scene": manifest[sid],
            "skin_ratio_pct": f"{ratio:.2f}",
            "mask_nonempty": int(skin.max() > 0),
        })
    with open(out_path, "w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["sample_id", "scene",
                                           "skin_ratio_pct", "mask_nonempty"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    empty = sum(1 for r in rows if not r["mask_nonempty"])
    print(f"[stats] 已输出 {out_path}（{len(rows)} 行，空掩码 {empty} 张）")


def main():
    if not os.path.isdir(SUBSET):
        raise SystemExit(f"未找到数据子集 {SUBSET}，请先运行 prepare_data.py fetch。")
    if not os.path.exists(MANIFEST):
        raise SystemExit(f"未找到评测清单 {MANIFEST}。")
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = load_manifest(MANIFEST)
    build_panel(manifest, os.path.join(OUT_DIR, "skin_mask_panel.png"))
    build_contact(manifest, os.path.join(OUT_DIR, "skin_mask_contact.png"))
    write_stats(manifest, os.path.join(OUT_DIR, "skin_mask_stats.csv"))


if __name__ == "__main__":
    main()