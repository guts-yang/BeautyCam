#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CelebAMask-HQ 数据预处理：合并分散掩码 + 抽取小规模子集 + 生成固定评测清单。

用法示例：
    # 1) 合并全部 30000 张掩码（约 30 分钟、需额外磁盘空间）
    python src/prepare_data.py merge --root data/input/CelebAMask-HQ

    # 2) 抽取 40 张作为本课题的工作子集（含原图 + 合并掩码 + 皮肤掩码 + manifest）
    python src/prepare_data.py subset --root data/input/CelebAMask-HQ \
        --out data/input/subset --num 40 --manifest assets/evaluation_manifest.csv

依赖：opencv-python-headless、numpy

说明（对齐 docs/baseline/arch.md 5.2）：
- 文件枚举先排序，再按固定随机种子抽样，保证同一数据集下结果可复现；
- 输出 `assets/evaluation_manifest.csv`，字段为 `sample_id,scene`，不含机器绝对路径与图像像素；
- 场景标签为占位值 `unlabeled`，需人工核验后填写（五类场景各 8 张须人工确认，不凭随机种子宣称已分层）。
"""

import argparse
import csv
import os
import random
import shutil

import cv2
import numpy as np

# 与官方 face_parsing 一致的 18 个组件，顺序决定合并后的索引（+1，0 留给背景）
LABEL_LIST = [
    "skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow",
    "l_ear", "r_ear", "mouth", "u_lip", "l_lip", "hair", "hat",
    "ear_r", "neck_l", "neck", "cloth",
]
SKIN_INDEX = LABEL_LIST.index("skin") + 1  # = 1

# 评测场景预设类别（PRD 第 8 节：五类场景各 8 张，共 40 张）
SCENES = ["normal", "backlight", "lowlight", "blemish", "blur"]

MANIFEST_FIELDS = ["sample_id", "scene"]


def merge_masks(root, img_num=30000, save_dir=None):
    """把 CelebAMask-HQ-mask-anno 下分散的组件 png 合并成单张索引掩码。"""
    src = os.path.join(root, "CelebAMask-HQ-mask-anno")
    save_dir = save_dir or os.path.join(root, "CelebAMask-HQ-mask")
    os.makedirs(save_dir, exist_ok=True)

    for k in range(img_num):
        folder_num = k // 2000  # 每 2000 张一个子目录
        merged = np.zeros((512, 512), dtype=np.uint8)
        for idx, label in enumerate(LABEL_LIST):
            fname = os.path.join(src, str(folder_num), f"{k:05d}_{label}.png")
            if not os.path.exists(fname):
                continue
            comp = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if comp is None:
                continue
            merged[comp != 0] = idx + 1
        cv2.imwrite(os.path.join(save_dir, f"{k}.png"), merged)
        if (k + 1) % 1000 == 0:
            print(f"[merge] {k + 1}/{img_num}")
    print(f"[merge] 完成，输出目录：{save_dir}")


def _list_dataset_ids(img_dir):
    """枚举数据集图片 ID，先排序再返回，保证可复现（不依赖 readdir 顺序）。"""
    ids = []
    for name in os.listdir(img_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() in {".jpg", ".jpeg", ".png"} and stem.isdigit():
            ids.append(int(stem))
    return sorted(ids)


def _assign_scenes(count, scenes=SCENES):
    """按轮转方式给样本分配场景占位标签，供人工核验时替换。

    不宣称已实现真实场景分层：分组只是把评测集均匀铺到五类，最终标签须人工确认。
    """
    return [scenes[i % len(scenes)] for i in range(count)]


def build_subset(root, out, num=40, seed=42, manifest_path=None, scenes=None):
    """从全量数据抽取子集，输出原图、合并掩码、皮肤掩码与固定评测清单。"""
    img_dir = os.path.join(root, "CelebA-HQ-img")
    mask_dir = os.path.join(root, "CelebAMask-HQ-mask")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(
            f"未找到合并掩码目录 {mask_dir}，请先执行 merge 子命令。"
        )

    # 关键：先排序再固定种子抽样，避免枚举顺序导致不可复现。
    ids = _list_dataset_ids(img_dir)
    random.Random(seed).shuffle(ids)
    ids = ids[:num]

    out_img = os.path.join(out, "images")
    out_mask = os.path.join(out, "masks")
    out_skin = os.path.join(out, "skin_masks")
    for d in (out_img, out_mask, out_skin):
        os.makedirs(d, exist_ok=True)

    written = []
    for k in ids:
        src_img = os.path.join(img_dir, f"{k}.jpg")
        src_mask = os.path.join(mask_dir, f"{k}.png")
        if not (os.path.exists(src_img) and os.path.exists(src_mask)):
            continue
        shutil.copy(src_img, os.path.join(out_img, f"{k}.jpg"))
        mask = cv2.imread(src_mask, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(out_mask, f"{k}.png"), mask)
        skin = np.where(mask == SKIN_INDEX, 255, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(out_skin, f"{k}.png"), skin)
        written.append(k)

    written.sort()
    labels = _assign_scenes(len(written), scenes or SCENES)
    manifest_rows = [{"sample_id": str(k), "scene": labels[i]}
                     for i, k in enumerate(written)]

    if manifest_path:
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"[subset] 已写出评测清单：{manifest_path}（scene 为占位标签，须人工核验）")

    print(f"[subset] 已输出 {len(written)} 张到 {out}")


def main():
    ap = argparse.ArgumentParser(description="CelebAMask-HQ 数据预处理")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("merge", help="合并分散掩码")
    p1.add_argument("--root", required=True, help="CelebAMask-HQ 根目录")
    p1.add_argument("--img-num", type=int, default=30000)
    p1.add_argument("--save-dir", default=None)

    p2 = sub.add_parser("subset", help="抽取子集并生成评测清单")
    p2.add_argument("--root", required=True, help="CelebAMask-HQ 根目录")
    p2.add_argument("--out", required=True, help="子集输出目录")
    p2.add_argument("--num", type=int, default=40)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--manifest", default="assets/evaluation_manifest.csv",
                    help="评测清单输出路径（相对仓库根目录）")

    args = ap.parse_args()
    if args.cmd == "merge":
        merge_masks(args.root, args.img_num, args.save_dir)
    else:
        build_subset(args.root, args.out, args.num, args.seed, args.manifest)


if __name__ == "__main__":
    main()