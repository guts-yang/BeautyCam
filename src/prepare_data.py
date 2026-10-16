#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CelebAMask-HQ 数据预处理：远程按需抽取 / 合并掩码 / 生成固定评测清单。

顶层约束（不可违反，见 README「项目约束基线」与 docs/baseline/arch.md 5.2）：
所有数据集脚本一律只获取工作所需的最小样本子集，禁止下载完整数据集或做全量抽取。
本课题不训练深度模型，工作子集为 20~50 张，默认走「远程按需抽取」：只拉取
评测所需的那几十张，不下载 3GB 整包。个别样本缺标注时按跳过处理，不得因此改为全量拉取。

用法示例：
    # 1)（推荐，默认路径）从公开镜像按 HTTP Range 抽取少量样本，不下载整包
    python src/prepare_data.py fetch --num 40 --out data/input/subset \
        --manifest assets/evaluation_manifest.csv

    # 2)（仅当本地已自行具备整包时）合并分散掩码
    python src/prepare_data.py merge --root data/input/CelebAMask-HQ

    # 3)（仅当本地已自行具备整包时）从本地抽取子集
    python src/prepare_data.py subset --root data/input/CelebAMask-HQ \
        --out data/input/subset --num 40 --manifest assets/evaluation_manifest.csv

依赖：opencv-python-headless、numpy

说明（对齐 docs/baseline/arch.md 5.2）：
- 文件枚举先排序，再按固定随机种子抽样，保证同一数据集下结果可复现；
- 输出 `assets/evaluation_manifest.csv`，字段为 `sample_id,scene`，不含机器绝对路径与图像像素；
- 场景标签为占位值，需人工核验后填写（五类场景各 8 张须人工确认，不凭随机种子宣称已分层）。

远程抽取的代价：zip 中央目录约 50MB 必须读取（40 万个条目），
其余字节按需 Range 拉取；实测 3 张约 50MB、40 张约 66MB，远小于整包 3.15GB。
"""

import argparse
import csv
import io
import os
import random
import shutil
import urllib.request
import zipfile

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

# 公开镜像（HuggingFace 上的官方压缩包副本），支持 HTTP Range 分段读取。
DEFAULT_ZIP_URL = (
    "https://huggingface.co/datasets/liusq/CelebAMask-HQ/resolve/main/CelebAMask-HQ.zip"
)
ZIP_PREFIX = "CelebAMask-HQ/"

class DataError(RuntimeError):
    """数据准备过程中的可读错误。"""

# --- 合并分散掩码（本地整包场景） -----------------------------------------

def merge_masks(root, ids, save_dir=None):
    """把 CelebAMask-HQ-mask-anno 下分散的组件 png 合并成单张索引掩码。

    顶层约束要求只处理工作所需的最小样本，因此本函数只合并显式传入的 `ids`，
    不做全量扫描。若确实需要整包，请自行决定并明确传入全部 ID。
    """
    src = os.path.join(root, "CelebAMask-HQ-mask-anno")
    save_dir = save_dir or os.path.join(root, "CelebAMask-HQ-mask")
    os.makedirs(save_dir, exist_ok=True)

    ids = sorted(ids)
    for done, k in enumerate(ids, 1):
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
        if done % 100 == 0:
            print(f"[merge] {done}/{len(ids)}")
    print(f"[merge] 完成 {len(ids)} 张，输出目录：{save_dir}")

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

def _write_manifest(written_ids, manifest_path):
    """写出 `sample_id,scene` 评测清单（不含绝对路径与像素）。"""
    if not manifest_path:
        return
    written_ids = sorted(written_ids)
    labels = _assign_scenes(len(written_ids))
    rows = [{"sample_id": str(k), "scene": labels[i]}
            for i, k in enumerate(written_ids)]
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[manifest] 已写出评测清单：{manifest_path}"
          f"（{len(rows)} 行，scene 为占位标签，须人工核验）")

def _write_subset_dirs(written, out_img, out_mask, out_skin):
    """确保子集三个输出目录存在。"""
    for d in (out_img, out_mask, out_skin):
        os.makedirs(d, exist_ok=True)

def _merge_components(comp_arrays):
    """把 {label: 二值图} 合并为索引掩码（组件缺失即跳过）。"""
    merged = np.zeros((512, 512), dtype=np.uint8)
    for idx, label in enumerate(LABEL_LIST):
        comp = comp_arrays.get(label)
        if comp is None:
            continue
        if comp.shape[:2] != (512, 512):
            comp = cv2.resize(comp, (512, 512), interpolation=cv2.INTER_NEAREST)
        merged[comp != 0] = idx + 1
    return merged

def _emit_sample(sample_id, image_bgr, merged_mask, out_img, out_mask, out_skin):
    """写出单张样本的原图、合并掩码与皮肤掩码。"""
    cv2.imwrite(os.path.join(out_img, f"{sample_id}.jpg"), image_bgr)
    cv2.imwrite(os.path.join(out_mask, f"{sample_id}.png"), merged_mask)
    skin = np.where(merged_mask == SKIN_INDEX, 255, 0).astype(np.uint8)
    cv2.imwrite(os.path.join(out_skin, f"{sample_id}.png"), skin)
    return skin

def build_subset(root, out, num=40, seed=42, manifest_path=None, scenes=None):
    """从本地整包目录抽取子集，输出原图、合并掩码、皮肤掩码与固定评测清单。

    顶层约束：只处理抽中的样本，不对全量 30000 张做扫描或合并。
    """
    img_dir = os.path.join(root, "CelebA-HQ-img")
    anno_dir = os.path.join(root, "CelebAMask-HQ-mask-anno")
    if not os.path.isdir(anno_dir):
        raise DataError(f"未找到标注目录 {anno_dir}，"
                        f"或改用 fetch 子命令从远程按需抽取。")

    # 关键：先排序再固定种子抽样，避免枚举顺序导致不可复现。
    ids = _list_dataset_ids(img_dir)
    random.Random(seed).shuffle(ids)
    ids = ids[:num]

    out_img = os.path.join(out, "images")
    out_mask = os.path.join(out, "masks")
    out_skin = os.path.join(out, "skin_masks")
    _write_subset_dirs(None, out_img, out_mask, out_skin)

    written = []
    for k in ids:
        src_img = os.path.join(img_dir, f"{k}.jpg")
        if not os.path.exists(src_img):
            continue
        arr = cv2.imread(src_img, cv2.IMREAD_COLOR)
        if arr is None:
            continue
        # 只合并这一张的组件，不全量扫描。
        folder = k // 2000
        comps = {}
        for label in LABEL_LIST:
            fname = os.path.join(anno_dir, str(folder), f"{k:05d}_{label}.png")
            if not os.path.exists(fname):
                continue
            comp = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if comp is not None:
                comps[label] = comp
        merged = _merge_components(comps)
        _emit_sample(k, arr, merged, out_img, out_mask, out_skin)
        written.append(k)

    _write_manifest(written, manifest_path)
    print(f"[subset] 已输出 {len(written)} 张到 {out}")
    return written

# --- 远程按需抽取（默认路径，不下载整包） ---------------------------------

class HttpRangeFile(io.RawIOBase):
    """把远程文件伪装成可 seek 的本地文件，只按需拉取字节区间。

    zipfile 需要随机访问：先读尾部中央目录，再按条目偏移取数据。
    本类把这两类访问都转成 HTTP Range 请求，避免整包下载。
    """

    def __init__(self, url, size, retries=4, timeout=180):
        self.url = url
        self.size = size
        self.retries = retries
        self.timeout = timeout
        self.pos = 0
        self.requests = 0
        self.bytes_fetched = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        last = None
        for _ in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = resp.read()
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
        else:
            raise DataError(f"Range 读取失败 bytes={self.pos}-{end}：{last}")
        self.pos += len(data)
        self.requests += 1
        self.bytes_fetched += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)

def _resolve_zip(url):
    """跟随重定向拿到可 Range 的真实地址与总大小。"""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.geturl(), int(resp.headers["Content-Length"])

def fetch_subset(out, num=40, seed=42, manifest_path=None, zip_url=None,
                 pool=30000):
    """从远程 zip 按需抽取子集，只拉取需要的样本字节。

    抽样规则与 build_subset 完全一致（排序 + 固定种子），因此两者在同一数据集下
    产出的 sample_id 相同，可互相校验。`pool` 只是候选 ID 池，用于抽样，
    不会下载或处理全量数据。
    """
    zip_url = zip_url or DEFAULT_ZIP_URL
    final_url, size = _resolve_zip(zip_url)
    print(f"[fetch] 远程包 {size/1e6:.0f}MB，按需抽取 {num} 张（不下载整包）")

    rf = HttpRangeFile(final_url, size)
    try:
        zf = zipfile.ZipFile(rf)
    except zipfile.BadZipFile as exc:
        raise DataError(f"远程 zip 无法解析：{exc}") from exc
    names = set(zf.namelist())
    print(f"[fetch] 条目 {len(names)}，中央目录已读取 {rf.bytes_fetched/1e6:.1f}MB")

    ids = list(range(pool))
    random.Random(seed).shuffle(ids)

    out_img = os.path.join(out, "images")
    out_mask = os.path.join(out, "masks")
    out_skin = os.path.join(out, "skin_masks")
    _write_subset_dirs(None, out_img, out_mask, out_skin)

    written = []
    skipped_no_skin = 0
    for k in ids:
        if len(written) >= num:
            break
        folder = k // 2000
        img_name = f"{ZIP_PREFIX}CelebA-HQ-img/{k}.jpg"
        skin_name = (f"{ZIP_PREFIX}CelebAMask-HQ-mask-anno/"
                     f"{folder}/{k:05d}_skin.png")
        if img_name not in names:
            continue
        if skin_name not in names:
            # 官方数据集中个别样本缺皮肤标注，跳过以保证 GT 完整。
            skipped_no_skin += 1
            continue

        with zf.open(img_name) as fp:
            arr = cv2.imdecode(np.frombuffer(fp.read(), np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            continue

        comps = {}
        for label in LABEL_LIST:
            cn = (f"{ZIP_PREFIX}CelebAMask-HQ-mask-anno/"
                  f"{folder}/{k:05d}_{label}.png")
            if cn not in names:
                continue
            with zf.open(cn) as fp:
                comp = cv2.imdecode(np.frombuffer(fp.read(), np.uint8),
                                    cv2.IMREAD_GRAYSCALE)
            if comp is not None:
                comps[label] = comp
        merged = _merge_components(comps)

        skin = _emit_sample(k, arr, merged, out_img, out_mask, out_skin)
        written.append(k)
        print(f"[fetch] id={k} 组件 {len(comps)}/{len(LABEL_LIST)} "
              f"皮肤占比={skin.mean()/255*100:.1f}% 已拉取={rf.bytes_fetched/1e6:.1f}MB")

    _write_manifest(written, manifest_path)
    print(f"[fetch] 已输出 {len(written)} 张到 {out}"
          f"（跳过缺皮肤标注 {skipped_no_skin} 张）")
    print(f"[fetch] Range 请求 {rf.requests} 次，共拉取 {rf.bytes_fetched/1e6:.1f}MB，"
          f"相比整包 {size/1e6:.0f}MB 节省 {100*(1-rf.bytes_fetched/size):.1f}%")
    return written

# --- CLI ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="CelebAMask-HQ 数据预处理")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("fetch", help="从远程按需抽取子集（推荐，不下载整包）")
    p0.add_argument("--out", required=True, help="子集输出目录")
    p0.add_argument("--num", type=int, default=40, help="抽取张数")
    p0.add_argument("--seed", type=int, default=42)
    p0.add_argument("--manifest", default="assets/evaluation_manifest.csv")
    p0.add_argument("--zip-url", default=None, help="覆盖默认的公开镜像地址")
    p0.add_argument("--pool", type=int, default=30000,
                    help="候选 ID 池大小（仅用于抽样，不下载/不处理全量）")

    p1 = sub.add_parser("merge", help="合并指定样本的分散掩码（仅限已抽取的样本）")
    p1.add_argument("--root", required=True, help="CelebAMask-HQ 根目录")
    p1.add_argument("--ids", required=True,
                    help="要合并的样本 ID，逗号分隔（如 10507,11300,12002）")
    p1.add_argument("--save-dir", default=None)

    p2 = sub.add_parser("subset", help="从本地目录抽取子集并生成评测清单")
    p2.add_argument("--root", required=True, help="CelebAMask-HQ 根目录")
    p2.add_argument("--out", required=True, help="子集输出目录")
    p2.add_argument("--num", type=int, default=40)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--manifest", default="assets/evaluation_manifest.csv",
                    help="评测清单输出路径（相对仓库根目录）")

    args = ap.parse_args()
    if args.cmd == "fetch":
        fetch_subset(args.out, args.num, args.seed, args.manifest,
                     args.zip_url, args.pool)
    elif args.cmd == "merge":
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
        merge_masks(args.root, ids, args.save_dir)
    else:
        build_subset(args.root, args.out, args.num, args.seed, args.manifest)

if __name__ == "__main__":
    main()