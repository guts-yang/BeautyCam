#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""main.py —— CLI 薄入口（A 独占）。

只调用 core 层，不直接调用任何算法模块。
支持单图处理与目录批处理；失败项写入 results/<run_id>/failures.csv 后继续下一张。
"""

from __future__ import annotations

import argparse
import os
import sys

from beauty.core.config import load_module_config, load_pipeline_config
from beauty.core.pipeline import run_pipeline
from beauty.io.reader import read_image, read_mask

_MODULES = ["skin_detect", "exposure", "smooth", "freq_enhance", "restore", "color", "metrics"]
_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _load_all_module_cfgs() -> dict:
    return {n: load_module_config(n) for n in _MODULES}


def _run_single(args, pipeline_cfg, module_cfgs, sample_id: str | None, output_root: str) -> bool:
    img = read_image(args.input)
    mask = None
    if getattr(args, "skin_mask", None) and os.path.isfile(args.skin_mask):
        mask = read_mask(args.skin_mask, img.shape[:2])
    regions = {"skin_gt": mask} if mask is not None else {}
    result = run_pipeline(img, pipeline_cfg, module_cfgs, mode=args.mode,
                          output_root=output_root, sample_id=sample_id, regions=regions)
    print(result.status_text())
    return result.succeeded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beautycam",
        description="焕颜相机 BeautyCam CLI：人脸美颜单图/批处理")
    parser.add_argument("--input", required=True, help="输入图片路径或目录（批处理）")
    parser.add_argument("--output-root", default=None,
                        help="项目根目录（决定 data/output 与 results 落点），默认当前目录")
    parser.add_argument("--skin-mask", default=None, help="可选：GT 皮肤掩码路径（评测用）")
    parser.add_argument("--mode", choices=["process", "evaluation"], default="process")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = os.path.abspath(args.output_root or os.getcwd())
    pipeline_cfg = load_pipeline_config()
    module_cfgs = _load_all_module_cfgs()

    if os.path.isdir(args.input):
        files = sorted(f for f in os.listdir(args.input)
                       if os.path.splitext(f)[1].lower() in _EXTS)
        ok = 0
        for idx, name in enumerate(files):
            single = argparse.Namespace(**vars(args))
            single.input = os.path.join(args.input, name)
            try:
                if _run_single(single, pipeline_cfg, module_cfgs, f"{idx:04d}", output_root):
                    ok += 1
            except Exception as exc:  # 失败隔离：记录后继续下一个样本
                print(f"[失败] {name}: {type(exc).__name__}: {exc}")
                continue
        print(f"批处理完成：成功 {ok}/{len(files)}")
        return 0

    return 0 if _run_single(args, pipeline_cfg, module_cfgs, None, output_root) else 1


if __name__ == "__main__":
    sys.exit(main())