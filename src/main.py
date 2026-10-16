#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""main.py —— CLI 薄入口（A 独占）。

本文件在 PR#1（E0 地基）阶段仅提供参数骨架，保证 `python src/main.py --help`
可运行（I-01 验收）；实际单图/批处理编排在 PR#2（E1，I-06）中接入 core.pipeline。
"""

from __future__ import annotations

import argparse
import sys


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
    print(f"BeautyCam CLI（地基骨架）：input={args.input} mode={args.mode}")
    print("主流程编排将于 E1 阶段接入 core.pipeline。")
    return 0


if __name__ == "__main__":
    sys.exit(main())