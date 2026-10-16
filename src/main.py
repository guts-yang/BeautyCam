#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""main.py —— CLI 薄入口（A 独占）。

只调用 core 层，不直接调用任何算法模块。
支持单图处理与目录批处理；失败项汇总写入 results/<run_id>/failures.csv 后继续下一张。
对齐 arch 7.3：一次批处理共享一个 run_id，每个样本各有 sample_id。
"""

from __future__ import annotations

import argparse
import os
import sys

from beauty.core.config import load_module_config, load_pipeline_config
from beauty.core.contracts import BatchOutcome
from beauty.core.pipeline import new_run_id, run_pipeline
from beauty.io.reader import ImageReadError, read_image, read_mask
from beauty.io.writer import write_failures_csv

_MODULES = ["skin_detect", "exposure", "smooth", "freq_enhance", "restore", "color", "metrics"]
_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _load_all_module_cfgs() -> dict:
    return {n: load_module_config(n) for n in _MODULES}


def _scrub_paths(message: str, abs_to_rel: dict) -> str:
    """把消息中的绝对路径替换为相对标识，避免泄露本机路径（arch 7.2/7.3）。"""
    for abs_p, rel_p in abs_to_rel.items():
        if abs_p:
            message = message.replace(abs_p, rel_p)
    return message


def _failure_row(run_id: str, rel_path: str, stage: str,
                 error_type: str, error_message: str) -> dict:
    """构造一条 failures.csv 记录；路径保持相对，不含本机绝对路径。"""
    return {
        "run_id": run_id,
        "input_relative_path": rel_path,
        "stage": stage,
        "error_type": error_type,
        "error_message": error_message,
    }


def _scrub_failures(rows, abs_to_rel):
    """逐条清洗失败记录 error_message 中的绝对路径。"""
    scrubbed = []
    for row in rows:
        item = dict(row)
        item["error_message"] = _scrub_paths(str(item.get("error_message", "")), abs_to_rel)
        scrubbed.append(item)
    return scrubbed


def _run_single(args, pipeline_cfg, module_cfgs, sample_id, output_root, run_id,
                input_relative_path=None):
    """处理单张图，返回 PipelineResult；读图失败时抛出并由调用方记录。"""
    img = read_image(args.input)
    mask = None
    if getattr(args, "skin_mask", None) and os.path.isfile(args.skin_mask):
        mask = read_mask(args.skin_mask, img.shape[:2])
    regions = {"skin_gt": mask} if mask is not None else {}
    return run_pipeline(img, pipeline_cfg, module_cfgs, mode=args.mode,
                        output_root=output_root, sample_id=sample_id, regions=regions,
                        run_id=run_id, input_relative_path=input_relative_path)


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


def _abs_to_rel_map(input_path: str, skin_mask: str | None = None) -> dict:
    """构造绝对路径 -> 相对标识 的映射，用于失败消息脱敏。"""
    mapping = {os.path.abspath(input_path): os.path.basename(os.path.abspath(input_path))}
    if skin_mask:
        mapping[os.path.abspath(skin_mask)] = os.path.basename(skin_mask)
    return mapping


def _run_batch(args, pipeline_cfg, module_cfgs, output_root):
    """目录批处理：共享同一 run_id，逐张处理并汇总失败项。"""
    files = sorted(f for f in os.listdir(args.input)
                   if os.path.splitext(f)[1].lower() in _EXTS)
    run_id = new_run_id()
    outcome = BatchOutcome(run_id=run_id, total=len(files), succeeded=0, failed=0)
    input_root_name = os.path.basename(os.path.abspath(args.input))
    abs_to_rel = _abs_to_rel_map(args.input, getattr(args, "skin_mask", None))

    for idx, name in enumerate(files):
        rel_path = os.path.join(input_root_name, name)
        single = argparse.Namespace(**vars(args))
        single.input = os.path.join(args.input, name)
        try:
            result = _run_single(single, pipeline_cfg, module_cfgs, f"{idx:04d}",
                                 output_root, run_id, rel_path)
        except (ImageReadError, OSError) as exc:  # 读图失败也记录后继续
            message = _scrub_paths(str(exc), abs_to_rel)
            outcome.failures.append(
                _failure_row(run_id, rel_path, "read", type(exc).__name__, message))
            outcome.failed += 1
            outcome.messages.append(f"[失败] {name}: {type(exc).__name__}: {message}")
            continue
        outcome.messages.append(result.status_text())
        outcome.failures.extend(_scrub_failures(result.failures, abs_to_rel))
        if result.succeeded:
            outcome.succeeded += 1
        else:
            outcome.failed += 1

    if outcome.failures:
        write_failures_csv(os.path.join(output_root, "results", run_id, "failures.csv"),
                           outcome.failures)
    return outcome


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = os.path.abspath(args.output_root or os.getcwd())
    pipeline_cfg = load_pipeline_config()
    module_cfgs = _load_all_module_cfgs()

    if os.path.isdir(args.input):
        outcome = _run_batch(args, pipeline_cfg, module_cfgs, output_root)
        for line in outcome.messages:
            print(line)
        print(outcome.status_text())
        return 0

    abs_to_rel = _abs_to_rel_map(args.input, getattr(args, "skin_mask", None))
    try:
        result = _run_single(args, pipeline_cfg, module_cfgs, None, output_root, None,
                             os.path.basename(args.input))
    except (ImageReadError, OSError) as exc:
        print(f"处理失败：{type(exc).__name__}: {_scrub_paths(str(exc), abs_to_rel)}")
        return 1
    if result.failures:
        write_failures_csv(
            os.path.join(output_root, "results", result.run_id or "unknown", "failures.csv"),
            _scrub_failures(result.failures, abs_to_rel))
    print(result.status_text())
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    sys.exit(main())