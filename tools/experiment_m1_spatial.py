"""M1 传统 GrabCut 的离线候选试验，不修改生效 schema 或算法。

附加空间参数必须来自显式 JSON；GT 仅用于预测完成后的计分。
CLI 在每次候选前重置 OpenCV RNG，便于开发复现；不能原样搬进纯计算模块。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

import cv2
import numpy as np
import yaml


def spatial_prediction(image, cfg, options):
    """只接收图像/颜色配置/实验参数，不接收 GT 或文件路径。"""
    from beauty.modules import skin_detect

    limit = options["working_size"]
    scale = min(1.0, limit / max(image.shape[:2]))
    working = cv2.resize(image, (max(1, round(image.shape[1] * scale)),
                                max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    color_cfg = deepcopy(cfg)
    color_cfg.setdefault("postprocess", {})["enabled"] = False
    color = skin_detect.detect_skin(working, color_cfg) != 0
    seed_cfg = deepcopy(color_cfg)
    if "seed_ycbcr" in options:
        seed_cfg["ycbcr"] = options["seed_ycbcr"]
    seed_color = skin_detect.detect_skin(working, seed_cfg) != 0
    height, width = working.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    xx, yy = xx / max(width - 1, 1), yy / max(height - 1, 1)
    cx, cy = options["center"]
    rx, ry = options["roi_radius"]
    sx, sy = options["seed_radius"]
    broad = ((xx - cx) / rx)**2 + ((yy - cy) / ry)**2 <= 1
    seed = ((xx - cx) / sx)**2 + ((yy - cy) / sy)**2 <= 1
    mark = np.full(color.shape, cv2.GC_PR_BGD, np.uint8)
    mark[color & broad] = cv2.GC_PR_FGD
    mark[~broad] = cv2.GC_BGD
    mark[seed_color & seed & broad] = cv2.GC_FGD
    status = "ok"
    if not np.any(mark == cv2.GC_FGD) or not np.any(mark == cv2.GC_BGD):
        predicted = np.zeros_like(color)
        status = "insufficient_seeds_zero_prediction"
    else:
        cv2.grabCut(working, mark, None, np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), options["iterations"], cv2.GC_INIT_WITH_MASK)
        predicted = color & ((mark == cv2.GC_FGD) | (mark == cv2.GC_PR_FGD))
    mask = cv2.resize(predicted.astype(np.uint8) * 255, (image.shape[1], image.shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    return mask, status


def _candidates(items):
    if not isinstance(items, list) or not items:
        raise ValueError("候选 JSON 必须是非空列表")
    seen = set()
    for item in items:
        name = item.get("name", "")
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name) or name in seen:
            raise ValueError("候选名称必须是不重复的 lowercase_ascii_name")
        seen.add(name)
        for key in ("working_size", "iterations", "rng_seed"):
            value = item.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "rng_seed" else 1):
                raise ValueError(f"{key} 必须为合法整数")
        for key in ("center", "roi_radius", "seed_radius"):
            values = item.get(key)
            if (not isinstance(values, list) or len(values) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not np.isfinite(v) or not 0 < v <= 1 for v in values)):
                raise ValueError(f"{key} 必须为两个 (0,1] 有限数值")
        if any(seed > roi for seed, roi in zip(item["seed_radius"], item["roi_radius"])):
            raise ValueError("seed_radius 不能大于 roi_radius")
    return items


def _read(path, mode):
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), mode)
    if image is None:
        raise ValueError(f"无法解码：{path.name}")
    return image


def _write_png(path, data):
    ok, encoded = cv2.imencode(".png", data)
    if not ok:
        raise ValueError("编码失败")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def evaluate(data_root, manifest, cfg, candidates, output):
    data_root, manifest, output = Path(data_root), Path(manifest), Path(output)
    options = _candidates(candidates)
    if output.exists():
        raise ValueError("输出目录已存在，请使用新目录")
    contents = manifest.read_bytes()
    rows = list(csv.DictReader(contents.decode("utf-8-sig").splitlines()))
    if not rows or any("sample_id" not in row for row in rows):
        raise ValueError("manifest 必须含 sample_id 和至少一个样本")
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids) or any(not s.isascii() or not s.isdecimal() for s in ids):
        raise ValueError("sample_id 必须是不重复的数字 ID")
    for sample in ids:
        for folder, suffix in (("images", ".jpg"), ("skin_masks", ".png")):
            if not (data_root / folder / (sample + suffix)).is_file():
                raise FileNotFoundError(f"固定样本 {sample} 缺少 {folder}")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for row in rows:
        sample = row["sample_id"]
        image = _read(data_root / "images" / f"{sample}.jpg", cv2.IMREAD_COLOR)
        gt = _read(data_root / "skin_masks" / f"{sample}.png", cv2.IMREAD_GRAYSCALE)
        if not set(np.unique(gt)).issubset({0, 255}):
            raise ValueError("皮肤 GT 必须是 0/255")
        gt = cv2.resize(gt, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) != 0
        original = image.copy()
        for item in options:
            # RNG 写操作限定在实验 CLI 调度层；正式模块仍保持原实现。
            cv2.setRNGSeed(item["rng_seed"])
            pred, status = spatial_prediction(image, cfg, item)
            if (pred.shape != gt.shape or pred.dtype != np.uint8
                    or not set(np.unique(pred)).issubset({0, 255})
                    or not np.array_equal(image, original)):
                raise AssertionError("候选输入/输出契约失败")
            prediction = pred != 0
            tp, fp = int(np.count_nonzero(prediction & gt)), int(np.count_nonzero(prediction & ~gt))
            gt_count = int(gt.sum())
            records.append({"candidate": item["name"], "sample_id": sample, "status": status,
                            "iou": tp / (gt_count + fp) if gt_count else None,
                            "recall": tp / gt_count if gt_count else None,
                            "tp": tp, "fp": fp, "gt_pixels": gt_count})
            _write_png(output / "predictions" / item["name"] / f"{sample}.png", pred)
        print(f"evaluated {sample}", flush=True)
    scores = []
    for item in options:
        valid = [r for r in records if r["candidate"] == item["name"] and r["iou"] is not None]
        scores.append({"name": item["name"], "valid_samples": len(valid),
                       "iou_macro_mean": float(np.mean([r["iou"] for r in valid])) if valid else None,
                       "recall_macro_mean": float(np.mean([r["recall"] for r in valid])) if valid else None,
                       "seed_failures": sum(r["status"] != "ok" for r in records if r["candidate"] == item["name"])})
    scored = [r for r in scores if r["valid_samples"] == len(rows) and r["recall_macro_mean"] >= 0.80]
    best = max(scored, key=lambda r: r["iou_macro_mean"]) if scored else None
    summary = {"scope": "same fixed-set development experiments, no held-out validation; not production",
               "parameters": options, "color_config": cfg, "total_samples": len(rows),
               "candidates": scores, "best_with_recall_at_least_0_8": best,
               "manifest_sha256": hashlib.sha256(contents).hexdigest(),
               "experiment_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "runtime": {"python": sys.version.split()[0], "opencv": cv2.__version__, "numpy": np.__version__},
               "gt_usage": "scoring only after prediction; never passed to spatial_prediction",
               "rng_note": "CLI resets global OpenCV RNG before each experimental prediction; needs redesign before module integration"}
    for filename, data in (("samples.csv", records), ("candidate_scores.csv", scores)):
        with (output / filename).open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    return summary


def main(argv=None):
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repo / "data/input/subset")
    parser.add_argument("--manifest", type=Path, default=repo / "assets/evaluation_manifest.csv")
    parser.add_argument("--config", type=Path, default=repo / "config/skin_detect.yaml")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or repo / "results" / ("m1_spatial_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                               + "_" + uuid.uuid4().hex[:6])
    summary = evaluate(args.data_root, args.manifest, yaml.safe_load(args.config.read_text(encoding="utf-8")),
                       json.loads(args.candidates.read_text(encoding="utf-8")), output)
    print(json.dumps(summary["candidates"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
