"""B四模块真实组合复查：使用M1预测掩码，保存全部固定样本和曝光边界测量。

GT仅在处理完成后计算M1覆盖率，不传给任何处理模块。此工具不替代隔离
定量门槛、两名非实现者的视觉评分或A的M1–M7整套程序验收。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml


def boundary_change(before, after, mask):
    """测量新增亮度在掩码内侧一像素边界的幅度，避免混淆原图固有边缘。"""
    inner = cv2.erode(mask, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=255)
    edge = (mask != 0) & (inner == 0)
    if not np.any(edge):
        return {"pixels": 0, "mean": None, "p95": None, "max": None}
    y_before = cv2.cvtColor(before, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    y_after = cv2.cvtColor(after, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    values = np.abs(y_after[edge] - y_before[edge])
    return {"pixels": int(len(values)), "mean": float(values.mean()),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def _read(path, mode):
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), mode)
    if image is None:
        raise ValueError(f"无法解码 {path}")
    return image


def _save(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"无法编码 {path}")
    path.write_bytes(data.tobytes())


def main(argv=None):
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repo/"data/input/subset")
    parser.add_argument("--manifest", type=Path, default=repo/"assets/evaluation_manifest.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        samples = [row["sample_id"] for row in csv.DictReader(stream)]
    if (not samples or len(set(samples)) != len(samples)
            or any(not value.isascii() or not value.isdecimal() for value in samples)):
        raise ValueError("清单必须含不重复的十进制样本编号")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(repo/"src"))
    from beauty.modules import skin_detect, exposure, smooth, freq_enhance

    names = ["skin_detect", "exposure", "smooth", "freq_enhance"]
    cfg = {name: yaml.safe_load((repo/"config"/f"{name}.yaml").read_text(encoding="utf-8")) for name in names}
    hard_cfg = deepcopy(cfg["exposure"])
    hard_cfg["mask_blend"] = {"radius_ratio": 0}
    records, panels = [], []
    for sample in samples:
        before = _read(args.data_root/"images"/f"{sample}.jpg", cv2.IMREAD_COLOR)
        mask = skin_detect.detect_skin(before, cfg["skin_detect"])
        exposed = exposure.process(before, mask, cfg["exposure"])
        hard = exposure.process(before, mask, hard_cfg)
        result = freq_enhance.process(smooth.process(exposed, mask, cfg["smooth"]), mask, cfg["freq_enhance"])
        if result.shape != before.shape or result.dtype != np.uint8:
            raise AssertionError("组合输出格式错误")
        if not np.array_equal(before[mask == 0], result[mask == 0]):
            raise AssertionError("组合处理越过M1掩码")
        # GT只在推理全部结束后用于离线覆盖率计算。
        gt = _read(args.data_root/"skin_masks"/f"{sample}.png", cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (before.shape[1], before.shape[0]), interpolation=cv2.INTER_NEAREST) != 0
        pred = mask != 0
        intersection, union = np.count_nonzero(pred & gt), np.count_nonzero(pred | gt)
        soft_boundary, hard_boundary = boundary_change(before, exposed, mask), boundary_change(before, hard, mask)
        record = {"sample_id": sample, "iou": intersection/union if union else None,
                  "recall": intersection/np.count_nonzero(gt) if np.any(gt) else None,
                  "mask_pixels": int(pred.sum()), "boundary_pixels": soft_boundary["pixels"],
                  "exposure_boundary_mean": soft_boundary["mean"], "exposure_boundary_p95": soft_boundary["p95"],
                  "hard_control_boundary_mean": hard_boundary["mean"], "hard_control_boundary_p95": hard_boundary["p95"]}
        records.append(record)
        overlay = before.copy()
        overlay[pred] = (.6*overlay[pred]+.4*np.array([0, 220, 0])).astype(np.uint8)
        tiles = []
        for label, image in [("input", before), ("predicted skin", overlay), ("B M1-M4", result)]:
            tile = cv2.copyMakeBorder(cv2.resize(image, (384, 384)), 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(250, 250, 250))
            cv2.putText(tile, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .6, (30, 30, 30), 1)
            tiles.append(tile)
        panel = cv2.copyMakeBorder(np.hstack(tiles), 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(panel, sample, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .6, (30, 30, 30), 1)
        panels.append(panel)
        _save(args.output/"figures"/f"{sample}.png", panel)
        _save(args.output/"outputs"/f"{sample}.png", result)
        _save(args.output/"masks"/f"{sample}.png", mask)
        print(json.dumps(record), flush=True)
    with (args.output/"samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for start in range(0, len(panels), 8):
        _save(args.output/f"contact_{start//8+1:02d}.png", np.vstack(panels[start:start+8]))
    summary = {"scope": "B M1-M4 actual predicted-mask composite; not full M1-M7 pipeline",
               "samples": len(records), "iou_macro_mean": float(np.mean([r["iou"] for r in records if r["iou"] is not None])),
               "recall_macro_mean": float(np.mean([r["recall"] for r in records if r["recall"] is not None])),
               "subjective_acceptance": "pending; boundary measurements do not establish visual acceptance",
               "config": cfg, "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               "algorithm_sha256": {name: hashlib.sha256((repo/"src/beauty/modules"/f"{name}.py").read_bytes()).hexdigest() for name in names}}
    (args.output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
