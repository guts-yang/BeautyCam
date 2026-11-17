"""M1 离线隔离评测：固定清单、逐图 IoU/Recall、后处理消融与对照图。

从仓库根目录运行：
    python tools/evaluate_m1.py --data-root data/input/subset
仅评测 M1，不调用其他图像处理模块；结果默认写入被 Git 忽略的 results/。
本工具用于 B 的 #17/#18 证据，不替代 C 的 M7 正式指标接口。
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

import cv2
import numpy as np
import yaml


def mask_scores(predicted: np.ndarray, ground_truth: np.ndarray) -> dict:
    """空 GT 无分数；其余逐图计算，零预测和零交集均是有效的零分。"""
    if predicted.ndim != 2 or predicted.shape != ground_truth.shape:
        raise ValueError("预测与 GT 必须为同尺寸二维掩码")
    pred, gt = predicted != 0, ground_truth != 0
    if not np.any(gt):
        return {"iou": None, "recall": None, "valid": False, "reason": "empty_gt"}
    tp = int(np.count_nonzero(pred & gt))
    union = int(np.count_nonzero(pred | gt))
    return {"iou": tp / union, "recall": tp / int(np.count_nonzero(gt)),
            "valid": True, "reason": ""}


def summarize_scores(records: list[dict]) -> dict:
    """每张图等权，排除空 GT，保留有效数和中位数。"""
    valid = [record for record in records if record["valid"]]
    result = {"total_samples": len(records), "valid_samples": len(valid),
              "empty_gt_samples": sum(r["reason"] == "empty_gt" for r in records)}
    for key in ("iou", "recall"):
        values = [record[key] for record in valid]
        result[f"{key}_macro_mean"] = float(np.mean(values)) if values else None
        result[f"{key}_median"] = float(np.median(values)) if values else None
    return result


def _read(path: Path, flag: int) -> np.ndarray:
    # imdecode 支持包含中文的 Windows 路径。
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), flag)
    if image is None:
        raise ValueError(f"无法解码：{path.name}")
    return image


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"编码失败：{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def _panel(image: np.ndarray, gt: np.ndarray, raw: np.ndarray,
           processed: np.ndarray, sample_id: str, scores: dict) -> np.ndarray:
    overlay = image.copy()
    overlay[processed != 0] = (
        0.6 * overlay[processed != 0] + 0.4 * np.array([0, 220, 0])
    ).astype(np.uint8)
    panels = []
    for title, data in [("input", image), ("GT", gt), ("post off", raw),
                        ("post on", processed), ("prediction overlay", overlay)]:
        if data.ndim == 2:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        tile = cv2.resize(data, (256, 256), interpolation=(
            cv2.INTER_NEAREST if title in {"GT", "post off", "post on"} else cv2.INTER_AREA))
        tile = cv2.copyMakeBorder(tile, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
        cv2.putText(tile, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
        panels.append(tile)
    panel = np.hstack(panels)
    panel = cv2.copyMakeBorder(panel, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    label = (f'{sample_id}  IoU={scores["iou"]:.3f} Recall={scores["recall"]:.3f}'
             if scores["valid"] else f"{sample_id}  empty GT: excluded")
    cv2.putText(panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    return panel


def evaluate(data_root: Path, manifest: Path, cfg: dict, output: Path) -> dict:
    """对固定 manifest 完整评测；缺少图像/GT 时失败，不静默缩小样本集。"""
    from beauty.modules import skin_detect

    manifest_bytes = manifest.read_bytes()
    rows = list(csv.DictReader(manifest_bytes.decode("utf-8-sig").splitlines()))
    if not rows or any("sample_id" not in row or "scene" not in row for row in rows):
        raise ValueError("manifest 必须包含 sample_id,scene 和至少一个样本")
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids) or any(not x.isascii() or not x.isdecimal() for x in ids):
        raise ValueError("sample_id 必须是不重复的数字 ID")
    if output.exists():
        raise ValueError("输出目录已存在，请使用新目录，避免覆盖评测证据")
    off, on = deepcopy(cfg), deepcopy(cfg)
    off.setdefault("postprocess", {})["enabled"] = False
    on.setdefault("postprocess", {})["enabled"] = True
    records, raw_scores, processed_scores, panels, masks = [], [], [], [], []
    for row in rows:
        sample = row["sample_id"]
        image = _read(data_root / "images" / f"{sample}.jpg", cv2.IMREAD_COLOR)
        gt = _read(data_root / "skin_masks" / f"{sample}.png", cv2.IMREAD_GRAYSCALE)
        if not set(np.unique(gt)).issubset({0, 255}):
            raise ValueError(f"{sample} 的皮肤 GT 必须是 0/255 二值掩码")
        if gt.shape != image.shape[:2]:
            gt = cv2.resize(gt, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        raw, processed = skin_detect.detect_skin(image, off), skin_detect.detect_skin(image, on)
        for mask in (raw, processed):
            if mask.dtype != np.uint8 or mask.shape != gt.shape or not set(np.unique(mask)).issubset({0, 255}):
                raise AssertionError("M1 输出契约失败")
        before, after = mask_scores(raw, gt), mask_scores(processed, gt)
        raw_scores.append(before)
        processed_scores.append(after)
        records.append({"sample_id": sample, "scene": row["scene"],
                        "valid": after["valid"], "not_applicable_reason": after["reason"],
                        "iou_post_off": before["iou"], "recall_post_off": before["recall"],
                        "iou_post_on": after["iou"], "recall_post_on": after["recall"],
                        "gt_pixels": int(np.count_nonzero(gt))})
        panels.append(_panel(image, gt, raw, processed, sample, after))
        for name, mask in (("post_off", raw), ("post_on", processed)):
            ok, encoded = cv2.imencode(".png", mask)
            if not ok:
                raise ValueError("预测掩码编码失败")
            masks.append((f"{sample}_{name}.png", encoded.tobytes()))
        # 先收集完整清单再落盘，避免缺图被误认为全量评测。
    raw_summary, processed_summary = summarize_scores(raw_scores), summarize_scores(processed_scores)
    delta = (processed_summary["iou_macro_mean"] - raw_summary["iou_macro_mean"]
             if processed_summary["valid_samples"] else None)
    summary = {"profile": "module_isolated", "target_module": "skin_detect",
               "algorithm_sha256": hashlib.sha256(Path(skin_detect.__file__).read_bytes()).hexdigest(),
               "runtime": {"python": sys.version.split()[0], "opencv": cv2.__version__, "numpy": np.__version__},
               "ground_truth": "skin_masks, nearest-neighbor alignment",
               "scene_labels": "unverified; copied from manifest, not evidence of stratification",
               "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "config_post_off": off, "config_post_on": on,
               "post_off": raw_summary, "post_on": processed_summary,
               "postprocess_iou_delta": delta,
               "thresholds": {"iou": 0.75, "recall": 0.80, "min_postprocess_iou_delta": -0.02},
               "acceptance": {
                   "iou": processed_summary["iou_macro_mean"] is not None and processed_summary["iou_macro_mean"] >= 0.75,
                   "recall": processed_summary["recall_macro_mean"] is not None and processed_summary["recall_macro_mean"] >= 0.80,
                   "postprocess_regression": delta is not None and delta >= -0.02}}
    output.mkdir(parents=True)
    with (output / "samples.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row, panel in zip(rows, panels):
        _write_png(output / "figures" / f'{row["sample_id"]}.png', panel)
    (output / "predictions").mkdir()
    for name, encoded in masks:
        (output / "predictions" / name).write_bytes(encoded)
    ranked = sorted(range(len(rows)), key=lambda i: processed_scores[i]["iou"] if processed_scores[i]["valid"] else -1)
    selected = list(dict.fromkeys(ranked[:4] + ranked[-4:]))
    _write_png(output / "contact_sheet.png", np.vstack([panels[i] for i in selected]))
    return summary


def main(argv=None) -> int:
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repo / "data/input/subset")
    parser.add_argument("--manifest", type=Path, default=repo / "assets/evaluation_manifest.csv")
    parser.add_argument("--config", type=Path, default=repo / "config/skin_detect.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output or repo / "results" / ("m1_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6])
    summary = evaluate(args.data_root, args.manifest, cfg, output)
    print(json.dumps({"output": str(output), "post_off": summary["post_off"],
                      "post_on": summary["post_on"], "acceptance": summary["acceptance"]},
                     ensure_ascii=False, indent=2))
    return 0 if all(summary["acceptance"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
