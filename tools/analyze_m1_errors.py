"""M1 误检来源与连通域选择能力诊断；不修改检测算法或正式验收口径。

python tools/analyze_m1_errors.py --data-root data/input/subset
GT 仅用于离线归因。连通域 oracle 是诊断上界，绝不是可部署预测成绩。
"""
from __future__ import annotations

import argparse
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


def component_selection_ceiling(predicted, ground_truth):
    """仅允许保留/删除既有整块连通域时，用 GT 求最大可达 IoU。

    IoU=sum(TP)/(GT+sum(FP))；按 TP/FP 降序尝试各前缀即可找到最优。
    FP=0 的真阳性组件排最前；全假阳性组件不会改善分数。
    该函数接收 GT，仅能离线诊断，禁止拿它当运行时后处理。
    """
    if predicted.ndim != 2 or predicted.shape != ground_truth.shape:
        raise ValueError("预测与 GT 必须为同尺寸二维掩码")
    gt = ground_truth != 0
    gt_count = int(gt.sum())
    if not gt_count:
        return {"iou_ceiling": None, "component_count": None, "reason": "empty_gt"}
    number, labels, stats, _ = cv2.connectedComponentsWithStats(
        (predicted != 0).astype(np.uint8), connectivity=8)
    if number == 1:
        return {"iou_ceiling": 0.0, "component_count": 0, "reason": ""}
    tp = np.bincount(labels[gt], minlength=number)[1:].astype(np.float64)
    fp = stats[1:, cv2.CC_STAT_AREA].astype(np.float64) - tp
    ratio = np.divide(tp, fp, out=np.full_like(tp, np.inf), where=fp != 0)
    order = np.argsort(-ratio, kind="stable")
    ious = np.cumsum(tp[order]) / (gt_count + np.cumsum(fp[order]))
    return {"iou_ceiling": float(np.max(ious)), "component_count": number - 1, "reason": ""}


def label_error_counts(predicted, labels, skin_gt):
    """逐标签计数；确认 skin_gt 与本项目合并标签的 skin==1 相同。"""
    if predicted.ndim != 2 or predicted.shape != labels.shape or labels.shape != skin_gt.shape:
        raise ValueError("预测、标签、皮肤 GT 必须同尺寸二维")
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels > 18):
        raise ValueError("合并标签必须为 0..18 整数")
    if not np.array_equal(skin_gt != 0, labels == 1):
        raise ValueError("skin_gt 与合并标签 skin==1 不一致")
    pred, gt = predicted != 0, skin_gt != 0
    counts = np.bincount(labels.ravel(), minlength=19)
    selected = np.bincount(labels[pred], minlength=19)
    fp = selected.copy()
    fp[1] = 0
    tp = int(selected[1])
    gt_count = int(counts[1])
    union = gt_count + int(fp.sum())
    return {"class_pixels": counts, "selected_pixels": selected, "fp_pixels": fp,
            "tp": tp, "fp": int(fp.sum()), "fn": gt_count - tp,
            "iou": tp / union if gt_count else None,
            "recall": tp / gt_count if gt_count else None,
            "precision": tp / int(pred.sum()) if np.any(pred) else 0.0}


def _read(path, mode):
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), mode)
    if image is None:
        raise ValueError(f"无法解码：{path.name}")
    return image


def _write_png(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("图像编码失败")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def _panel(image, skin, prediction, sample, scores):
    pred, gt = prediction != 0, skin != 0
    error = image.copy()
    error[pred & gt] = (40, 190, 40)
    error[pred & ~gt] = (30, 30, 230)
    error[~pred & gt] = (230, 70, 30)
    tiles = []
    for label, data in (("input", image), ("skin GT", skin), ("prediction", prediction),
                        ("TP green / FP red / FN blue", error)):
        if data.ndim == 2:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        tile = cv2.resize(data, (320, 320), interpolation=cv2.INTER_AREA if label == "input" else cv2.INTER_NEAREST)
        tile = cv2.copyMakeBorder(tile, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(250, 250, 250))
        cv2.putText(tile, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(tile)
    panel = cv2.copyMakeBorder(np.hstack(tiles), 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    title = (f'{sample} | IoU={scores["iou"]:.4f} | Recall={scores["recall"]:.4f}'
             if scores["iou"] is not None else f"{sample} | empty GT, no score")
    cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    return panel


def analyze(data_root, manifest, cfg, output, predictions=None):
    from beauty.modules import skin_detect

    data_root, manifest, output = Path(data_root), Path(manifest), Path(output)
    predictions = Path(predictions) if predictions is not None else None
    if output.exists():
        raise ValueError("输出目录已存在，请使用新目录")
    contents = manifest.read_bytes()
    samples = list(csv.DictReader(contents.decode("utf-8-sig").splitlines()))
    if not samples or any(not {"sample_id", "scene"} <= row.keys() for row in samples):
        raise ValueError("manifest 必须含 sample_id,scene 和至少一个样本")
    ids = [row["sample_id"] for row in samples]
    if len(ids) != len(set(ids)) or any(not s.isascii() or not s.isdecimal() for s in ids):
        raise ValueError("sample_id 必须是不重复的数字 ID")
    for sample in ids:
        for folder, suffix in (("images", ".jpg"), ("skin_masks", ".png"), ("masks", ".png")):
            if not (data_root / folder / (sample + suffix)).is_file():
                raise FileNotFoundError(f"固定样本 {sample} 缺少 {folder}")
        if predictions is not None and not (predictions / f"{sample}.png").is_file():
            raise FileNotFoundError(f"固定样本 {sample} 缺少保存的预测")
    names = ["background", "skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow",
             "l_ear", "r_ear", "mouth", "u_lip", "l_lip", "hair", "hat", "ear_r",
             "neck_l", "neck", "cloth"]
    records, classes = [], []
    total_class, total_fp = np.zeros(19, np.int64), np.zeros(19, np.int64)
    prediction_digest = hashlib.sha256()
    output.mkdir(parents=True, exist_ok=False)
    for row in samples:
        sample = row["sample_id"]
        image = _read(data_root / "images" / f"{sample}.jpg", cv2.IMREAD_COLOR)
        skin = _read(data_root / "skin_masks" / f"{sample}.png", cv2.IMREAD_GRAYSCALE)
        labels = _read(data_root / "masks" / f"{sample}.png", cv2.IMREAD_GRAYSCALE)
        if not set(np.unique(skin)).issubset({0, 255}):
            raise ValueError("皮肤 GT 必须是 0/255")
        size = (image.shape[1], image.shape[0])
        skin = cv2.resize(skin, size, interpolation=cv2.INTER_NEAREST)
        labels = cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST)
        original = image.copy()
        pred = (skin_detect.detect_skin(image, cfg) if predictions is None else
                _read(predictions / f"{sample}.png", cv2.IMREAD_GRAYSCALE))
        if (pred.dtype != np.uint8 or pred.shape != skin.shape
                or not set(np.unique(pred)).issubset({0, 255})
                or not np.array_equal(image, original)):
            raise AssertionError("M1 输出/输入不变性契约失败")
        prediction_digest.update(sample.encode("ascii") + b"\0" + pred.tobytes())
        errors = label_error_counts(pred, labels, skin)
        ceiling = component_selection_ceiling(pred, skin)
        records.append({"sample_id": sample, "scene": row["scene"],
                        **{key: errors[key] for key in ("tp", "fp", "fn", "iou", "recall", "precision")},
                        **ceiling})
        for index, name in enumerate(names):
            classes.append({"sample_id": sample, "label_id": index, "label": name,
                            "class_pixels": int(errors["class_pixels"][index]),
                            "selected_pixels": int(errors["selected_pixels"][index]),
                            "fp_pixels": int(errors["fp_pixels"][index])})
        total_class += errors["class_pixels"]
        total_fp += errors["fp_pixels"]
        _write_png(output / "figures" / f"{sample}.png", _panel(image, skin, pred, sample, errors))
    valid = [row for row in records if row["iou"] is not None]
    source = [{"label_id": index, "label": name, "class_pixels": int(total_class[index]),
               "fp_pixels": int(total_fp[index]),
               "share_of_all_fp": float(total_fp[index] / total_fp.sum()) if total_fp.sum() else 0.0}
              for index, name in enumerate(names)]
    source.sort(key=lambda row: (-row["fp_pixels"], row["label_id"]))
    metrics = {key + "_macro_mean": float(np.mean([row[key] for row in valid])) if valid else None
               for key in ("iou", "recall", "precision")}
    summary = {
        "scope": "offline diagnosis; GT never supplied to detect_skin",
        "config": cfg, "total_samples": len(records), "valid_samples": len(valid),
        "prediction_source": "current M1 module" if predictions is None else "saved candidate masks",
        "prediction_metrics": metrics,
        "baseline": metrics if predictions is None else None,
        "component_selection_oracle_iou_macro_mean":
            float(np.mean([row["iou_ceiling"] for row in valid])) if valid else None,
        "oracle_scope": "exact GT-assisted upper bound only for selecting whole 8-connected components "
                        "of the current fixed prediction; not a deployable algorithm or a bound on all M1 methods",
        "fp_pixels": int(total_fp.sum()), "false_positive_sources_pixel_weighted": source,
        "gt_consistency": "all skin masks exactly equal nearest-aligned merged labels==1",
        "manifest_sha256": hashlib.sha256(contents).hexdigest(),
        "algorithm_sha256": (hashlib.sha256(Path(skin_detect.__file__).read_bytes()).hexdigest()
                             if predictions is None else None),
        "predicted_arrays_sha256": prediction_digest.hexdigest(),
        "runtime": {"python": sys.version.split()[0], "opencv": cv2.__version__, "numpy": np.__version__},
        "scene_labels": "unverified manifest placeholders",
    }
    for name, rows in (("samples.csv", records), ("label_errors.csv", classes), ("fp_sources.csv", source)):
        with (output / name).open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
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
    parser.add_argument("--predictions", type=Path, help="可选：审计已保存的原尺寸二值预测，不能提供 GT 代替预测")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or repo / "results" / ("m1_errors_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                               + "_" + uuid.uuid4().hex[:6])
    summary = analyze(args.data_root, args.manifest, yaml.safe_load(args.config.read_text(encoding="utf-8")), output,
                      args.predictions)
    print(json.dumps({"output": str(output), "metrics": summary["prediction_metrics"],
                      "component_selection_oracle_iou_macro_mean":
                          summary["component_selection_oracle_iou_macro_mean"],
                      "top_fp_sources": summary["false_positive_sources_pixel_weighted"][:5]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
