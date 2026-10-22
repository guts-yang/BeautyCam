"""M2 隔离评测：固定清单、同一皮肤 GT 的亮度/过曝统计与直方图。

python tools/evaluate_m2.py --data-root data/input/subset
仅对输入皮肤 GT 平均 Y < 90 的真实欠曝样本判定提亮阈值。
现有 gamma/linear 分支不等于完成 Issue #19 的 CLAHE 要求。
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


def exposure_statistics(before, after, skin):
    """PRD 3.4：同一图、同一 GT，阈值固定 Y<90 / Y>250。"""
    region = skin != 0
    if not np.any(region):
        return {"mean_before_y": None, "mean_after_y": None, "relative_gain": None,
                "overexposure_before_ratio": None, "overexposure_after_ratio": None,
                "underexposed": False, "valid": False, "reason": "empty_skin_gt"}
    first, second = (cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0][region]
                     for image in (before, after))
    mean_before, mean_after = float(first.mean()), float(second.mean())
    return {"mean_before_y": mean_before, "mean_after_y": mean_after,
            "relative_gain": (mean_after - mean_before) / max(mean_before, 1),
            "overexposure_before_ratio": float(np.mean(first > 250)),
            "overexposure_after_ratio": float(np.mean(second > 250)),
            "underexposed": mean_before < 90, "valid": True, "reason": ""}


def _aggregate(values):
    return {"macro_mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None, "valid_samples": len(values)}


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


def _figure(path, sample, before, after, before_hist, after_hist, stats):
    # 使用项目已 pin 的 OpenCV 绘制线性坐标直方图，不引入可选绘图库到 CI。
    canvas = np.full((950, 1200, 3), 255, np.uint8)
    for position, image, title in ((0, before, "Input"), (1, after, "M2 output")):
        scale = min(570 / image.shape[1], 530 / image.shape[0])
        tile = cv2.resize(image, (max(1, round(image.shape[1] * scale)),
                                  max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        x, y = 15 + position * 600 + (570 - tile.shape[1]) // 2, 60
        canvas[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
        cv2.putText(canvas, title, (15 + position * 600, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (30, 30, 30), 2, cv2.LINE_AA)
    denominator = max(int(before_hist.sum()), 1)
    distributions = [before_hist / denominator, after_hist / denominator]
    peak = max(max(float(values.max()) for values in distributions), 1 / denominator)
    left, right, top, bottom = 105, 1150, 670, 870
    for tick in range(5):
        y = round(bottom - (bottom - top) * tick / 4)
        cv2.line(canvas, (left, y), (right, y), (225, 225, 225), 1)
        cv2.putText(canvas, f"{peak * tick / 4:.4f}", (12, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
    for value in (0, 64, 128, 192, 255):
        x = round(left + value * (right - left) / 255)
        cv2.line(canvas, (x, bottom), (x, bottom + 5), (60, 60, 60), 1)
        cv2.putText(canvas, str(value), (x - 14, bottom + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (left, top), (right, bottom), (80, 80, 80), 1)
    for values, color, label, label_x in zip(distributions, ((190, 90, 20), (30, 100, 210)),
                                            ("Before", "After"), (815, 1000)):
        points = np.column_stack((left + np.arange(256) * (right - left) / 255,
                                  bottom - values * (bottom - top) / peak))
        cv2.polylines(canvas, [np.rint(points).astype(np.int32)], False, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (label_x, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Pixel fraction per Y bin", (20, 650), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, "8-bit Y (same skin GT)", (455, 930), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (30, 30, 30), 1, cv2.LINE_AA)
    mean_label = ("empty skin GT" if not stats["valid"] else
                  f'mean Y: {stats["mean_before_y"]:.2f} -> {stats["mean_after_y"]:.2f}; '
                  f'gain: {100 * stats["relative_gain"]:.2f}%; '
                  f'after Y>250: {100 * stats["overexposure_after_ratio"]:.3f}%')
    cv2.putText(canvas, f"Sample {sample} | {mean_label}", (15, 615), cv2.FONT_HERSHEY_SIMPLEX,
                0.51, (30, 30, 30), 1, cv2.LINE_AA)
    _write_png(path, canvas)


def evaluate(data_root, manifest, cfg, output):
    from beauty.modules import exposure

    data_root, manifest, output = Path(data_root), Path(manifest), Path(output)
    if output.exists():
        raise ValueError("输出目录已存在，请使用新目录")
    manifest_bytes = manifest.read_bytes()
    samples = list(csv.DictReader(manifest_bytes.decode("utf-8-sig").splitlines()))
    if not samples or any(not {"sample_id", "scene"} <= row.keys() for row in samples):
        raise ValueError("manifest 必须含 sample_id,scene 和至少一个样本")
    ids = [row["sample_id"] for row in samples]
    if len(ids) != len(set(ids)) or any(not s.isascii() or not s.isdecimal() for s in ids):
        raise ValueError("sample_id 必须是不重复的数字 ID")
    for sample in ids:
        for folder, suffix in (("images", ".jpg"), ("skin_masks", ".png")):
            if not (data_root / folder / (sample + suffix)).is_file():
                raise FileNotFoundError(f"固定样本 {sample} 缺少 {folder}")
    output.mkdir(parents=True, exist_ok=False)
    records, histograms = [], []
    for row in samples:
        sample = row["sample_id"]
        before = _read(data_root / "images" / f"{sample}.jpg", cv2.IMREAD_COLOR)
        skin = _read(data_root / "skin_masks" / f"{sample}.png", cv2.IMREAD_GRAYSCALE)
        if not set(np.unique(skin)).issubset({0, 255}):
            raise ValueError("皮肤 GT 必须是 0/255")
        if skin.shape != before.shape[:2]:
            skin = cv2.resize(skin, (before.shape[1], before.shape[0]), interpolation=cv2.INTER_NEAREST)
        original, original_skin = before.copy(), skin.copy()
        after = exposure.process(before, skin, cfg)
        if after.shape != before.shape or after.dtype != np.uint8 or np.shares_memory(after, before):
            raise AssertionError("M2 图像输出契约失败")
        if not np.array_equal(before, original) or not np.array_equal(skin, original_skin):
            raise AssertionError("M2 修改输入")
        if not np.array_equal(after[skin == 0], before[skin == 0]):
            raise AssertionError("M2 修改了掩码外像素")
        stats = exposure_statistics(before, after, skin)
        records.append({"sample_id": sample, "scene": row["scene"], **stats})
        hist = [np.bincount(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0][skin != 0],
                           minlength=256) for image in (before, after)]
        histograms.extend({"sample_id": sample, "y_bin": value,
                           "before_count": int(hist[0][value]), "after_count": int(hist[1][value])}
                          for value in range(256))
        _write_png(output / "outputs" / f"{sample}.png", after)
        _figure(output / "figures" / f"{sample}.png", sample, before, after, *hist, stats)

    valid = [row for row in records if row["valid"]]
    under = [row for row in valid if row["underexposed"]]
    invalid_count = len(records) - len(valid)
    passed = bool(under) and invalid_count == 0
    summary = {
        "profile": "module_isolated", "target_module": "exposure", "mask_source": "skin_gt",
        "implementation_scope": "adaptive gamma/linear only; CLAHE schema change pending",
        "config": cfg, "total_samples": len(records), "valid_samples": len(valid),
        "invalid_samples": invalid_count,
        "underexposed_sample_ids": [row["sample_id"] for row in under],
        "underexposed_relative_gain": _aggregate([row["relative_gain"] for row in under]),
        "underexposed_after_overexposure": _aggregate([row["overexposure_after_ratio"] for row in under]),
        "all_samples_relative_gain": _aggregate([row["relative_gain"] for row in valid]),
        "all_samples_after_overexposure": _aggregate([row["overexposure_after_ratio"] for row in valid]),
        "preexisting_overexposure_ids": [row["sample_id"] for row in valid
                                       if row["overexposure_before_ratio"] > 0.01],
        "output_overexposure_exceedance_ids": [row["sample_id"] for row in valid
                                              if row["overexposure_after_ratio"] > 0.01],
        "quantitative_acceptance": {
            "underexposed_gain": passed and all(row["relative_gain"] >= 0.15 for row in under),
            "underexposed_overexposure": passed and all(row["overexposure_after_ratio"] <= 0.01 for row in under)},
        "full_issue_19_acceptance": False,
        "full_issue_19_pending_reason": "CLAHE plus adaptive gamma not yet implemented; schema review required",
        "protocol": {"underexposed_input_mean_y_lt": 90, "overexposed_output_y_gt": 250,
                     "overexposure_ratio_limit": 0.01, "relative_gain_min": 0.15,
                     "gt_resize": "nearest", "same_region_for_all_statistics": True,
                     "acceptance_scope": "each real underexposed sample, no synthetic replacements"},
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "algorithm_sha256": hashlib.sha256(Path(exposure.__file__).read_bytes()).hexdigest(),
        "runtime": {"python": sys.version.split()[0], "opencv": cv2.__version__, "numpy": np.__version__},
        "scene_labels": "unverified manifest placeholders",
    }
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("samples.csv", records), ("histograms.csv", histograms)):
        with (output / filename).open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    (output / "README.md").write_text(
        "# M2 隔离评测\n\n全部图像和统计使用同一皮肤 GT；非空 GT 的直方图面积归一化为 1。"
        "仅输入皮肤 GT 平均 Y<90 的真实样本参与欠曝验收，不用 scene 占位值筛选。"
        "samples.csv 保留全部样本；histograms.csv 保存 256 个亮度桶的前后原始计数。"
        "无欠曝样本或存在无效 GT 时不得宣称定量通过。\n\n"
        "当前实现仅包含自适应 gamma/linear；即使定量通过，也没有完成 #19 的 CLAHE 要求。\n",
        encoding="utf-8")
    return summary


def main(argv=None):
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repo / "data/input/subset")
    parser.add_argument("--manifest", type=Path, default=repo / "assets/evaluation_manifest.csv")
    parser.add_argument("--config", type=Path, default=repo / "config/exposure.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output or repo / "results" / ("m2_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                               + "_" + uuid.uuid4().hex[:6])
    summary = evaluate(args.data_root, args.manifest, cfg, output)
    print(json.dumps({"output": str(output), "underexposed_sample_ids": summary["underexposed_sample_ids"],
                      "underexposed_relative_gain": summary["underexposed_relative_gain"],
                      "underexposed_after_overexposure": summary["underexposed_after_overexposure"],
                      "quantitative_acceptance": summary["quantitative_acceptance"],
                      "full_issue_19_acceptance": summary["full_issue_19_acceptance"]},
                     ensure_ascii=False, indent=2))
    # 退出码只表示定量结果，不能作为整个 Issue 已完成的凭据。
    return 0 if all(summary["quantitative_acceptance"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
