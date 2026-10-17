"""M3 模块隔离评测：skin GT、7x7 局部方差、五官 Sobel EPI。

python tools/evaluate_m3.py --data-root data/input/subset
只验证定量指标；痘印 ROI 与 10 张固定样例仍需两名非实现者检查。
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


def _luminance(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]


def variance_drop(before, after, skin):
    """PRD 3.4：7x7 方差图，GT 以 11x11 椭圆核腐蚀后选中心。"""
    region = cv2.erode(skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                       borderType=cv2.BORDER_REFLECT_101) != 0
    if not np.any(region):
        return {"value": None, "before": None, "after": None, "reason": "empty_eroded_skin_gt"}
    values = []
    for image in (before, after):
        y = _luminance(image).astype(np.float64)
        mean = cv2.boxFilter(y, -1, (7, 7), borderType=cv2.BORDER_REFLECT_101)
        square_mean = cv2.boxFilter(y * y, -1, (7, 7), borderType=cv2.BORDER_REFLECT_101)
        values.append(float(np.mean(np.maximum(square_mean - mean * mean, 0)[region])))
    if values[0] < 1e-6:
        return {"value": None, "before": values[0], "after": values[1], "reason": "zero_input_variance"}
    return {"value": (values[0] - values[1]) / values[0],
            "before": values[0], "after": values[1], "reason": ""}


def edge_preservation(before, after, features):
    """项目自定义 EPI = sum(min gradients)/(sum(max gradients)+1e-8)。"""
    region = features != 0
    if not np.any(region):
        return {"value": None, "reason": "empty_face_feature_gt"}
    gradients = []
    for image in (before, after):
        y = _luminance(image)
        gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
        gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
        gradients.append(cv2.magnitude(gx, gy)[region])
    denominator = float(np.sum(np.maximum(*gradients), dtype=np.float64))
    if denominator < 1e-8:
        return {"value": None, "reason": "zero_feature_gradient"}
    value = float(np.sum(np.minimum(*gradients), dtype=np.float64)) / (denominator + 1e-8)
    return {"value": value, "reason": ""}


def _aggregate(values):
    valid = [value for value in values if value is not None]
    return {"macro_mean": float(np.mean(valid)) if valid else None,
            "median": float(np.median(valid)) if valid else None,
            "valid_samples": len(valid), "invalid_samples": len(values) - len(valid)}


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


def evaluate(data_root, manifest, cfg, output):
    from beauty.modules import smooth

    data_root, manifest, output = Path(data_root), Path(manifest), Path(output)
    if output.exists():
        raise ValueError("输出目录已存在，请使用新目录")
    manifest_bytes = manifest.read_bytes()
    samples = list(csv.DictReader(manifest_bytes.decode("utf-8-sig").splitlines()))
    if not samples or any(not {"sample_id", "scene"} <= row.keys() for row in samples):
        raise ValueError("manifest 必须含 sample_id,scene 和至少一个样本")
    ids = [row["sample_id"] for row in samples]
    if len(ids) != len(set(ids)) or any(not x.isascii() or not x.isdecimal() for x in ids):
        raise ValueError("sample_id 必须是不重复的数字 ID")
    # 完整清单的前置条件先检查，禁止静默删去缺失的困难样本。
    for sample in ids:
        for folder, suffix in (("images", ".jpg"), ("skin_masks", ".png"), ("masks", ".png")):
            if not (data_root / folder / (sample + suffix)).is_file():
                raise FileNotFoundError(f"固定样本 {sample} 缺少 {folder}")
    records, roi_candidates = [], []
    for row in samples:
        sample = row["sample_id"]
        before = _read(data_root / "images" / f"{sample}.jpg", cv2.IMREAD_COLOR)
        skin = _read(data_root / "skin_masks" / f"{sample}.png", cv2.IMREAD_GRAYSCALE)
        labels = _read(data_root / "masks" / f"{sample}.png", cv2.IMREAD_GRAYSCALE)
        if not set(np.unique(skin)).issubset({0, 255}):
            raise ValueError("皮肤 GT 必须是 0/255")
        if skin.shape != before.shape[:2]:
            skin = cv2.resize(skin, (before.shape[1], before.shape[0]), interpolation=cv2.INTER_NEAREST)
        if labels.shape != before.shape[:2]:
            labels = cv2.resize(labels, (before.shape[1], before.shape[0]), interpolation=cv2.INTER_NEAREST)
        features = np.isin(labels, [3, 4, 5, 6, 7, 10, 11, 12]).astype(np.uint8) * 255
        original, original_skin = before.copy(), skin.copy()
        after = smooth.process(before, skin, cfg)
        if after.shape != before.shape or after.dtype != np.uint8 or np.shares_memory(after, before):
            raise AssertionError("M3 图像输出契约失败")
        if not np.array_equal(before, original) or not np.array_equal(skin, original_skin):
            raise AssertionError("M3 修改输入")
        if not np.array_equal(after[skin == 0], before[skin == 0]):
            raise AssertionError("M3 修改了掩码外像素")
        variance, epi = variance_drop(before, after, skin), edge_preservation(before, after, features)
        records.append({"sample_id": sample, "scene": row["scene"],
                        "variance_before": variance["before"], "variance_after": variance["after"],
                        "variance_drop": variance["value"], "variance_reason": variance["reason"],
                        "epi": epi["value"], "epi_reason": epi["reason"]})
        tiles = []
        diff = cv2.convertScaleAbs(cv2.absdiff(before, after), alpha=4)
        for title, image in (("input", before), ("M3 output", after), ("absolute difference x4", diff)):
            tile = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
            tile = cv2.copyMakeBorder(tile, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
            cv2.putText(tile, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1)
            tiles.append(tile)
        _write_png(output / "figures" / f"{sample}.png", np.hstack(tiles))
        _write_png(output / "outputs" / f"{sample}.png", after)
        # 仅生成待人工核验的暗斑候选，不能将算法改动区域当成真实痘印 GT。
        y = _luminance(before)
        diameter = int(cfg.get("bilateral_d", 9))
        radius = diameter // 2
        if radius:
            padded = cv2.copyMakeBorder(y, radius, radius, radius, radius, cv2.BORDER_REFLECT_101)
            median = cv2.medianBlur(padded, diameter)[radius:-radius, radius:-radius]
            residual = cv2.subtract(median, y)
            interior = cv2.erode(skin, np.ones((diameter, diameter), np.uint8)) != 0
            values = residual[interior]
            if values.size and np.any(values):
                threshold, _ = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                _, lab, stats, _ = cv2.connectedComponentsWithStats(
                    ((residual > threshold) & interior).astype(np.uint8), connectivity=8)
                residual_sums = np.bincount(lab.ravel(), weights=residual.ravel(), minlength=len(stats))
                candidates = []
                for index, (x, y0, width, height, area) in enumerate(stats[1:], 1):
                    if (area <= diameter * diameter and width <= diameter and height <= diameter
                            and np.all(skin[y0:y0 + height, x:x + width] != 0)):
                        score = float(residual_sums[index] / area)
                        candidates.append((score, int(x), int(y0), int(width), int(height)))
                for score, x, y0, width, height in sorted(candidates, reverse=True)[:2]:
                    roi_candidates.append({"sample_id": sample, "x": x, "y": y0, "width": width, "height": height,
                                           "review_status": "unreviewed_candidate", "residual_mean": score})
    variance_summary = _aggregate([row["variance_drop"] for row in records])
    epi_summary = _aggregate([row["epi"] for row in records])
    v, e = variance_summary["macro_mean"], epi_summary["macro_mean"]
    summary = {"profile": "module_isolated", "target_module": "smooth", "mask_source": "skin_gt",
               "config": cfg, "total_samples": len(samples), "variance_drop": variance_summary, "epi": epi_summary,
               "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "algorithm_sha256": hashlib.sha256(Path(smooth.__file__).read_bytes()).hexdigest(),
               "runtime": {"python": sys.version.split()[0], "opencv": cv2.__version__, "numpy": np.__version__},
               "protocol": {"variance_window": 7, "skin_erode_kernel": 11, "feature_labels": [3, 4, 5, 6, 7, 10, 11, 12]},
               "quantitative_acceptance": {"variance_drop": v is not None and 0.20 <= v <= 0.40,
                                            "epi": e is not None and e >= 0.80},
               "fixed_visual_review_sample_ids": ids[:10],
               "human_review_files": ["visual_review.csv", "roi_review.csv", "REVIEW.md"],
               "subjective_review": "pending two non-implementers; ROI candidates are not verified blemishes",
               "scene_labels": "unverified manifest placeholders"}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "samples.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    with (output / "roi_candidates.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["sample_id", "x", "y", "width", "height", "review_status", "residual_mean"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(roi_candidates)
    visual_fields = ["sample_id"] + [
        f"reviewer_{reviewer}_{criterion}" for reviewer in (1, 2)
        for criterion in ("no_halo", "no_double_edge", "no_contour_shift")]
    with (output / "visual_review.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=visual_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({"sample_id": sample} for sample in ids[:10])
    roi_fields = ["sample_id", "x", "y", "width", "height", "selected", "confirmed_blemish"] + [
        f"reviewer_{reviewer}_{criterion}" for reviewer in (1, 2)
        for criterion in ("visibility_before", "visibility_after", "naturalness_after", "feature_protection_after")]
    with (output / "roi_review.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=roi_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: candidate[key] for key in ("sample_id", "x", "y", "width", "height")}
                         for candidate in roi_candidates)
    (output / "REVIEW.md").write_text(
        "# M3 人工验收材料\n\n"
        "visual_review.csv 固定使用 manifest 前 10 张。两位非实现者独立对照原图与 outputs/ 中原尺寸结果，"
        "按 100% 比例检查光晕、双边缘、轮廓位移。填 true/false，不使用缩略图代替原尺寸检查；"
        "至少 9/10 张必须由两人共同判定三个问题均不存在。\n\n"
        "roi_review.csv 是输入暗残差自动产生的候选，并非已确认的痘印。先人工确认真实痘印，"
        "冻结至少 10 个 ROI、覆盖至少 5 张图；不要按处理效果挑选。坐标以输入原图像素为准，"
        "1024x1024 数据可直接用于课程坐标协议。selected 和 confirmed_blemish 未填写前不计入验收。\n\n"
        "两名非实现者独立评分：可见度 1=不可见、5=清晰可见；自然度/五官保护 1=明显破坏、5=自然保留。"
        "至少 80% 已冻结 ROI 中，两人都应判定可见度前后下降 >=1 分，且处理后自然度与五官保护均 >=4。\n\n"
        "候选和评分表所有人工字段初始留空，不能把定量达标或自动候选当成人工验收通过。\n",
        encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repo / "data/input/subset")
    parser.add_argument("--manifest", type=Path, default=repo / "assets/evaluation_manifest.csv")
    parser.add_argument("--config", type=Path, default=repo / "config/smooth.yaml")
    parser.add_argument("--strength", type=float, help="仅覆盖本次已有 strength 参数，不回写 YAML")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.strength is not None:
        cfg["strength"] = args.strength
    output = args.output or repo / "results" / ("m3_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6])
    summary = evaluate(args.data_root, args.manifest, cfg, output)
    print(json.dumps({"output": str(output), "variance_drop": summary["variance_drop"], "epi": summary["epi"],
                      "quantitative_acceptance": summary["quantitative_acceptance"],
                      "subjective_review": summary["subjective_review"]}, ensure_ascii=False, indent=2))
    return 0 if all(summary["quantitative_acceptance"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
