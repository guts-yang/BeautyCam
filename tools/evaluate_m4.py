"""M4 独立评测：高频 RMS、五官 EPI、FFT 频谱/分层图与人工检查空表。

python tools/evaluate_m4.py --data-root data/input/subset
只验证定量指标和诊断契约；无光晕/双边缘仍需两位非实现者检查。
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


def _y(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]


def high_frequency_retention(before, after, skin):
    """PRD 3.4：腐蚀皮肤 GT 内，Y-GaussianBlur(Y,sigma=3) 的 RMS 比率。"""
    region = cv2.erode(skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                       borderType=cv2.BORDER_REFLECT_101) != 0
    if not np.any(region):
        return {"value": None, "before_rms": None, "after_rms": None, "reason": "empty_eroded_skin_gt"}
    energies = []
    for image in (before, after):
        y = _y(image).astype(np.float64)
        high = y - cv2.GaussianBlur(y, (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_REFLECT_101)
        energies.append(float(np.sqrt(np.mean(high[region]**2))))
    if energies[0] < 1e-6:
        return {"value": None, "before_rms": energies[0], "after_rms": energies[1], "reason": "zero_input_hf_energy"}
    return {"value": energies[1] / energies[0], "before_rms": energies[0], "after_rms": energies[1], "reason": ""}


def edge_preservation(before, after, features):
    """与 PRD/M3 相同的项目自定义 Sobel EPI，而非通用同名分数。"""
    region = features != 0
    if not np.any(region):
        return {"value": None, "reason": "empty_face_feature_gt"}
    gradients = []
    for image in (before, after):
        y = _y(image)
        gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
        gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
        gradients.append(cv2.magnitude(gx, gy)[region])
    denominator = float(np.sum(np.maximum(*gradients), dtype=np.float64))
    if denominator < 1e-8:
        return {"value": None, "reason": "zero_feature_gradient"}
    return {"value": float(np.sum(np.minimum(*gradients), dtype=np.float64)) / (denominator + 1e-8), "reason": ""}


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


def log_spectrum(image):
    magnitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(_y(image).astype(float)))))
    extent = float(np.max(magnitude) - np.min(magnitude))
    gray = (np.zeros(magnitude.shape, np.uint8) if extent == 0 else
            np.clip(np.rint((magnitude - np.min(magnitude)) * 255 / extent), 0, 255).astype(np.uint8))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def evaluate(data_root, manifest, cfg, output):
    from beauty.modules import freq_enhance

    data_root, manifest, output = Path(data_root), Path(manifest), Path(output)
    if output.exists():
        raise ValueError("输出目录已存在，请使用新目录")
    manifest_bytes = manifest.read_bytes()
    samples = list(csv.DictReader(manifest_bytes.decode("utf-8-sig").splitlines()))
    if not samples or any(not {"sample_id", "scene"} <= row.keys() for row in samples):
        raise ValueError("manifest 必须含 sample_id,scene 和至少一个样本")
    ids = [row["sample_id"] for row in samples]
    if len(set(ids)) != len(ids) or any(not x.isascii() or not x.isdecimal() for x in ids):
        raise ValueError("sample_id 必须是不重复的数字 ID")
    for sample in ids:
        for folder, suffix in (("images", ".jpg"), ("skin_masks", ".png"), ("masks", ".png")):
            if not (data_root / folder / (sample + suffix)).is_file():
                raise FileNotFoundError(f"固定样本 {sample} 缺少 {folder}")
    records = []
    for row in samples:
        sample = row["sample_id"]
        before = _read(data_root / "images" / f"{sample}.jpg", cv2.IMREAD_COLOR)
        skin = _read(data_root / "skin_masks" / f"{sample}.png", cv2.IMREAD_GRAYSCALE)
        labels = _read(data_root / "masks" / f"{sample}.png", cv2.IMREAD_GRAYSCALE)
        if not set(np.unique(skin)).issubset({0, 255}):
            raise ValueError("皮肤 GT 必须为 0/255")
        target = (before.shape[1], before.shape[0])
        if skin.shape != before.shape[:2]:
            skin = cv2.resize(skin, target, interpolation=cv2.INTER_NEAREST)
        if labels.shape != before.shape[:2]:
            labels = cv2.resize(labels, target, interpolation=cv2.INTER_NEAREST)
        features = np.isin(labels, [3, 4, 5, 6, 7, 10, 11, 12]).astype(np.uint8) * 255
        original, original_mask = before.copy(), skin.copy()
        after = freq_enhance.process(before, skin, cfg)
        diagnostics = freq_enhance.get_frequency_diagnostics(before, cfg)
        repeated = freq_enhance.process(before, skin, cfg)
        if not np.array_equal(after, repeated):
            raise AssertionError("诊断改变了 M4 主输出")
        if not np.array_equal(before, original) or not np.array_equal(skin, original_mask):
            raise AssertionError("M4 修改输入")
        if after.shape != before.shape or after.dtype != np.uint8 or np.shares_memory(after, before):
            raise AssertionError("M4 图像输出契约失败")
        if not np.array_equal(after[skin == 0], before[skin == 0]):
            raise AssertionError("M4 修改掩码外像素")
        if set(diagnostics) != {"low_frequency", "high_frequency"} or any(
                image.shape != before.shape or image.dtype != np.uint8 for image in diagnostics.values()):
            raise AssertionError("M4 诊断契约失败")
        ratio = high_frequency_retention(before, after, skin)
        epi = edge_preservation(before, after, features)
        feature_values = _y(before)[features != 0]
        contrast = float(np.percentile(feature_values, 95) - np.percentile(feature_values, 5)) if feature_values.size else 0
        records.append({"sample_id": sample, "scene": row["scene"],
                        "hf_before_rms": ratio["before_rms"], "hf_after_rms": ratio["after_rms"],
                        "hf_ratio": ratio["value"], "hf_reason": ratio["reason"],
                        "epi": epi["value"], "epi_reason": epi["reason"], "input_feature_contrast": contrast})
        spectrum = log_spectrum(before)
        tiles = []
        for title, image in (("input", before), ("M4 output", after),
                             ("low-frequency Y", diagnostics["low_frequency"]),
                             ("signed high-frequency Y +128", diagnostics["high_frequency"]),
                             ("input log spectrum", spectrum)):
            tile = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            tile = cv2.copyMakeBorder(tile, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
            cv2.putText(tile, title, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (20, 20, 20), 1)
            tiles.append(tile)
        _write_png(output / "figures" / f"{sample}.png", np.hstack(tiles))
        _write_png(output / "outputs" / f"{sample}.png", after)
        for name, image in diagnostics.items():
            _write_png(output / "diagnostics" / sample / f"{name}.png", image)
        _write_png(output / "spectra" / f"{sample}.png", spectrum)
        if len(records) % 10 == 0:
            print(f"[M4] {len(records)}/{len(samples)}", flush=True)
    ratio_summary = _aggregate([row["hf_ratio"] for row in records])
    epi_summary = _aggregate([row["epi"] for row in records])
    ratio, epi = ratio_summary["macro_mean"], epi_summary["macro_mean"]
    review_rows = sorted(records, key=lambda row: (-row["input_feature_contrast"], int(row["sample_id"])))[:10]
    summary = {"profile": "module_isolated", "target_module": "freq_enhance", "mask_source": "skin_gt",
               "config": cfg, "total_samples": len(samples), "high_frequency_retention": ratio_summary, "epi": epi_summary,
               "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "algorithm_sha256": hashlib.sha256(Path(freq_enhance.__file__).read_bytes()).hexdigest(),
               "runtime": {"python": sys.version.split()[0], "opencv": cv2.__version__, "numpy": np.__version__},
               "protocol": {"hf_sigma": 3, "skin_erode_kernel": 11, "feature_labels": [3, 4, 5, 6, 7, 10, 11, 12]},
               "quantitative_acceptance": {"high_frequency_retention": ratio is not None and 0.70 <= ratio <= 1.30,
                                            "epi": epi is not None and epi >= 0.80},
               "diagnostic_invariance_samples": len(records),
               "visual_review_selection": "top 10 by input-only feature Y 95th-5th percentile contrast, ties by ID",
               "fixed_visual_review_sample_ids": [row["sample_id"] for row in review_rows],
               "subjective_review": "pending two non-implementers; thumbnails are not 100% inspection",
               "scene_labels": "unverified manifest placeholders"}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "samples.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    fields = ["sample_id", "input_feature_contrast"] + [
        f"reviewer_{reviewer}_{criterion}" for reviewer in (1, 2) for criterion in ("no_halo", "no_double_edge")]
    with (output / "visual_review.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in ("sample_id", "input_feature_contrast")} for row in review_rows)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "REVIEW.md").write_text(
        "# M4 人工检查\n\n固定 10 张样例按处理前五官亮度对比度排序选择，未使用处理后分数挑样本。"
        "请两位非实现者对照原始 1024x1024 图和 outputs/ 中原尺寸结果，按 100% 显示比例检查眼、眉、唇边界。"
        "visual_review.csv 中分别填写是否无光晕、无双边缘；至少 9/10 张须两人都通过。"
        "人工字段初始为空；定量达标不代表人工检查通过。figures/ 缩略图仅用于导航。\n\n"
        "diagnostics/ 的低频图为亮度；高频图中 128 表示零残差，大于/小于 128 分别为正/负。"
        "spectra/ 使用每图归一化的 log(1+abs(FFT))，只作显示，不用于高频能量计分。\n",
        encoding="utf-8")
    return summary


def main(argv=None):
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repo / "data/input/subset")
    parser.add_argument("--manifest", type=Path, default=repo / "assets/evaluation_manifest.csv")
    parser.add_argument("--config", type=Path, default=repo / "config/freq_enhance.yaml")
    parser.add_argument("--highpass-gain", type=float, help="仅覆盖本次已有参数，不回写 YAML")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.highpass_gain is not None:
        cfg["highpass_gain"] = args.highpass_gain
    output = args.output or repo / "results" / ("m4_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6])
    summary = evaluate(args.data_root, args.manifest, cfg, output)
    print(json.dumps({"output": str(output), "high_frequency_retention": summary["high_frequency_retention"],
                      "epi": summary["epi"], "quantitative_acceptance": summary["quantitative_acceptance"]},
                     ensure_ascii=False, indent=2))
    return 0 if all(summary["quantitative_acceptance"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
