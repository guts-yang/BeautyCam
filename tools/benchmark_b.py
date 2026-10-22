"""固定清单上的 B 四阶段计时；不包含 M5–M7 或 Streamlit，不代替 R11 整机验收。

输入解码、缩放、模型导入和一次预热不计入处理时间。所有正式样本均保留，
不更改 OpenCV 的全局线程数/RNG；记录硬件、软件版本、配置及实现哈希。
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import uuid

import cv2
import numpy as np
import yaml


def main(argv=None):
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repo / "data/input/subset")
    parser.add_argument("--manifest", type=Path, default=repo / "assets/evaluation_manifest.csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)
    if args.size < 1 or args.repeats < 1:
        parser.error("size 和 repeats 必须为正整数")
    output = args.output or repo / "results" / ("b_timing_" + uuid.uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        ids = [row["sample_id"] for row in csv.DictReader(stream)]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("固定清单不能为空或含重复 sample_id")
    if any(not sample.isascii() or not sample.isdecimal() for sample in ids):
        raise ValueError("sample_id 必须为十进制数据编号")
    sys.path.insert(0, str(repo / "src"))
    from beauty.modules import skin_detect, exposure, smooth, freq_enhance

    names = ["skin_detect", "exposure", "smooth", "freq_enhance"]
    configs = {name: yaml.safe_load((repo / "config" / f"{name}.yaml").read_text(encoding="utf-8"))
               for name in names}

    def read_image(sample):
        path = args.data_root / "images" / f"{sample}.jpg"
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法解码 {path}")
        return cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_AREA)

    def process(image):
        start = perf_counter()
        mask = skin_detect.detect_skin(image, configs["skin_detect"])
        m1 = perf_counter()
        result = exposure.process(image, mask, configs["exposure"])
        m2 = perf_counter()
        result = smooth.process(result, mask, configs["smooth"])
        m3 = perf_counter()
        result = freq_enhance.process(result, mask, configs["freq_enhance"])
        end = perf_counter()
        return {"m1_seconds": m1-start, "m2_seconds": m2-m1,
                "m3_seconds": m3-m2, "m4_seconds": end-m3, "b_seconds": end-start}

    process(read_image(ids[0]))
    records = []
    for repeat in range(args.repeats):
        for sample in ids:
            image = read_image(sample)
            record = {"sample_id": sample, "repeat": repeat+1, **process(image)}
            records.append(record)
            print(json.dumps(record), flush=True)
    with (output / "timing.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    source_paths = [args.manifest] + [repo / "config" / f"{name}.yaml" for name in names]
    source_paths += [repo / "src/beauty/modules" / f"{name}.py" for name in names]
    source_paths.append(Path(__file__).resolve())
    times = np.array([row["b_seconds"] for row in records])
    summary = {
        "scope": "B M1-M4 sequential prefix only; excludes I/O, resize, M5-M7 and UI",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "size": [args.size, args.size], "samples": len(ids), "repeats": args.repeats,
        "warmup": "one unmeasured complete B pass using first manifest input",
        "device": {"platform": platform.platform(), "processor": platform.processor(),
                   "logical_cpus": os.cpu_count(), "opencv_threads": cv2.getNumThreads()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__},
        "source_sha256": {str(path.relative_to(repo) if path.is_relative_to(repo) else path):
                          hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "b_seconds_mean": float(times.mean()), "b_seconds_median": float(np.median(times)),
        "b_seconds_p95": float(np.percentile(times, 95)), "b_seconds_max": float(times.max()),
        "b_passes_within_two_seconds": int(np.count_nonzero(times <= 2)),
        "whole_pipeline_r11_verified": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0  # 超过 P1 目标保留实测，不冒充 UI 或算法定量门禁。


if __name__ == "__main__":
    raise SystemExit(main())
