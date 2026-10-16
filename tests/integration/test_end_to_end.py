# -*- coding: utf-8 -*-
"""端到端集成回归（A 维护）。

与 test_main_flow.py 的区别：
- 本文件的用例全部标记 `end_to_end`，用合成数据 + 临时目录跑真实的 CLI 全链路，
  因此**不需要本地数据集**，会随 CI 的 `integration-regression` job 真正执行（非空跑）；
- test_main_flow.py 中标记 `integration` 的用例需要本地数据集与固定 manifest，
  仅在本地显式配置 BEAUTYCAM_DATA_ROOT 后才执行。

覆盖：单图落盘、批处理产物契约、evaluation 模式 metrics.csv 表头、
      preview 模式零副作用、失败脱敏。
"""

from __future__ import annotations

import csv
import os

import cv2
import numpy as np
import pytest

from beauty.io.writer import FAILURES_FIELDS, METRICS_FIELDS

pytestmark = pytest.mark.end_to_end

_MODULES = ["skin_detect", "exposure", "smooth", "freq_enhance", "restore", "color", "metrics"]


def _write_image(path: str, h: int = 32, w: int = 40, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    assert cv2.imwrite(path, img)


def _cli():
    from main import main as cli_main
    return cli_main


def test_single_image_end_to_end_single_sample(tmp_path):
    """单图 process 模式：返回 0，且 final.png 落在 data/output/<run_id>/samples/sample/。"""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    src = in_dir / "face.png"
    _write_image(str(src), seed=1)

    out_root = tmp_path / "out"
    code = _cli()(["--input", str(src), "--output-root", str(out_root), "--mode", "process"])
    assert code == 0

    finals = list((out_root / "data" / "output").glob("*/samples/sample/final.png"))
    assert len(finals) == 1, "单图应产出且仅产出一个 final.png"
    written = cv2.imread(str(finals[0]))
    assert written is not None and written.shape[2] == 3


def test_single_image_does_not_create_failures_csv(tmp_path):
    """单图成功时不应产生 failures.csv（失败记录是运行级、按需落盘）。"""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    src = in_dir / "face.png"
    _write_image(str(src), seed=2)

    out_root = tmp_path / "out"
    assert _cli()(["--input", str(src), "--output-root", str(out_root), "--mode", "process"]) == 0
    assert not list((out_root / "results").glob("*/failures.csv")) if (out_root / "results").exists() \
        else True


def test_batch_produces_one_run_shared_outputs(tmp_path):
    """批处理：三张图共享一个 run_id，各自产出 final.png，目录名 0000/0001/0002。"""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        _write_image(str(in_dir / name), seed=hash(name) % 100)

    out_root = tmp_path / "out"
    assert _cli()(["--input", str(in_dir), "--output-root", str(out_root), "--mode", "process"]) == 0

    run_dirs = list((out_root / "data" / "output").glob("*"))
    assert len(run_dirs) == 1, "一次批处理只应产生一个 run 目录（共享 run_id）"
    samples = sorted(p.name for p in (run_dirs[0] / "samples").iterdir())
    assert samples == ["0000", "0001", "0002"]
    for s in samples:
        assert (run_dirs[0] / "samples" / s / "final.png").is_file()


def test_batch_failure_row_fields_and_scrubbing(tmp_path):
    """批处理含坏文件：failures.csv 字段顺序正确、记录坏文件且不含本机绝对路径。"""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _write_image(str(in_dir / "good.png"), seed=3)
    (in_dir / "bad.png").write_text("not an image", encoding="utf-8")

    out_root = tmp_path / "out"
    assert _cli()(["--input", str(in_dir), "--output-root", str(out_root), "--mode", "process"]) == 0

    csvs = list((out_root / "results").glob("*/failures.csv"))
    assert len(csvs) == 1
    text = csvs[0].read_text(encoding="utf-8")
    with csvs[0].open(encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert list(rows[0].keys()) == FAILURES_FIELDS
    assert any("bad.png" in r["input_relative_path"] for r in rows)
    assert str(tmp_path) not in text, "failures.csv 不得包含本机绝对路径"


def test_evaluation_mode_writes_metrics_header(tmp_path):
    """evaluation 模式：即使指标为空，也必须产出带固定表头的 metrics.csv。"""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    src = in_dir / "face.png"
    _write_image(str(src), seed=4)

    out_root = tmp_path / "out"
    assert _cli()(["--input", str(src), "--output-root", str(out_root),
                   "--mode", "evaluation"]) == 0

    metric_csvs = list((out_root / "results").glob("*/metrics.csv"))
    assert len(metric_csvs) == 1, "evaluation 模式必须产出 metrics.csv"
    header = metric_csvs[0].read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == METRICS_FIELDS


def test_preview_mode_has_no_disk_side_effects(tmp_path):
    """preview 模式零副作用：不创建 data/output 与 results 目录。"""
    from beauty.core.config import load_module_config, load_pipeline_config
    from beauty.core.pipeline import run_pipeline

    rng = np.random.default_rng(9)
    img = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
    module_cfgs = {n: load_module_config(n) for n in _MODULES}
    result = run_pipeline(img, load_pipeline_config(), module_cfgs,
                          mode="preview", output_root=str(tmp_path))
    assert result.succeeded and result.run_id is None
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "results").exists()


def test_missing_input_returns_nonzero(tmp_path):
    """输入文件不存在：CLI 返回非 0，不产生崩溃堆栈。"""
    out_root = tmp_path / "out"
    code = _cli()(["--input", str(tmp_path / "nope.png"),
                   "--output-root", str(out_root), "--mode", "process"])
    assert code == 1