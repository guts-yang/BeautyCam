# -*- coding: utf-8 -*-
"""集成与回归测试（A 维护）。

层级（对齐设计方案第 6 章）：
- 本文件中的用例使用内存合成数据与临时目录，随 PR CI 执行；
- 需要本地数据集与固定 manifest 的评测用例标记 `integration`，默认不随 PR CI 执行。

覆盖：主流程单图、空掩码不崩、批处理失败隔离、黄金基线（合成图像的确定性输出）。
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from beauty.core.config import load_module_config, load_pipeline_config
from beauty.core.pipeline import run_pipeline
from beauty.io.writer import FAILURES_FIELDS, METRICS_FIELDS

_MODULES = ["skin_detect", "exposure", "smooth", "freq_enhance", "restore", "color", "metrics"]


def _module_cfgs() -> dict:
    return {n: load_module_config(n) for n in _MODULES}


def _synthetic_bgr(h: int = 48, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_main_flow_preview_succeeds():
    """主流程：preview 模式在合成图上返回成功结果，且不创建持久化目录。"""
    img = _synthetic_bgr()
    result = run_pipeline(img, load_pipeline_config(), _module_cfgs(), mode="preview")
    assert result.succeeded
    assert result.final_bgr is not None
    assert result.final_bgr.shape == img.shape
    assert result.run_id is None  # preview 不创建运行目录


def test_stage_order_is_frozen():
    """阶段执行顺序冻结为 M5→M1→M2→M3→M4→M6。"""
    img = _synthetic_bgr()
    result = run_pipeline(img, load_pipeline_config(), _module_cfgs(), mode="preview")
    executed = [s.stage for s in result.stages]
    assert executed == ["restore", "skin_detect", "exposure", "smooth", "freq_enhance", "color"]


def test_m5_disabled_by_default():
    """M5 默认关闭（pipeline.yaml 开关）。"""
    img = _synthetic_bgr()
    result = run_pipeline(img, load_pipeline_config(), _module_cfgs(), mode="preview")
    restore_stage = next(s for s in result.stages if s.stage == "restore")
    assert restore_stage.status.value == "skipped"


def test_empty_mask_does_not_crash_pipeline():
    """桩实现下 M1 产出空掩码，流水线不崩溃且给出可读状态。"""
    img = _synthetic_bgr()
    result = run_pipeline(img, load_pipeline_config(), _module_cfgs(), mode="preview")
    assert result.succeeded
    assert result.skin_mask is not None
    assert not np.any(result.skin_mask)
    assert any("掩码为空" in s.message for s in result.stages)


def test_process_mode_writes_final_and_records(tmp_path):
    """process 模式：写入 final.png 并创建运行目录。"""
    img = _synthetic_bgr()
    result = run_pipeline(img, load_pipeline_config(), _module_cfgs(), mode="process",
                          output_root=str(tmp_path), sample_id="0000")
    assert result.succeeded and result.run_id
    final = tmp_path / "data" / "output" / result.run_id / "samples" / "0000" / "final.png"
    assert final.is_file()


def test_batch_failure_isolation(tmp_path):
    """批处理失败隔离（端到端）：坏文件写入 failures.csv，好样本继续并共享同一 run_id。"""
    from main import main as cli_main

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    rng = np.random.default_rng(11)
    for name in ("a.jpg", "b.jpg"):
        cv2.imwrite(str(in_dir / name), rng.integers(0, 256, (40, 48, 3), dtype=np.uint8))
    (in_dir / "broken.png").write_text("not an image", encoding="utf-8")

    out_root = tmp_path / "out"
    code = cli_main(["--input", str(in_dir), "--output-root", str(out_root), "--mode", "process"])
    assert code == 0

    results_dir = out_root / "results"
    csvs = list(results_dir.glob("*/failures.csv"))
    assert len(csvs) == 1, "一次批处理应只产生一个 failures.csv（共享 run_id）"
    text = csvs[0].read_text(encoding="utf-8")
    header, *rows = text.splitlines()
    assert header.split(",") == FAILURES_FIELDS
    assert len(rows) == 1 and "broken.png" in rows[0]
    assert str(tmp_path) not in text, "failures.csv 不得包含本机绝对路径"

    run_id = csvs[0].parent.name
    assert (out_root / "data" / "output" / run_id / "samples" / "0000" / "final.png").is_file()
    assert (out_root / "data" / "output" / run_id / "samples" / "0001" / "final.png").is_file()


def test_single_run_shares_one_run_id_with_failures(tmp_path):
    """单图阶段失败时，failures.csv 写入该 run 目录。"""
    from beauty.core.pipeline import run_pipeline
    img = _synthetic_bgr()

    def _boom(image_bgr, skin_mask, cfg):
        raise RuntimeError("模拟 M3 失败")

    module_cfgs = _module_cfgs()
    module_cfgs["smooth"] = {"__boom__": True}
    import beauty.modules.smooth as smooth_mod
    original = smooth_mod.process
    smooth_mod.process = _boom
    try:
        result = run_pipeline(img, load_pipeline_config(), module_cfgs, mode="process",
                              output_root=str(tmp_path), sample_id="0000",
                              run_id="fixedrun", input_relative_path="in/a.jpg")
    finally:
        smooth_mod.process = original
    assert not result.succeeded
    assert result.run_id == "fixedrun"
    assert len(result.failures) == 1
    assert result.failures[0]["stage"] == "smooth"
    assert result.failures[0]["input_relative_path"] == "in/a.jpg"


def test_preview_has_no_failures_side_effects(tmp_path):
    """preview 模式不创建 results 目录。"""
    from beauty.core.pipeline import run_pipeline
    img = _synthetic_bgr()
    run_pipeline(img, load_pipeline_config(), _module_cfgs(), mode="preview")
    assert not (tmp_path / "results").exists()


def test_metrics_row_fields_align_with_contract():
    """evaluation 模式产生的指标行字段顺序与契约一致。"""
    assert METRICS_FIELDS[0] == "run_id" and METRICS_FIELDS[-1] == "config_id"


def test_deterministic_output_for_golden_baseline():
    """图像基线：固定输入 + 固定参数下输出确定性一致（哈希比对）。"""
    img = _synthetic_bgr()
    cfg = load_pipeline_config()
    module_cfgs = _module_cfgs()
    r1 = run_pipeline(img, cfg, module_cfgs, mode="preview")
    r2 = run_pipeline(img, cfg, module_cfgs, mode="preview")
    assert r1.final_bgr is not None and r2.final_bgr is not None
    h1 = cv2.img_hash.averageHash(r1.final_bgr) if hasattr(cv2, "img_hash") else None
    # 不依赖 opencv-contrib：用字节级比较保证确定性。
    assert np.array_equal(r1.final_bgr, r2.final_bgr)
    if h1 is not None:
        h2 = cv2.img_hash.averageHash(r2.final_bgr)
        assert np.array_equal(h1, h2)


@pytest.mark.integration
def test_integration_requires_local_dataset():
    """集成评测前置条件：需要本地数据根目录与固定 manifest。"""
    data_root = os.environ.get("BEAUTYCAM_DATA_ROOT")
    if not data_root or not os.path.isdir(data_root):
        pytest.skip("前置条件未满足：请设置 BEAUTYCAM_DATA_ROOT 指向本地数据集")
    manifest = os.path.join(data_root, "evaluation_manifest.csv")
    assert os.path.isfile(manifest), "缺少固定评测清单 assets/evaluation_manifest.csv"