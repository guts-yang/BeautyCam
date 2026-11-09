"""把既有 M3/M4 评测材料整理为可离线交接的目录；不填写或判断任何人工分数。"""
from __future__ import annotations

import argparse
import csv
from html import escape
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np


def _rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _copy_native(source, target):
    image = cv2.imdecode(np.frombuffer(source.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (1024, 1024):
        raise ValueError(f"人工评审坐标协议要求1024×1024图像：{source}")
    shutil.copyfile(source, target)


def prepare(data_root, m3_run, m4_run, output):
    sources = {"m3": m3_run, "m4": m4_run}
    visual = {module: _rows(root / "visual_review.csv") for module, root in sources.items()}
    candidates = _rows(m3_run / "roi_candidates.csv")
    samples = {row["sample_id"] for rows in visual.values() for row in rows}
    samples.update(row["sample_id"] for row in candidates)
    if any(not value.isascii() or not value.isdecimal() for value in samples):
        raise ValueError("样本编号必须是十进制数据编号")
    for module, rows in visual.items():
        if len(rows) != 10 or len({row["sample_id"] for row in rows}) != 10:
            raise ValueError(f"{module} 必须保留固定10张不同样例")
    output.mkdir(parents=True, exist_ok=False)
    for sample in sorted(samples, key=int):
        target = output / "originals" / f"{sample}.jpg"
        target.parent.mkdir(exist_ok=True)
        _copy_native(data_root / "images" / target.name, target)
    for module, root in sources.items():
        folder = output / module
        (folder / "outputs").mkdir(parents=True)
        filenames = ["REVIEW.md", "visual_review.csv", "summary.json", "samples.csv"]
        if module == "m3":
            filenames += ["roi_candidates.csv", "roi_review.csv"]
        for name in filenames:
            shutil.copyfile(root / name, folder / name)
        selected = samples if module == "m3" else {row["sample_id"] for row in visual[module]}
        for sample in sorted(selected, key=int):
            _copy_native(root / "outputs" / f"{sample}.png", folder / "outputs" / f"{sample}.png")

    pages = []
    for module, rows in visual.items():
        links = []
        for row in rows:
            sample = row["sample_id"]
            page = f"{module}_{sample}.html"
            (output / page).write_text(
                '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                f'<title>{module.upper()} {sample} 原尺寸检查</title>'
                '<style>body{font-family:sans-serif}main{display:flex;gap:16px}figure{margin:0}'
                'img{display:block;width:1024px;height:1024px;max-width:none}</style>'
                f'<p><a href="index.html">返回目录</a> · {module.upper()} / {sample} · '
                '浏览器缩放须为100%，可水平滚动；左原图，右处理后。请检查眼/眉/唇边界。</p>'
                f'<main><figure><figcaption>原图</figcaption><img src="originals/{sample}.jpg" alt="原图"></figure>'
                f'<figure><figcaption>处理后</figcaption><img src="{module}/outputs/{sample}.png" alt="处理后"></figure></main>'
                '</html>', encoding="utf-8")
            links.append(f'<li><a href="{page}">{escape(sample)}</a></li>')
        pages.append(f'<h2>{module.upper()} 固定10张边缘检查</h2><ol>'+''.join(links)+'</ol>'
                     f'<p>评分表：<a href="{module}/visual_review.csv">{module}/visual_review.csv</a></p>')
    roi_rows = []
    for row in candidates:
        sample = row["sample_id"]
        x, y, w, h = [int(row[key]) for key in ("x", "y", "width", "height")]
        if x < 0 or y < 0 or w < 1 or h < 1 or x+w > 1024 or y+h > 1024:
            raise ValueError(f"候选ROI越界：{row}")
        left, top = max(0, min(1024-96, x+w//2-48)), max(0, min(1024-96, y+h//2-48))
        roi_rows.append(f'<tr><td>{sample}<br>({x},{y},{w},{h})</td><td>'
                        f'<div class="crop"><img src="originals/{sample}.jpg" style="left:{-left}px;top:{-top}px" alt="仅原图候选上下文">'
                        f'<span style="left:{x-left}px;top:{y-top}px;width:{w}px;height:{h}px"></span></div></td>'
                        f'<td><a href="originals/{sample}.jpg">查看原图</a> · '
                        f'<a href="m3/outputs/{sample}.png">冻结ROI后查看处理图</a></td></tr>')
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>B模块人工评审</title>'
        '<style>body{font-family:sans-serif;line-height:1.6;margin:24px;max-width:1000px}'
        'td{border:1px solid #ccc;padding:8px}.crop{position:relative;width:96px;height:96px;overflow:hidden}'
        '.crop img{position:absolute;width:1024px;height:1024px;max-width:none}'
        '.crop span{position:absolute;border:1px solid red;box-sizing:border-box}</style>'
        '<h1>B模块人工评审入口</h1><p>评审状态：未完成。本材料不包含自动生成的人工分数。</p>'
        '<p>由 A 和 C 两位非实现者分别独立记录结果；不要先互相对答案。'
        '保留各自原始副本，统一汇总时记录对应评价者。资料仅供本地课程评审。</p>'
        '<p>M3：两人均确认无光晕、双边缘、轮廓位移的样例至少9/10；'
        'M4：两人均确认无光晕、双边缘的样例至少9/10。表内填写 true/false，空白代表待评审。</p>'
        + ''.join(pages) +
        '<h2>M3 痘印候选核验</h2><p>以下是输入暗结构候选，可能是雀斑、阴影或正常纹理，尚未确认痘印。'
        '只根据原图确认真实痘印，在 m3/roi_review.csv 填 selected、confirmed_blemish，'
        '冻结至少10个ROI、覆盖至少5张图后，再查看处理图评分。若候选不足，需重新标注，不能强行凑数。'
        '坐标是1024×1024原图的 x,y,width,height，红框为候选，周围96×96上下文以原尺寸显示。</p>'
        '<p>两人分别填写1–5分：visibility_before/after（1=不可见，5=清晰）；'
        'naturalness_after、feature_protection_after（1=明显破坏，5=自然保留）。'
        '至少80%冻结ROI必须由两人均判定可见度下降≥1，且自然度、五官保护均≥4。'
        '<a href="m3/roi_review.csv">打开ROI评分表</a>。</p>'
        '<table><tr><th>编号/ROI坐标</th><th>仅原图上下文</th><th>原尺寸文件</th></tr>'
        + ''.join(roi_rows) + '</table></html>', encoding="utf-8")
    hashes = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(output.rglob("*")) if path.is_file()}
    (output / "package_manifest.json").write_text(json.dumps(
        {"subjective_acceptance": "pending", "sha256_before_review": hashes},
        ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return output / "index.html"


def main(argv=None):
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repo / "data/input/subset")
    parser.add_argument("--m3-run", type=Path, required=True)
    parser.add_argument("--m4-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(prepare(args.data_root, args.m3_run, args.m4_run, args.output))


if __name__ == "__main__":
    main()
