"""Package OpenCV's installed classic Haar XML into B's YAML (offline tool only).

Run from the repository root: python tools/package_haar_config.py
The full XML, including its original licence header, is retained losslessly.
M1 consumes only the in-memory configuration; this tool is never imported by M1.
"""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import textwrap
import zlib

import cv2
import yaml


def main():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "config/skin_detect.yaml"
    text = path.read_text(encoding="utf-8")
    marker = "# BEGIN AUTHORIZED CLASSIC HAAR MODELS"
    text = text.split(marker, 1)[0].rstrip()
    if "face_locator" in yaml.safe_load(text):
        raise ValueError("existing face_locator outside generated block")
    lines = [marker,
             "# 用户2026-10-31批准传统预训练Haar；无深度学习。见report/lht_haar_20261031.md。",
             "# 完整XML（含原许可）无损压缩；运行时仅从cfg内存解码，不读取模型文件。",
             "face_locator:", "  enabled: true", "  working_size: 512", "  models:"]
    names = {"face_default": "haarcascade_frontalface_default.xml",
             "face_alt2": "haarcascade_frontalface_alt2.xml",
             "eyes": "haarcascade_eye_tree_eyeglasses.xml"}
    for role, filename in names.items():
        data = (Path(cv2.data.haarcascades) / filename).read_bytes()
        lines.extend([f"    {role}:", f"      filename: {filename}",
                      f"      sha256: {hashlib.sha256(data).hexdigest()}",
                      f"      packaged_opencv_version: '{cv2.__version__}'",
                      f"      source: https://github.com/opencv/opencv/blob/4.10.0/data/haarcascades/{filename}",
                      "      xml_zlib_base64: |"])
        packed = base64.b64encode(zlib.compress(data, 9)).decode("ascii")
        lines.extend("        " + part for part in textwrap.wrap(packed, 100))
    path.write_text(text + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"Packaged {len(names)} classic models from OpenCV {cv2.__version__}; {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
