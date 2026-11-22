"""Inspect Haar resources shipped with the installed OpenCV package.

Run from the repository root with ``python tools/inspect_haar_models.py``.
This offline diagnostic prints provenance without rewriting module configuration.
"""
from __future__ import annotations

import hashlib
from importlib.resources import files
import json
from pathlib import Path

import cv2
import yaml


def main():
    repo = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repo / "config/skin_detect.yaml").read_text(encoding="utf-8"))
    records = {}
    for role, name in config["face_locator"]["models"].items():
        data = files("cv2.data").joinpath(name).read_bytes()
        storage = cv2.FileStorage(data.decode("utf-8"), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        try:
            model = cv2.CascadeClassifier()
            if not model.read(storage.getFirstTopLevelNode()) or model.empty():
                raise RuntimeError(f"OpenCV package resource could not be loaded: {name}")
        finally:
            storage.release()
        records[role] = {"resource": name, "bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest(),
                         "contains_original_licence": b"Intel Corporation" in data}
    print(json.dumps({"opencv": cv2.__version__, "models": records}, indent=2))


if __name__ == "__main__":
    main()
