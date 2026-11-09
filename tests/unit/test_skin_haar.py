"""Authorized classic Haar: in-memory models, localization fallback and skin continuity."""
import base64
from copy import deepcopy
import hashlib
from pathlib import Path
import zlib

import cv2
import numpy as np
import pytest
import yaml

from beauty.modules import skin_detect


@pytest.fixture
def locator():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[2]/"config/skin_detect.yaml").read_text(encoding="utf-8"))
    return skin_detect._haar_configuration(cfg)


def test_all_packaged_models_load_from_memory_and_keep_original_licences(locator, monkeypatch):
    real_storage = cv2.FileStorage
    calls = []

    def storage(xml, flags):
        assert xml.lstrip().startswith("<?xml")
        assert flags & cv2.FILE_STORAGE_MEMORY
        calls.append(flags)
        return real_storage(xml, flags)

    monkeypatch.setattr(cv2, "FileStorage", storage)
    for payload in locator["models"].values():
        xml = zlib.decompress(base64.b64decode(payload["xml_zlib_base64"]))
        assert hashlib.sha256(xml).hexdigest() == payload["sha256"]
        assert b"Redistribution" in xml and b"Intel Corporation" in xml
        assert not skin_detect._haar_classifier(payload).empty()
    assert len(calls) == 3


def test_localizer_does_not_read_files_modify_configuration_or_consume_rng(locator, monkeypatch):
    image = np.full((128, 160, 3), (240, 20, 20), np.uint8)
    image.setflags(write=False)
    saved = deepcopy(locator)

    def no_files(*args, **kwargs):
        raise AssertionError("algorithm attempted file I/O")

    monkeypatch.setattr("builtins.open", no_files)
    cv2.setRNGSeed(19)
    expected = cv2.randu(np.empty((5, 5), np.float32), 0, 1)
    cv2.setRNGSeed(19)
    assert skin_detect._haar_localize(image, locator["models"]) is None
    np.testing.assert_array_equal(cv2.randu(np.empty((5, 5), np.float32), 0, 1), expected)
    assert locator == saved


def test_unreliable_localization_preserves_original_mask(locator, monkeypatch):
    image = np.full((128, 160, 3), 100, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[30:90, 40:100] = 255
    original = mask.copy()
    monkeypatch.setattr(skin_detect, "_haar_localize", lambda *_: None)
    result = skin_detect._haar_refine(image, mask, locator)
    np.testing.assert_array_equal(result, original)


def test_continuity_fills_highlight_and_nose_but_protects_actual_eye_locations(locator, monkeypatch):
    image = np.full((256, 256, 3), (240, 20, 20), np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.ellipse(mask, (128, 125), (68, 95), 0, 0, 360, 255, -1)
    image[mask != 0] = (80, 120, 180)
    mask[115:135, 122:134] = 0
    mask[55:65, 122:134] = 0
    image[55:65, 122:134] = (240, 240, 240)
    eyes = [np.array([85, 85, 28, 28.]), np.array([143, 85, 28, 28.])]
    monkeypatch.setattr(skin_detect, "_haar_localize", lambda *_: (np.array([55, 40, 146, 146.]), eyes))
    result = skin_detect._haar_refine(image, mask, locator)
    assert result[125, 128] == 255
    assert result[60, 128] == 255
    assert result[99, 99] == result[99, 157] == 0
    assert result[10, 10] == result[240, 128] == 0


def test_dynamic_path_tracks_continuous_image_boundary():
    expected = np.array([4, 4, 5, 6, 7, 7, 6, 5, 4])
    cost = np.full((12, len(expected)), 100., np.float32)
    cost[expected, np.arange(len(expected))] = 0
    np.testing.assert_array_equal(skin_detect._contour_path(cost), expected)


@pytest.mark.parametrize("cfg", [{"face_locator": None}, {"face_locator": {"enabled": "yes"}},
    {"face_locator": {"enabled": True}},
    {"face_locator": {"enabled": True, "working_size": 0}},
    {"face_locator": {"enabled": True, "models": {"face_default": "path.xml"}}}])
def test_invalid_locator_configuration_is_rejected(cfg):
    with pytest.raises(ValueError, match="skin_detect.face_locator"):
        skin_detect._haar_configuration(cfg)


def test_corrupt_models_raise_clear_error(locator):
    payload = dict(locator["models"]["eyes"])
    payload["sha256"] = "0"*64
    with pytest.raises(ValueError, match="SHA256"):
        skin_detect._haar_classifier(payload)
    payload["xml_zlib_base64"] = "not a base64 model"
    with pytest.raises(ValueError, match="模型无效"):
        skin_detect._haar_classifier(payload)


def test_disabled_or_absent_locator_retains_legacy_behavior(monkeypatch):
    def forbidden(*args):
        raise AssertionError("disabled locator must not load classifiers")
    monkeypatch.setattr(skin_detect, "_haar_classifier", forbidden)
    image = np.full((90, 110, 3), (80, 120, 180), np.uint8)
    expected = skin_detect.detect_skin(image, {})
    np.testing.assert_array_equal(skin_detect.detect_skin(image, {"face_locator": {"enabled": False}}), expected)


def test_final_morphology_cleans_localized_mask_instead_of_running_before_it(monkeypatch):
    image = np.full((96, 96, 3), (80, 120, 180), np.uint8)
    localized = np.zeros(image.shape[:2], np.uint8)
    localized[25:70, 25:70] = 255
    localized[10, 10] = 255
    monkeypatch.setattr(skin_detect, "_haar_configuration", lambda _: {})
    monkeypatch.setattr(skin_detect, "_haar_refine", lambda *_: localized.copy())
    assert skin_detect.detect_skin(image, {"postprocess": {"enabled": False}})[10, 10] == 255
    cleaned = skin_detect.detect_skin(image, {})
    assert cleaned[10, 10] == 0 and cleaned[45, 45] == 255
