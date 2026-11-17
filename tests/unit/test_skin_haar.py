"""Classic Haar package resources, independent switches and skin continuity."""
from copy import deepcopy
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from beauty.modules import skin_detect


@pytest.fixture
def locator():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[2]/"config/skin_detect.yaml").read_text(encoding="utf-8"))
    return skin_detect._haar_configuration(cfg)


def test_models_load_from_opencv_package_instead_of_yaml_payloads(locator, monkeypatch):
    real_storage = cv2.FileStorage
    calls = []

    def storage(xml, flags):
        assert xml.lstrip().startswith("<?xml") and flags & cv2.FILE_STORAGE_MEMORY
        calls.append(flags)
        return real_storage(xml, flags)

    expected = {
        "face_default": "0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0",
        "face_alt2": "7b0c967d9abbdfbde025eb9c786947d151b6426040d07a8f9562ed8fd90724b4",
        "eyes": "e32f9c67935c33e9d1331eb14fa58554ff17835c03742663bcb97a892e936a57",
    }
    monkeypatch.setattr(cv2, "FileStorage", storage)
    for role, resource in locator["models"].items():
        assert isinstance(resource, str) and len(resource) < 60
        xml = (Path(cv2.data.haarcascades) / resource).read_bytes()
        assert hashlib.sha256(xml).hexdigest() == expected[role]
        assert b"Redistribution" in xml and b"Intel Corporation" in xml
        assert not skin_detect._haar_classifier(resource).empty()
    assert len(calls) == 3


def test_localizer_does_not_read_user_images_modify_config_or_consume_rng(locator, monkeypatch):
    image = np.full((128, 160, 3), (240, 20, 20), np.uint8)
    image.setflags(write=False)
    saved = deepcopy(locator)

    def no_files(*args, **kwargs):
        raise AssertionError("algorithm attempted file I/O")

    monkeypatch.setattr(cv2, "imread", no_files)
    monkeypatch.setattr(cv2, "imwrite", no_files)
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


def test_missing_package_model_has_clear_error(locator, monkeypatch):
    class MissingClassifier:
        def read(self, node):
            return True
        def empty(self):
            return True
    monkeypatch.setattr(cv2, "CascadeClassifier", lambda: MissingClassifier())
    with pytest.raises(ValueError, match="skin_detect.face_locator.*包内级联资源"):
        skin_detect._haar_classifier(locator["models"]["eyes"])


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
    monkeypatch.setattr(skin_detect, "_haar_configuration", lambda _: None)
    monkeypatch.setattr(skin_detect, "_haar_refine", lambda *_: localized.copy())
    assert skin_detect.detect_skin(image, {"postprocess": {"enabled": False}})[10, 10] == 255
    cleaned = skin_detect.detect_skin(image, {})
    assert cleaned[10, 10] == 0 and cleaned[45, 45] == 255


def test_mouth_hole_survives_without_restoring_false_nose_or_chin_exclusions():
    image = np.full((160, 160, 3), (90, 130, 165), np.uint8)
    filled = np.zeros(image.shape[:2], bool)
    filled[35:133, 42:116] = True
    selected = filled.copy()
    lip = np.s_[101:106, 64:94]
    nose_shadow = np.s_[75:80, 64:94]
    chin_shadow = np.s_[112:118, 64:94]
    for region in (lip, nose_shadow, chin_shadow):
        selected[region] = False
        image[region] = (45, 50, 142)

    face = np.array([28., 20., 100., 100.])
    eyes = [np.array([50., 53., 20., 20.]), np.array([86., 53., 20., 20.])]
    protected = skin_detect._mouth_holes(image, selected, filled, face, eyes)
    expected = np.zeros_like(protected)
    expected[lip] = True
    np.testing.assert_array_equal(protected, expected)


def test_default_fixed_geometry_is_disabled_and_off_center_skin_is_retained(locator, monkeypatch):
    cfg = yaml.safe_load((Path(__file__).resolve().parents[2]/"config/skin_detect.yaml").read_text(encoding="utf-8"))
    assert cfg["spatial"]["enabled"] is False
    assert cfg["face_locator"]["enabled"] is True
    monkeypatch.setattr(skin_detect, "_haar_localize", lambda *_: None)
    image = np.full((128, 160, 3), (255, 0, 0), np.uint8)
    image[4:40, 5:50] = (80, 120, 180)
    mask = skin_detect.detect_skin(image, cfg)
    expected = skin_detect.detect_skin(image, {"postprocess": cfg["postprocess"]})
    np.testing.assert_array_equal(mask, expected)
    assert np.all(mask[8:36, 9:46] == 255)
    assert not np.any(mask[50:])


def test_detected_face_segmentation_seeds_translate_with_face(locator):
    working = np.zeros((512, 512, 3), np.uint8)
    first = skin_detect._face_segmentation((working, (np.array([120., 150., 240., 240.]), [])), locator)
    moved = skin_detect._face_segmentation((working, (np.array([55., 65., 240., 240.]), [])), locator)
    np.testing.assert_allclose(np.subtract(moved["center"], first["center"]), [-65/511, -85/511])
    assert first["roi_radius"] == moved["roi_radius"]
    assert first["seed_radius"] == moved["seed_radius"]


@pytest.mark.parametrize("bad_name", ["../secret.xml", "C:/model.xml", "model.xml", "haarcascade_face.xml/extra"])
def test_model_resources_cannot_point_to_user_files(bad_name):
    cfg = {"face_locator": {"enabled": True, "models": {
        "face_default": bad_name, "face_alt2": "haarcascade_frontalface_alt2.xml",
        "eyes": "haarcascade_eye_tree_eyeglasses.xml"}}}
    with pytest.raises(ValueError, match="包内haarcascade"):
        skin_detect._haar_configuration(cfg)
