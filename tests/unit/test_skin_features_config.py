"""Feature settings must control processing, preserve defaults and reject invalid geometry."""
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from beauty.modules import skin_detect


def test_shipped_feature_settings_equal_legacy_defaults():
    config = yaml.safe_load((Path(__file__).resolve().parents[2] / "config/skin_detect.yaml").read_text(encoding="utf-8"))
    assert skin_detect._feature_configuration(config) == skin_detect._feature_configuration({})


def test_dark_feature_threshold_is_configurable_without_mutating_input():
    image = np.full((128, 144, 3), (240, 20, 20), np.uint8)
    face = np.zeros(image.shape[:2], np.uint8)
    cv2.ellipse(face, (78, 65), (36, 49), 0, 0, 360, 1, -1)
    image[face != 0] = (80, 120, 180)
    cv2.ellipse(image, (62, 48), (8, 3), 0, 0, 360, (10, 10, 10), -1)
    image.flags.writeable = False
    default = skin_detect._content_features(image, face != 0)
    config = {"features": {"dark": {"local_difference": 255}}}
    original = deepcopy(config)
    tuned = skin_detect._content_features(image, face != 0, parameters=skin_detect._feature_configuration(config))
    assert not default[48, 62] and tuned[48, 62]
    assert config == original


def test_eye_protection_parameters_reach_public_detect_entrypoint(monkeypatch):
    image = np.full((160, 160, 3), (80, 120, 180), np.uint8)
    face = np.array([25., 20., 110., 110.])
    eyes = [np.array([45., 50., 24., 24.]), np.array([92., 50., 24., 24.])]
    monkeypatch.setattr(skin_detect, "_haar_localize", lambda *_: (face, eyes))
    config = {"postprocess": {"enabled": False}, "face_locator": {"enabled": True,
              "models": {"face_default": "haarcascade_frontalface_default.xml",
                         "face_alt2": "haarcascade_frontalface_alt2.xml",
                         "eyes": "haarcascade_eye_tree_eyeglasses.xml"}}}
    default = skin_detect.detect_skin(image, config)
    config["features"] = {"eyes": {"radius_height": .4, "max_radius_y_distance": .4}}
    tuned = skin_detect.detect_skin(image, config)
    assert default[68, 57] == 255 and tuned[68, 57] == 0


@pytest.mark.parametrize("features", [
    None, {"dark": None}, {"typo": {}}, {"dark": {"typo": 1}},
    {"dark": {"local_difference": True}}, {"dark": {"local_difference": float("nan")}},
    {"dark": {"local_radius": 2**2000}}, {"dark": {"close_kernel": [2, 3]}},
    {"dark": {"search_bands": [[.6, .4]]}}, {"dark": {"search_bands": []}},
    {"lips": {"min_width": .7, "max_width": .5}}, {"lips": {"residual_percentile": 101}},
    {"eyes": {"radius_width": 0}}, {"dilate_kernel": 2},
])
def test_invalid_feature_settings_include_key_path(features):
    with pytest.raises(ValueError, match="skin_detect[.]features"):
        skin_detect._feature_configuration({"features": features})
