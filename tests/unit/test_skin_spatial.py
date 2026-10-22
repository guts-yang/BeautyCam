"""正式 M1 空间分割：实际误检抑制、确定性、RNG 隔离与配置契约。"""
from copy import deepcopy

import cv2
import numpy as np
import pytest

from beauty.modules import skin_detect


def _scene():
    image = np.full((64, 64, 3), (255, 0, 0), np.uint8)
    image[16:48, 16:48] = (80, 120, 180)
    image[3:9, 3:9] = (80, 120, 180)
    return image


def _cfg():
    return {"postprocess": {"enabled": False}, "spatial": {
        "enabled": True, "center": [0.5, 0.5], "roi_radius": [0.45, 0.45],
        "seed_radius": [0.12, 0.12], "iterations": 1, "gmm_iterations": 3,
        "vote_threshold": 1, "profiles": [
            {"space": "bgr", "working_size": 64, "trusted": True, "bright_seed": False}]}}


def test_spatial_segmentation_removes_disconnected_skin_colored_background():
    image = _scene()
    raw = skin_detect.detect_skin(image, {"postprocess": {"enabled": False}})
    prediction = skin_detect.detect_skin(image, _cfg())
    assert np.all(raw[3:9, 3:9] == 255)
    assert not np.any(prediction[3:9, 3:9])
    assert np.all(prediction[24:40, 24:40] == 255)
    assert prediction.dtype == np.uint8 and prediction.shape == image.shape[:2]


def test_geometry_protects_configured_feature_area_and_does_not_modify_inputs():
    image = _scene()
    image.setflags(write=False)
    cfg = _cfg()
    cfg["spatial"]["exclude_ellipses"] = [[0.5, 0.5, 0.06, 0.08]]
    original, original_cfg = image.copy(), deepcopy(cfg)
    prediction = skin_detect.detect_skin(image, cfg)
    assert not np.any(prediction[30:34, 30:34])
    assert prediction[24, 24] == 255
    np.testing.assert_array_equal(image, original)
    assert cfg == original_cfg


def test_spatial_processing_is_repeatable_and_preserves_global_opencv_rng():
    image = _scene()
    cfg = _cfg()
    cv2.setRNGSeed(123)
    expected_random = cv2.randu(np.empty((8, 8), np.float32), 0, 1)
    cv2.setRNGSeed(123)
    first = skin_detect.detect_skin(image, cfg)
    actual_random = cv2.randu(np.empty((8, 8), np.float32), 0, 1)
    np.testing.assert_array_equal(actual_random, expected_random)
    cv2.setRNGSeed(456)
    second = skin_detect.detect_skin(image, cfg)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("space,trusted,bright", [
    ("gray", False, False), ("bgr", True, False), ("bgr", True, True)])
def test_all_deployed_profiles_obey_readonly_contract(space, trusted, bright):
    image = _scene()[::2, ::2]
    image.setflags(write=False)
    original = image.copy()
    cfg = _cfg()
    cfg["spatial"]["profiles"] = [{"space": space, "working_size": 32,
                                  "trusted": trusted, "bright_seed": bright}]
    result = skin_detect.detect_skin(image, cfg)
    assert result.shape == image.shape[:2] and result.dtype == np.uint8
    assert set(np.unique(result)) <= {0, 255}
    np.testing.assert_array_equal(image, original)


def test_disabled_spatial_is_byte_exact_with_legacy_color_detection():
    image = _scene()
    legacy = skin_detect.detect_skin(image, {})
    disabled = skin_detect.detect_skin(image, {"spatial": {"enabled": False}})
    np.testing.assert_array_equal(legacy, disabled)


def test_parallel_profiles_equal_serial_and_do_not_consume_rng():
    image = _scene()
    cfg = _cfg()
    profile = cfg["spatial"]["profiles"][0]
    cfg["spatial"]["profiles"] = [dict(profile) for _ in range(3)]
    cfg["spatial"]["vote_threshold"] = 2
    serial = skin_detect.detect_skin(image, cfg)
    cfg["spatial"]["workers"] = 3
    cv2.setRNGSeed(97)
    expected = cv2.randu(np.empty((8, 8), np.float32), 0, 1)
    cv2.setRNGSeed(97)
    parallel = skin_detect.detect_skin(image, cfg)
    np.testing.assert_array_equal(parallel, serial)
    np.testing.assert_array_equal(cv2.randu(np.empty((8, 8), np.float32), 0, 1), expected)


def test_gmm_models_are_finite_positive_definite_and_input_only():
    pixels = np.array([[10, 30, 70], [50, 60, 80], [180, 160, 150]] * 5, np.uint8)
    pixels.setflags(write=False)
    model = skin_detect._gmm_model(pixels, 5)
    assert model.shape == (1, 65) and model.dtype == np.float64
    assert np.isfinite(model).all()
    assert model[0, :5].sum() == pytest.approx(1)
    for index in np.flatnonzero(model[0, :5]):
        covariance = model[0, 20 + index * 9:29 + index * 9].reshape(3, 3)
        assert np.linalg.eigvalsh(covariance).min() > 0
    np.testing.assert_array_equal(model, skin_detect._gmm_model(pixels, 5))


@pytest.mark.parametrize("spatial", [
    None, {"enabled": "true"}, {"enabled": True, "center": [0.5]},
    {"enabled": True, "roi_radius": [0, 0.4]},
    {"enabled": True, "seed_radius": [0.8, 0.8]},
    {"enabled": True, "profiles": []}, {"enabled": True, "profiles": ["bad"]},
    {"enabled": True, "profiles": [{"space": "lab"}]},
    {"enabled": True, "profiles": [{"space": "bgr", "working_size": 0}]},
    {"enabled": True, "profiles": [{"space": "bgr", "bright_seed": "false"}]},
    {"enabled": True, "vote_threshold": 10},
    {"enabled": True, "gmm_iterations": 0},
    {"enabled": True, "iterations": 2000},
    {"enabled": True, "face_ellipse": [0.5, 0.5, -0.1, 0.4]},
    {"enabled": True, "exclude_ellipses": None},
    {"enabled": True, "center": [2**2000, 0.5]},
])
def test_invalid_spatial_configuration_has_key_path(spatial):
    with pytest.raises(ValueError, match="skin_detect[.]spatial"):
        skin_detect.detect_skin(_scene(), {"spatial": spatial})
