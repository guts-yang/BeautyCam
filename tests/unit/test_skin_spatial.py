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


def test_coverage_recovery_retains_face_outside_old_final_ellipse():
    image = np.full((96, 96, 3), (255, 0, 0), np.uint8)
    image[15:78, 25:83] = (80, 120, 180)
    cfg = _cfg()
    cfg["spatial"].update(center=[0.5, 0.48], roi_radius=[0.47, 0.48],
                           face_ellipse=[0.5, 0.52, 0.34, 0.4])
    cfg["spatial"]["profiles"][0]["working_size"] = 96
    clipped = skin_detect.detect_skin(image, cfg)
    cfg["spatial"]["face_ellipse"] = None
    cfg["spatial"]["coverage_recovery"] = {"enabled": True, "working_size": 96}
    restored = skin_detect.detect_skin(image, cfg)
    assert clipped[69, 78] == 0
    assert restored[69, 78] == 255
    assert not np.any(restored[:10])
    assert not np.any(restored[:, :20])


def test_recovery_recognizes_shadowed_skin_without_growing_into_blue_background():
    image = np.full((128, 128, 3), (240, 20, 20), np.uint8)
    shape = np.zeros((128, 128), np.uint8)
    cv2.ellipse(shape, (64, 60), (34, 46), 0, 0, 360, 255, -1)
    for row in range(128):
        t = np.clip((row-40)/60, 0, 1)
        image[row, shape[row] != 0] = np.rint((1-t)*np.array([80, 120, 180]) + t*np.array([40, 75, 115]))
    image[119:128] = (40, 75, 115)  # ROI外也有同色区域，不能用于污染背景模型。
    cfg = _cfg()
    cfg["spatial"].update(center=[0.5, 0.4], roi_radius=[0.47, 0.53], seed_radius=[0.1, 0.1],
                           coverage_recovery={"enabled": True, "working_size": 128})
    cfg["spatial"]["profiles"][0]["working_size"] = 128
    cfg["spatial"]["iterations"] = 3
    cfg["spatial"]["gmm_iterations"] = 10
    prediction = skin_detect.detect_skin(image, cfg)
    assert prediction[97, 64] == 255
    assert not np.any(prediction[:, :20])
    assert not np.any(prediction[119:])


def test_recovery_is_readonly_repeatable_serial_parallel_and_rng_isolated():
    image = _scene()
    image.setflags(write=False)
    cfg = _cfg()
    cfg["spatial"]["coverage_recovery"] = {"enabled": True, "working_size": 64}
    original_cfg = deepcopy(cfg)
    cv2.setRNGSeed(83)
    expected = cv2.randu(np.empty((8, 8), np.float32), 0, 1)
    cv2.setRNGSeed(83)
    first = skin_detect.detect_skin(image, cfg)
    np.testing.assert_array_equal(cv2.randu(np.empty((8, 8), np.float32), 0, 1), expected)
    cfg["spatial"]["workers"] = 3
    np.testing.assert_array_equal(first, skin_detect.detect_skin(image, cfg))
    cfg["spatial"].pop("workers")
    assert cfg == original_cfg


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


def _expanded_pixel_gmm(pixels, iterations):
    """逐像素参考，用于验证颜色计数加权保留 Lloyd 拟合的数值结果。"""
    data = pixels.astype(np.float64)
    order = np.argsort(data.mean(axis=1), kind="stable")
    means = data[order[np.rint(np.linspace(0, len(data) - 1, 5)).astype(int)]].copy()
    previous = None
    for _ in range(iterations):
        labels = np.argmin(np.sum((data[:, None, :] - means[None, :, :])**2, axis=2), axis=1)
        if previous is not None and np.array_equal(labels, previous):
            break
        previous = labels.copy()
        for k in range(5):
            if np.any(labels == k):
                means[k] = data[labels == k].mean(axis=0)
    model = np.zeros((1, 65), np.float64)
    for k in range(5):
        group = data[labels == k]
        if len(group):
            center = group.mean(axis=0)
            delta = group - center
            model[0, k] = len(group) / len(data)
            model[0, 5 + 3*k:8 + 3*k] = center
            model[0, 20 + 9*k:29 + 9*k] = (delta.T @ delta / len(group) + np.eye(3)/12).ravel()
    return model


@pytest.mark.parametrize("iterations", [1, 3, 10])
@pytest.mark.parametrize("case", ["repeated", "gray", "unique", "ties", "single"])
def test_weighted_gmm_matches_expanded_pixels_including_quantile_ties(case, iterations):
    rng = np.random.default_rng(47)
    if case == "repeated":
        palette = rng.integers(0, 256, (40, 3), dtype=np.uint8)
        pixels = np.repeat(palette, rng.integers(1, 30, len(palette)), axis=0)
    elif case == "gray":
        gray = rng.integers(0, 256, (800, 1), dtype=np.uint8)
        pixels = np.repeat(gray, 3, axis=1)
    elif case == "ties":
        pixels = np.tile(np.array([[0, 128, 255], [255, 128, 0], [128, 0, 255]], np.uint8), (100, 1))
    elif case == "single":
        pixels = np.array([[255, 0, 255]], np.uint8)
    else:
        pixels = rng.integers(0, 256, (1000, 3), dtype=np.uint8)
    pixels = np.repeat(pixels, 2, axis=0)[::2]  # 非连续只读视图也应成立。
    pixels.setflags(write=False)
    np.testing.assert_allclose(skin_detect._gmm_model(pixels, iterations),
                               _expanded_pixel_gmm(pixels, iterations), rtol=1e-12, atol=2e-10)


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
    {"enabled": True, "coverage_recovery": None},
    {"enabled": True, "coverage_recovery": {"enabled": "true"}},
    {"enabled": True, "coverage_recovery": {"enabled": True, "working_size": 0}},
    {"enabled": True, "coverage_recovery": {"lower_quantile": True}},
    {"enabled": True, "coverage_recovery": {"lower_quantile": float("nan")}},
    {"enabled": True, "coverage_recovery": {"horizontal_padding": -0.1}},
    {"enabled": True, "coverage_recovery": {"enabled": True}, "profiles": [{"space": "gray"}]},
])
def test_invalid_spatial_configuration_has_key_path(spatial):
    with pytest.raises(ValueError, match="skin_detect[.]spatial"):
        skin_detect.detect_skin(_scene(), {"spatial": spatial})
