"""M2 实际提亮、参数作用、过曝预算与纯函数契约。"""
from copy import deepcopy

import cv2
import numpy as np
import pytest

from beauty.modules import exposure


def _image(value=60, shape=(24, 32)):
    return np.full((*shape, 3), value, np.uint8)


def _mask(image):
    return np.full(image.shape[:2], 255, np.uint8)


def _mean(image):
    return float(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0].mean())


@pytest.mark.parametrize("method", ["gamma", "linear"])
def test_dark_constant_is_brightened_toward_target(method):
    image = _image()
    result = exposure.process(image, _mask(image), {"method": method})
    # desired=60+0.3*(150-60)=87，不只是输出格式正确。
    assert _mean(result) == pytest.approx(87, abs=1)
    assert result.dtype == np.uint8 and result.shape == image.shape
    assert not np.shares_memory(image, result)
    assert np.all(image == 60)


def test_strength_target_and_maximum_gain_have_real_effects():
    image = _image()
    mask = _mask(image)
    standard = _mean(exposure.process(image, mask, {}))
    assert _mean(exposure.process(image, mask, {"gain": 0.6})) > standard
    assert _mean(exposure.process(image, mask, {"target_y": 120})) < standard
    assert _mean(exposure.process(image, mask, {"max_gain": 1.2})) == pytest.approx(72, abs=1)
    assert _mean(exposure.process(image, mask, {"gamma": 0.8})) > standard
    assert _mean(exposure.process(image, mask, {"gamma": 2})) == 60


@pytest.mark.parametrize("patch", [{"gain": 0}, {"max_gain": 1}, {"target_y": 50}])
def test_explicit_noop_is_byte_exact(patch):
    image = np.full((16, 20, 3), (30, 65, 110), np.uint8)
    result = exposure.process(image, _mask(image), patch)
    np.testing.assert_array_equal(result, image)
    assert not np.shares_memory(result, image)


def test_mask_blend_removes_brightness_step_without_changing_outside():
    image = _image(60, (128, 128))
    mask = np.zeros((128, 128), np.uint8)
    mask[20:108, 20:108] = 255
    hard = exposure.process(image, mask, {})
    soft = exposure.process(image, mask, {"mask_blend": {"radius_ratio": 0.1}})
    hard_y = cv2.cvtColor(hard, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(float)
    soft_y = cv2.cvtColor(soft, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(float)
    assert hard_y[64, 20] - hard_y[64, 19] > 20
    assert soft_y[64, 20] - soft_y[64, 19] <= 1
    assert np.max(np.abs(np.diff(soft_y[64, 16:65]))) <= 5
    assert soft_y[64, 64] > soft_y[64, 22] > soft_y[64, 20]
    np.testing.assert_array_equal(soft[mask == 0], image[mask == 0])
    assert _mean(soft[20:108, 20:108]) >= 80


def test_mask_blend_does_not_restore_hard_step_on_narrow_fragment():
    image = _image(60, (128, 128))
    mask = np.zeros((128, 128), np.uint8)
    mask[20:108, 63:66] = 255
    result = exposure.process(image, mask, {"mask_blend": {"radius_ratio": 0.1}})
    assert int(result[64, 63, 0]) - 60 <= 2


def test_zero_blend_is_legacy_exact_and_full_mask_is_unchanged():
    image = _image(60, (32, 32))
    mask = _mask(image)
    expected = exposure.process(image, mask, {})
    np.testing.assert_array_equal(exposure.process(image, mask, {"mask_blend": {"radius_ratio": 0}}), expected)
    np.testing.assert_array_equal(exposure.process(image, mask, {"mask_blend": {"radius_ratio": 0.03}}), expected)


@pytest.mark.parametrize("blend", [None, {"radius_ratio": True}, {"radius_ratio": -0.1},
                                  {"radius_ratio": float("inf")}, {"radius_ratio": 1.1},
                                  {"max_compensation": 0.5}, {"max_compensation": 5}])
def test_invalid_mask_blend_configuration_is_rejected(blend):
    image = _image()
    with pytest.raises(ValueError, match="exposure[.]mask_blend"):
        exposure.process(image, _mask(image), {"mask_blend": blend})


@pytest.mark.parametrize("mask_kind", ["none", "empty"])
def test_empty_mask_does_not_brighten_entire_image(mask_kind):
    image = _image()
    mask = None if mask_kind == "none" else np.zeros(image.shape[:2], np.uint8)
    np.testing.assert_array_equal(exposure.process(image, mask, {"clahe": {"enabled": True}}), image)


def test_readonly_strided_arrays_and_config_stay_unchanged():
    image = np.full((24, 48, 3), (35, 60, 95), np.uint8)[:, ::2]
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[4:20, 5:18] = 255
    image.setflags(write=False)
    mask.setflags(write=False)
    original, original_mask = image.copy(), mask.copy()
    cfg = {"method": "gamma", "gain": 0.3}
    original_cfg = deepcopy(cfg)
    result = exposure.process(image, mask, cfg)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(mask, original_mask)
    np.testing.assert_array_equal(result[mask == 0], image[mask == 0])
    assert _mean(result[4:20, 5:18]) > _mean(image[4:20, 5:18])
    assert cfg == original_cfg


@pytest.mark.parametrize("method", ["gamma", "linear"])
@pytest.mark.parametrize("use_clahe", [False, True])
def test_overexposure_guard_checks_final_bgr_luminance(method, use_clahe):
    image = _image(70, (20, 20))
    image[0, :8] = 250
    mask = _mask(image)
    cfg = {"method": method, "gain": 1, "clahe": {"enabled": use_clahe}}
    loose = exposure.process(image, mask, {**cfg, "overexposure_ratio_limit": 1})
    strict = exposure.process(image, mask, {**cfg, "overexposure_ratio_limit": 0})
    loose_y = cv2.cvtColor(loose, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    strict_y = cv2.cvtColor(strict, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    assert np.mean(loose_y > 250) > 0
    assert np.mean(strict_y > 250) == 0
    assert _mean(strict) > _mean(image)
    assert _mean(strict) < _mean(loose)


def test_preexisting_overexposure_is_not_hidden():
    image = _image()
    image[0, :16] = 255
    np.testing.assert_array_equal(exposure.process(image, _mask(image), {"clahe": {"enabled": True}}), image)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1)])
def test_tiny_and_black_images(shape):
    for value in (0, 60, 180, 255):
        image = _image(value, shape)
        result = exposure.process(image, _mask(image), {})
        assert result.shape == image.shape
        if value != 60:
            np.testing.assert_array_equal(result, image)


@pytest.mark.parametrize("cfg", [
    None, [], {"gain": -0.1}, {"gain": 1.1}, {"gain": True},
    {"target_y": 0}, {"max_gain": 0.9}, {"gamma": 0}, {"gamma": float("nan")},
    {"overexposure_y": 256}, {"overexposure_ratio_limit": -0.1},
    {"method": "clahe"}, {"method": []},
])
def test_bad_configuration_fails_with_key_path(cfg):
    image = _image()
    with pytest.raises(ValueError, match="exposure[.]"):
        exposure.process(image, _mask(image), cfg)


@pytest.mark.parametrize("image", [
    None, np.zeros((0, 2, 3), np.uint8), np.zeros((4, 5), np.uint8),
    np.zeros((4, 5, 3), float),
])
def test_bad_images(image):
    with pytest.raises(ValueError, match="image_bgr"):
        exposure.process(image, None, {})


@pytest.mark.parametrize("mask", [
    np.ones((24, 32), np.uint8), np.zeros((24, 32), bool),
    np.zeros((24, 32, 1), np.uint8), np.zeros((3, 4), np.uint8),
])
def test_bad_masks(mask):
    with pytest.raises(ValueError, match="skin_mask"):
        exposure.process(_image(), mask, {})


@pytest.mark.parametrize("method", ["gamma", "linear"])
def test_clahe_missing_or_disabled_preserves_legacy_curve_byte_exactly(method):
    rng = np.random.default_rng(7)
    image = rng.integers(15, 110, (32, 40, 3), dtype=np.uint8)
    mask = _mask(image)
    cfg = {"method": method, "gain": 0.3}
    missing = exposure.process(image, mask, cfg)
    disabled = exposure.process(image, mask, {**cfg, "clahe": {
        "enabled": False, "clip_limit": 8.0, "tile_grid_size": [2, 2]}})
    np.testing.assert_array_equal(missing, disabled)


def _textured_image():
    y, x = np.mgrid[:96, :128]
    gray = (55 + x // 4 + ((x // 4 + y // 4) % 2) * 5).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def test_clahe_increases_contrast_on_low_contrast_texture_and_parameters_work():
    image = _textured_image()
    mask = _mask(image)
    plain = exposure.process(image, mask, {})
    mild = exposure.process(image, mask, {"clahe": {"enabled": True, "clip_limit": 2, "tile_grid_size": [8, 8]}})
    strong = exposure.process(image, mask, {"clahe": {"enabled": True, "clip_limit": 8, "tile_grid_size": [8, 8]}})
    other_grid = exposure.process(image, mask, {"clahe": {"enabled": True, "clip_limit": 2, "tile_grid_size": [2, 2]}})
    contrast = lambda value: float(cv2.cvtColor(value, cv2.COLOR_BGR2YCrCb)[:, :, 0].std())
    assert contrast(mild) > contrast(plain)
    assert contrast(strong) > contrast(mild)
    assert not np.array_equal(mild, other_grid)
    assert np.all(cv2.cvtColor(strong, cv2.COLOR_BGR2YCrCb)[:, :, 0] <= 250)


def test_clahe_preserves_partial_mask_readonly_inputs_and_nested_configuration():
    image = _textured_image()[:, ::2]
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[12:84, 8:56] = 255
    cfg = {"clahe": {"enabled": True, "clip_limit": 2, "tile_grid_size": [4, 4]}}
    original_cfg = deepcopy(cfg)
    original, original_mask = image.copy(), mask.copy()
    image.setflags(write=False)
    mask.setflags(write=False)
    result = exposure.process(image, mask, cfg)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(mask, original_mask)
    np.testing.assert_array_equal(result[mask == 0], image[mask == 0])
    assert np.any(result[mask != 0] != image[mask != 0])
    assert cfg == original_cfg


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (2, 3)])
def test_clahe_tiny_images_and_oversized_grid_remain_safe(shape):
    y = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape) + 40
    image = np.repeat(y[:, :, None], 3, axis=2)
    result = exposure.process(image, _mask(image), {"clahe": {
        "enabled": True, "clip_limit": 1e308, "tile_grid_size": [2**60, 2**60]}})
    assert result.shape == image.shape and result.dtype == np.uint8
    np.testing.assert_array_equal(image[:, :, 0], y)


@pytest.mark.parametrize("clahe", [
    None, [], {"enabled": "true"}, {"clip_limit": 0}, {"clip_limit": float("inf")},
    {"clip_limit": 2**2000}, {"tile_grid_size": [0, 8]}, {"tile_grid_size": [8]},
    {"tile_grid_size": [True, 8]}, {"tile_grid_size": [2.5, 8]},
])
def test_invalid_clahe_configuration_has_key_path(clahe):
    image = _image()
    with pytest.raises(ValueError, match="exposure[.]clahe"):
        exposure.process(image, _mask(image), {"clahe": clahe})
