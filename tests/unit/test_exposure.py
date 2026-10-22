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


@pytest.mark.parametrize("mask_kind", ["none", "empty"])
def test_empty_mask_does_not_brighten_entire_image(mask_kind):
    image = _image()
    mask = None if mask_kind == "none" else np.zeros(image.shape[:2], np.uint8)
    np.testing.assert_array_equal(exposure.process(image, mask, {}), image)


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
def test_overexposure_guard_checks_final_bgr_luminance(method):
    image = _image(70, (20, 20))
    image[0, :8] = 250
    mask = _mask(image)
    loose = exposure.process(image, mask, {"method": method, "gain": 1, "overexposure_ratio_limit": 1})
    strict = exposure.process(image, mask, {"method": method, "gain": 1, "overexposure_ratio_limit": 0})
    loose_y = cv2.cvtColor(loose, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    strict_y = cv2.cvtColor(strict, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    assert np.mean(loose_y > 250) > 0
    assert np.mean(strict_y > 250) == 0
    assert _mean(strict) > _mean(image)
    assert _mean(strict) < _mean(loose)


def test_preexisting_overexposure_is_not_hidden():
    image = _image()
    image[0, :16] = 255
    np.testing.assert_array_equal(exposure.process(image, _mask(image), {}), image)


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
