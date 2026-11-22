"""M3 的保边、局部暗点、掩码隔离和输入契约。"""
from copy import deepcopy

import cv2
import numpy as np
import pytest

from beauty.modules.smooth import process


def _noise_scene():
    rng = np.random.default_rng(17)
    gray = np.clip(150 + rng.normal(0, 12, (64, 64)), 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def test_smoothing_reduces_noise_with_given_mask():
    image = _noise_scene()
    mask = np.full(image.shape[:2], 255, np.uint8)
    out = process(image, mask, {})
    assert np.var(out[8:-8, 8:-8, 0]) < np.var(image[8:-8, 8:-8, 0])
    assert out.shape == image.shape and out.dtype == np.uint8
    assert not np.shares_memory(out, image)


@pytest.mark.parametrize("mask_kind", ["empty", "none"])
def test_missing_or_empty_mask_returns_unchanged_new_image(mask_kind):
    image = _noise_scene()
    mask = np.zeros(image.shape[:2], np.uint8) if mask_kind == "empty" else None
    out = process(image, mask, {})
    np.testing.assert_array_equal(out, image)
    assert not np.shares_memory(out, image)


def test_outside_mask_and_input_arrays_unchanged():
    image = _noise_scene()[::2, ::2]
    image.flags.writeable = False
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[8:24, 8:24] = 255
    mask.flags.writeable = False
    original, original_mask = image.copy(), mask.copy()
    cfg = {"strength": 0.5, "bilateral_d": 5}
    original_cfg = deepcopy(cfg)
    out = process(image, mask, cfg)
    np.testing.assert_array_equal(out[mask == 0], image[mask == 0])
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(mask, original_mask)
    assert cfg == original_cfg
    assert np.any(out[mask != 0] != image[mask != 0])


def test_small_dark_spot_is_lightened_but_not_large_feature():
    image = np.full((64, 64, 3), 180, np.uint8)
    image[18:21, 18:21] = 80
    image[35:51, 30:50] = 30
    mask = np.full((64, 64), 255, np.uint8)
    out = process(image, mask, {"strength": 1.0})
    assert np.mean(out[18:21, 18:21]) > 130
    assert np.mean(out[39:47, 34:46]) <= 32


def test_small_spot_outside_skin_mask_is_not_repaired():
    image = np.full((40, 40, 3), 180, np.uint8)
    image[3:6, 3:6] = 80
    mask = np.zeros((40, 40), np.uint8)
    mask[12:34, 12:34] = 255
    out = process(image, mask, {"strength": 1.0})
    np.testing.assert_array_equal(out[3:6, 3:6], image[3:6, 3:6])


def test_high_contrast_step_edge_retained():
    image = np.full((48, 48, 3), 40, np.uint8)
    image[:, 24:] = 210
    out = process(image, np.full((48, 48), 255, np.uint8), {"strength": 1.0})
    assert float(np.mean(out[:, 24])) - float(np.mean(out[:, 23])) >= 0.9 * 170


@pytest.mark.parametrize("cfg", [{"strength": 0}, {"bilateral_d": 1}])
def test_noop_settings_return_copy(cfg):
    image = _noise_scene()
    out = process(image, np.full(image.shape[:2], 255, np.uint8), cfg)
    np.testing.assert_array_equal(out, image)
    assert out is not image


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (3, 2)])
def test_tiny_image_stays_valid(shape):
    image = np.full((*shape, 3), (90, 130, 180), np.uint8)
    out = process(image, np.full(shape, 255, np.uint8), {})
    np.testing.assert_array_equal(out, image)


def test_same_input_configuration_is_deterministic():
    image = _noise_scene()
    mask = np.full(image.shape[:2], 255, np.uint8)
    np.testing.assert_array_equal(process(image, mask, {}), process(image, mask, {}))


@pytest.mark.parametrize("cfg", [None, [], {"strength": -1}, {"strength": 1.1},
                                 {"strength": True}, {"sigma_color": 0},
                                 {"sigma_space": float("nan")}, {"bilateral_d": 4},
                                 {"bilateral_d": 0}, {"bilateral_d": True},
                                 {"border_type": "BORDER_CONSTANT"}])
def test_invalid_configuration(cfg):
    with pytest.raises(ValueError, match="smooth"):
        process(_noise_scene(), np.zeros((64, 64), np.uint8), cfg)


@pytest.mark.parametrize("mask", [np.zeros((63, 64), np.uint8),
                                  np.ones((64, 64), np.uint8),
                                  np.zeros((64, 64), bool),
                                  np.zeros((64, 64, 1), np.uint8)])
def test_invalid_mask(mask):
    with pytest.raises(ValueError, match="skin_mask"):
        process(_noise_scene(), mask, {})


@pytest.mark.parametrize("image", [None, [], np.zeros((0, 1, 3), np.uint8),
                                   np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3), float)])
def test_invalid_image(image):
    with pytest.raises(ValueError, match="BGR uint8"):
        process(image, None, {})
