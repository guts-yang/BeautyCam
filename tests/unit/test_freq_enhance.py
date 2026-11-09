"""M4 频率响应、边界、回注和只读诊断契约。"""
from copy import deepcopy

import numpy as np
import pytest

from beauty.modules.freq_enhance import get_frequency_diagnostics, process


def _grating(k, width=65, height=32):
    gray = np.rint(128 + 40 * np.cos(np.pi * k * np.arange(width) / (width - 1))).astype(np.uint8)
    return np.broadcast_to(gray[None, :, None], (height, width, 3)).copy()


def test_gaussian_frequency_response_on_known_cosine_mode():
    image = _grating(8)
    out = process(image, np.full(image.shape[:2], 255, np.uint8),
                  {"lowpass_sigma": 3, "highpass_gain": 0})
    cosine = np.cos(np.pi * 8 * np.arange(65) / 64)
    before = float(np.dot(image[0, :, 0].astype(float) - 128, cosine))
    after = float(np.dot(out[0, :, 0].astype(float) - 128, cosine))
    expected = np.exp(-2 * np.pi**2 * 3**2 * (8 / 128)**2)
    assert after / before == pytest.approx(expected, abs=0.02)


def test_lowpass_attenuates_high_frequency_more_than_low_frequency():
    amplitudes = []
    for k in (2, 16):
        image = _grating(k)
        out = process(image, np.full(image.shape[:2], 255, np.uint8), {"highpass_gain": 0})
        amplitudes.append(float(np.std(out[:, :, 0])))
    assert amplitudes[1] < 0.1 * amplitudes[0]


def test_detail_injection_strength_changes_texture_without_brightness_shift():
    image = _grating(16)
    mask = np.full(image.shape[:2], 255, np.uint8)
    soft = process(image, mask, {"highpass_gain": 0.2})
    strong = process(image, mask, {"highpass_gain": 0.9})
    assert np.std(strong[:, :, 0]) > 3 * np.std(soft[:, :, 0])
    assert abs(float(np.mean(strong)) - float(np.mean(image))) < 1


def test_unit_gains_reconstruct_color_image_exactly():
    image = np.random.default_rng(4).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    out = process(image, np.full(image.shape[:2], 255, np.uint8),
                  {"lowpass_gain": 1, "highpass_gain": 1})
    np.testing.assert_array_equal(out, image)
    assert not np.shares_memory(out, image)


@pytest.mark.parametrize("shape", [(1, 1), (1, 8), (8, 1), (2, 3)])
def test_constant_color_and_tiny_images_stay_unchanged(shape):
    image = np.full((*shape, 3), (80, 120, 180), np.uint8)
    out = process(image, np.full(shape, 255, np.uint8), {})
    np.testing.assert_array_equal(out, image)
    diag = get_frequency_diagnostics(image, {})
    assert np.all(diag["high_frequency"] == 128)


def test_reflection_avoids_opposite_edge_wraparound():
    image = np.full((32, 64, 3), 40, np.uint8)
    image[:, 32:] = 200
    out = process(image, np.full((32, 64), 255, np.uint8), {"highpass_gain": 0})
    assert np.all(out[:, 0] == 40) and np.all(out[:, -1] == 200)


@pytest.mark.parametrize("mask", [None, np.zeros((32, 65), np.uint8)])
def test_empty_or_missing_mask_returns_copy(mask):
    image = _grating(8)
    out = process(image, mask, {})
    np.testing.assert_array_equal(out, image)
    assert out is not image


def test_masks_and_configuration_unchanged_and_no_outside_changes():
    image = _grating(8)[::2, ::2]
    image.flags.writeable = False
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[3:-3, 5:-5] = 255
    mask.flags.writeable = False
    old_image, old_mask = image.copy(), mask.copy()
    cfg = {"diagnostics": {"low_frequency": False, "high_frequency": False}}
    old_cfg = deepcopy(cfg)
    first = process(image, mask, cfg)
    diag = get_frequency_diagnostics(image, cfg)
    diag["low_frequency"][:] = 0
    second = process(image, mask, cfg)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[mask == 0], image[mask == 0])
    np.testing.assert_array_equal(image, old_image)
    np.testing.assert_array_equal(mask, old_mask)
    assert cfg == old_cfg
    assert np.any(first[mask != 0] != image[mask != 0])


def test_diagnostics_return_real_layers_and_preserve_signed_high_frequency():
    image = _grating(16)
    original = image.copy()
    diag = get_frequency_diagnostics(image, {})
    assert set(diag) == {"low_frequency", "high_frequency"}
    for data in diag.values():
        assert data.shape == image.shape and data.dtype == np.uint8
    assert float(np.std(diag["low_frequency"])) < 2
    assert np.min(diag["high_frequency"]) < 128 < np.max(diag["high_frequency"])
    reconstruction = diag["low_frequency"].astype(int) + diag["high_frequency"].astype(int) - 128
    np.testing.assert_allclose(reconstruction, image, atol=1)
    np.testing.assert_array_equal(image, original)
    assert not np.shares_memory(diag["low_frequency"], diag["high_frequency"])


@pytest.mark.parametrize("cfg", [None, [], {"lowpass_sigma": 0}, {"lowpass_sigma": -1},
                                 {"highpass_gain": -1}, {"lowpass_gain": float("inf")},
                                 {"highpass_gain": float("nan")}, {"highpass_gain": True},
                                 {"highpass_gain": 1e308}, {"diagnostics": None},
                                 {"diagnostics": {"low_frequency": "yes"}}])
def test_invalid_configuration(cfg):
    with pytest.raises(ValueError, match="freq_enhance"):
        process(_grating(8), None, cfg)


@pytest.mark.parametrize("image", [None, [], np.zeros((0, 1, 3), np.uint8),
                                   np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3), float)])
def test_invalid_image(image):
    with pytest.raises(ValueError, match="BGR uint8"):
        get_frequency_diagnostics(image, {})


@pytest.mark.parametrize("mask", [np.zeros((31, 65), np.uint8), np.ones((32, 65), np.uint8),
                                  np.zeros((32, 65), bool), np.zeros((32, 65, 1), np.uint8)])
def test_invalid_mask(mask):
    with pytest.raises(ValueError, match="skin_mask"):
        process(_grating(8), mask, {})
