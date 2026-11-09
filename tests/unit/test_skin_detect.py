"""M1 颜色判定、后处理和输入契约；只使用内存合成图。"""

from copy import deepcopy

import cv2
import numpy as np
import pytest

from beauty.modules.skin_detect import detect_skin


def _scene():
    image = np.full((64, 64, 3), (255, 0, 0), dtype=np.uint8)
    image[16:48, 16:48] = (80, 120, 180)
    return image


def test_detects_skin_patch_and_excludes_blue_background():
    image = _scene()
    mask = detect_skin(image, {"postprocess": {"enabled": False}})
    expected = np.zeros(image.shape[:2], dtype=np.uint8)
    expected[16:48, 16:48] = 255
    np.testing.assert_array_equal(mask, expected)
    assert mask.dtype == np.uint8
    assert not np.shares_memory(mask, image)


@pytest.mark.parametrize("color", [(0, 0, 0), (255, 255, 255), (128, 128, 128),
                                   (255, 0, 0), (0, 255, 0)])
def test_non_skin_or_extreme_light_can_return_empty_mask(color):
    image = np.full((24, 32, 3), color, dtype=np.uint8)
    assert not np.any(detect_skin(image, {}))


def test_thresholds_are_inclusive_and_use_opencv_cr_cb_order():
    image = np.full((3, 4, 3), (80, 120, 180), dtype=np.uint8)
    _, cr, cb = map(int, cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[0, 0])
    cfg = {"ycbcr": {"cr_min": cr, "cr_max": cr, "cb_min": cb, "cb_max": cb},
           "postprocess": {"enabled": False}}
    assert np.all(detect_skin(image, cfg) == 255)
    cfg["ycbcr"]["cr_min"] = cr + 1
    cfg["ycbcr"]["cr_max"] = cr + 1
    assert not np.any(detect_skin(image, cfg))


def test_configuration_changes_result_without_global_state():
    image = np.full((8, 8, 3), 128, dtype=np.uint8)
    custom = {"ycbcr": {"cr_min": 128, "cr_max": 128, "cb_min": 128, "cb_max": 128}}
    assert not np.any(detect_skin(image, {}))
    assert np.all(detect_skin(image, custom) == 255)
    assert not np.any(detect_skin(image, {}))


def test_opening_removes_isolated_noise_and_preserves_main_patch():
    image = _scene()
    image[4, 4] = (80, 120, 180)
    raw = detect_skin(image, {"postprocess": {"enabled": False}})
    cleaned = detect_skin(image, {})
    assert raw[4, 4] == 255 and cleaned[4, 4] == 0
    assert np.all(cleaned[20:44, 20:44] == 255)
    assert set(np.unique(cleaned)) == {0, 255}


def test_connected_area_filter_uses_image_area_and_keeps_larger_regions():
    image = np.full((64, 64, 3), (255, 0, 0), dtype=np.uint8)
    image[20:40, 20:40] = (80, 120, 180)
    image[4:7, 4:7] = (80, 120, 180)
    cfg = {"postprocess": {"morph_kernel": 1, "min_area_ratio": 0.01}}
    mask = detect_skin(image, cfg)
    assert np.all(mask[20:40, 20:40] == 255)
    assert not np.any(mask[4:7, 4:7])
    cfg["postprocess"]["enabled"] = False
    assert np.all(detect_skin(image, cfg)[4:7, 4:7] == 255)


def test_component_with_area_equal_to_cutoff_is_retained():
    image = np.full((10, 10, 3), (255, 0, 0), dtype=np.uint8)
    image[2:4, 2:4] = (80, 120, 180)
    cfg = {"postprocess": {"morph_kernel": 1, "min_area_ratio": 0.04}}
    assert np.count_nonzero(detect_skin(image, cfg)) == 4


def test_large_dark_occlusion_remains_outside_skin():
    image = np.full((64, 64, 3), (80, 120, 180), dtype=np.uint8)
    image[20:44, 20:44] = 0
    mask = detect_skin(image, {})
    assert not np.any(mask[24:40, 24:40])
    assert np.all(mask[:16, :] == 255)


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (2, 3)])
def test_small_images_are_supported(shape):
    image = np.full((*shape, 3), (80, 120, 180), dtype=np.uint8)
    mask = detect_skin(image, {})
    assert mask.shape == shape and mask.dtype == np.uint8
    assert np.all(mask == 255)


def test_input_and_nested_configuration_unchanged_with_readonly_strided_array():
    image = _scene()[::2, ::2]
    image.flags.writeable = False
    original = image.copy()
    cfg = {"ycbcr": {"cr_min": 133}, "postprocess": {"morph_kernel": 3}}
    original_cfg = deepcopy(cfg)
    first = detect_skin(image, cfg)
    second = detect_skin(image, cfg)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(first, second)
    assert cfg == original_cfg
    assert not np.shares_memory(first, second)


@pytest.mark.parametrize("image", [None, [], np.empty((0, 4, 3), np.uint8),
                                   np.empty((4, 0, 3), np.uint8),
                                   np.zeros((4, 4), np.uint8),
                                   np.zeros((4, 4, 4), np.uint8),
                                   np.zeros((4, 4, 3), np.float32)])
def test_invalid_image_has_readable_error(image):
    with pytest.raises(ValueError, match="BGR uint8"):
        detect_skin(image, {})


@pytest.mark.parametrize("cfg", [None, [], {"ycbcr": None}, {"postprocess": []},
                                 {"ycbcr": {"cr_min": -1}},
                                 {"ycbcr": {"cb_max": 256}},
                                 {"ycbcr": {"cr_min": 200, "cr_max": 100}},
                                 {"ycbcr": {"cr_min": True}},
                                 {"postprocess": {"enabled": "false"}},
                                 {"postprocess": {"morph_kernel": 0}},
                                 {"postprocess": {"morph_kernel": 4}},
                                 {"postprocess": {"min_area_ratio": -0.1}},
                                 {"postprocess": {"min_area_ratio": 1.1}},
                                 {"postprocess": {"min_area_ratio": float("nan")}}])
def test_invalid_configuration_rejected(cfg):
    with pytest.raises(ValueError, match="skin_detect"):
        detect_skin(_scene(), cfg)
