"""Continuous skin coverage must not create a fixed nose cutout or clip corners."""
from copy import deepcopy

import cv2
import numpy as np
import pytest

from beauty.modules import skin_detect


def _settings(**patch):
    cfg = {"spatial": {"enabled": True, "surface": {"enabled": True, **patch}}}
    return skin_detect._spatial_configuration(cfg)


def _scene():
    image = np.full((128, 144, 3), (240, 20, 20), np.uint8)
    face = np.zeros(image.shape[:2], np.uint8)
    cv2.ellipse(face, (78, 65), (36, 49), 0, 0, 360, 1, -1)
    image[face != 0] = (80, 120, 180)
    return image, face != 0


def test_continuous_surface_restores_nose_and_temple_without_background_growth():
    image, face = _scene()
    core = face.copy()
    core[55:72, 73:83] = False  # enclosed artificial nose gap
    core[43:66, 103:] = False   # old right-angle notch
    core[108:] = False
    result = skin_detect._skin_surface(image, core, face, _settings())
    assert result[63, 78]
    assert result[56, 108]
    assert result[110, 78]
    assert not result[65, 10]
    assert not result[10, 78]


def test_content_features_protect_actual_eyes_and_lips_but_keep_nose():
    image, face = _scene()
    cv2.ellipse(image, (62, 48), (8, 3), 0, 0, 360, (10, 10, 10), -1)
    cv2.ellipse(image, (94, 48), (8, 3), 0, 0, 360, (10, 10, 10), -1)
    image[90:97, 62:95] = (40, 30, 160)
    result = skin_detect._content_features(image, face)
    assert not result[48, 62]
    assert not result[48, 94]
    assert not result[93, 78]
    assert result[68, 78]
    assert result[78, 58]


def test_highlight_recovery_does_not_fill_enclosed_white_eye_holes():
    image, face = _scene()
    face[44:51, 57:67] = False
    image[44:51, 57:67] = 240
    result = skin_detect._skin_surface(image, face, face, _settings(protect_features=False))
    assert not result[47, 62]
    assert result[67, 78]


def test_reddish_nose_tip_is_not_mistaken_for_lips_or_eyes():
    image, face = _scene()
    image[70:78, 71:86] = (70, 70, 170)
    image[90:97, 62:95] = (40, 30, 160)
    result = skin_detect._content_features(image, face)
    assert result[74, 78]
    assert not result[93, 78]


def test_connected_bilateral_eye_structures_are_not_rejected_as_one_wide_blob():
    image, face = _scene()
    image[44:50, 55:102] = (10, 10, 10)
    result = skin_detect._content_features(image, face)
    assert not result[47, 62]
    assert not result[47, 94]
    assert result[68, 78]


def test_single_weak_profile_cannot_expand_surface_into_background(monkeypatch):
    image, face = _scene()
    settings = _settings(protect_features=False, recover_highlights=False)
    settings["coverage_recovery"]["enabled"] = True

    def prediction(image, bounds, cfg, profile):
        result = face.astype(np.uint8)
        if profile["space"] == "gray":
            result[64:67, 113:120] = 1
        return result

    monkeypatch.setattr(skin_detect, "_spatial_profile", prediction)
    result = skin_detect._spatial_mask(image, face.astype(np.uint8)*255,
                                      (133, 173, 77, 127), settings)
    assert result[65, 78] == 255
    assert result[65, 118] == 0


def test_recovery_profile_can_recognize_skin_beyond_initial_seed_roi():
    image, face = _scene()
    cfg = {"spatial": {"enabled": True, "center": [.5, .45], "roi_radius": [.22, .30],
                        "seed_radius": [.10, .15], "surface": {"enabled": True}}}
    settings = skin_detect._spatial_configuration(cfg)
    profile = {"space": "bgr", "working_size": 144, "trusted": True, "bright_seed": False}
    clipped = skin_detect._spatial_profile(image, (133, 173, 77, 127), settings, profile)
    # 第二轮接收的是多路共识主体；本例显式提供粗主体，不能用单路退化的窄种子代替。
    coarse = face & skin_detect._ellipse(face.shape, (.5, .45, .22, .30))
    recovered = skin_detect._spatial_profile(image, (133, 173, 77, 127), settings,
                                            {**profile, "non_skin_background": True,
                                             "recovery_core": coarse})
    assert clipped[110, 78] == 0
    assert recovered[110, 78] == 1
    assert recovered[65, 105] == 1
    assert not np.any(recovered[:, :15])


def test_complete_skin_pipeline_covers_plain_face_without_treating_blue_background():
    image, face = _scene()
    cfg = {"spatial": {"enabled": True, "surface": {"enabled": True},
                        "coverage_recovery": {"enabled": True}, "workers": 1},
           "postprocess": {"enabled": True, "morph_kernel": 3}}
    result = skin_detect.detect_skin(image, cfg) != 0
    assert np.count_nonzero(result & face)/np.count_nonzero(face) > .95
    assert np.count_nonzero(result & ~face)/np.count_nonzero(~face) < .01


def test_recovered_neck_and_reddish_chin_do_not_move_or_duplicate_lip_protection():
    image, face = _scene()
    surface = face.copy()
    surface[100:, 64:93] = True
    image[surface] = (80, 120, 180)
    image[90:97, 62:95] = (40, 30, 160)
    image[105:113, 68:89] = (60, 80, 160)
    result = skin_detect._content_features(image, surface, reference=face)
    assert not result[93, 78]
    assert result[109, 78]
    assert result[68, 78]


def test_under_eye_skin_shadow_is_not_protected_as_an_eye():
    image, face = _scene()
    cv2.ellipse(image, (62, 48), (8, 3), 0, 0, 360, (10, 10, 10), -1)
    image[56:64, 52:72] = (60, 80, 110)
    result = skin_detect._content_features(image, face)
    assert not result[48, 62]
    assert result[59, 62]


def test_surface_is_readonly_repeatable_and_follows_translated_subject():
    image, face = _scene()
    settings = _settings()
    before = deepcopy(settings)
    image.setflags(write=False)
    face.setflags(write=False)
    result = skin_detect._skin_surface(image, face, face, settings)
    np.testing.assert_array_equal(result, skin_detect._skin_surface(image, face, face, settings))
    shifted = np.roll(image, -12, axis=1)
    shifted_face = np.roll(face, -12, axis=1)
    moved = skin_detect._skin_surface(shifted, shifted_face, shifted_face, settings)
    np.testing.assert_array_equal(moved[:, :120], np.roll(result, -12, axis=1)[:, :120])
    assert settings == before


@pytest.mark.parametrize("surface", [None, {"enabled": "true"}, {"protect_features": 1},
                                   {"recover_highlights": None}, {"horizontal_radius": True},
                                   {"vertical_radius": -1}, {"horizontal_radius": float("nan")},
                                   {"vertical_radius": 2**2000}])
def test_invalid_surface_configuration_has_path(surface):
    with pytest.raises(ValueError, match="skin_detect[.]spatial[.]surface"):
        skin_detect._spatial_configuration({"spatial": {"enabled": True, "surface": surface}})
