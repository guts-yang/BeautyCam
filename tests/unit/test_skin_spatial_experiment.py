"""离线 GrabCut 候选的图像契约与无 GT 预测边界。"""
from copy import deepcopy
import importlib.util
import inspect
from pathlib import Path

import cv2
import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("experiment_m1_spatial", Path(__file__).resolve().parents[2] / "tools/experiment_m1_spatial.py")
_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tool)


def _options():
    return {"name": "fixture", "center": [0.5, 0.5], "roi_radius": [0.45, 0.45],
            "seed_radius": [0.12, 0.12], "working_size": 64, "iterations": 2, "rng_seed": 0}


def test_spatial_prediction_does_not_accept_gt_and_preserves_input():
    assert list(inspect.signature(_tool.spatial_prediction).parameters) == ["image", "cfg", "options"]
    image = np.full((64, 64, 3), (255, 0, 0), np.uint8)
    image[16:48, 16:48] = (80, 120, 180)
    image.setflags(write=False)
    original = image.copy()
    cfg, options = {"spatial": {"enabled": True, "profiles": []}}, _options()
    saved_cfg, saved_options = deepcopy(cfg), deepcopy(options)
    cv2.setRNGSeed(options["rng_seed"])
    first, status = _tool.spatial_prediction(image, cfg, options)
    cv2.setRNGSeed(options["rng_seed"])
    second, second_status = _tool.spatial_prediction(image, cfg, options)
    assert status == second_status == "ok"
    assert first.shape == image.shape[:2] and first.dtype == np.uint8
    assert set(np.unique(first)) == {0, 255}
    assert np.all(first[24:40, 24:40] == 255)
    assert not np.any(first[:12])
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(first, second)
    assert cfg == saved_cfg and options == saved_options


def test_no_foreground_seeds_is_reported_and_scored_as_zero_prediction():
    image = np.full((24, 32, 3), (255, 0, 0), np.uint8)
    result, status = _tool.spatial_prediction(image, {}, _options())
    assert not np.any(result) and status == "insufficient_seeds_zero_prediction"


@pytest.mark.parametrize("patch", [{"name": "../escape"}, {"iterations": 0},
                                   {"seed_radius": [0.6, 0.2]}, {"center": [float("nan"), 0.5]}])
def test_invalid_candidate_configuration_is_rejected(patch):
    with pytest.raises(ValueError):
        _tool._candidates([{**_options(), **patch}])


def test_duplicate_candidates_are_rejected():
    with pytest.raises(ValueError, match="不重复"):
        _tool._candidates([_options(), _options()])
