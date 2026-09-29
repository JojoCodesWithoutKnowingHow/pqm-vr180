import numpy as np
import pytest

from vr180 import depthmap, placement

X = np.linspace(0, 1, 101)


@pytest.mark.parametrize("level", [-3, -2, -1, 1, 2, 3, -2.5, 1.5])
def test_foreground_scale_keeps_the_ends_and_order(level):
    y = depthmap.foreground_scale(X.copy(), level)
    assert y[0] == pytest.approx(0, abs=1e-3) and y[-1] == pytest.approx(1, abs=1e-3)
    assert np.all(np.diff(y) >= -1e-9)


def test_negative_foreground_scale_pushes_the_background_back():
    # IW3's inv_mul_2 (the author's -2): a steep jump over the farthest tenth,
    # then a gentle rise, so the background separates from everything in front.
    y = depthmap.foreground_scale(X.copy(), -2)
    assert y[10] == pytest.approx(0.49, abs=0.01)
    slope = np.diff(y) / np.diff(X)
    assert slope[:5].mean() > 3 and 0.3 < slope[20:].mean() < 0.8
    assert np.all(y >= X - 1e-9)
    p2 = depthmap.foreground_scale(X.copy(), 2)             # the reverse
    assert np.all(p2 <= X + 1e-9) and p2[50] < 0.25
    assert np.allclose(depthmap.foreground_scale(X.copy(), 0), X)


def test_fractional_level_sits_between_its_neighbours():
    a, b = depthmap.foreground_scale(X.copy(), -2), depthmap.foreground_scale(X.copy(), -3)
    m = depthmap.foreground_scale(X.copy(), -2.5)
    assert np.all(m >= np.minimum(a, b) - 1e-6) and np.all(m <= np.maximum(a, b) + 1e-6)


def _scene():
    """A character (0.30-0.60) in the middle of a wall (0.20), with a floor far
    nearer (up to 1.0) filling the bottom third, like a widened pano's nadir."""
    d = np.full((90, 180), 0.2, np.float32)
    d[30:60, 70:110] = np.linspace(0.3, 0.6, 40)[None, :]
    d[60:] = np.linspace(0.5, 1.0, 30)[:, None]
    region = np.zeros(d.shape, bool)
    region[25:62, 60:120] = True
    return d, region


def test_the_floor_takes_the_range_globally_but_not_by_region():
    d, region = _scene()
    g = depthmap.normalise(d.copy(), None)
    r = depthmap.normalise(d.copy(), region)
    char_g = np.ptp(g[30:60, 70:110])
    char_r = np.ptp(r[30:60, 70:110])
    assert char_r > 1.5 * char_g                   # the character gets depth back
    assert r[-1].min() > 0.85 and r.max() <= 1.0   # the nearer floor sits in the knee
    assert np.all(np.diff(r[60:, 0]) >= -1e-6)     # and keeps its order


def test_region_from_source_grows_by_degrees():
    m = np.zeros((180, 360), np.uint8)
    m[80:100, 170:190] = 255
    r = depthmap.region_from_source(m, 10)
    rows, cols = np.flatnonzero(r.any(1)), np.flatnonzero(r.any(0))
    assert (rows[0], rows[-1], cols[0], cols[-1]) == (70, 109, 160, 199)


@pytest.mark.parametrize("w,h,deg", [(1536, 640, 75), (1344, 768, 70), (1024, 1024, 60),
                                     (832, 1216, 50), (768, 1344, 45), (900, 1100, 55),
                                     (1216, 832, 70), (1152, 896, 65)])
def test_ratio_presets(w, h, deg):
    assert placement.by_ratio(w, h)[0] == deg


def test_shot_sizes_a_whole_figure_and_falls_back_when_cut():
    seg = np.zeros((1216, 832), bool)
    seg[108:1108, 300:500] = True                      # a whole figure, 82% of the height
    deg, why = placement.by_shot(832, 1216, seg, [])
    assert 50 < deg < 60 and "whole figure" in why
    small = np.zeros_like(seg)
    small[500:804, 380:460] = True                     # 25%: a long shot, so the frame is wide
    assert placement.by_shot(832, 1216, small, [])[0] == placement.LIMITS[1]
    deg, why = placement.by_shot(832, 1216, seg, ["bottom"])
    assert deg == placement.by_ratio(832, 1216)[0] and "cut at bottom" in why
    assert "no subject" in placement.by_shot(832, 1216, None, [])[1]


def test_camera_placement_uses_the_estimated_focal_length_and_falls_back():
    # A 1024-wide image seen through a focal length of 512 px spans 90 degrees.
    deg, why = placement.by_camera(1024, 768, {"focal_px": 512.0, "subject_m": 1.8})
    assert deg == pytest.approx(90.0, abs=0.01) and "1.8 m" in why
    assert placement.by_camera(1024, 768, {"focal_px": 5000.0})[0] == placement.LIMITS[0]
    deg, why = placement.by_camera(1024, 768, {"error": "Traceback...\nOSError: no model"})
    assert deg == 60.0 and "OSError: no model" in why
    assert placement.by_camera(1024, 768, None)[0] == 60.0


def test_detail_match_and_rim_bring_the_seam_ratio_toward_one():
    from vr180 import post
    rng = np.random.default_rng(0)
    base = rng.integers(60, 200, (400, 400, 3)).astype(np.uint8)
    import cv2
    soft = cv2.GaussianBlur(base, (0, 0), 1.5)
    source = np.zeros((400, 400), bool)
    source[100:300, 100:300] = True
    pano = np.where(source[..., None], base, soft)          # crisp source, soft fill
    before = post.detail_ratio(pano, source)
    sharpened, amount = post.detail_match(pano, source)
    assert before > 1.5 and amount > 0
    assert abs(post.detail_ratio(sharpened, source) - 1) < abs(before - 1)
    assert np.array_equal(sharpened[source], pano[source])  # the source untouched
    rimmed = post.soften_rim(pano, source, 12)
    assert np.array_equal(rimmed[~source], pano[~source])   # the rim touches only the source
    assert np.array_equal(rimmed[130:270, 130:270], pano[130:270, 130:270])


def test_moge_placement_takes_the_long_side_and_falls_back():
    assert placement.by_moge(1344, 768, {"hfov": 58.4, "vfov": 35.4})[0] == pytest.approx(58.4)
    assert placement.by_moge(832, 1216, {"hfov": 55.8, "vfov": 75.4})[0] == pytest.approx(75.4)
    assert placement.by_moge(1024, 1024, {"hfov": 120.0, "vfov": 120.0})[0] == placement.LIMITS[1]
    assert placement.by_moge(1024, 1024, {"error": "x\nOSError: no model"}) == (60.0, "moge estimate failed (OSError: no model); 60")


def test_moge_is_the_default_placement_and_a_number_overrides_it():
    from vr180.cli import parse
    assert parse(["s.png", "-o", "o.jpg", "--checkpoint", "c"]).long_side == "moge"
    assert parse(["s.png", "-o", "o.jpg", "--checkpoint", "c", "--long-side", "70"]).long_side == "70"
