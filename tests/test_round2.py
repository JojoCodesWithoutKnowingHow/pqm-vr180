"""Round 2 of the outpaint work: the one-sided grown canvas (and its off-centre
placement), the full-resolution pass, scene-first, and the seam repaint. Offline."""
import json
import math

import numpy as np
import pytest
from PIL import Image

from test_outpaint import ColourForge, checker, her_segment
from vr180 import grow, seam, sphere


def test_place_focal_centred_is_place():
    src = checker(416, 608)
    a, ma, _f = sphere.place(src, 1024, 60.0)
    b, mb = sphere.place_focal(src, 1024, sphere.focal_px(416, 608, 60.0), 208, 304)
    assert np.array_equal(ma, mb) and np.array_equal(a, b)


def test_flat_of_undoes_place_focal():
    src = checker(400, 500, cell=40)
    f = sphere.focal_px(400, 500, 60.0)
    pano, _m = sphere.place_focal(src, 4096, f, 200, 150)      # off-centre on purpose
    back = sphere.flat_of(pano, 400, 500, f, 200, 150)
    inner = (slice(20, 480), slice(20, 380))
    assert np.median(np.abs(back[inner].astype(int) - src[inner].astype(int))) < 6


def test_growth_is_one_sided_and_capped_by_angle():
    f = grow.focal(832, 1216, 60.0)
    assert grow.side_growth(832, 1216, ["bottom"], 1.0, f) == {"bottom": 1216}   # 60 deg: room
    add = grow.side_growth(832, 1216, ["bottom"], 2.0, f)
    assert list(add) == ["bottom"]
    # Two source heights would put the edge past 70 degrees: capped there.
    edge = math.degrees(math.atan((1216 / 2 + add["bottom"]) / f))
    assert add["bottom"] < 2432 and edge == pytest.approx(70, abs=0.2)
    small = grow.side_growth(832, 1216, ["bottom", "left"], 0.3, f)
    assert small == {"bottom": int(0.3 * 1216), "left": int(0.3 * 832)}


def test_extend_side_keeps_the_source_and_lands_it_where_it_was():
    src = checker(416, 608)
    fake = ColourForge()
    g = grow.extend_side(src, ["bottom"], 0.6, 60.0, fake, ("1girl",), ["indoors"], "indoors",
                         seed=1, refine_denoise=0.35)
    x0, y0, w, h = g.rect
    assert (x0, y0) == (0, 0) and g.image.shape[:2] == (608 + int(0.6 * 608), 416)
    rim = g.log["rim"]
    assert np.array_equal(g.image[rim:h - rim, rim:w - rim], src[rim:h - rim, rim:w - rim])
    gen = [c for c in fake.calls if not c["touch_up"]]
    ref = [c for c in fake.calls if c["touch_up"]]
    assert len(gen) == 1 and gen[0]["control"] and gen[0]["her"]
    assert ref and all(c["denoise"] == 0.35 and not c["control"] for c in ref)
    # The grown canvas keeps the source's optical centre: on the sphere the source
    # sits exactly where it would alone.
    alone, m1, _f = sphere.place(src, 1024, 60.0)
    grown, _m2 = sphere.place_focal(g.image, 1024, g.focal, *g.centre)
    inner = sphere.place_focal(np.full((h - 2 * rim, w - 2 * rim), 255, np.uint8), 1024, g.focal,
                               g.centre[0] - rim, g.centre[1] - rim)[1] > 0
    assert np.median(np.abs(alone[inner].astype(int) - grown[inner].astype(int))) < 6


def test_nothing_cut_grows_nothing():
    assert grow.extend_side(checker(64, 64), [], 1.0, 60.0, ColourForge(), (), [], None, 1) is None


def test_scene_first_paints_the_body_only_in_its_fan():
    src = checker(416, 608)
    seg = np.zeros(src.shape[:2], bool)
    seg[304:, 166:250] = True                        # legs cut by the bottom edge
    f = grow.focal(416, 608, 60.0)
    add = grow.side_growth(416, 608, ["bottom"], 0.6, f)
    W, H, x0, y0 = grow.canvas_geometry(416, 608, add)
    scene = np.full((H, W, 3), (200, 40, 40), np.uint8)     # the laid-out scene: red
    g = grow.extend_side(src, ["bottom"], 0.6, 60.0, ColourForge(), ("1girl",), ["room"], None,
                         seed=1, refine_denoise=0, scene=scene, seg=seg)
    below = g.image[608:]
    green = (below[..., 1] > 200) & (below[..., 0] < 40)
    red = (below[..., 0] > 150) & (below[..., 1] < 80)
    cols = np.flatnonzero(green.any(0))
    assert green.any() and red.mean() > 0.4                  # the body, over the scene
    assert cols[0] > 60 and cols[-1] < 356                   # a fan from her legs, not the width
    assert g.log["order"] == "scene-first"


def test_seam_repaint_touches_only_the_band():
    src = checker(416, 608)
    pano, mask, _f = sphere.place(src, 2048, 60.0)
    pano[mask == 0] = (40, 40, 200)
    before = pano.copy()
    fake = ColourForge()
    log = seam.repaint(pano, mask > 0, fake, "1girl", "", seed=1, denoise=0.4, S=256,
                       inner=4, outer=12)
    assert 3 <= log["views"] <= 16
    assert all(c["touch_up"] and c["denoise"] == 0.4 and not c["control"] for c in fake.calls)
    changed = np.abs(pano.astype(int) - before.astype(int)).max(-1) > 0
    deep_in = mask > 0
    import cv2
    deep_in = cv2.erode(deep_in.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    far_out = cv2.dilate((mask > 0).astype(np.uint8), np.ones((61, 61), np.uint8)) == 0
    assert changed.any() and not changed[deep_in].any() and not changed[far_out & (sphere.off_axis_deg(2048) < 80)].any()


@pytest.mark.parametrize("flags", [
    ["--join", "hard", "--extend-side", "0.6", "--layout", "fisheye", "--layout-px", "256",
     "--layout-strong", "--seam-repaint", "0.4", "--soften-rim", "0"],
    ["--join", "hard", "--extend-side", "0.6", "--layout", "fisheye", "--layout-px", "256",
     "--layout-strong", "--seam-repaint", "0.4", "--soften-rim", "0", "--order", "scene-first"],
])
def test_cli_runs_round_two(tmp_path, monkeypatch, flags):
    from vr180 import cli, forge, subject

    src = checker(416, 608)
    Image.fromarray(src).save(tmp_path / "src.png")
    seg_model = tmp_path / "seg.onnx"
    seg_model.write_bytes(b"x")
    fake = ColourForge()

    class FakeSeg:
        def __init__(self, path):
            self.fn = her_segment(src)

        def __call__(self, rgb, threshold=0.5):
            return self.fn(rgb, threshold)

    monkeypatch.setattr(subject, "Segmenter", FakeSeg)
    monkeypatch.setattr(forge.Forge, "resolve", lambda self, s: [])
    monkeypatch.setattr(forge.Forge, "inpaint",
                        lambda self, image, mask, prompt, negative, seed, s, **kw:
                        fake(image, mask, prompt, negative, seed, **kw))
    out = tmp_path / "o_180_LR.jpg"
    rc = cli.main([str(tmp_path / "src.png"), "-o", str(out), "--checkpoint", "c",
                   "--tags", "indoors, room", "--subject-tags", "1girl, skirt", "--long-side", "60",
                   "--width", "1024", "--view-px", "256", "--segment-model", str(seg_model),
                   "--pano-only", *flags])
    assert rc == 0
    work = tmp_path / "o_180_LR.work"
    log = json.loads((work / "log.json").read_text(encoding="utf-8"))
    assert log["extend_side"]["added"] == {"bottom": int(0.6 * 608)}
    assert log["extend_side"]["order"] == ("scene-first" if "scene-first" in flags else "body-first")
    assert log["seam"]["repaint"]["views"] >= 3 and "rim_px" not in log["seam"]
    assert "(fisheye:1.3)" in log["layout"]["prompt"]
    # The seam remedies measure at the original source.
    mask = np.array(Image.open(work / "source_mask.png")) > 127
    _p, m1, _f = sphere.place(src, 1024, 60.0)
    assert abs(int(mask.sum()) - int((m1 > 0).sum())) < 0.03 * (m1 > 0).sum()
