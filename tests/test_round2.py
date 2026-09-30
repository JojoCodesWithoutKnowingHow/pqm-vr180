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


def green_is_her(rgb, threshold=0.5):
    return (rgb[..., 1] > 200) & (rgb[..., 0] < 40) & (rgb[..., 2] < 40)


def test_extend_side_keeps_the_source_and_lands_it_where_it_was():
    src = checker(416, 608)
    fake = ColourForge()
    # Her legs run off the bottom; the fake paints her fan green and green counts as
    # her, so the body keeps reaching the new edge: two steps of 0.3 to the limit.
    g = grow.extend_side(src, ["bottom"], 0.6, 60.0, fake, ("1girl",), ["indoors"], "indoors",
                         seed=1, refine_denoise=0.35, step=0.3, segment=her_segment(src))
    x0, y0, w, h = g.rect
    assert (x0, y0) == (0, 0) and g.log["added"] == {"bottom": int(0.6 * 608)}
    # Her legs cross the edge 83 px wide: the first step reaches 1.5x that, not a
    # full step; later steps follow the (fake, fan-shaped) body as it widens.
    assert g.log["passes"][0]["added"] == 124 and len(g.log["passes"]) == 3
    rim = g.log["rim"]
    assert np.array_equal(g.image[rim:h - rim, rim:w - rim], src[rim:h - rim, rim:w - rim])
    gen = [c for c in fake.calls if not c["touch_up"]]
    ref = [c for c in fake.calls if c["touch_up"]]
    # Each step: her fan with her tags, then the rest of the band as scenery.
    assert [c["her"] for c in gen] == [True, False] * 3
    assert all(c["control"] for c in gen)
    assert all("no humans" in c["prompt"] for c in gen if not c["her"])
    assert ref and all(c["denoise"] == 0.35 and not c["control"] for c in ref)
    # The grown canvas keeps the source's optical centre: on the sphere the source
    # sits exactly where it would alone.
    alone, m1, _f = sphere.place(src, 1024, 60.0)
    grown, _m2 = sphere.place_focal(g.image, 1024, g.focal, *g.centre)
    inner = sphere.place_focal(np.full((h - 2 * rim, w - 2 * rim), 255, np.uint8), 1024, g.focal,
                               g.centre[0] - rim, g.centre[1] - rim)[1] > 0
    assert np.median(np.abs(alone[inner].astype(int) - grown[inner].astype(int))) < 6


def test_growth_stops_once_the_body_ends():
    # Round 2's first try grew a full height at once and painted a second body in the
    # empty space. Here the body is finished within the first step: no more is grown.
    src = checker(416, 608)
    fake = ColourForge()
    g = grow.extend_side(src, ["bottom"], 1.0, 60.0, fake, ("1girl",), [], None, seed=1,
                         refine_denoise=0, step=0.3,
                         segment=lambda rgb, threshold=0.5: np.zeros(rgb.shape[:2], bool))
    assert g.log["added"] == {"bottom": int(0.3 * 608)}
    assert [c["her"] for c in fake.calls] == [False]         # no body crossing: scenery only
    assert g.log["passes"][0]["body_reaches_edge"] is False


def test_a_new_figure_in_the_band_is_not_the_body():
    # Round 2b: a figure painted in the new band, touching the edge but not joined
    # to her, kept the canvas growing. The body is only what connects to her.
    seg = np.zeros((100, 100), bool)
    seg[:50, 40:60] = True                       # her, down to row 50
    seg[70:100, 10:25] = True                    # someone else, at the bottom edge
    prev = np.zeros((100, 100), bool)
    prev[:40, 40:60] = True
    body = grow.track_body(seg, prev)
    assert body[:50, 40:60].all() and not body[70:, 10:25].any()
    assert not grow.body_reaches(body, "bottom")


def test_nothing_cut_grows_nothing():
    assert grow.extend_side(checker(64, 64), [], 1.0, 60.0, ColourForge(), (), [], None, 1) is None


def test_scene_first_paints_the_body_only_in_its_fan():
    src = checker(416, 608)
    seg = np.zeros(src.shape[:2], bool)
    seg[304:, 166:250] = True                        # legs cut by the bottom edge

    def segment(rgb, threshold=0.5):
        if rgb.shape[:2] == src.shape[:2]:
            return seg
        return green_is_her(rgb)

    def scene_of(W, H, cx, cy):
        return np.full((H, W, 3), (200, 40, 40), np.uint8)     # the laid-out scene: red

    g = grow.extend_side(src, ["bottom"], 0.3, 60.0, ColourForge(), ("1girl",), ["room"], None,
                         seed=1, refine_denoise=0, step=0.3, segment=segment, scene_of=scene_of)
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


def test_compose_only_details_the_layout():
    # G: once the layout is made around the finished picture the scene is final;
    # every view is a detail pass over it -- no subject view, no inpaint ControlNet.
    from vr180 import layout, widen
    src = checker(416, 608)
    pano, mask, _f = sphere.place(src, 1024, 60.0)
    lay, _log = layout.make_layout(pano, mask > 0, ["indoors", "room"], "indoors", ColourForge(),
                                   seed=1, S=256)
    fake = ColourForge()
    opt = widen.Options(width=1024, view_px=256, seam_px=8, subject_tags=("1girl",), taper=False,
                        join="hard", layout_denoise=0.35, compose=True)
    r = widen.widen(src, ["indoors", "room"], fake, opt, None, say=lambda s: None,
                    segment=her_segment(src), layout=lay)
    assert fake.calls and all(c["touch_up"] and c["denoise"] == 0.35 for c in fake.calls)
    assert not any(c["her"] for c in fake.calls)
    assert {v["kind"] for v in r.log["views"]} == {"scene"}
    assert r.log["front_unfilled"] < 0.002


def test_hires_layout_refines_in_tiles_and_compose_zero_calls_nothing(tmp_path):
    # Round 4: the layout composed at S, scaled up and refined in tiles (the hires
    # fix); with --compose 0 the views make no calls -- the hires layout is the scene.
    from vr180 import layout, widen
    src = checker(416, 608)
    pano, mask, _f = sphere.place(src, 1024, 60.0)
    fake = ColourForge()
    lay, log = layout.make_layout(pano, mask > 0, ["indoors"], "indoors", fake, seed=1, S=256,
                                  hires=512, hires_denoise=0.4, work=tmp_path)
    first, rest = fake.calls[0], fake.calls[1:]
    assert first["shape"] == (256, 256) and not first["touch_up"]
    assert rest and all(c["touch_up"] and c["denoise"] == 0.4 and c["shape"] == (512, 512)
                        for c in rest)
    assert log["hires"] == 512 and log["hires_tiles"] == len(rest)
    fake2 = ColourForge()
    opt = widen.Options(width=1024, view_px=256, seam_px=8, taper=False, layout_denoise=0.0,
                        compose=True)
    r = widen.widen(src, ["indoors"], fake2, opt, None, say=lambda s: None, layout=lay)
    assert fake2.calls == [] and r.log["front_unfilled"] < 0.002


def test_no_dark_rim_at_merge_edges():
    # Round 4 (H0): interpolating across the black of empty pixels darkened the edge
    # of what was known, and the blend pasted it back as a faint dotted line at
    # every merge. A uniform scene and a uniform source must come out uniform.
    from vr180 import widen
    src = np.full((608, 416, 3), 150, np.uint8)
    lay = np.full((512, 1024, 3), 150, np.uint8)
    opt = widen.Options(width=1024, view_px=256, seam_px=8, taper=False, layout_denoise=0.0,
                        compose=True)
    r = widen.widen(src, ["indoors"], ColourForge(), opt, None, say=lambda s: None, layout=lay)
    front = sphere.off_axis_deg(1024) <= 95
    assert r.pano[front].min() >= 146


def test_extend_in_layout_paints_only_the_fan_and_keeps_the_source():
    # Round 6: the body is continued inside the laid-out fisheye, the fan the only
    # mask, then refined flat at full resolution.
    from vr180 import inlayout, layout
    src = checker(416, 608)
    seg = np.zeros(src.shape[:2], bool)
    seg[304:, 166:250] = True                        # legs cut by the bottom edge
    pano, mask, _f = sphere.place(src, 1024, 60.0)
    fake = ColourForge()
    lay, _log, fish = layout.make_layout(pano, mask > 0, ["room"], None, fake, seed=1, S=256,
                                         return_fisheye=True)
    before = fish.copy()
    fake2 = ColourForge()
    g, fish2, log = inlayout.extend(src, seg, ["bottom"], 60.0, 1024, fish, 100.0, fake2,
                                    ("1girl",), ["room"], None, seed=1, grow_frac=0.6,
                                    refine_denoise=0.5)
    first = fake2.calls[0]
    assert first["her"] and first["control"] and not first["touch_up"]
    assert first["masked"] < 0.5                      # the fan, not the whole crop
    assert all(c["touch_up"] and c["denoise"] == 0.5 for c in fake2.calls[1:])
    changed = np.abs(fish2.astype(int) - before.astype(int)).max(-1) > 0
    assert changed.any() and changed.mean() < 0.1     # only the fan's part of the fisheye
    x0, y0, w, h = g.rect
    assert np.array_equal(g.image[y0 + 8:y0 + h - 8, x0 + 8:x0 + w - 8], src[8:h - 8, 8:w - 8])
    assert log["order"] == "in-layout" and log["added"] == {"bottom": int(0.6 * 608)}


def test_cli_runs_extend_in_layout(tmp_path, monkeypatch):
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
                   "--pano-only", "--join", "hard", "--extend-in-layout", "--extend-side", "0.6",
                   "--layout", "fisheye", "--layout-px", "256", "--layout-hires", "512",
                   "--compose", "0", "--seam-repaint", "0.4", "--soften-rim", "0"])
    assert rc == 0
    work = tmp_path / "o_180_LR.work"
    log = json.loads((work / "log.json").read_text(encoding="utf-8"))
    assert log["extend_in_layout"]["order"] == "in-layout"
    assert (work / "layout_fisheye_with_body.png").exists()
    assert {v["kind"] for v in log["widen"]["views"]} <= {"scene"}
    assert "repaint_inner" in log["seam"]
