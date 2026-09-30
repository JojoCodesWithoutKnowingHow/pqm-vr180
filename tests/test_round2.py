"""Round 2 of the outpaint work: the one-sided grown canvas (and its off-centre
placement), the full-resolution pass, scene-first, and the seam repaint. Offline."""
import json
import math

import cv2
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


def test_forge_sends_regions_to_forge_couple(monkeypatch):
    from vr180 import forge
    f = forge.Forge("http://x")
    sent = {}

    def fake_call(method, path, payload=None, tries=3):
        sent.update(payload)
        return {"images": [forge.b64png(np.full((32, 32, 3), 90, np.uint8))]}

    monkeypatch.setattr(f, "_call", fake_call)
    her = np.zeros((32, 32), bool)
    her[16:] = True
    img = np.full((32, 32, 3), 128, np.uint8)
    f.inpaint(img, np.full((32, 32), 255, np.uint8), "ignored", "n", 1,
              forge.Settings(checkpoint="c"), regions=[("scene, no humans", ~her), ("1girl", her)])
    args = sent["alwayson_scripts"]["forge couple"]["args"]
    assert len(args) == 17 and args[:4] == [True, True, "Mask", "\n"] and args[5] == "None"
    assert len(args[7]) == 2 and all(set(m) == {"mask", "weight"} for m in args[7])
    assert sent["prompt"] == "scene, no humans\n1girl"


def test_regional_layout_paints_her_only_in_her_region_and_hires_leaves_it():
    # Round 7: the layout generated once with two regional prompts; the hires pass
    # (scene, no people) never touches her region.
    from vr180 import inlayout, layout
    src = checker(416, 608)
    seg = np.zeros(src.shape[:2], bool)
    seg[304:, 166:250] = True
    her = inlayout.her_on_fisheye(src, seg, ["bottom"], 60.0, 1024, 256, 100.0, grow_frac=0.6)
    assert her is not None and 0.005 < her.mean() < 0.4
    pano, mask, _f = sphere.place(src, 1024, 60.0)
    fake = ColourForge()
    _lay, _log, fish = layout.make_layout(pano, mask > 0, ["room"], None, fake, seed=1, S=256,
                                          hires=512, return_fisheye=True,
                                          region=("1girl, skirt", her))
    first = fake.calls[0]
    assert first["regions"] is not None and len(first["regions"]) == 2
    assert all(c["regions"] is None and not c["her"] for c in fake.calls[1:])  # the hires tiles
    # Where her region was painted (not her own source pixels), it holds her body.
    known = layout.to_fisheye((mask > 0).astype(np.uint8) * 255, 256, 100.0, cv2.INTER_NEAREST) > 127
    painted_her = cv2.resize((her & ~known).astype(np.uint8), (512, 512),
                             interpolation=cv2.INTER_NEAREST) > 0
    green = (fish[..., 1] > 200) & (fish[..., 0] < 40) & (fish[..., 2] < 40)
    inner = cv2.erode(painted_her.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    assert inner.any() and green[inner].mean() > 0.5   # her region kept its green body
    g, _fish2, log = inlayout.extend(src, seg, ["bottom"], 60.0, 1024, fish, 100.0,
                                     ColourForge(), ("1girl",), ["room"], None, seed=1,
                                     grow_frac=0.6, paint=False)
    assert log["order"] == "regional"


def test_body_reach_follows_the_crossing_not_the_source():
    # Round 7: a region sized as a fraction of the source was filled with her.
    from vr180 import inlayout
    seg = np.zeros((600, 400), bool)
    seg[300:, 150:230] = True                    # legs 80 px wide at the bottom edge
    seg[100:140, 390:] = True                    # a hand, 40 px, at the right edge
    reach = inlayout.body_reach(seg, {"bottom": 600, "right": 400, "top": 600})
    assert reach == {"bottom": 120, "right": 60}  # 1.5x each crossing; nothing at the top
    assert inlayout.body_reach(seg, {"bottom": 50}) == {"bottom": 50}   # never past the limit


@pytest.mark.parametrize("variant", ["J", "K"])
def test_cli_runs_round_nine(tmp_path, monkeypatch, variant):
    # J: the extension gives only her body, the layout owns the scenery; K: layout
    # first on the current pipeline (scene-first growth, hires layout, compose 0).
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
    base = [str(tmp_path / "src.png"), "-o", str(tmp_path / "o_180_LR.jpg"), "--checkpoint", "c",
            "--tags", "indoors, room", "--subject-tags", "1girl, skirt", "--long-side", "60",
            "--width", "1024", "--view-px", "256", "--segment-model", str(seg_model), "--pano-only",
            "--join", "hard", "--extend-side", "0.6", "--layout", "fisheye", "--layout-px", "256",
            "--layout-hires", "512", "--compose", "0", "--seam-repaint", "0.4", "--soften-rim", "0"]
    extra = ["--layout-owns-scenery"] if variant == "J" else ["--order", "scene-first"]
    assert cli.main(base + extra) == 0
    log = json.loads((tmp_path / "o_180_LR.work" / "log.json").read_text(encoding="utf-8"))
    if variant == "J":
        assert "layout_owns_scenery" in log and "repaint_silhouette" in log["seam"]
    else:
        assert log["extend_side"]["order"] == "scene-first"
    assert {v["kind"] for v in log["widen"]["views"]} <= {"scene"}


def test_extension_records_where_it_painted_her():
    src = checker(416, 608)
    g = grow.extend_side(src, ["bottom"], 0.6, 60.0, ColourForge(), ("1girl",), ["room"], None,
                         seed=1, refine_denoise=0, step=0.3, segment=her_segment(src))
    x0, y0, w, h = g.rect
    assert g.fans is not None and g.fans.shape == g.image.shape[:2]
    assert g.fans[y0 + h:].any() and not g.fans[y0:y0 + h, x0:x0 + w].any()


def test_the_source_fades_back_in_with_no_step_at_its_rim():
    # Round 12: a hard cut 8 px inside the source left a strip with an edge on both
    # sides -- the original's rectangle traced in lines. Across the rim the change
    # from the extension to the source must be gradual.
    src = np.full((608, 416, 3), 100, np.uint8)
    g = grow.extend_side(src, ["bottom"], 0.3, 60.0, ColourForge(), ("1girl",), ["room"], None,
                         seed=1, refine_denoise=0.35, step=0.3,
                         segment=lambda rgb, threshold=0.5: np.zeros(rgb.shape[:2], bool))
    x0, y0, w, h = g.rect
    col = g.image[y0 + h - 40:y0 + h + 10, x0 + w // 2].astype(int)   # down across the bottom rim
    assert np.abs(np.diff(col, axis=0)).max() <= 40                   # no single-pixel step
    assert (g.image[y0 + 16:y0 + h - 16, x0 + 16:x0 + w - 16] == 100).all()   # exact inside


def test_pose_words_drop_the_crop():
    from vr180 import prompts
    assert prompts.pose_words("lying, on side, upper body, reaching towards viewer, cowboy shot") == \
        ["lying", "on side", "reaching towards viewer"]


class InkForge:
    """Paints every mask a flat grey and, like the real model, inks a black line
    along the mask's edge."""

    def __call__(self, image, mask, prompt, negative, seed, **kw):
        out = image.copy()
        m = mask > 0
        out[m] = 150
        edge = m & ~(cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        out[edge] = 0
        return out


def test_no_ink_line_survives_a_repaint():
    # Round 14: the model inks a line along every inpaint mask's edge, and every
    # paste gave it about half weight -- one line per growth step, along the
    # original's edges, and at every repaint. A uniform scene must stay uniform.
    src = np.full((608, 416, 3), 150, np.uint8)
    for refine in (0.0, 0.35):       # the growth steps alone, then with the tiles
        g = grow.extend_side(src, ["bottom", "left"], 0.6, 60.0, InkForge(), ("1girl",), ["room"],
                             None, seed=1, refine_denoise=refine, step=0.3,
                             segment=lambda rgb, threshold=0.5: np.zeros(rgb.shape[:2], bool))
        assert g.image.min() >= 140, (refine, g.image.min())


def test_adetail_repaints_her_whole_figure_once_and_keeps_the_source():
    # Round 16 (the author): an ADetailer pass over her whole figure, source part
    # included, in one generation at low denoise; the source is restored after.
    from vr180 import inlayout
    H, W = 900, 700
    canvas = np.full((H, W, 3), 90, np.uint8)
    rect = (100, 100, 400, 500)                       # x, y, w, h of the source
    person = np.zeros((H, W), bool)
    person[300:800, 250:350] = True                   # her body, running out of the source
    fake = ColourForge()
    log = inlayout.adetail(canvas, rect, lambda rgb, threshold=0.5: person, fake, "1girl", "",
                           seed=1, denoise=0.2)
    assert len(fake.calls) == 1
    c = fake.calls[0]
    assert c["touch_up"] and c["denoise"] == 0.2 and not c["control"]
    changed = np.abs(canvas.astype(int) - 90).max(-1) > 0
    assert changed[400:700, 280:320].all()             # her figure, in and out of the source
    assert not changed[:150].any() and not changed[:, :150].any()   # nothing far from her
    assert log["figure_px"] == int(person.sum())


def test_adetail_skips_when_no_figure_joins_the_source():
    from vr180 import inlayout
    canvas = np.full((400, 400, 3), 90, np.uint8)
    fake = ColourForge()
    log = inlayout.adetail(canvas, (0, 0, 100, 100),
                           lambda rgb, threshold=0.5: np.zeros(rgb.shape[:2], bool),
                           fake, "1girl", "", seed=1, denoise=0.2)
    assert fake.calls == [] and "skipped" in log


def test_cli_runs_j_with_adetail(tmp_path, monkeypatch):
    # Round 17: the ADetailer pass on J's composed picture (not round 8's layout).
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
    rc = cli.main([str(tmp_path / "src.png"), "-o", str(tmp_path / "o_180_LR.jpg"), "--checkpoint",
                   "c", "--tags", "indoors, room", "--subject-tags", "1girl, skirt",
                   "--subject-framing", "sitting, cowboy shot", "--long-side", "60",
                   "--width", "1024", "--view-px", "256", "--segment-model", str(seg_model),
                   "--pano-only", "--join", "hard", "--extend-side", "0.6", "--layout", "fisheye",
                   "--layout-px", "256", "--layout-hires", "512", "--compose", "0",
                   "--seam-repaint", "0.4", "--soften-rim", "0", "--layout-owns-scenery",
                   "--adetail", "0.2"])
    assert rc == 0
    log = json.loads((tmp_path / "o_180_LR.work" / "log.json").read_text(encoding="utf-8"))
    assert log["adetail"]["denoise"] == 0.2 and "box" in log["adetail"]
    assert any(c["denoise"] == 0.2 and "sitting" in c["prompt"] and "cowboy shot" not in c["prompt"]
               for c in fake.calls)


def test_cli_layout_full_prompt_then_adetail_no_extension(tmp_path, monkeypatch):
    # Round 18 (the author's pipeline): no extension step; the layout gets the full
    # prompt and draws her; an ADetailer pass refines her figure.
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
    rc = cli.main([str(tmp_path / "src.png"), "-o", str(tmp_path / "o_180_LR.jpg"), "--checkpoint",
                   "c", "--tags", "indoors, room", "--subject-tags", "1girl, skirt",
                   "--subject-framing", "sitting, cowboy shot", "--long-side", "60",
                   "--width", "1024", "--view-px", "256", "--segment-model", str(seg_model),
                   "--pano-only", "--join", "hard", "--extend-in-layout", "--layout-full-prompt",
                   "--extend-side", "0.6", "--layout", "fisheye", "--layout-px", "256",
                   "--layout-hires", "512", "--layout-strong", "--compose", "0",
                   "--seam-repaint", "0.4", "--soften-rim", "0", "--adetail", "0.2"])
    assert rc == 0
    log = json.loads((tmp_path / "o_180_LR.work" / "log.json").read_text(encoding="utf-8"))
    lp = log["layout"]["prompt"]
    assert "1girl" in lp and "sitting" in lp and "no humans" not in lp and "fisheye" in lp
    assert "cowboy shot" not in lp
    ad = log["extend_in_layout"]["adetail"]           # the pass ran (the fake segmenter
    assert ad is not None                              # may not see the fake's figure)
    assert "extend_side" not in log                         # no extension step
    first = fake.calls[0]
    assert first["her"] and first["control"] and not first["touch_up"]   # the layout itself


def test_her_region_reach_widens_her_region():
    # Round 21: a wider reach (3 crossing widths) for the layout's her region.
    from vr180 import inlayout
    src = checker(416, 608)
    seg = np.zeros(src.shape[:2], bool)
    seg[304:, 166:250] = True
    narrow = inlayout.her_on_fisheye(src, seg, ["bottom"], 60.0, 1024, 256, 100.0, grow_frac=1.0)
    wide = inlayout.her_on_fisheye(src, seg, ["bottom"], 60.0, 1024, 256, 100.0, grow_frac=1.0,
                                   reach=3.0)
    assert (narrow & ~wide).sum() == 0 and wide.sum() > narrow.sum() * 1.2


def test_adetail_stays_inside_its_limit():
    # Round 18: a giant second Yamato the layout drew touched her, and the pass took
    # the whole canvas; outside ``limit`` nothing is repainted.
    from vr180 import inlayout
    H, W = 900, 700
    canvas = np.full((H, W, 3), 90, np.uint8)
    person = np.zeros((H, W), bool)
    person[300:800, 250:350] = True                   # her body, out of the source
    person[750:900, 0:700] = True                     # a giant joined to it
    limit = np.zeros((H, W), bool)
    limit[:700] = True
    fake = ColourForge()
    log = inlayout.adetail(canvas, (100, 100, 400, 500), lambda rgb, threshold=0.5: person, fake,
                           "1girl", "", seed=1, denoise=0.2, limit=limit)
    changed = np.abs(canvas.astype(int) - 90).max(-1) > 0
    assert not changed[790:].any()
    assert changed[400:650, 280:320].all()
    assert log["figure_px"] == int((person & limit).sum())


def test_cli_layout_her_region_uses_pose_words_and_scenery_beyond(tmp_path, monkeypatch):
    # Round 21: her tags and pose words only in her region, scenery beyond.
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
    rc = cli.main([str(tmp_path / "src.png"), "-o", str(tmp_path / "o_180_LR.jpg"), "--checkpoint",
                   "c", "--tags", "indoors, room", "--subject-tags", "1girl, skirt",
                   "--subject-framing", "sitting, cowboy shot", "--long-side", "60",
                   "--width", "1024", "--view-px", "256", "--segment-model", str(seg_model),
                   "--pano-only", "--join", "hard", "--extend-in-layout", "--layout-her-region",
                   "3", "--extend-side", "0.6", "--layout", "fisheye", "--layout-px", "256",
                   "--layout-hires", "512", "--layout-strong", "--compose", "0",
                   "--seam-repaint", "0.4", "--soften-rim", "0", "--adetail", "0.2"])
    assert rc == 0
    first = fake.calls[0]
    assert first["regions"] is not None
    her_p = [p for p, _m in first["regions"] if "1girl" in p]
    rest = [p for p, _m in first["regions"] if "1girl" not in p]
    assert len(her_p) == 1 and "sitting" in her_p[0] and "cowboy shot" not in her_p[0]
    assert rest and all("no humans" in p for p in rest)
    log = json.loads((tmp_path / "o_180_LR.work" / "log.json").read_text(encoding="utf-8"))
    assert log["extend_in_layout"]["adetail"] is not None


def test_layout_close_negative_reaches_the_full_prompt_layout(tmp_path, monkeypatch):
    # Round 21: the close-up negative goes to the layout and its hires tiles.
    from vr180 import layout, prompts
    src = checker(416, 608)
    pano, mask, _f = sphere.place(src, 1024, 60.0)
    fake = ColourForge()
    neg = prompts.CLOSE_NEGATIVE + ", " + prompts.SUBJECT_NEGATIVE
    layout.make_layout(pano, mask > 0, ["room"], None, fake, seed=1, S=256, hires=512,
                       full_prompt="1girl, sitting", full_negative=neg)
    assert all(c["negative"] == neg for c in fake.calls)
    assert "giantess" in fake.calls[0]["negative"]
