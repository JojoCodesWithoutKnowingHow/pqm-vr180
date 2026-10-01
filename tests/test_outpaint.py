"""The outpaint work, round 1: the hard join (L1/S1), the flat extension (L2) and the
fisheye layout (S2). Offline, with a fake Forge."""
import math

import numpy as np
import pytest

from vr180 import flatext, layout, prompts, sphere, widen


#: Written against the CLI's defaults before 0.5.0 (conftest.V1_DEFAULTS).
pytestmark = pytest.mark.usefixtures("v1_defaults")


def checker(w, h, cell=32):
    y, x = np.mgrid[0:h, 0:w]
    v = (((x // cell) + (y // cell)) % 2 * 200 + 30).astype(np.uint8)
    return np.dstack([v, np.full_like(v, 90), 255 - v])


class ColourForge:
    """Paints a subject pass pure green and a scene pass pure blue, and records
    each call's mask against the picture it was given."""

    def __init__(self):
        self.calls = []

    def __call__(self, image, mask, prompt, negative, seed, control=None, reference=None,
                 steps=None, denoise=None, touch_up=False, regions=None, mask_blur=None,
                 soft=False):
        her = "no humans" not in prompt
        green_in = (image[..., 1] > 200) & (image[..., 0] < 40) & (image[..., 2] < 40)
        self.calls.append({"her": her, "prompt": prompt, "negative": negative,
                           "mask_blur": mask_blur, "soft": soft, "denoise": denoise, "touch_up": touch_up,
                           "control": control is not None,
                           "repaints_body": bool((green_in & (mask > 0)).any()),
                           "masked": float((mask > 0).mean()), "shape": image.shape[:2],
                           "regions": regions})
        out = image.copy()
        if regions:
            # Regional prompts: each region painted by its own prompt.
            for rprompt, rmask in regions:
                sel = (mask > 0) & (rmask > 0)
                out[sel] = (0, 0, 255) if "no humans" in rprompt else (0, 255, 0)
            return out
        out[mask > 0] = (0, 255, 0) if her else (0, 0, 255)
        return out


def her_segment(src):
    """The source's lower middle is her; after that, anything green is."""
    def segment(rgb, threshold=0.5):
        if rgb.shape[:2] == src.shape[:2] and not (rgb[..., 1] == 255).any():
            m = np.zeros(rgb.shape[:2], bool)
            m[rgb.shape[0] // 2:, rgb.shape[1] * 2 // 5: rgb.shape[1] * 3 // 5] = True
            return m
        return (rgb[..., 1] > 200) & (rgb[..., 0] < 40) & (rgb[..., 2] < 40)
    return segment


def grey_seed(view, unknown):
    """The hole starts grey, so green going into a call is a body already painted
    (the real pre-fill smears the edge's colours into the hole)."""
    out = view.copy()
    out[unknown] = 128
    return out


def run(join, monkeypatch, **kw):
    monkeypatch.setattr(widen, "_seed", grey_seed)
    src = checker(416, 608)
    fake = ColourForge()
    opt = widen.Options(width=1024, view_px=256, seam_px=8, subject_tags=("1girl",),
                        taper=False, join=join, **kw)
    r = widen.widen(src, ["indoors", "room"], fake, opt, None, say=lambda s: None,
                    segment=her_segment(src))
    return r, fake


def test_hard_join_never_repaints_a_painted_body(monkeypatch):
    r, fake = run("hard", monkeypatch)
    assert sum(c["her"] for c in fake.calls) >= 2               # the body went on
    assert not any(c["repaints_body"] for c in fake.calls)
    assert r.log["join"] == "hard" and r.log["front_unfilled"] < 0.002


def test_the_old_blend_does_repaint_it(monkeypatch):
    # What the hard join fixes: 0.4.0's seam band ran over the body it had painted.
    _r, fake = run("blend", monkeypatch)
    assert any(c["repaints_body"] for c in fake.calls)


def test_fan_zone_widens_with_depth_and_keeps_the_root():
    S = 256
    known = np.zeros((S, S), bool)
    known[:128] = True
    subj = np.zeros((S, S), bool)
    subj[60:128, 110:146] = True
    straight = widen.continuation_zone(subj, ~known, ~known, S)
    fan = widen.continuation_zone(subj, ~known, ~known, S, spread_deg=30)
    assert fan is not None and not fan[:128].any()

    def width(z, row):
        cols = np.flatnonzero(z[row])
        return 0 if cols.size == 0 else cols[-1] - cols[0]

    deep = 128 + 60
    assert width(fan, deep) > width(straight, deep) + 30
    assert abs(width(fan, 130) - width(straight, 130)) <= 8
    assert (fan | ~straight).all()                   # the fan holds the old strip


def test_canvas_grows_only_the_cut_axes():
    assert flatext.canvas_for(400, 600, ["bottom"], 1.5) == (400, 900)
    assert flatext.canvas_for(400, 600, ["left", "bottom"], 1.5) == (600, 900)
    assert flatext.canvas_for(400, 600, [], 1.5) is None
    assert flatext.canvas_for(400, 600, ["bottom"], 1.0) is None


def test_ext_long_side_keeps_the_focal_length():
    got = flatext.ext_long_side(600, 600, 60.0, 600, 1200)
    assert got == pytest.approx(2 * math.degrees(math.atan(2 * math.tan(math.radians(30)))))


def test_flat_extend_is_one_call_and_the_source_lands_where_it_did():
    src = checker(416, 608)
    fake = ColourForge()
    ext = flatext.extend(src, ["bottom"], 1.5, fake, ("1girl", "skirt"), ["indoors"], "indoors",
                         seed=1)
    assert len(fake.calls) == 1 and fake.calls[0]["her"]
    gh, gw = fake.calls[0]["shape"]
    assert gh * gw <= 1280 * 1024 * 1.1 and gh % 64 == 0 and gw % 64 == 0
    x, y, w, h = ext.rect
    assert np.array_equal(ext.image[y:y + h, x:x + w], src)       # never regenerated
    assert ext.image.shape[:2] == (912, 416)
    # On the sphere, the source sits at the same angles as it would alone.
    long2 = flatext.ext_long_side(416, 608, 60.0, 416, 912)
    alone, m1, _f = sphere.place(src, 1024, 60.0)
    grown, _m2, _f = sphere.place(ext.image, 1024, long2)
    inner = (m1 > 0)
    assert np.median(np.abs(alone[inner].astype(int) - grown[inner].astype(int))) < 8


def test_fisheye_round_trip():
    pano, _m, _f = sphere.place(checker(1024, 1024), 2048, 90.0)
    fish = layout.to_fisheye(pano, 1024, 100.0)
    back, cover = layout.from_fisheye(fish, 2048, 100.0)
    near = cover & (sphere.off_axis_deg(2048) < 40)
    assert np.median(np.abs(back[near].astype(int) - pano[near].astype(int))) < 10
    assert not cover[sphere.off_axis_deg(2048) > 101].any()


def test_layout_is_one_call_and_views_refine_it():
    src = checker(416, 608)
    lay_fake = ColourForge()
    pano, mask, _f = sphere.place(src, 1024, 60.0)
    lay, log = layout.make_layout(pano, mask > 0, ["indoors", "room"], "indoors", lay_fake, seed=1,
                                  S=256)
    assert len(lay_fake.calls) == 1
    p = lay_fake.calls[0]["prompt"].split(", ")
    assert "fisheye" in p and "no humans" in p and "room" in p
    assert (lay.reshape(-1, 3).max(1) > 0).mean() > 0.99            # nothing left black
    fake = ColourForge()
    opt = widen.Options(width=1024, view_px=256, seam_px=8, subject_tags=("1girl",), taper=False,
                        join="hard", layout_denoise=0.6)
    r = widen.widen(src, ["indoors", "room"], fake, opt, None, say=lambda s: None,
                    segment=her_segment(src), layout=lay)
    assert r.log["layout"] is True
    scene = [c for c in fake.calls if not c["her"]]
    body = [c for c in fake.calls if c["her"]]
    assert scene and all(c["denoise"] == 0.6 for c in scene)
    assert body and all(c["denoise"] is None for c in body)        # a body starts fresh
    assert r.log["front_unfilled"] < 0.002


def test_layout_prompt_names_no_direction():
    p = layout.layout_prompt(["indoors", "bedroom"], "indoors", quality="")
    assert "ceiling" not in p and "from above" not in p and p.startswith("fisheye")
    assert "scenery" not in layout.layout_prompt(["simple background"], "plain", quality="")
    assert prompts.QUALITY in layout.layout_prompt([], None)


@pytest.mark.parametrize("flags", [
    [],
    ["--join", "hard"],
    ["--join", "hard", "--flat-extend", "1.5"],
    ["--join", "hard", "--flat-extend", "1.5", "--layout", "fisheye", "--layout-px", "256"],
])
def test_cli_runs_every_round_one_variant(tmp_path, monkeypatch, flags):
    import json

    from PIL import Image

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
    assert log["widen"]["join"] == ("hard" if "hard" in flags else "blend")
    mask = np.array(Image.open(work / "source_mask.png"))
    placed, m1, _f = sphere.place(src, 1024, 60.0)
    # The seam remedies measure at the original source, extended or not.
    assert abs(int((mask > 127).sum()) - int((m1 > 0).sum())) < 0.03 * (m1 > 0).sum()
    if "--flat-extend" in flags:
        assert log["flat_extend"]["canvas"] == [416, 912]
        assert (work / "flat_extended.png").exists()
    if "fisheye" in flags:
        assert log["layout"]["max_deg"] == 100.0 and (work / "layout_fisheye.png").exists()
