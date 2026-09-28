import json
import math

import cv2
import numpy as np
import pytest

from vr180 import forge, plan, prompts, sphere, widen


def checker(w, h, cell=32):
    y, x = np.mgrid[0:h, 0:w]
    v = (((x // cell) + (y // cell)) % 2 * 200 + 30).astype(np.uint8)
    return np.dstack([v, np.full_like(v, 90), 255 - v])


def test_place_spans_the_expected_angles():
    pano, mask, (hfov, vfov) = sphere.place(checker(1216, 832), 2048, 90.0)
    assert hfov == 90.0 and vfov == pytest.approx(2 * math.degrees(math.atan(832 / 1216)), 1e-6)
    lon = (np.nonzero(mask.any(0))[0][[0, -1]] + 0.5) / 2048 * 360 - 180
    assert lon[0] == pytest.approx(-45, abs=0.5) and lon[1] == pytest.approx(45, abs=0.5)


def test_view_and_back_project_round_trip():
    pano, _m, _f = sphere.place(checker(1024, 1024), 2048, 90.0)
    v = sphere.view_of(pano, 0, 0, 60, 512)
    img, cover = sphere.back_project(v, 0, 0, 60, 2048)
    diff = np.abs(img[cover].astype(int) - pano[cover].astype(int))
    assert np.median(diff) < 8


def test_views_are_level_and_face_where_asked():
    d = sphere.view_dirs(90, 0, 90, 64)
    centre = d[32, 32]
    assert centre[0] > 0.99                     # yaw 90 looks right (+x)
    up = sphere.view_dirs(0, 60, 90, 64)[32, 32]
    assert math.degrees(math.asin(up[1])) == pytest.approx(60, abs=2)
    row = sphere.view_dirs(0, 0, 90, 64)[32]    # the middle row stays on the horizon
    assert np.abs(row[:, 1]).max() < 0.03


def test_planner_covers_the_hemisphere_from_a_source():
    _p, mask, _f = sphere.place(checker(1216, 832), 1024, 90.0)
    known = mask > 0
    pl = plan.Planner(90.0, 100.0)
    seen = []
    for _ in range(60):
        v = pl.next_view(pl.small(known))
        if v is None:
            break
        seen.append(v)
        _img, cover = sphere.back_project(np.zeros((8, 8), np.uint8), v.yaw, v.pitch, 90.0, 1024)
        known |= cover
    assert pl.remaining(pl.small(known)) < 0.002
    assert 6 <= len(seen) <= 30
    assert all(v.new <= 0.45 for v in seen[:3])   # it starts by growing from the edge


class FakeForge:
    """Stands in for Forge: returns the seeded view, tinted inside the mask, so the
    test can tell fill from source."""

    def __init__(self):
        self.calls = []

    def __call__(self, image, mask, prompt, negative, seed, control=None, reference=None):
        self.calls.append({"prompt": prompt, "negative": negative, "masked": float((mask > 0).mean()),
                           "control_black": None if control is None
                           else bool((control[mask > 0] == 0).all()),
                           "reference": reference is not None})
        out = image.copy()
        out[mask > 0] = (out[mask > 0] * 0.5 + np.array([0, 60, 0])).astype(np.uint8)
        return out


def test_widen_fills_the_front_and_never_touches_the_source(tmp_path):
    src = checker(608, 416)
    fake = FakeForge()
    opt = widen.Options(width=1024, view_px=256, seam_px=8)
    r = widen.widen(src, ["indoors", "wooden floor", "couch"], fake, opt, tmp_path, say=lambda s: None)
    placed, mask, _f = sphere.place(src, 1024, opt.long_side)
    source = mask > 0
    assert np.array_equal(r.pano[source], placed[source])
    assert r.log["front_unfilled"] < 0.002
    assert r.log["setting"] == "indoors"
    assert len(fake.calls) == len(r.log["views"]) <= 30
    ups = [v for v in r.log["views"] if v["pitch"] >= 45]
    downs = [v for v in r.log["views"] if v["pitch"] <= -45]
    assert ups and all("ceiling" in v["prompt"] for v in ups)
    assert downs and all("floor" in v["prompt"].split(", ") for v in downs)
    assert (r.pano.reshape(-1, 3).max(1) > 0).mean() > 0.99    # the back is filled too
    json.dumps(r.log)


@pytest.mark.parametrize("tags,expect", [
    (["outdoors", "city"], "outdoors"),
    (["grey background", "simple background"], "plain"),
    (["room", "curtains"], "indoors"),
    (["sky", "bed"], None),
    (["sweat"], None),
])
def test_setting(tags, expect):
    assert prompts.setting(tags)[0] == expect


def test_plain_background_gets_no_scenery_words():
    p = prompts.view_prompt(["grey background", "simple background"], "plain", 80)
    assert "sky" not in p and "ceiling" not in p and "scenery" not in p


def test_forge_payload_uses_promax_inpaint(monkeypatch):
    f = forge.Forge("http://x")
    sent = {}

    def fake_call(method, path, payload=None, tries=3):
        if path == "/sdapi/v1/sd-models":
            return [{"model_name": "waiIllustriousSDXL_v140",
                     "title": "waiIllustriousSDXL_v140.safetensors [bdb59bac77]"}]
        if path == "/controlnet/model_list":
            return {"model_list": ["controlnet-union-sdxl-promax [abc]"]}
        sent.update(payload)
        return {"images": [forge.b64png(np.zeros((64, 64, 3), np.uint8))]}

    monkeypatch.setattr(f, "_call", fake_call)
    s = forge.Settings(checkpoint="waiIllustriousSDXL_v140", method="cn")
    assert f.resolve(s) == []
    out = f.inpaint(np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64), np.uint8), "p", "n", 1, s)
    unit = sent["alwayson_scripts"]["ControlNet"]["args"][0]
    assert unit["module"] == "inpaint_only+lama" and unit["type_filter"] == "Inpaint"
    assert unit["model"].startswith("controlnet-union-sdxl-promax") and "image" not in unit
    assert out.shape == (64, 64, 3)
    assert f.resolve(forge.Settings(checkpoint="waiIllustriousSDXL_v140.safetensors")) == []
    assert f.resolve(forge.Settings(checkpoint="waiIllustrious")) != []   # no substring
    s2 = forge.Settings(checkpoint="missing")
    monkeypatch.setattr(f, "_call", lambda m, p, payload=None, tries=3:
                        [] if "sd-models" in p else {"model_list": []})
    assert len(f.resolve(s2)) == 1                       # plain needs no ControlNet
    assert len(f.resolve(forge.Settings(checkpoint="missing", method="cn"))) == 2


@pytest.mark.parametrize("yaw,pitch", [(0, 0), (170, 10), (-175, -30), (60, 70), (0, -90),
                                       (-120, 50), (90, -60)])
def test_bounds_hold_everything_a_view_covers(yaw, pitch):
    W = 512
    view = checker(128, 128, 16)
    full_img, full_cover = sphere.back_project(view, yaw, pitch, 90, W)
    region = sphere.bounds(yaw, pitch, 90, W)
    img, cover = sphere.back_project(view, yaw, pitch, 90, W, region=region)
    inside = np.zeros_like(full_cover)
    inside[region[0], region[1]] = True
    assert not (full_cover & ~inside).any()
    assert np.array_equal(full_cover[region[0]][:, region[1]], cover)
    assert np.array_equal(full_img[region[0]][:, region[1]][cover], img[cover])


def test_views_facing_down_or_up_drop_the_other_half():
    tags = ["landscape", "mountain", "lake", "sky", "cloud", "sunset", "grass", "reflection"]
    down = prompts.view_prompt(tags, "outdoors", -50).split(", ")
    up = prompts.view_prompt(tags, "outdoors", 50).split(", ")
    level = prompts.view_prompt(tags, "outdoors", 0).split(", ")
    assert "ground" in down and not {"sky", "cloud", "sunset", "mountain"} & set(down)
    assert "lake" in down and "grass" in down
    assert "sky" in up and not {"lake", "grass", "reflection"} & set(up)
    assert set(tags) <= set(level)
    assert "horizon" in prompts.view_negative("x", "outdoors", -50)
    assert prompts.view_negative("x", "outdoors", 0) == "x"
    assert prompts.view_negative("x", "plain", -80) == "x"



def test_plain_background_views_are_extended_without_diffusion(tmp_path):
    src = checker(416, 608)
    fake = FakeForge()
    opt = widen.Options(width=1024, view_px=256, seam_px=8)
    r = widen.widen(src, ["simple background", "grey background"], fake, opt, None,
                    say=lambda s: None)
    assert r.log["setting"] == "plain"
    assert fake.calls == []
    assert {v["kind"] for v in r.log["views"]} == {"plain"}
    assert r.log["front_unfilled"] < 0.002


def test_a_subject_cut_by_the_frame_is_continued(tmp_path):
    # The "subject" is the bottom half of the source, running off its bottom edge.
    src = checker(416, 608)

    def segment(rgb):
        m = np.zeros(rgb.shape[:2], bool)
        m[rgb.shape[0] // 2:, rgb.shape[1] // 4: 3 * rgb.shape[1] // 4] = True
        return m

    fake = FakeForge()
    opt = widen.Options(width=1024, view_px=256, seam_px=8, subject_tags=("1girl", "skirt"),
                        reference=True)
    r = widen.widen(src, ["simple background", "grey background"], fake, opt, None,
                    say=lambda s: None, segment=segment)
    assert r.log["subject"]["cut_at"] == ["bottom"]
    kinds = [v["kind"] for v in r.log["views"]]
    assert "subject" in kinds and "plain" in kinds
    subj = [c for c in fake.calls]
    assert subj and all(c["prompt"].split(", ")[5] == "1girl" for c in subj)
    assert all("no humans" not in c["prompt"] and "1girl" not in c["negative"] for c in subj)
    assert all(c["control_black"] and c["reference"] for c in subj)
    down = [v for v in r.log["views"] if v["kind"] == "subject"]
    assert min(v["pitch"] for v in down) < 0          # it went on below the frame


def test_black_fill_is_refused(monkeypatch):
    f = forge.Forge("http://x")
    monkeypatch.setattr(f, "_call", lambda m, p, payload=None, tries=3:
                        {"images": [forge.b64png(np.zeros((32, 32, 3), np.uint8))]})
    mask = np.zeros((32, 32), np.uint8)
    mask[8:24, 8:24] = 255
    with pytest.raises(forge.ForgeError, match="black"):
        f.inpaint(np.full((32, 32, 3), 128, np.uint8), mask, "p", "n", 1,
                  forge.Settings(checkpoint="c"))


def test_noob_sends_its_own_black_control_and_a_reference(monkeypatch):
    f = forge.Forge("http://x")
    sent = {}

    def fake_call(method, path, payload=None, tries=3):
        if path == "/sdapi/v1/sd-models":
            return [{"model_name": "c", "title": "c.safetensors [x]"}]
        if path == "/controlnet/model_list":
            return {"model_list": ["None", "noobaiInpainting_v10.fp16", "noobIPAMARK1_mark1"]}
        sent.update(payload)
        return {"images": [forge.b64png(np.full((32, 32, 3), 90, np.uint8))]}

    monkeypatch.setattr(f, "_call", fake_call)
    s = forge.Settings(checkpoint="c", method="noob", ipa_model="noobipa")
    assert f.resolve(s) == [] and s.cn_model.startswith("noobai") and s.ipa_model.startswith("noobIPA")
    img = np.full((32, 32, 3), 128, np.uint8)
    f.inpaint(img, np.full((32, 32), 255, np.uint8), "p", "n", 1, s, control=img, reference=img)
    units = sent["alwayson_scripts"]["ControlNet"]["args"]
    assert units[0]["module"] == "None" and "image" in units[0]
    assert units[1]["module"] == forge.IPA_MODULE and units[1]["weight"] == 0.5
