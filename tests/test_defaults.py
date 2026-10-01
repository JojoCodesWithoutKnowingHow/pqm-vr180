"""0.5.0: a run with no flags is the outpaint rework's baseline, and LoRA calls go
only on the passes that draw her."""
import pytest

from vr180 import cli, prompts
from vr180.forge import Forge

#: The baseline as the author approved it (rounds 47-50), as PQM's V.2.5 lists it.
BASELINE = ("--checkpoint Waifu-Inpaint-XL --method plain --denoise 1.0 --join hard "
            "--auto-pipeline --extend-side 1.0 --extend-refine 0.35 --layout fisheye "
            "--layout-strong --layout-hires 2048 --layout-hires-denoise 0.4 --compose 0.3 "
            "--seam-repaint 0.4 --soften-rim 0 --keep-threshold 0.15 --source-fade 32 "
            "--adetail 0.27 --layout-mask-grow 0 --layout-mask-blur 4 "
            "--touch-mask-weight-j 0 --no-detail-match").split()


def test_no_flags_is_the_baseline():
    bare = vars(cli.parse(["src.png", "-o", "out_180_LR.jpg"]))
    spelled = vars(cli.parse(["src.png", "-o", "out_180_LR.jpg"] + BASELINE))
    assert bare == spelled


def test_the_losing_flags_stay_off():
    a = cli.parse(["src.png", "-o", "o_180_LR.jpg"])
    assert not (a.layout_soft or a.extend_regional or a.layout_full_prompt
                or a.layout_close_negative or a.extend_in_layout or a.layout_owns_scenery)
    assert (a.layout_fine, a.layout_joint, a.redraw, a.extend_guided, a.tone_match,
            a.trim_source, a.source_feather, a.layout_hires_overlap, a.flat_extend) == (0,) * 9
    assert a.touch_mask_weight is None and a.detail_match is False


def test_baseline_switches_can_be_turned_off():
    a = cli.parse(["s.png", "-o", "o_180_LR.jpg", "--no-auto-pipeline", "--no-layout-strong",
                   "--detail-match"])
    assert (a.auto_pipeline, a.layout_strong, a.detail_match) == (False, False, True)


def test_lora_calls_parse_and_refuse_prose():
    assert prompts.lora_calls("<lora:tifa_v2:0.8>, <lyco:style:1>") == (
        "<lora:tifa_v2:0.8>", "<lyco:style:1>")
    assert prompts.lora_calls("") == ()
    with pytest.raises(ValueError):
        prompts.lora_calls("<lora:a:1>, 1girl")


def test_loras_go_on_her_passes_only(monkeypatch):
    monkeypatch.setattr(prompts, "HER_LORAS", ("<lora:tifa:0.8>",))
    her = prompts.subject_prompt(["1girl", "tifa lockhart"], ["indoors", "bar"], "indoors", 0.0)
    room = prompts.view_prompt(["indoors", "bar"], "indoors", 0.0)
    assert her.endswith("<lora:tifa:0.8>")
    assert "<lora" not in room


def test_a_lora_forge_does_not_list_is_named(monkeypatch):
    f = Forge("http://forge")
    calls = []

    def fake(method, path, payload=None, tries=3):
        calls.append(path)
        if path.endswith("/loras"):
            return [{"name": "tifa_v2", "alias": "Tifa"}]
        return None

    monkeypatch.setattr(f, "_call", fake)
    assert f.missing_loras(("<lora:TIFA_v2:0.8>", "<lora:tifa:1>")) == []
    assert f.missing_loras(("<lora:nami:0.7>",)) == ["LoRA 'nami' is not in Forge's list"]
    assert f.missing_loras(()) == []


def test_a_default_run_puts_her_loras_on_her_passes_only(tmp_path, monkeypatch):
    # The baseline end to end on the fake Forge (sizes cut down so it runs offline):
    # every prompt with her tags carries the LoRA, no scenery prompt does.
    import json

    from PIL import Image

    from test_outpaint import ColourForge, checker, her_segment
    from vr180 import forge, subject

    src = checker(416, 608)
    Image.fromarray(src).save(tmp_path / "src.png")
    seg_model = tmp_path / "seg.onnx"
    seg_model.write_bytes(b"x")
    fake, seen = ColourForge(), []

    class FakeSeg:
        def __init__(self, path):
            self.fn = her_segment(src)

        def __call__(self, rgb, threshold=0.5):
            return self.fn(rgb, threshold)

    def inpaint(self, image, mask, prompt, negative, seed, s, **kw):
        seen.append(prompt)
        return fake(image, mask, prompt, negative, seed, **kw)

    monkeypatch.setattr(subject, "Segmenter", FakeSeg)
    monkeypatch.setattr(forge.Forge, "resolve", lambda self, s: [])
    monkeypatch.setattr(forge.Forge, "missing_loras", lambda self, calls: [])
    monkeypatch.setattr(forge.Forge, "inpaint", inpaint)
    rc = cli.main([str(tmp_path / "src.png"), "-o", str(tmp_path / "o_180_LR.jpg"),
                   "--tags", "indoors, room", "--subject-tags", "1girl, skirt", "--long-side", "60",
                   "--width", "1024", "--view-px", "256", "--layout-px", "256",
                   "--layout-hires", "512", "--segment-model", str(seg_model), "--pano-only",
                   "--loras", "<lora:her_v1:0.8>"])
    assert rc == 0
    log = json.loads((tmp_path / "o_180_LR.work" / "log.json").read_text(encoding="utf-8"))
    assert log["loras"] == ["<lora:her_v1:0.8>"] and "auto_pipeline" in log
    hers = [p for p in seen if "1girl" in p]
    scenery = [p for p in seen if "no humans" in p]
    assert hers and scenery
    assert all("<lora:her_v1:0.8>" in p for p in hers)
    assert not any("<lora" in p for p in scenery)


def test_a_lora_forge_lacks_stops_the_run(tmp_path, monkeypatch, capsys):
    from PIL import Image

    from test_outpaint import checker
    from vr180 import forge

    Image.fromarray(checker(64, 64)).save(tmp_path / "src.png")
    monkeypatch.setattr(forge.Forge, "resolve", lambda self, s: [])
    monkeypatch.setattr(forge.Forge, "missing_loras",
                        lambda self, calls: ["LoRA 'nami' is not in Forge's list"])
    rc = cli.main([str(tmp_path / "src.png"), "-o", str(tmp_path / "o_180_LR.jpg"),
                   "--long-side", "60", "--loras", "<lora:nami:0.7>"])
    assert rc == 2 and "LoRA 'nami' is not in Forge's list" in capsys.readouterr().err
