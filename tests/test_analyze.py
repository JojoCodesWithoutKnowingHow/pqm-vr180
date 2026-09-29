"""V.2: the report PQM's VR extension decides from, the tagger behind it, and the
Forge restart the companion does for itself."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vr180 import analyze, placement, tagger

HERE = Path(__file__).resolve().parent.parent


class FakeInput:
    name = "input"
    shape = ["batch", 448, 448, 3]


class FakeSession:
    """The tagger's ONNX session: records what it was fed, answers set scores."""

    def __init__(self, scores):
        self.scores = np.array(scores, np.float32)
        self.fed = None

    def get_inputs(self):
        return [FakeInput()]

    def run(self, _outputs, feed):
        self.fed = feed["input"]
        return [self.scores[None]]


def tags_csv(tmp_path):
    rows = ["tag_id,name,category,count",
            "1,general,9,10", "2,explicit,9,10",
            "3,indoors,0,10", "4,wooden_floor,0,10", "5,^_^,0,10",
            "6,1girl,0,10", "7,nami_(one_piece),4,10", "8,couch,0,10"]
    (tmp_path / "selected_tags.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return tmp_path


def test_tags_are_read_in_order_with_spaces_and_kaomoji_kept(tmp_path):
    names = tagger.read_tags(tags_csv(tmp_path) / "selected_tags.csv")
    assert names == [("general", "rating"), ("explicit", "rating"), ("indoors", "general"),
                     ("wooden floor", "general"), ("^_^", "general"), ("1girl", "general"),
                     ("nami (one piece)", "character"), ("couch", "general")]


def test_the_input_is_a_white_padded_bgr_square():
    rgb = np.zeros((20, 40, 3), np.uint8)
    rgb[..., 0] = 200                            # red in RGB
    x = tagger.prepare(rgb, 40)
    assert x.shape == (1, 40, 40, 3) and x.dtype == np.float32
    assert tuple(x[0, 0, 0]) == (255, 255, 255)  # padding is white
    assert tuple(x[0, 20, 20]) == (0, 0, 200)    # BGR, 0-255


def test_thresholds_per_kind_and_one_rating(tmp_path):
    session = FakeSession([0.9, 0.2, 0.97, 0.36, 0.1, 0.99, 0.84, 0.34])
    t = tagger.Tagger(tags_csv(tmp_path), session=session)
    found = t(np.zeros((64, 48, 3), np.uint8))
    assert session.fed.shape == (1, 448, 448, 3)
    assert [(x.name, x.kind) for x in found] == [
        ("1girl", "general"), ("indoors", "general"), ("wooden floor", "general"),
        ("general", "rating")]   # the character at 0.84 is under 0.85; couch under 0.35


def test_the_report_carries_tags_subject_and_raw_placement(tmp_path):
    src = tmp_path / "src.png"
    Image.new("RGB", (120, 80), (40, 40, 40)).save(src)
    seg = np.zeros((80, 120), bool)
    seg[30:, 50:70] = True                       # runs off the bottom edge
    fake_tags = [tagger.Tag("indoors", 0.9, "general"), tagger.Tag("1girl", 0.99, "general")]
    report = analyze.analyze(src, tagger=lambda rgb: fake_tags, segment=lambda rgb: seg,
                             estimate=lambda p: {"hfov": 118.0, "vfov": 90.0})
    assert report["width"] == 120 and report["height"] == 80
    assert report["tags"][0] == {"name": "indoors", "score": 0.9, "kind": "general"}
    assert report["subject"]["found"] and report["subject"]["cut"] == ["bottom"]
    p = report["placement"]
    assert p["estimated"] and p["raw_deg"] == 118.0
    assert p["long_side"] == placement.LIMITS[1] and "clamped" in p["why"]
    json.dumps(report)


def test_a_part_that_fails_is_reported_and_the_rest_still_report(tmp_path):
    src = tmp_path / "src.png"
    Image.new("RGB", (64, 64)).save(src)

    def broken(_rgb):
        raise RuntimeError("no model")

    report = analyze.analyze(src, tagger=broken, segment=None,
                             estimate=lambda p: {"error": "MoGe crashed\nCUDA out of memory"})
    assert report["tags"] == [] and "no model" in report["tags_error"]
    assert report["subject"] == {"found": False, "fraction": 0.0, "cut": []}
    assert report["placement"]["estimated"] is False
    assert "CUDA out of memory" in report["placement"]["error"]
    skipped = analyze.analyze(src, estimate=None)
    assert skipped["placement"] == {"estimated": False, "skipped": True}


def _restart_module():
    spec = importlib.util.spec_from_file_location("restart_forge", HERE / "setup" / "restart_forge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_restart_finds_forge_by_its_argv_and_never_itself():
    mod = _restart_module()
    assert mod.is_webui(["bash", "/workspace/forge/sd-webui-forge-neo/webui.sh", "--port", "7861"])
    assert mod.is_launch(["/workspace/forge/venv/bin/python3.11", "launch.py", "--listen"])
    assert not mod.is_webui(["python3", "restart_forge.py", "webui.sh"])
    assert not mod.is_launch(["python3", "restart_forge.py", "launch.py"])
    assert not mod.is_launch(["grep", "launch.py"])


def test_pod_setup_pins_the_tagger_and_restarts_before_verifying():
    text = (HERE / "setup" / "pod_setup.sh").read_bytes().decode()
    assert "\r" not in text, "a CRLF script dies on the pod (V.1)"
    assert "wd-eva02-large-tagger-v3/resolve/b25b82a03f7282e41aa2f257a52c7583b710bd1c" in text
    assert "all) companion; stereo360; models; moge; restart; verify ;;" in text
    assert 'VR180_DIR=${VR180_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}' in text
