"""What the companion can see in one image before widening it, as JSON (V.2).

    python -m vr180.analyze SRC.png -o analysis.json

PQM's VR extension runs this first, on the pod, and decides from it -- with its own
classifier -- what to fill the surroundings with, what the subject is, and where to
place the source; an image it is not sure of is set aside for the user instead of
guessed (``EXT-VR`` -> *Hands off*). So this reports and never decides:

- ``tags``: the WD tagger's tags with their scores (:mod:`vr180.tagger`)
- ``subject``: whether anime-seg finds a character, how much of the frame it takes,
  and which edges of the frame cut it (the edges a body would be continued past)
- ``placement``: MoGe-2's field of view for the image, the long side
  :func:`vr180.placement.by_moge` would place it at, and the **raw** estimate
  before clamping -- an estimate outside the clamp is a reason for doubt

Each part that fails says so under ``error`` and the rest still report: a missing
MoGe estimate is a decision for PQM, not a crash here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__, placement, stereo, widen
from .cli import SEG_MODEL
from .tagger import TAGGER_DIR


def parse(argv=None):
    p = argparse.ArgumentParser(prog="vr180.analyze", description=__doc__.split("\n\n")[0])
    p.add_argument("src")
    p.add_argument("-o", "--out", required=True, help="where the JSON goes")
    p.add_argument("--tagger-dir", default=TAGGER_DIR)
    p.add_argument("--segment-model", default=SEG_MODEL)
    p.add_argument("--est-python", default="/workspace/venvs/est/bin/python",
                   help="the venv with MoGe-2")
    p.add_argument("--no-moge", dest="moge", action="store_false",
                   help="skip the field-of-view estimate (the user fixed the angle)")
    return p.parse_args(argv)


def subject_of(seg: np.ndarray | None) -> dict:
    if seg is None:
        return {"found": False, "fraction": 0.0, "cut": []}
    fraction = float(seg.mean())
    found = fraction >= 0.01
    return {"found": found, "fraction": round(fraction, 4),
            "cut": widen._cut_edges(seg) if found else []}


def placement_of(w: int, h: int, est: dict | None) -> dict:
    long_side, why = placement.by_moge(w, h, est)
    ok = bool(est) and "hfov" in est
    out = {"estimated": ok, "long_side": round(float(long_side), 2), "why": why}
    if ok:
        out["raw_deg"] = round(float(est["hfov"] if w >= h else est["vfov"]), 2)
        out["limits"] = list(placement.LIMITS)
    else:
        out["error"] = str((est or {}).get("error", "no estimate")).strip()[-300:]
    return out


def analyze(src: Path, *, tagger=None, segment=None, estimate=None) -> dict:
    """The report for ``src``. ``tagger`` and ``segment`` are callables on an RGB
    array; ``estimate`` returns MoGe's dict (or ``None`` to skip it)."""
    t0 = time.time()
    rgb = np.array(Image.open(src).convert("RGB"))
    h, w = rgb.shape[:2]
    report: dict = {"version": __version__, "src": src.name, "width": w, "height": h}
    try:
        report["tags"] = [t.as_dict() for t in tagger(rgb)] if tagger else []
    except Exception as exc:  # a report, not a run: say what failed
        report["tags"], report["tags_error"] = [], "%s: %s" % (type(exc).__name__, exc)
    try:
        report["subject"] = subject_of(segment(rgb) if segment else None)
    except Exception as exc:
        report["subject"] = dict(subject_of(None), error="%s: %s" % (type(exc).__name__, exc))
    if estimate is not None:
        report["placement"] = placement_of(w, h, estimate(src))
    else:
        report["placement"] = {"estimated": False, "skipped": True}
    report["seconds"] = round(time.time() - t0, 2)
    return report


def main(argv=None) -> int:
    a = parse(argv)
    src, out = Path(a.src), Path(a.out)
    tagger = segment = None
    problems = []
    try:
        from .tagger import Tagger
        tagger = Tagger(a.tagger_dir)
    except Exception as exc:
        problems.append("tagger: %s" % exc)
    if Path(a.segment_model).exists():
        from .subject import Segmenter
        segment = Segmenter(a.segment_model)
    else:
        problems.append("no segmentation model at %s" % a.segment_model)
    if problems:
        print("not ready: " + "; ".join(problems), file=sys.stderr)
        return 2
    work = out.parent
    estimate = (lambda p: stereo.estimate_moge(p, python=a.est_python, work=work)) if a.moge else None
    report = analyze(src, tagger=tagger, segment=segment, estimate=estimate)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    tmp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    tmp.replace(out)
    print("analyzed %s in %.1fs: %d tags, subject %s, placement %s" % (
        src.name, report["seconds"], len(report["tags"]),
        "cut at " + ",".join(report["subject"]["cut"]) if report["subject"]["cut"]
        else ("whole" if report["subject"]["found"] else "none"),
        report["placement"].get("long_side", "skipped")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
