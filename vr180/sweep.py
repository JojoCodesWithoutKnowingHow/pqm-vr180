"""Rerun only the stereo step on a finished widening, once per variant.

    python -m vr180.sweep WORK_DIR OUT_DIR VARIANTS.json [--stereo-python P --stereo-dir D]

VARIANTS.json is ``{"name": {"norm": "region", "fg_scale": -2, "strength": 1.0,
"extra": "--depth-backend depth-anything"}, ...}``. Each writes
``<stem>_<name>_180_LR.jpg`` in OUT_DIR, where ``<stem>`` is WORK_DIR's name
without ``_180_LR.work``, and appends to ``sweep.json`` in WORK_DIR. The widening
is paid once; a variant costs one stereo run (~10 s on a 5090).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import stereo


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vr180.sweep")
    p.add_argument("work")
    p.add_argument("out_dir")
    p.add_argument("variants")
    p.add_argument("--stereo-python", default="/workspace/venvs/stereo360/bin/python")
    p.add_argument("--stereo-dir", default="/workspace/stereo360")
    a = p.parse_args(argv)
    work, out_dir = Path(a.work), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = work.name.replace("_180_LR.work", "").replace(".work", "")
    variants = json.loads(Path(a.variants).read_text(encoding="utf-8"))
    log_path = work / "sweep.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    for name, v in variants.items():
        d = stereo.Depth(norm=v.get("norm", "global"), grow=float(v.get("grow", 15)),
                         knee=float(v.get("knee", 0.85)), fg_scale=float(v.get("fg_scale", 0)),
                         strength=float(v.get("strength", 1.0)),
                         extra=stereo.Depth.parse_extra(v.get("extra", "")))
        target = out_dir / ("%s_%s_180_LR.jpg" % (stem, name))
        try:
            info = stereo.to_vr180(work / "pano.png", target, python=a.stereo_python,
                                   checkout=a.stereo_dir, depth=d,
                                   source_mask=work / "source_mask.png")
            print("SWEEP %s %s %.1fs" % (stem, name, info["seconds"]), flush=True)
        except Exception as exc:  # one variant failing must not stop the rest
            info = {"error": str(exc)[-800:]}
            print("SWEEP %s %s FAILED %s" % (stem, name, str(exc)[-300:]), flush=True)
        log[name] = dict(info, variant=v, out=target.name)
        log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
