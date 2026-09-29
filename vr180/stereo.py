"""Depth, the second eye and VR180 packaging: stereo360, wrapped.

V.0 ran it and the author judged the result "working great" in a Quest 3. It
lives in its own venv (it brings torch and transformers), so it is called as a
program, never imported. With a depth option set it runs through
``stereo_drive``, which swaps in the companion's normalisation (``depthmap``).
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

#: The commit V.0's trial ran was not recorded; this is the one V.1 pinned.
STEREO360_REPO = "https://github.com/LeonG-ZA/stereo360"
STEREO360_REF = "23d1e256a8562fe7ebaf55dc41c809a8f43e774b"
HERE = Path(__file__).resolve().parent.parent


@dataclass
class Depth:
    """How depth becomes disparity. The defaults are stereo360's own behaviour."""
    norm: str = "global"          # "global" (stereo360) or "region" (the source's)
    grow: float = 15.0            # degrees the region reaches past the source
    knee: float = 0.85
    fg_scale: float = 0.0         # IW3's foreground scale, -3..3
    strength: float = 1.0
    extra: list = field(default_factory=list)   # any other stereo360 arguments

    def driven(self) -> bool:
        return self.norm == "region" or self.fg_scale != 0

    @staticmethod
    def parse_extra(text: str) -> list:
        return shlex.split(text) if text else []


def estimate_camera(src: Path, *, python: str, checkout: str, subject=None,
                    work: Path | None = None, timeout: float = 600) -> dict:
    """Depth Pro's estimate of the source's camera (``vr180.estimate``), or
    ``{"error": ...}`` when it cannot be had; never raises."""
    import json
    work = work or src.parent
    out = work / "camera.json"
    cmd = [python, "-m", "vr180.estimate", str(src), str(out)]
    if subject is not None:
        import numpy as np
        from PIL import Image
        mask = work / "subject_mask.png"
        Image.fromarray((np.asarray(subject) * 255).astype("uint8")).save(mask)
        cmd += ["--subject", str(mask)]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [checkout, str(HERE)] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    try:
        r = subprocess.run(cmd, cwd=checkout, capture_output=True, text=True, timeout=timeout,
                           env=env)
        if r.returncode != 0 or not out.exists():
            return {"error": (r.stderr or r.stdout)[-600:]}
        return json.loads(out.read_text(encoding="utf-8"))
    except Exception as exc:  # an estimate must never stop a run
        return {"error": str(exc)[-600:]}


def estimate_moge(src: Path, *, python: str, work: Path | None = None,
                  timeout: float = 600) -> dict:
    """MoGe-2's estimate of the source's camera (``vr180.estimate_alt --only moge``,
    in its own venv), or ``{"error": ...}``; never raises."""
    import json
    work = work or src.parent
    out = work / "moge.json"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(HERE)] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    try:
        r = subprocess.run([python, "-m", "vr180.estimate_alt", "--only", "moge", str(out), str(src)],
                           cwd=str(HERE), capture_output=True, text=True, timeout=timeout, env=env)
        if r.returncode != 0 or not out.exists():
            return {"error": (r.stderr or r.stdout)[-600:]}
        return json.loads(out.read_text(encoding="utf-8")).get(str(src), {}).get(
            "moge", {"error": "no result"})
    except Exception as exc:  # an estimate must never stop a run
        return {"error": str(exc)[-600:]}


def to_vr180(pano: Path, out: Path, *, python: str, checkout: str, depth: Depth | None = None,
             source_mask: Path | None = None, inpaint: str = "learned",
             timeout: float = 1800) -> dict:
    """Run stereo360 on a 360 equirect ``pano``; write a side-by-side VR180 ``out``."""
    d = depth or Depth()
    s360 = [str(pano), "-o", str(out), "--output-mode", "vr180", "--inpaint", inpaint,
            "--strength", str(d.strength)] + list(d.extra)
    env = None
    if d.driven():
        pre = ["--knee", str(d.knee), "--fg-scale", str(d.fg_scale), "--grow", str(d.grow)]
        if d.norm == "region" and source_mask is not None:
            pre += ["--region", str(source_mask)]
        cmd = [python, "-m", "vr180.stereo_drive"] + pre + ["--"] + s360
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            [checkout, str(HERE)] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    else:
        cmd = [python, "-m", "stereo360"] + s360
    t0 = time.time()
    r = subprocess.run(cmd, cwd=checkout, capture_output=True, text=True, timeout=timeout, env=env)
    info = {"cmd": " ".join(cmd[1:]), "exit": r.returncode, "seconds": round(time.time() - t0, 1)}
    if r.returncode != 0 or not out.exists():
        raise RuntimeError("stereo360 failed (exit %d): %s" % (r.returncode,
                                                              (r.stderr or r.stdout)[-1500:]))
    return info
