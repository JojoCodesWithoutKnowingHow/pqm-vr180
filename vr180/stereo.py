"""Depth, the second eye and VR180 packaging: stereo360, wrapped.

V.0 ran it and the author judged the result "working great" in a Quest 3. It
lives in its own venv (it brings torch and transformers), so it is called as a
program, never imported.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

#: The commit V.0's trial ran was not recorded; this is the one V.1 pinned.
STEREO360_REPO = "https://github.com/LeonG-ZA/stereo360"
STEREO360_REF = "23d1e256a8562fe7ebaf55dc41c809a8f43e774b"


def to_vr180(pano: Path, out: Path, *, python: str, checkout: str, strength: float = 1.0,
             inpaint: str = "learned", timeout: float = 1800) -> dict:
    """Run stereo360 on a 360 equirect ``pano``; write a side-by-side VR180 ``out``."""
    cmd = [python, "-m", "stereo360", str(pano), "-o", str(out), "--output-mode", "vr180",
           "--inpaint", inpaint, "--strength", str(strength)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=checkout, capture_output=True, text=True, timeout=timeout)
    info = {"cmd": " ".join(cmd[1:]), "exit": r.returncode, "seconds": round(time.time() - t0, 1)}
    if r.returncode != 0 or not out.exists():
        raise RuntimeError("stereo360 failed (exit %d): %s" % (r.returncode,
                                                              (r.stderr or r.stdout)[-1500:]))
    return info
