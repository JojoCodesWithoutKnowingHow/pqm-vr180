"""Run stereo360 with the companion's depth normalisation (``depthmap``).

Runs *inside stereo360's venv*, from its checkout, with this repo on the path:

    python -m vr180.stereo_drive --region SOURCE_MASK.png --grow 15 --knee 0.85 \\
        --fg-scale -2 -- PANO.png -o OUT_180_LR.jpg --output-mode vr180 --inpaint learned

Everything after ``--`` is stereo360's own command line, unchanged. The hook
replaces ``stereo360.warp.normalize_inv_depth``, which the still-image path calls
through the module (pinned commit 23d1e256).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import depthmap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: stereo_drive [options] -- <stereo360 arguments>")
    cut = argv.index("--")
    p = argparse.ArgumentParser(prog="vr180.stereo_drive")
    p.add_argument("--region", help="source mask PNG (255 = source); default: whole sphere")
    p.add_argument("--grow", type=float, default=15.0, help="degrees to grow the region by")
    p.add_argument("--knee", type=float, default=0.85)
    p.add_argument("--fg-scale", type=float, default=0.0, help="IW3 foreground scale, -3..3")
    a = p.parse_args(argv[:cut])
    rest = argv[cut + 1:]

    region = None
    if a.region:
        from PIL import Image
        region = depthmap.region_from_source(np.array(Image.open(a.region).convert("L")), a.grow)

    from stereo360 import cli as s360_cli, warp  # the venv's stereo360
    import cv2

    def normalise(inv_depth: np.ndarray) -> np.ndarray:
        r = None
        if region is not None:
            h, w = inv_depth.shape[:2]
            r = cv2.resize(region.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        x = depthmap.normalise(inv_depth, r, knee=a.knee)
        x = depthmap.foreground_scale(x, a.fg_scale).astype(np.float32)
        inv_depth[...] = x          # stereo360 expects its buffer back, filled
        return inv_depth

    warp.normalize_inv_depth = normalise
    rc = s360_cli.main(rest)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
