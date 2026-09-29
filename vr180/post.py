"""Hide the seam between the source and the fill, on a finished panorama.

V.1: the author saw the source's edge on the flat panorama. It is a *sharpness*
step, not a colour one: the source is shrunk into place (which crisps it) while
the fill is generated at the panorama's own density (softer). Two remedies:

- ``detail_match``: measure fine-detail energy just inside the source's rim and
  just outside it, and sharpen the fill by the shortfall. The source is untouched.
- ``soften_rim``: blend the source's outermost pixels toward a slightly blurred
  copy, fading to nothing a few pixels in. Touches the source's edge a little,
  which the author allowed.

    python -m vr180.post PANO.png SOURCE_MASK.png OUT.png [--detail] [--rim 12]
"""
from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np

SIGMA = 1.2      # what counts as fine detail: the part a blur of this size removes


def _rings(source: np.ndarray, px: int):
    m = source.astype(np.uint8)
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    inner = source & ~(cv2.erode(m, k) > 0)
    outer = (cv2.dilate(m, k) > 0) & ~source
    return inner, outer


def _detail(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    return f - cv2.GaussianBlur(f, (0, 0), SIGMA)


def detail_ratio(pano: np.ndarray, source: np.ndarray, px: int = 24) -> float:
    """Fine-detail energy just inside the source's rim over just outside it."""
    inner, outer = _rings(source, px)
    hf = np.abs(_detail(pano)).mean(-1)
    return float(hf[inner].mean() / max(hf[outer].mean(), 1e-3))


def detail_match(pano: np.ndarray, source: np.ndarray, px: int = 24,
                 max_amount: float = 1.5) -> tuple[np.ndarray, float]:
    """Sharpen everything but the source by the measured shortfall; (image, amount)."""
    amount = float(np.clip(detail_ratio(pano, source, px) - 1.0, 0.0, max_amount))
    if amount <= 0.02:
        return pano, 0.0
    f = pano.astype(np.float32)
    sharp = np.clip(f + amount * _detail(pano), 0, 255)
    out = np.where(source[..., None], f, sharp)
    return out.round().astype(np.uint8), amount


def soften_rim(pano: np.ndarray, source: np.ndarray, px: int = 12,
               sigma: float = 1.5) -> np.ndarray:
    """Blend the source's outermost ``px`` pixels toward a blurred copy: fully at
    the edge, fading to nothing ``px`` pixels in."""
    if px <= 0:
        return pano
    dist = cv2.distanceTransform(source.astype(np.uint8), cv2.DIST_L2, 5)
    w = np.clip(1.0 - dist / px, 0.0, 1.0) * source
    f = pano.astype(np.float32)
    soft = cv2.GaussianBlur(f, (0, 0), sigma)
    out = f * (1 - w[..., None]) + soft * w[..., None]
    return out.round().astype(np.uint8)


def main(argv=None) -> int:
    from PIL import Image
    p = argparse.ArgumentParser(prog="vr180.post")
    p.add_argument("pano")
    p.add_argument("source_mask")
    p.add_argument("out")
    p.add_argument("--detail", action="store_true")
    p.add_argument("--rim", type=int, default=0)
    a = p.parse_args(argv)
    pano = np.array(Image.open(a.pano).convert("RGB"))
    source = np.array(Image.open(a.source_mask).convert("L")) > 127
    info = {"ratio_before": round(detail_ratio(pano, source), 3)}
    if a.detail:
        pano, info["amount"] = detail_match(pano, source)
    if a.rim:
        pano = soften_rim(pano, source, a.rim)
        info["rim_px"] = a.rim
    info["ratio_after"] = round(detail_ratio(pano, source), 3)
    Image.fromarray(pano).save(a.out)
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
