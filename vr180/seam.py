"""Repaint the seam where the picture meets the fill, on the sphere (round 2).

Round 1: the author saw the seam in every image, A included -- a band of blur at the
source's edge (V.1's ``soften_rim`` blurs its outermost 12 px on purpose, and the
fill is softer), and misaligned lineart where a body crosses it. Here the blur is
not used; instead a few views are laid along the picture's boundary and a narrow
band straddling it (``inner`` px of the picture, ``outer`` px of the fill) is
repainted at low denoise, so detail and lines run across.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from . import sphere


def view_centres(boundary: np.ndarray, spacing_deg: float = 22.0, limit: int = 16) -> list:
    """(yaw, pitch) of views along the boundary: greedy, each covering what is
    within ``spacing_deg`` of its centre."""
    H, W = boundary.shape
    ys, xs = np.nonzero(boundary)
    if ys.size == 0:
        return []
    step = max(1, ys.size // 4000)
    ys, xs = ys[::step], xs[::step]
    lon = (xs + 0.5) / W * 2 * np.pi - np.pi
    lat = np.pi / 2 - (ys + 0.5) / H * np.pi
    v = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], -1)
    left = np.ones(len(v), bool)
    cos_r = math.cos(math.radians(spacing_deg))
    out = []
    while left.any() and len(out) < limit:
        i = int(np.flatnonzero(left)[0])
        out.append((math.degrees(lon[i]), math.degrees(lat[i])))
        left &= v @ v[i] < cos_r
    return out


def repaint(pano: np.ndarray, picture: np.ndarray, inpaint, prompt: str, negative: str,
            seed: int, denoise: float = 0.4, fov: float = 60.0, S: int = 1024,
            inner: int = 8, outer: int = 24, steps: int | None = None) -> dict:
    """Repaint, in place, the band straddling the edge of ``picture`` (a bool mask
    on the equirect). Returns a log."""
    W = pano.shape[1]
    m8 = picture.astype(np.uint8)
    boundary = picture & ~(cv2.erode(m8, np.ones((3, 3), np.uint8)) > 0)
    centres = view_centres(boundary)
    for n, (yaw, pitch) in enumerate(centres):
        view = sphere.view_of(pano, yaw, pitch, fov, S)
        mv = sphere.view_of(m8 * 255, yaw, pitch, fov, S, cv2.INTER_NEAREST) > 127
        band = ((cv2.dilate(mv.astype(np.uint8), np.ones((2 * outer + 1,) * 2, np.uint8)) > 0)
                & ~(cv2.erode(mv.astype(np.uint8), np.ones((2 * inner + 1,) * 2, np.uint8)) > 0))
        if band.mean() < 0.001:
            continue
        gen = inpaint(view, (band * 255).astype(np.uint8), prompt, negative, seed + n,
                      steps=steps, denoise=denoise, touch_up=True)
        # Zero at both of the band's edges (the model inks a line along a mask's
        # edge; round 14), full in its middle.
        from .grow import edge_ramp
        w = edge_ramp(band, max(2, (inner + outer) // 4))
        region = sphere.bounds(yaw, pitch, fov, W)
        img, cover = sphere.back_project(gen, yaw, pitch, fov, W, region=region)
        ws, _c = sphere.back_project(w, yaw, pitch, fov, W, interp=cv2.INTER_LINEAR, region=region)
        ws = np.where(cover, np.clip(ws, 0, 1), 0).astype(np.float32)
        rows, cols = region
        sub = pano[rows][:, cols].astype(np.float32)
        pano[rows, cols] = (sub * (1 - ws[..., None]) + img * ws[..., None]).round().astype(np.uint8)
    return {"views": len(centres), "denoise": denoise, "inner": inner, "outer": outer}
