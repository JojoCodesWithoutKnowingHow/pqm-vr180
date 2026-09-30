"""Lay the whole scene out in one generation, before the views refine it (S2).

V.2's views each saw a 90-degree window and the tags, and nothing decided the room as
a whole: a second bed appeared beside the first, a beach under a ship's deck. Here the
front of the sphere, out to ``max_deg`` off straight ahead, is drawn as **one
equidistant fisheye image** with the source (and any flat extension) in its middle,
and one inpaint fills all of it: one generation decides where the walls, the bed and
the horizon go. Illustrious knows ``fisheye`` from its training tags.

The layout is coarse (1024 px across 200 degrees, a third of the panorama's density),
so it is only a start: ``widen(layout=...)`` seeds each scene view with it and repaints
it at ``layout_denoise``, keeping the arrangement and adding the detail.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from . import prompts, sphere


def fisheye_dirs(S: int, max_deg: float) -> np.ndarray:
    """World direction of every pixel of an S x S equidistant fisheye looking
    straight ahead, whose edge midpoints are ``max_deg`` off-axis. Shape (S, S, 3)."""
    g = (np.arange(S, dtype=np.float32) + 0.5) / S * 2 - 1
    X, Y = np.meshgrid(g, -g)                   # Y up
    r = np.hypot(X, Y)
    th = r * math.radians(max_deg)
    safe = np.maximum(r, 1e-6)
    return np.stack([np.sin(th) * X / safe, np.sin(th) * Y / safe, np.cos(th)], -1)


def to_fisheye(pano: np.ndarray, S: int, max_deg: float,
               interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    H, W = pano.shape[:2]
    d = fisheye_dirs(S, max_deg)
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    mx = ((lon + np.pi) / (2 * np.pi) * W - 0.5) % W
    my = np.clip((np.pi / 2 - lat) / np.pi * H - 0.5, 0, H - 1)
    return cv2.remap(pano, mx.astype(np.float32), my.astype(np.float32), interp,
                     borderMode=cv2.BORDER_WRAP)


def from_fisheye(fish: np.ndarray, W: int, max_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """The fisheye back on a W-wide equirect: (image, covered as bool). Covered is
    the disc within ``max_deg`` of straight ahead."""
    S = fish.shape[0]
    d = sphere.equirect_dirs(W)
    th = np.arccos(np.clip(d[..., 2], -1, 1))
    rxy = np.maximum(np.hypot(d[..., 0], d[..., 1]), 1e-6)
    r = th / math.radians(max_deg)
    X, Y = r * d[..., 0] / rxy, r * d[..., 1] / rxy
    mx = ((X + 1) / 2 * S - 0.5).astype(np.float32)
    my = ((1 - Y) / 2 * S - 0.5).astype(np.float32)
    img = cv2.remap(fish, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img, th <= math.radians(max_deg)


def layout_prompt(fill_tags: list[str], where: str | None, quality: str = prompts.QUALITY,
                  strong: bool = False) -> str:
    """The whole scene at once: every tag, no direction words (up and down are both
    in the picture), and the projection named. ``strong`` (round 2): weighted, with
    ``fisheye lens`` and ``curved lines``, since round 1's Yamato layout came back as
    an ordinary wide-angle picture and bent the room when read as a fisheye. (An
    automatic straight-line check was tried on round 1's twenty layouts and could not
    tell that one from a true fisheye, so it is not used.)"""
    lens = ["(fisheye:1.3)", "fisheye lens", "curved lines"] if strong else ["fisheye"]
    extra = lens + ["wide shot", "no humans"] + ([] if where == "plain" else ["scenery"])
    words = list(dict.fromkeys(extra + list(fill_tags)))
    return ", ".join(([quality] if quality else []) + words)


def make_layout(pano: np.ndarray, known: np.ndarray, fill_tags: list[str], where: str | None,
                inpaint, seed: int, max_deg: float = 100.0, S: int = 1024,
                quality: str = prompts.QUALITY, negative: str = prompts.NEGATIVE,
                reference: np.ndarray | None = None, steps: int | None = None,
                work=None, strong: bool = False) -> tuple[np.ndarray, dict]:
    """(layout equirect the size of ``pano``, log). Outside the fisheye's disc the
    layout is the disc's edge carried outward and blurred, so a view there still
    starts from something of the scene's colour."""
    from . import widen   # the pre-fill and the back fill live there
    H, W = known.shape
    fish = to_fisheye(pano, S, max_deg)
    kf = to_fisheye((known * 255).astype(np.uint8), S, max_deg, cv2.INTER_NEAREST) > 127
    unknown = ~kf
    mask = cv2.dilate(unknown.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    seeded = widen._seed(fish, mask > 0)
    control = seeded.copy()
    control[mask > 0] = 0
    prompt = layout_prompt(fill_tags, where, quality, strong=strong)
    gen = inpaint(seeded, mask, prompt, negative, seed, control=control, reference=reference,
                  steps=steps)
    img, cover = from_fisheye(gen, W, max_deg)
    out = np.zeros_like(pano)
    out[cover] = img[cover]
    out = widen.fill_back(out, cover)
    if work is not None:
        from PIL import Image
        Image.fromarray(gen).save(work / "layout_fisheye.png")
    return out, {"max_deg": max_deg, "px": S, "prompt": prompt,
                 "known_frac": round(float(kf.mean()), 3)}
