"""Finish a cut-off body on the flat image, before anything goes on the sphere (L2).

V.2's outputs showed the continued body failing in two ways: a leg stopping at the
source's edge under a scene view's floor, and a leg drawn piecemeal across views of
different perspective that came apart where they met. Both come from the body being
painted a slice at a time. Here it is painted **once**: the source's canvas is grown
in the direction(s) the frame cuts the subject, symmetrically about its centre so the
picture's optical centre stays where ``sphere.place`` assumes it is, and one inpaint
fills the new canvas with the subject's tags and the scene's. The extended picture is
then placed on the sphere as the source; the source's own pixels are pasted back
exactly, so nothing of it is regenerated.

The fill is made at about ``budget`` pixels (SDXL's comfortable size) and scaled up to
the canvas, so it is softer than the source; ``post.detail_match`` measures that step
at the *original* source's rim, as before.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import prompts

#: Which axis a cut edge grows.
AXIS = {"top": "v", "bottom": "v", "left": "h", "right": "h"}


@dataclass
class Extended:
    image: np.ndarray            # the grown canvas, source pasted back exactly
    rect: tuple[int, int, int, int]   # where the source sits in it: x, y, w, h
    log: dict = field(default_factory=dict)

    def source_mask(self) -> np.ndarray:
        """uint8, 255 over the original source in the grown canvas."""
        m = np.zeros(self.image.shape[:2], np.uint8)
        x, y, w, h = self.rect
        m[y:y + h, x:x + w] = 255
        return m


def ext_long_side(w: int, h: int, long_side: float, W: int, H: int) -> float:
    """The long-side angle of a ``W x H`` canvas grown about the centre of a
    ``w x h`` picture whose long side spans ``long_side`` degrees: the same focal
    length, more picture."""
    f = (max(w, h) / 2) / math.tan(math.radians(long_side) / 2)
    return math.degrees(2 * math.atan((max(W, H) / 2) / f))


def canvas_for(w: int, h: int, cut: list[str], factor: float) -> tuple[int, int] | None:
    """The grown canvas's size, or None when nothing is cut."""
    axes = {AXIS[c] for c in cut if c in AXIS}
    if not axes or factor <= 1.0:
        return None
    W = int(round(w * factor / 2)) * 2 if "h" in axes else w
    H = int(round(h * factor / 2)) * 2 if "v" in axes else h
    return W, H


def _prefill(img: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Navier-Stokes from the hole's border, blurred: a start for the sampler."""
    h, w = hole.shape
    s = 256 / max(h, w)
    sw, sh = max(8, int(w * s)), max(8, int(h * s))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    sh_ = (cv2.resize(hole.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0)
    sh_ = cv2.dilate(sh_.astype(np.uint8), np.ones((3, 3), np.uint8))
    filled = cv2.inpaint(small, sh_ * 255, 5, cv2.INPAINT_NS)
    filled = cv2.GaussianBlur(filled, (0, 0), 3)
    big = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)
    out = img.copy()
    out[hole] = big[hole]
    return out


def extend(src: np.ndarray, cut: list[str], factor: float, inpaint, subject_tags, fill_tags,
           where: str | None, seed: int, quality: str = prompts.QUALITY,
           reference: np.ndarray | None = None, steps: int | None = None,
           budget: int = 1280 * 1024) -> Extended | None:
    h, w = src.shape[:2]
    size = canvas_for(w, h, cut, factor)
    if size is None:
        return None
    W, H = size
    x0, y0 = (W - w) // 2, (H - h) // 2
    scale = min(1.0, math.sqrt(budget / (W * H)))
    gw, gh = max(64, int(round(W * scale / 64)) * 64), max(64, int(round(H * scale / 64)) * 64)
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[y0:y0 + h, x0:x0 + w] = src
    hole = np.ones((H, W), bool)
    hole[y0:y0 + h, x0:x0 + w] = False
    small = cv2.resize(canvas, (gw, gh), interpolation=cv2.INTER_AREA)
    shole = cv2.resize(hole.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST) > 0
    # A pixel or two of the source's rim into the hole: resampling blurred it there.
    shole = cv2.dilate(shole.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    seeded = _prefill(small, shole)
    control = seeded.copy()
    control[shole] = 0                          # NoobAI Inpainting: the hole pure black
    prompt = prompts.subject_prompt(list(subject_tags), list(fill_tags), where, 0.0, quality)
    gen = inpaint(seeded, (shole * 255).astype(np.uint8), prompt, prompts.SUBJECT_NEGATIVE,
                  seed, control=control, reference=reference, steps=steps)
    big = cv2.resize(gen, (W, H), interpolation=cv2.INTER_LANCZOS4)
    out = np.where(hole[..., None], big, canvas)
    return Extended(out, (x0, y0, w, h),
                    {"cut": cut, "factor": factor, "canvas": [W, H], "generated_at": [gw, gh],
                     "prompt": prompt})
