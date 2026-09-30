"""Continue the body inside the laid-out scene (round 6, the author's idea).

Round 5's H0 builds the character's extension on a flat canvas that knows nothing
of the room, then lays the room out around it. The author, after round 4: lay the
scene out first, then inpaint the extension *in the full layout*, masked to where
the body goes on, so the model sees the floor, the furniture and the light when it
draws her legs. Scene first was tried as F in round 2 and grew extra people; it
painted in strips without her face in view and let the sphere views go on painting
"her". Here she is painted once, with her whole body and the room in view.

1. The layout (hires) is made around the source alone; its fisheye is kept.
2. Where the frame cuts the body, a fan continues it across the grown canvas
   (``grow.body_zone``); the fan is carried onto the fisheye. A crop holding her
   and the fan with room around them is inpainted there, the fan the only mask:
   nothing else in the scene can change.
3. In the fisheye she is small, so the grown flat canvas is taken back out of the
   updated layout (``sphere.flat_of``) and the fan refined at full resolution at
   ``refine_denoise`` with her tags: the pose and footing from step 2, the detail
   of the source.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from . import grow, layout, prompts, sphere


def extend(src: np.ndarray, seg: np.ndarray, cut: list[str], long_side: float, W: int,
           fish: np.ndarray, max_deg: float, inpaint, subject_tags, fill_tags,
           where: str | None, seed: int, quality: str = prompts.QUALITY,
           reference: np.ndarray | None = None, steps: int | None = None,
           grow_frac: float = 1.0, max_side_deg: float = 70.0, budget: int = 1280 * 1024,
           refine_denoise: float = 0.5, work=None):
    """(grown picture as ``grow.Grown``, updated fisheye, log), or None when nothing
    is cut. ``fish`` is the layout's (hires) fisheye, ``max_deg`` its reach."""
    h, w = src.shape[:2]
    f = grow.focal(w, h, long_side)
    add = grow.side_growth(w, h, cut, grow_frac, f, max_side_deg)
    if not add:
        return None
    Wc, Hc, x0, y0 = grow.canvas_geometry(w, h, add)
    cx, cy = x0 + w / 2, y0 + h / 2
    zone = grow.body_zone(seg, (x0, y0, w, h), (Hc, Wc), add)
    if zone.sum() < 64:
        return None
    S = fish.shape[0]
    # The fan and the source, carried onto the fisheye.
    zp, _m = sphere.place_focal((zone * 255).astype(np.uint8), W, f, cx, cy)
    zf = layout.to_fisheye(zp, S, max_deg, cv2.INTER_NEAREST) > 127
    rect = np.zeros((Hc, Wc), np.uint8)
    rect[y0:y0 + h, x0:x0 + w] = 255
    sp, _m = sphere.place_focal(rect, W, f, cx, cy)
    sf = layout.to_fisheye(sp, S, max_deg, cv2.INTER_NEAREST) > 127
    ys, xs = np.nonzero(zf | sf)
    # A crop holding her and the fan, with a third again of room around them.
    y_a, y_b, x_a, x_b = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y_b - y_a, x_b - x_a) * 1.35)
    side = min(S, max(side, 512))
    ccy, ccx = (y_a + y_b) // 2, (x_a + x_b) // 2
    top = int(np.clip(ccy - side // 2, 0, S - side))
    left = int(np.clip(ccx - side // 2, 0, S - side))
    region = (slice(top, top + side), slice(left, left + side))
    crop = fish[region]
    m = zf[region]
    g = max(64, int(round(min(side, math.sqrt(budget)) / 64)) * 64)
    small = cv2.resize(crop, (g, g), interpolation=cv2.INTER_AREA)
    sm = cv2.resize(m.astype(np.uint8), (g, g), interpolation=cv2.INTER_NEAREST) > 0
    sm = cv2.dilate(sm.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    control = small.copy()
    control[sm] = 0                               # NoobAI Inpainting: the hole pure black
    her = prompts.subject_prompt(list(subject_tags), list(fill_tags), where, 0.0, quality)
    out = inpaint(small, (sm * 255).astype(np.uint8), her, prompts.SUBJECT_NEGATIVE, seed,
                  control=control, reference=reference, steps=steps)
    fish = fish.copy()
    big = cv2.resize(out, (side, side), interpolation=cv2.INTER_LANCZOS4)
    grow._paste(fish, region, big, m)
    # Back to a flat canvas at the source's density; the fan refined there.
    lay_eq, cover = layout.from_fisheye(fish, W, max_deg)
    canvas = sphere.flat_of(lay_eq, Wc, Hc, f, cx, cy)
    canvas[y0:y0 + h, x0:x0 + w] = src
    near = cv2.dilate(zone.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    near[y0 + 8:y0 + h - 8, x0 + 8:x0 + w - 8] = False
    tiles = grow.refine(canvas, near, inpaint, her, prompts.SUBJECT_NEGATIVE, seed + 500,
                        refine_denoise, steps=steps) if refine_denoise > 0 else 0
    canvas[y0 + 8:y0 + h - 8, x0 + 8:x0 + w - 8] = src[8:h - 8, 8:w - 8]
    if work is not None:
        from PIL import Image
        Image.fromarray(fish).save(work / "layout_fisheye_with_body.png")
    g_out = grow.Grown(canvas, (x0, y0, w, h), f, (cx, cy),
                       {"cut": cut, "added": add, "canvas": [Wc, Hc], "order": "in-layout",
                        "fisheye_crop": [top, left, side], "generated_at": g,
                        "refine_tiles": tiles, "refine_denoise": refine_denoise,
                        "prompt": her})
    g_out.body = None
    return g_out, fish, g_out.log
