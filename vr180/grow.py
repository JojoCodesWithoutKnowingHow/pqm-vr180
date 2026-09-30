"""Grow the picture on the side(s) the frame cuts the body, far enough, at full
resolution (round 2 of the outpaint work).

Round 1's flat extension (``flatext``) finished bodies better than anything before
it, but the author found, in the headset: feet and shins still failing (Do-S,
Mosquito Girl in concrete, Lucoa on the beach), a second body where it had no room
(Nami), and soft toes and hands, a divot or misaligned lineart at the join. The
causes, measured: its canvas grew symmetrically, so a cut side gained only +25% and
the feet fell to the sphere views; and it was generated at ~0.7 scale.

Here only the cut sides grow, each by ``grow`` times the source's size across it
(capped so the new edge stays within ``max_deg`` of straight ahead); the canvas
keeps the source's optical centre, so it is placed with ``sphere.place_focal``;
each side is generated in its own window with some of what is there as context,
legs first; and a tiled pass at low denoise brings the new pixels, and the source's
outermost ``rim`` pixels, to full resolution, which also joins the lineart across.

Scene first (``scene=``, the author's idea after round 1): the canvas starts as the
laid-out scene, and only a fan from where the frame cuts the body is painted with
the subject, over it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import prompts
from .flatext import _prefill

SIDES = ("bottom", "top", "left", "right")       # legs first


@dataclass
class Grown:
    image: np.ndarray
    rect: tuple[int, int, int, int]      # the source in the canvas: x, y, w, h
    focal: float                         # pixels
    centre: tuple[float, float]          # the source's optical centre in the canvas
    log: dict = field(default_factory=dict)
    body: np.ndarray | None = None       # the body we started with, in the canvas

    def source_mask(self) -> np.ndarray:
        m = np.zeros(self.image.shape[:2], np.uint8)
        x, y, w, h = self.rect
        m[y:y + h, x:x + w] = 255
        return m


def focal(w: int, h: int, long_side: float) -> float:
    return (max(w, h) / 2) / math.tan(math.radians(long_side) / 2)


def side_growth(w: int, h: int, cut: list[str], grow: float, f: float,
                max_deg: float = 70.0) -> dict:
    """Pixels to add on each cut side: ``grow`` times the source's size across that
    side, capped so the new edge stays within ``max_deg`` of straight ahead."""
    limit = f * math.tan(math.radians(max_deg))
    out = {}
    for side in SIDES:
        if side not in cut:
            continue
        vertical = side in ("top", "bottom")
        half, size = (h / 2, h) if vertical else (w / 2, w)
        add = int(min(grow * size, max(0.0, limit - half)))
        if add >= 16:
            out[side] = add
    return out


def canvas_geometry(w: int, h: int, add: dict):
    """(W, H, x0, y0) of the grown canvas, the source at (x0, y0)."""
    x0, y0 = add.get("left", 0), add.get("top", 0)
    return w + x0 + add.get("right", 0), h + y0 + add.get("bottom", 0), x0, y0


def _window(side: str, W: int, H: int, x0: int, y0: int, w: int, h: int, context: int):
    """The crop a side is generated in: its new band plus ``context`` pixels of what
    is already there, across the whole canvas."""
    if side == "bottom":
        return slice(max(0, y0 + h - context), H), slice(0, W)
    if side == "top":
        return slice(0, min(H, y0 + context)), slice(0, W)
    if side == "left":
        return slice(0, H), slice(0, min(W, x0 + context))
    return slice(0, H), slice(max(0, x0 + w - context), W)


def _gen_size(h: int, w: int, budget: int) -> tuple[int, int]:
    scale = min(1.0, math.sqrt(budget / (w * h)))
    return (max(64, int(round(w * scale / 64)) * 64), max(64, int(round(h * scale / 64)) * 64))


def _paste(canvas: np.ndarray, region, img: np.ndarray, mask: np.ndarray) -> None:
    """``img`` (the region's size) into ``canvas`` where ``mask``, feathered a pixel
    or two inside it so a pass leaves no hard edge of its own."""
    rs, cs = region
    m = mask.astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 1.5) * m
    sub = canvas[rs, cs].astype(np.float32)
    canvas[rs, cs] = (sub * (1 - m[..., None]) + img.astype(np.float32) * m[..., None]
                      ).round().astype(np.uint8)


def body_zone(seg: np.ndarray, rect, shape, add: dict, spread_deg: float = 12.0) -> np.ndarray:
    """Scene first: where the body may be painted over the laid-out scene -- a fan
    from where the frame cuts the subject, out across each grown side."""
    from .widen import _fan
    x0, y0, w, h = rect
    H, W = shape
    zone = np.zeros((H, W), bool)
    for side, n in add.items():
        e = np.zeros((H, W), np.uint8)
        if side == "bottom":
            e[y0 + h - 1, x0:x0 + w] = seg[-1]
            d = "down"
        elif side == "top":
            e[y0, x0:x0 + w] = seg[0]
            d = "up"
        elif side == "left":
            e[y0:y0 + h, x0] = seg[:, 0]
            d = "left"
        else:
            e[y0:y0 + h, x0 + w - 1] = seg[:, -1]
            d = "right"
        if e.sum() < 4:
            continue
        zone |= _fan(e, d, n, max(8, n // 12), math.tan(math.radians(spread_deg)))
    zone[y0:y0 + h, x0:x0 + w] = False
    return zone


def refine(canvas: np.ndarray, mask: np.ndarray, inpaint, prompt: str, negative: str,
           seed: int, denoise: float, tile: int = 1024, steps: int | None = None) -> int:
    """The full-resolution pass: ``tile`` px tiles over everything in ``mask``, each
    repainted at ``denoise`` with no inpaint ControlNet (``touch_up``). Returns the
    number of tiles run."""
    H, W = mask.shape
    th, tw = min(tile, H) // 8 * 8, min(tile, W) // 8 * 8

    def starts(total, t):
        s = list(range(0, max(1, total - t + 1), max(1, t * 7 // 8)))
        if s[-1] != total - t:
            s.append(total - t)
        return s

    n = 0
    for y in starts(H, th):
        for x in starts(W, tw):
            region = (slice(y, y + th), slice(x, x + tw))
            m = mask[region]
            if m.mean() < 0.002:
                continue
            out = inpaint(canvas[region].copy(), (m * 255).astype(np.uint8), prompt, negative,
                          seed + n, steps=steps, denoise=denoise, touch_up=True)
            _paste(canvas, region, out, m)
            n += 1
    return n


def _edge_rows(side: str, canvas: np.ndarray, band: int = 3) -> tuple:
    """The outermost ``band`` rows or columns of a side, as a slice pair."""
    H, W = canvas.shape[:2]
    return {"bottom": (slice(H - band, H), slice(0, W)), "top": (slice(0, band), slice(0, W)),
            "left": (slice(0, H), slice(0, band)), "right": (slice(0, H), slice(W - band, W))}[side]


def body_reaches(seg: np.ndarray, side: str, frac: float = 0.02) -> bool:
    """Does the body still run off this side of the canvas?"""
    return bool(seg[_edge_rows(side, seg)].mean() > frac)


def track_body(seg: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """The part of ``seg`` that is the body we started with: its connected pieces
    that touch ``prev`` (the body so far, in the same frame). Round 2b: a figure
    painted in the new band counted as "the body still reaches the edge", so the
    canvas kept growing and grew more figures; a figure not joined to the body is
    not the body."""
    n, labels = cv2.connectedComponents(seg.astype(np.uint8), connectivity=8)
    near = cv2.dilate(prev.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    keep = np.unique(labels[near & seg])
    keep = keep[keep > 0]
    return np.isin(labels, keep)


def extend_side(src: np.ndarray, cut: list[str], grow: float, long_side: float, inpaint,
                subject_tags, fill_tags, where: str | None, seed: int,
                quality: str = prompts.QUALITY, reference: np.ndarray | None = None,
                steps: int | None = None, budget: int = 1280 * 1024, max_deg: float = 70.0,
                refine_denoise: float = 0.35, rim: int = 8, step: float = 0.3,
                segment=None, scene_of=None, max_steps: int = 8) -> Grown | None:
    """Grow each cut side ``step`` times the source's size at a time, and stop a
    side as soon as the body no longer reaches its new edge, or at ``grow`` in all,
    or at ``max_deg``. None when nothing is cut or there is no segmenter.

    In each new band only a fan from where the body crosses the old edge is painted
    with the subject's tags; the rest of the band is scenery with no people -- the
    laid-out scene (``scene_of(W, H, cx, cy)``, scene first) or a scene pass.

    Round 2's first two tries, in order: growing a full source height at once gave
    Fubuki on the sofa a second body in the empty space below her; growing in steps
    with the subject's tags over the whole band still did, and a new figure touching
    the edge kept the growth going (``track_body``)."""
    if segment is None:
        return None
    h, w = src.shape[:2]
    f = focal(w, h, long_side)
    limit = side_growth(w, h, cut, grow, f, max_deg)
    if not limit:
        return None
    add = {s: 0 for s in limit}
    active = [s for s in SIDES if s in limit]
    canvas = src.copy()
    body = segment(src)
    her = prompts.subject_prompt(list(subject_tags), list(fill_tags), where, 0.0, quality)
    scene_p = prompts.view_prompt(list(fill_tags), where, 0.0, quality)
    scene_n = prompts.view_negative(prompts.NEGATIVE, where, 0.0)
    passes = []
    n = 0
    from .widen import _fan
    while active and n < max_steps:
        for side in list(active):
            size = h if side in ("top", "bottom") else w
            inc = min(int(step * size), limit[side] - add[side])
            if inc < 16:
                active.remove(side)
                continue
            old, old_body = canvas, body
            oh, ow = old.shape[:2]
            add[side] += inc
            W, H, x0, y0 = canvas_geometry(w, h, add)
            dx = inc if side == "left" else 0
            dy = inc if side == "top" else 0
            canvas = (np.zeros((H, W, 3), np.uint8) if scene_of is None
                      else scene_of(W, H, x0 + w / 2, y0 + h / 2).copy())
            known = np.zeros((H, W), bool)
            canvas[dy:dy + oh, dx:dx + ow] = old
            known[dy:dy + oh, dx:dx + ow] = True
            prev = np.zeros((H, W), bool)
            prev[dy:dy + oh, dx:dx + ow] = old_body
            band = ~known
            # Where the body crosses the old edge, and a fan from there into the band.
            rows = _edge_rows(side, np.zeros((oh, ow)))
            at_edge = np.zeros((oh, ow), bool)
            at_edge[rows] = True
            edge = np.zeros((H, W), np.uint8)
            edge[dy:dy + oh, dx:dx + ow] = (old_body & at_edge).astype(np.uint8)
            d = {"bottom": "down", "top": "up", "left": "left", "right": "right"}[side]
            zone = (_fan(edge, d, inc, max(8, inc // 12), math.tan(math.radians(12)))
                    if edge.sum() >= 4 else np.zeros((H, W), bool)) & band
            rest = band & ~zone
            region = _window(side, W, H, dx, dy, ow, oh, max(256, int(0.6 * inc)))
            jobs = [(zone, her, prompts.SUBJECT_NEGATIVE)]
            if scene_of is None:
                jobs.append((rest, scene_p, scene_n))       # scenery, no people
            painted = []
            for k, (todo, prompt, negative) in enumerate(jobs):
                m = todo[region]
                painted.append(round(float(m.mean()), 3))
                if m.mean() < 0.002:
                    continue
                crop = canvas[region]
                ch, cw = m.shape
                gw, gh = _gen_size(ch, cw, budget)
                small = cv2.resize(crop, (gw, gh), interpolation=cv2.INTER_AREA)
                sm = cv2.resize(m.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST) > 0
                sm = cv2.dilate(sm.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                # Over a laid-out scene the body starts from the scene; otherwise
                # from the blur (and the scene pass sees the body just painted).
                init = small if (scene_of is not None and k == 0) else _prefill(small, sm)
                control = init.copy()
                control[sm] = 0                  # NoobAI Inpainting: the hole pure black
                out = inpaint(init, (sm * 255).astype(np.uint8), prompt, negative,
                              seed + 10 * len(passes) + k, control=control,
                              reference=reference, steps=steps)
                big = cv2.resize(out, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
                _paste(canvas, region, big, m)
            body = track_body(segment(canvas), prev)
            reaches = body_reaches(body, side)
            passes.append({"side": side, "added": inc, "painted": painted,
                           "crossing_px": int(edge.sum()), "body_reaches_edge": reaches})
            if not reaches or add[side] >= limit[side]:
                active.remove(side)
        n += 1
    W, H, x0, y0 = canvas_geometry(w, h, add)
    canvas[y0:y0 + h, x0:x0 + w] = src
    hole = np.ones((H, W), bool)
    hole[y0:y0 + h, x0:x0 + w] = False
    tiles = 0
    if refine_denoise > 0:
        k3 = np.ones((2 * rim + 1, 2 * rim + 1), np.uint8)
        rim_band = (cv2.dilate(hole.astype(np.uint8), k3) > 0) & ~hole
        near_body = cv2.dilate(body.astype(np.uint8), np.ones((33, 33), np.uint8)) > 0
        todo = hole | rim_band
        # The body and its surroundings with her tags; the rest as scenery.
        tiles = refine(canvas, todo & near_body, inpaint, her, prompts.SUBJECT_NEGATIVE,
                       seed + 500, refine_denoise, steps=steps)
        tiles += refine(canvas, todo & ~near_body, inpaint, scene_p, scene_n,
                        seed + 700, refine_denoise, steps=steps)
        # Inside the rim, the source stays exactly as it was.
        canvas[y0 + rim:y0 + h - rim, x0 + rim:x0 + w - rim] = src[rim:h - rim, rim:w - rim]
    g = Grown(canvas, (x0, y0, w, h), f, (x0 + w / 2, y0 + h / 2),
              {"cut": cut, "grow": grow, "step": step, "limit": limit, "added": add,
               "canvas": [W, H], "passes": passes, "refine_tiles": tiles,
               "refine_denoise": refine_denoise, "rim": rim,
               "order": "body-first" if scene_of is None else "scene-first", "prompt": her})
    g.body = body
    return g
