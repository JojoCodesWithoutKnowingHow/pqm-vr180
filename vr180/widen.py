"""Widen a flat image to the front hemisphere, one perspective view at a time.

The rules V.0's trial and the author's verdict set:

- **The source is never regenerated.** It is placed once and no later step writes
  a pixel of it; the fill's seam is blended only on the fill side.
- **Each view is mostly known** (``plan.Planner``), so it continues the edge it
  touches instead of inventing a scene.
- **Each view knows where it looks** (``prompts.view_prompt``).
- **A subject the frame cuts off is continued**, not fenced off: views that run
  up against it are prompted with the subject (the author, V.1). Everything else
  is told "no humans", so no second person appears.
- **A plain background is extended, not generated**: its views take the colour
  from the edge, with no diffusion to invent objects on it (V.1's plain sources
  grew strange shapes at the subject's feet).
- **A body, once painted, is kept** (``join="hard"``, round 1 of the outpaint
  work): later views may not repaint or cross-fade over it. The 0.4.0 blend let a
  scene view paint floor over a continued leg, and its 32 px cross-fade left a
  see-through ghost where two views disagreed (V.2's outputs 001-003, 007).
- **Only what VR180 shows is generated**: the front hemisphere plus a margin for
  the second eye. The back of the sphere is a cheap blur, there only so the depth
  model has something plausible on the faces it reads.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np
from PIL import Image

from . import plan, prompts, sphere, subject


class Inpainter(Protocol):
    def __call__(self, image: np.ndarray, mask: np.ndarray, prompt: str, negative: str,
                 seed: int, control: np.ndarray | None = None,
                 reference: np.ndarray | None = None, steps: int | None = None,
                 **kw) -> np.ndarray: ...
    # ``denoise=`` is passed only over a layout (``widen(layout=...)``).


#: ``segment(rgb) -> bool mask`` of the subject, or None to skip subject handling.
Segment = Callable[[np.ndarray], np.ndarray]


@dataclass
class Options:
    width: int = 4096            # the equirect's width; VR180 then has W/2 per eye
    long_side: float = 60.0      # degrees the source's long side spans (V.1: 90 is twice life size)
    view_fov: float = 90.0
    view_px: int = 1024
    target_deg: float = 100.0    # fill out to this angle off straight ahead
    max_new: float = 0.45
    seam_px: int = 32            # band of earlier *fill* each view may repaint
    max_views: int = 40
    stop_left: float = 0.004     # stop planning views below this much target left;
                                 # the cracks that remain are filled in place
    seed: int = 1234
    quality: str = prompts.QUALITY   # the checkpoint's quality words (anime by default)
    negative: str = prompts.NEGATIVE
    subject_tags: tuple = ()         # the subject's own tags; empty = never continue a body
    plain_fill: bool = True          # plain background: extend the colour, no diffusion
    prompt_mode: str = "tags"        # "tags", or "minimal" (direction words only)
    reference: bool = False          # hand the source to the inpainter as a reference
    steps: int = 28
    taper: bool = True               # fewer steps and pixels far from the source (V.1: unseen)
    #: (degrees off the source's centre, steps): full steps within the first,
    #: easing to the last; linear between. Subject views always get full steps.
    taper_steps: tuple = ((40.0, 28), (60.0, 22), (80.0, 16), (100.0, 12))
    taper_far_deg: float = 60.0      # beyond this, a view renders at taper_far_px
    taper_far_px: int = 768
    #: "blend" is 0.4.0's join; "hard" keeps a painted body (never repainted or
    #: cross-faded), blends scene joins over ``hard_seam_px`` only, fans the
    #: continuation zone out by ``zone_spread_deg`` so a bending limb stays inside
    #: it, and grows the body with ``grow_segment`` (a lower threshold).
    join: str = "blend"
    hard_seam_px: int = 8
    zone_spread_deg: float = 30.0
    #: Over a layout (S2), how much a scene view may change it.
    layout_denoise: float = 0.7
    #: Compose (G, the author's idea after round 2): the scene is final once the
    #: layout is made around the finished picture, so every view is only a detail
    #: pass over it at ``layout_denoise`` -- no inpaint ControlNet, no view with the
    #: subject's tags, the full ``seam_px`` blend (the views agree, so nothing
    #: ghosts). Round 2's E repainted the layout at 0.7 and inked a line at every
    #: view's edge; F's subject views painted new bodies below the finished one.
    compose: bool = False


@dataclass
class Result:
    pano: np.ndarray
    source_mask: np.ndarray
    log: dict = field(default_factory=dict)


def off_centre_deg(yaw: float, pitch: float) -> float:
    """Angle between a view's centre and straight ahead (the source's centre)."""
    c = math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def taper_for(off_deg: float, kind: str, opt: "Options") -> tuple[int, int]:
    """(steps, view pixels) for a view ``off_deg`` from the source's centre.
    Foveated: the author's idea (V.1) -- the periphery is where a viewer looks
    least and the headset's lenses are softest, so it gets less work."""
    if not opt.taper or kind == "subject":
        return opt.steps, opt.view_px
    pts = opt.taper_steps
    if off_deg <= pts[0][0]:
        steps = pts[0][1]
    elif off_deg >= pts[-1][0]:
        steps = pts[-1][1]
    else:
        for (d0, s0), (d1, s1) in zip(pts, pts[1:]):
            if d0 <= off_deg <= d1:
                steps = s0 + (s1 - s0) * (off_deg - d0) / (d1 - d0)
                break
    steps = int(round(min(steps, opt.steps)))
    px = opt.taper_far_px if off_deg > opt.taper_far_deg else opt.view_px
    return steps, px


def continuation_zone(subj_known: np.ndarray, unknown: np.ndarray, hole: np.ndarray,
                      S: int, source: np.ndarray | None = None,
                      spread_deg: float = 0.0) -> np.ndarray | None:
    """Where a view continues the subject: the empty pixels just past its cut edge,
    extruded *outward* from the side of the frame that cut it (legs cut at the
    bottom continue downward, within the body's width plus a small margin). The
    reach follows the cut's width, clamped to [64 px, 25% of the view]; the body
    goes on in the next view as the painted part joins the subject.

    V.1: a round zone spreading from every cut edge reached deep into the room
    beside the character and invited a duplicate there."""
    edge = subj_known & (cv2.dilate(unknown.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    if edge.sum() < 8:
        return None
    ys, xs = np.nonzero(edge)
    span = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
    r = int(np.clip(0.5 * span, 64, 0.25 * S))
    # Each edge pixel goes the way the emptiness is: legs cut at the bottom go down,
    # an arm cut at the side goes sideways, each on its own (V.1: a majority vote
    # sent Tifa's cut torso sideways because her arms left the frame).
    u = unknown
    step = 3
    shift = {"down": np.zeros_like(u), "up": np.zeros_like(u),
             "right": np.zeros_like(u), "left": np.zeros_like(u)}
    shift["down"][:-step] = u[step:]
    shift["up"][step:] = u[:-step]
    shift["right"][:, :-step] = u[:, step:]
    shift["left"][:, step:] = u[:, :-step]
    zone = np.zeros(edge.shape, bool)
    for d, empty_that_way in shift.items():
        e = (edge & empty_that_way).astype(np.uint8)
        if e.sum() < 4:
            continue
        if spread_deg > 0:
            zone |= _fan(e, d, r, max(4, r // 4), math.tan(math.radians(spread_deg)))
            continue
        # A one-sided kernel: OpenCV's dilation extends *opposite* to where the
        # kernel's ones lie (ones in the top half reach down; checked in V.1).
        line = np.zeros(2 * r + 1, np.uint8)
        if d in ("down", "right"):
            line[:r + 1] = 1
        else:
            line[r:] = 1
        k = line[:, None] if d in ("down", "up") else line[None, :]
        ext = cv2.dilate(e, k)                                 # one-sided extrusion
        m = max(4, r // 4)                                     # a small side margin
        ext = cv2.dilate(ext, np.ones((1, 2 * m + 1) if d in ("down", "up") else (2 * m + 1, 1),
                                      np.uint8))
        zone |= ext > 0
    zone &= hole
    return zone if zone.any() else None


def _shift(a: np.ndarray, n: int, d: str) -> np.ndarray:
    """``a`` moved ``n`` pixels towards ``d``, the vacated rim empty."""
    if n == 0:
        return a.copy()
    out = np.zeros_like(a)
    if d == "down":
        out[n:] = a[:-n]
    elif d == "up":
        out[:-n] = a[n:]
    elif d == "right":
        out[:, n:] = a[:, :-n]
    else:
        out[:, :-n] = a[:, n:]
    return out


def _fan(e: np.ndarray, d: str, r: int, m: int, tan: float) -> np.ndarray:
    """A fan extruded from the edge pixels ``e`` towards ``d``: ``r`` deep, ``m``
    either side at the root, widening by ``tan`` per pixel of depth. Round 1 of the
    outpaint work: 0.4.0's straight strip cut off a limb that bent or angled, and
    the scene pass painted over what left it."""
    out = np.zeros(e.shape, np.uint8)
    step = max(2, r // 32)
    for depth in range(0, r + 1, step):
        half = int(round(m + depth * tan))
        k = np.ones((1, 2 * half + 1) if d in ("down", "up") else (2 * half + 1, 1), np.uint8)
        out |= _shift(cv2.dilate(e, k), depth, d)
    # Close the gaps between the sampled depths: a short one-sided line.
    line = np.zeros(2 * step + 1, np.uint8)
    if d in ("down", "right"):
        line[:step + 1] = 1
    else:
        line[step:] = 1
    k = line[:, None] if d in ("down", "up") else line[None, :]
    return cv2.dilate(out, k) > 0


def _feather(gen_mask: np.ndarray, unknown: np.ndarray, seam_px: int) -> np.ndarray:
    """Blend weight in the view: 1 over what was empty, ramping to 0 across the
    band of earlier fill that the view was allowed to repaint."""
    inside = (gen_mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    w = np.clip(dist / max(seam_px, 1), 0, 1)
    w[unknown] = 1.0
    w[inside == 0] = 0.0
    return w.astype(np.float32)


def _seed(view: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    """The hole's starting colours: Navier-Stokes from its border, then blurred,
    which is Krita AI Diffusion's pre-fill for expanding an image. Only a start for
    the sampler, so it is done at a quarter of the size and only the hole takes it."""
    S = view.shape[0]
    q = max(64, S // 4)
    small, hole = _shrink_known(view, ~unknown, (q, q))
    filled = cv2.inpaint(small, hole * 255, 5, cv2.INPAINT_NS)
    blurred = cv2.GaussianBlur(filled, (0, 0), max(1.0, q / 64))
    filled[hole > 0] = blurred[hole > 0]
    big = cv2.resize(filled, (S, S), interpolation=cv2.INTER_CUBIC)
    out = view.copy()
    out[unknown] = big[unknown]
    return out


def _shrink_known(img: np.ndarray, known: np.ndarray, size: tuple):
    """``img`` shrunk to ``size`` averaging only its known pixels (normalised
    convolution), and the small hole: where nothing known fell. A plain
    INTER_AREA shrink averages the empty (black) pixels into the known ones at the
    boundary, which pasted a dark dotted rim along every view's edge (V.1)."""
    k = known.astype(np.float32)
    num = cv2.resize(img.astype(np.float32) * k[..., None], size, interpolation=cv2.INTER_AREA)
    den = cv2.resize(k, size, interpolation=cv2.INTER_AREA)
    small = np.where(den[..., None] > 1e-3, num / np.maximum(den, 1e-3)[..., None], 0)
    hole = (den < 0.999).astype(np.uint8)
    return np.clip(small, 0, 255).astype(np.uint8), hole


def _push_pull(img: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Fill where ``weight`` is 0 by push-pull interpolation: average what is known
    into ever smaller images (normalised), then blend back up, so every gap takes
    a smooth mix of what surrounds it. No direction is preferred, so unlike
    Navier-Stokes it draws no streaks out of small tone changes along an edge."""
    img = img.astype(np.float32)
    w = weight.astype(np.float32)
    if w.min() > 0.999:
        return img
    if min(w.shape) <= 2:
        # The coarsest level: whatever still has nothing takes the known average.
        mean = (img * w[..., None]).sum((0, 1)) / max(float(w.sum()), 1e-4)
        return img * w[..., None] + mean * (1 - w[..., None])
    h2, w2 = (w.shape[0] + 1) // 2, (w.shape[1] + 1) // 2
    num = cv2.resize(img * w[..., None], (w2, h2), interpolation=cv2.INTER_AREA)
    den = cv2.resize(w, (w2, h2), interpolation=cv2.INTER_AREA)
    coarse = np.where(den[..., None] > 1e-4, num / np.maximum(den, 1e-4)[..., None], 0)
    coarse = _push_pull(coarse, np.clip(den * 4, 0, 1))
    up = cv2.resize(coarse, (w.shape[1], w.shape[0]), interpolation=cv2.INTER_LINEAR)
    return img * w[..., None] + up * (1 - w[..., None])


def fill_smooth(pano: np.ndarray, known: np.ndarray, todo: np.ndarray, width: int = 1024,
                sigma: float = 2.0) -> np.ndarray:
    """Fill ``todo`` in one pass over the whole sphere, by push-pull from what is
    known, at low resolution, and paste it only into ``todo``. For a plain
    background: view-by-view fills disagreed slightly and met in soft steps, and
    Navier-Stokes drew streaks out of the source's edge (V.1)."""
    H, W = known.shape
    w, h = width, width // 2
    small, hole = _shrink_known(pano, known, (w, h))
    pad = w // 8                      # wrap horizontally: the fill meets itself behind
    wide = np.concatenate([small[:, -pad:], small, small[:, :pad]], 1)
    hwide = np.concatenate([hole[:, -pad:], hole, hole[:, :pad]], 1)
    filled = _push_pull(wide, 1.0 - hwide.astype(np.float32))
    blurred = cv2.GaussianBlur(filled, (0, 0), sigma)
    filled[hwide > 0] = blurred[hwide > 0]
    filled = np.clip(filled, 0, 255).astype(np.uint8)
    big = cv2.resize(filled[:, pad:pad + w], (W, H), interpolation=cv2.INTER_CUBIC)
    out = pano.copy()
    out[todo] = big[todo]
    return out


def fill_back(pano: np.ndarray, known: np.ndarray) -> np.ndarray:
    """The rest of the sphere, cheaply: extend the edges at low resolution and blur."""
    H, W = known.shape
    sw = 512
    small = cv2.resize(pano, (sw, sw // 2), interpolation=cv2.INTER_AREA)
    ks = cv2.resize(known.astype(np.uint8), (sw, sw // 2), interpolation=cv2.INTER_NEAREST)
    # Wrap horizontally so the fill meets itself behind the viewer.
    wide = np.concatenate([small[:, -sw // 4:], small, small[:, :sw // 4]], 1)
    kwide = np.concatenate([ks[:, -sw // 4:], ks, ks[:, :sw // 4]], 1)
    filled = cv2.inpaint(wide, ((kwide == 0) * 255).astype(np.uint8), 8, cv2.INPAINT_TELEA)
    filled = cv2.GaussianBlur(filled, (0, 0), 6)[:, sw // 4: sw // 4 + sw]
    big = cv2.resize(filled, (W, H), interpolation=cv2.INTER_CUBIC)
    out = pano.copy()
    out[~known] = big[~known]
    return out


def widen(src: np.ndarray, fill_tags: list[str], inpaint: Inpainter, opt: Options,
          work: Path | None = None, say: Callable[[str], None] = print,
          segment: Segment | None = None, grow_segment: Segment | None = None,
          layout: np.ndarray | None = None,
          place: Callable[[np.ndarray], tuple] | None = None) -> Result:
    """``layout`` (S2): an equirect the size of the panorama whose front is a
    coarse fill of the whole scene from one generation (``vr180.layout``). Given,
    each scene view starts from it instead of a blur and changes it only by
    ``opt.layout_denoise``, so the views agree on where the room's walls, bed and
    horizon are. A body's continuation zone still starts from the blur.

    ``place(img) -> (pano, mask, (hfov, vfov))`` puts a picture the size of ``src``
    on the sphere; by default centred at ``opt.long_side``. A canvas grown on one
    side (round 2) is off-centre and brings its own."""
    t_all = time.time()
    hard = opt.join == "hard"
    grow = grow_segment if (hard and grow_segment is not None) else segment
    if place is None:
        def place(img):
            return sphere.place(img, opt.width, opt.long_side)
    pano, src_mask, (hfov, vfov) = place(src)
    source = src_mask > 0
    known = source.copy()
    where, why = prompts.setting(fill_tags)
    planner = plan.Planner(opt.view_fov, opt.target_deg, max_new=opt.max_new)
    log = {"hfov": round(hfov, 2), "vfov": round(vfov, 2), "setting": where,
           "setting_why": why, "fill_tags": fill_tags, "views": [], "join": opt.join,
           "layout": layout is not None}
    # The subject, on the sphere: from the source, then from every view that
    # continued it, so the next view down still knows the legs belong to it.
    subj = np.zeros(source.shape, bool)
    deferred = np.zeros(source.shape, bool)   # plain background, filled after the loop
    if segment is not None and opt.subject_tags:
        seg = segment(src)
        placed, _m, _f = place((seg * 255).astype(np.uint8))
        subj = (placed > 127) & source
        cut = _cut_edges(seg)
        log["subject"] = {"frac_of_source": round(float(seg.mean()), 3), "cut_at": cut}
        say("subject covers %.0f%% of the source; cut by the frame at %s"
            % (seg.mean() * 100, ", ".join(cut) or "no edge"))
    if work:
        (work / "views").mkdir(parents=True, exist_ok=True)
    say("source spans %.1f x %.1f deg; setting %s (%s)" % (hfov, vfov, where, why))

    for n in range(opt.max_views):
        small = planner.small(known)
        if planner.remaining(small) < opt.stop_left:
            break
        v = planner.next_view(small)
        if v is None:
            break
        F = opt.view_fov
        # What kind of view this is, cheaply (256 px), before choosing its size:
        # a body continued below the source can sit far off-centre and still
        # needs full size.
        q = 256
        kq = sphere.view_of((known * 255).astype(np.uint8), v.yaw, v.pitch, F, q,
                            cv2.INTER_NEAREST) > 127
        sq = sphere.view_of((subj * 255).astype(np.uint8), v.yaw, v.pitch, F, q,
                            cv2.INTER_NEAREST) > 127
        pre_kind = ("scene" if opt.compose
                    else "subject" if subject.touches(sq, ~kq, reach_px=12) else "scene")
        off = off_centre_deg(v.yaw, v.pitch)
        steps, S = taper_for(off, pre_kind, opt)
        short = hard and not opt.compose
        seam_px = max(4, int(round((opt.hard_seam_px if short else opt.seam_px) * S / opt.view_px)))
        view = sphere.view_of(pano, v.yaw, v.pitch, F, S)
        kv = sphere.view_of((known * 255).astype(np.uint8), v.yaw, v.pitch, F, S,
                            cv2.INTER_NEAREST) > 127
        sv = sphere.view_of(src_mask, v.yaw, v.pitch, F, S, cv2.INTER_NEAREST) > 127
        unknown = ~kv
        if unknown.mean() < 0.002:
            break
        # Repaint a band of earlier fill for the seam, never the source.
        band = np.ones((2 * seam_px + 1,) * 2, np.uint8)
        gen_mask = cv2.dilate(unknown.astype(np.uint8) * 255, band)
        gen_mask[sv] = 0
        subj_v = sphere.view_of((subj * 255).astype(np.uint8), v.yaw, v.pitch, F, S,
                                cv2.INTER_NEAREST) > 127
        subj_before = subj.copy()
        if hard:
            # A painted body is never repainted: its mask grown a pixel over what is
            # known, as the body's colours resample a pixel past its nearest-sampled mask.
            body = cv2.dilate((subj_v & kv).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            gen_mask[body & kv] = 0
        seeded = _seed(view, unknown)
        laid = None
        if layout is not None:
            lv = sphere.view_of(layout, v.yaw, v.pitch, F, S)
            laid = view.copy()
            laid[unknown] = lv[unknown]
        kind = ("subject" if not opt.compose
                and subject.touches(subj_v, unknown, reach_px=max(12, 48 * S // 1024))
                else "plain" if where == "plain" and opt.plain_fill else "scene")
        if kind == "subject" and pre_kind != "subject" and opt.taper:
            steps = opt.steps               # the full check found the subject after all
        t0 = time.time()
        if kind == "plain":
            # Deferred: the whole plain background is filled in one pass after the
            # loop (``fill_smooth``), so no view boundary can show in it.
            rows, cols = region = sphere.bounds(v.yaw, v.pitch, F, opt.width)
            _img, cover = sphere.back_project(np.zeros((8, 8), np.uint8), v.yaw, v.pitch, F,
                                              opt.width, region=region)
            newly = cover & ~known[rows][:, cols]
            deferred[rows, cols] |= newly
            known[rows, cols] |= cover
            log["views"].append({"yaw": v.yaw, "pitch": v.pitch, "new": round(v.new, 3),
                                 "kind": kind, "seconds": 0.0, "prompt": "", "negative": "",
                                 "target_left": round(planner.remaining(planner.small(known)), 4)})
            say("view %2d yaw %4.0f pitch %4.0f plain  : deferred to the one-pass fill"
                % (n, v.yaw, v.pitch))
            continue
        else:
            if kind == "subject":
                prompt = prompts.subject_prompt(list(opt.subject_tags), fill_tags, where,
                                                v.pitch, opt.quality)
                negative = prompts.SUBJECT_NEGATIVE
            elif opt.prompt_mode == "minimal":
                prompt = prompts.minimal_prompt(where, v.pitch, opt.quality)
                negative = prompts.view_negative(opt.negative, where, v.pitch)
            else:
                prompt = prompts.view_prompt(fill_tags, where, v.pitch, opt.quality)
                negative = prompts.view_negative(opt.negative, where, v.pitch)
            ref = src if opt.reference else None
            zone = None
            plain_rest = None
            ahead = None
            grown_a = None
            if kind == "subject":
                zone = continuation_zone(subj_v & kv, unknown, gen_mask > 0, S,
                                         spread_deg=opt.zone_spread_deg if hard else 0.0)
            if zone is not None and zone.any():
                # Two passes (V.1: a view that only grazed the character was given her
                # tags for its whole hole, and painted her again). Her tags paint only
                # the zone continuing her cut edge; the rest is scene, no people.
                zmask = (zone * 255).astype(np.uint8)
                control = seeded.copy()
                control[zone] = 0            # NoobAI Inpainting: the hole pure black
                gen = inpaint(seeded, zmask, prompt, negative, opt.seed + n,
                              control=control, reference=ref, steps=steps)
                rest = (gen_mask > 0) & ~zone
                if grow is not None:
                    # Where the body painted in the zone runs into the zone's far edge,
                    # keep the strip beyond it empty for the next view to continue.
                    # V.1: pass 2 filled it with scene, and the body stopped short.
                    grown_a = grow(gen) & zone
                    if grown_a.any():
                        ahead = continuation_zone(grown_a, rest, rest, S,
                                                  spread_deg=opt.zone_spread_deg if hard else 0.0)
                        if ahead is not None:
                            rest &= ~ahead
                if where == "plain" and opt.plain_fill:
                    plain_rest = rest            # joins the one-pass plain fill below
                elif rest.mean() > 0.005:
                    scene_prompt = prompts.view_prompt(fill_tags, where, v.pitch, opt.quality)
                    scene_negative = prompts.view_negative(opt.negative, where, v.pitch)
                    control = gen.copy()
                    control[rest] = 0
                    init, extra = gen, {}
                    if laid is not None:
                        init = gen.copy()
                        init[rest & unknown] = laid[rest & unknown]
                        extra = {"denoise": opt.layout_denoise}
                    gen = inpaint(init, (rest * 255).astype(np.uint8), scene_prompt,
                                  scene_negative, opt.seed + n + 1000, control=control,
                                  reference=ref, steps=steps, **extra)
                    prompt = prompt + "  ||  " + scene_prompt
            else:
                control = seeded.copy()
                control[gen_mask > 0] = 0    # NoobAI Inpainting: the hole pure black
                init, extra = seeded, {}
                if laid is not None and kind == "scene":
                    init, extra = laid, {"denoise": opt.layout_denoise}
                    if opt.compose:
                        extra["touch_up"] = True     # a detail pass: no inpaint ControlNet
                if opt.compose and laid is not None and kind == "scene" and opt.layout_denoise <= 0:
                    gen = laid                       # the hires layout as it is (round 4, H0)
                else:
                    gen = inpaint(init, gen_mask, prompt, negative, opt.seed + n,
                                  control=control, reference=ref, steps=steps, **extra)
        secs = time.time() - t0
        if kind == "subject" and grow is not None:
            # Grow the subject only from the continuation zone, so a stray figure
            # elsewhere can never be adopted as the subject and carried on.
            grown = grown_a if grown_a is not None else grow(gen) & (gen_mask > 0)
            if zone is not None:
                grown &= zone
            if grown.any():
                region = sphere.bounds(v.yaw, v.pitch, F, opt.width)
                sb, sc = sphere.back_project((grown * 255).astype(np.uint8), v.yaw, v.pitch, F,
                                             opt.width, interp=cv2.INTER_NEAREST, region=region)
                subj[region[0], region[1]] |= (sb > 127) & sc
        if kind == "subject" and ahead is not None and ahead.any():
            gen_mask = gen_mask.copy()
            gen_mask[ahead] = 0              # not pasted: it stays empty on the sphere
            w = _feather(gen_mask, unknown & ~ahead, seam_px)
        else:
            w = _feather(gen_mask, unknown, seam_px)
        rows, cols = region = sphere.bounds(v.yaw, v.pitch, F, opt.width)
        img, cover = sphere.back_project(gen, v.yaw, v.pitch, F, opt.width, region=region)
        ws, _ = sphere.back_project(w, v.yaw, v.pitch, F, opt.width,
                                   interp=cv2.INTER_LINEAR, region=region)
        keep = source[rows][:, cols]
        if hard:
            keep = keep | subj_before[rows][:, cols]
        ws = np.where(cover & ~keep, np.clip(ws, 0, 1), 0).astype(np.float32)
        sub = pano[rows][:, cols]
        pano[rows, cols] = (sub * (1 - ws[..., None]) + img * ws[..., None]).round().astype(np.uint8)
        known[rows, cols] |= ws > 0.5
        if kind == "subject" and plain_rest is not None and plain_rest.any():
            pr, pc = sphere.back_project((plain_rest * 255).astype(np.uint8), v.yaw, v.pitch, F,
                                         opt.width, interp=cv2.INTER_NEAREST, region=region)
            deferred[rows, cols] |= (pr > 127) & pc & ~source[rows][:, cols]
        rest = planner.remaining(planner.small(known))
        log["views"].append({"yaw": v.yaw, "pitch": v.pitch, "new": round(v.new, 3),
                             "kind": kind, "seconds": round(secs, 1), "prompt": prompt,
                             "off_deg": round(off, 1), "steps": steps, "px": S,
                             "reserved": round(float(ahead.mean()), 3)
                             if kind == "subject" and ahead is not None else 0.0,
                             "negative": negative, "target_left": round(rest, 4)})
        say("view %2d yaw %4.0f pitch %4.0f %-7s %2d steps %4d px: %2.0f%% new, %.1fs, "
            "%.1f%% of target left" % (n, v.yaw, v.pitch, kind, steps, S, v.new * 100, secs,
                                       rest * 100))
        if work:
            Image.fromarray(gen).save(work / "views" / ("%02d_y%d_p%d.png" % (n, v.yaw, v.pitch)))

    if deferred.any():
        painted = known & ~deferred
        pano = fill_smooth(pano, painted, deferred)
        log["plain_filled"] = round(float(deferred.mean()), 4)
    front = sphere.off_axis_deg(opt.width) <= 90
    cracks = front & ~known
    log["cracks_filled"] = round(float(cracks.mean() / front.mean()), 5)
    if cracks.any():
        # Slivers the views' rims left: Telea from their edges, in place.
        pano = cv2.inpaint(pano, (cracks * 255).astype(np.uint8), 5, cv2.INPAINT_TELEA)
        known |= cracks
    log["front_unfilled"] = round(float((front & ~known).mean() / front.mean()), 5)
    pano = fill_back(pano, known)
    log["seconds"] = round(time.time() - t_all, 1)
    return Result(pano, src_mask, log)


def _cut_edges(seg: np.ndarray, frac: float = 0.02) -> list[str]:
    """Which edges of the source the subject runs off (more than ``frac`` of it)."""
    edges = {"top": seg[0], "bottom": seg[-1], "left": seg[:, 0], "right": seg[:, -1]}
    return [name for name, e in edges.items() if e.mean() > frac]
