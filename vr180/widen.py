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
                 reference: np.ndarray | None = None, steps: int | None = None) -> np.ndarray: ...


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
    taper: bool = False              # fewer steps and pixels far from the source (off by default)
    #: (degrees off the source's centre, steps): full steps within the first,
    #: easing to the last; linear between. Subject views always get full steps.
    taper_steps: tuple = ((40.0, 28), (60.0, 22), (80.0, 16), (100.0, 12))
    taper_far_deg: float = 60.0      # beyond this, a view renders at taper_far_px
    taper_far_px: int = 768


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
          segment: Segment | None = None) -> Result:
    t_all = time.time()
    pano, src_mask, (hfov, vfov) = sphere.place(src, opt.width, opt.long_side)
    source = src_mask > 0
    known = source.copy()
    where, why = prompts.setting(fill_tags)
    planner = plan.Planner(opt.view_fov, opt.target_deg, max_new=opt.max_new)
    log = {"hfov": round(hfov, 2), "vfov": round(vfov, 2), "setting": where,
           "setting_why": why, "fill_tags": fill_tags, "views": []}
    # The subject, on the sphere: from the source, then from every view that
    # continued it, so the next view down still knows the legs belong to it.
    subj = np.zeros(source.shape, bool)
    deferred = np.zeros(source.shape, bool)   # plain background, filled after the loop
    if segment is not None and opt.subject_tags:
        seg = segment(src)
        placed, _m, _f = sphere.place((seg * 255).astype(np.uint8), opt.width, opt.long_side)
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
        pre_kind = "subject" if subject.touches(sq, ~kq, reach_px=12) else "scene"
        off = off_centre_deg(v.yaw, v.pitch)
        steps, S = taper_for(off, pre_kind, opt)
        seam_px = max(4, int(round(opt.seam_px * S / opt.view_px)))
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
        seeded = _seed(view, unknown)
        subj_v = sphere.view_of((subj * 255).astype(np.uint8), v.yaw, v.pitch, F, S,
                                cv2.INTER_NEAREST) > 127
        kind = ("subject" if subject.touches(subj_v, unknown, reach_px=max(12, 48 * S // 1024))
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
            control = seeded.copy()
            control[gen_mask > 0] = 0        # NoobAI Inpainting: the hole pure black
            gen = inpaint(seeded, gen_mask, prompt, negative, opt.seed + n,
                          control=control, reference=src if opt.reference else None,
                          steps=steps)
        secs = time.time() - t0
        if kind == "subject" and segment is not None:
            grown = segment(gen) & (gen_mask > 0)
            if grown.any():
                region = sphere.bounds(v.yaw, v.pitch, F, opt.width)
                sb, sc = sphere.back_project((grown * 255).astype(np.uint8), v.yaw, v.pitch, F,
                                             opt.width, interp=cv2.INTER_NEAREST, region=region)
                subj[region[0], region[1]] |= (sb > 127) & sc
        w = _feather(gen_mask, unknown, seam_px)
        rows, cols = region = sphere.bounds(v.yaw, v.pitch, F, opt.width)
        img, cover = sphere.back_project(gen, v.yaw, v.pitch, F, opt.width, region=region)
        ws, _ = sphere.back_project(w, v.yaw, v.pitch, F, opt.width,
                                   interp=cv2.INTER_LINEAR, region=region)
        ws = np.where(cover & ~source[rows][:, cols], np.clip(ws, 0, 1), 0).astype(np.float32)
        sub = pano[rows][:, cols]
        pano[rows, cols] = (sub * (1 - ws[..., None]) + img * ws[..., None]).round().astype(np.uint8)
        known[rows, cols] |= ws > 0.5
        rest = planner.remaining(planner.small(known))
        log["views"].append({"yaw": v.yaw, "pitch": v.pitch, "new": round(v.new, 3),
                             "kind": kind, "seconds": round(secs, 1), "prompt": prompt,
                             "off_deg": round(off, 1), "steps": steps, "px": S,
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
