"""Which view to fill next.

V.0's per-view fill failed partly because its 22 views were fixed in advance and
50-70% empty, so each one invented a scene instead of continuing one. Here the
next view is chosen from what is already known: of every candidate direction, the
one that fills the most of the still-empty target while staying *mostly known*
(at most ``max_new`` of it empty). The fill therefore grows outward from the
source in bands, and every band sees the edge it continues.

Planning runs on a small equirect (``PLAN_W``) and is recomputed after each view.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import sphere

PLAN_W = 256


@dataclass(frozen=True)
class View:
    yaw: float
    pitch: float
    new: float     # fraction of the view (by solid angle) not yet known
    gain: float    # solid angle of empty target it covers, steradians


class Planner:
    def __init__(self, fov: float, target_deg: float, step: float = 10.0,
                 max_new: float = 0.45, min_new: float = 0.02, inset: float = 0.98):
        self.fov, self.max_new, self.min_new = fov, max_new, min_new
        W = PLAN_W
        H = W // 2
        lat = np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi
        # Solid angle of each plan pixel: cos(lat) dlat dlon.
        self.area = (np.cos(lat)[:, None] * (np.pi / H) * (2 * np.pi / W)
                     * np.ones((1, W))).astype(np.float32)
        self.target = sphere.off_axis_deg(W) <= target_deg
        self.candidates: list[tuple[float, float]] = []
        covers = []
        dummy = np.zeros((8, 8), np.uint8)
        for pitch in np.arange(-90, 90 + 1e-6, step):
            yaws = [0.0] if abs(pitch) == 90 else np.arange(-180, 180, step)
            for yaw in yaws:
                _img, cover = sphere.back_project(dummy, float(yaw), float(pitch), fov, W,
                                                  inset=inset)
                self.candidates.append((float(yaw), float(pitch)))
                covers.append(cover.reshape(-1))
        self.covers = np.stack(covers)                          # (C, H*W) bool
        self.cover_area = self.covers @ self.area.reshape(-1)   # (C,)

    def small(self, known: np.ndarray) -> np.ndarray:
        """A full-size known mask (bool) at plan resolution: known where nearly
        all of a plan pixel's area is (a thin crack is filled in place later)."""
        import cv2
        k = cv2.resize(known.astype(np.float32), (PLAN_W, PLAN_W // 2),
                       interpolation=cv2.INTER_AREA)
        return k > 0.9

    def remaining(self, known_small: np.ndarray) -> float:
        """Fraction of the target, by solid angle, still empty."""
        empty = self.target & ~known_small
        return float((empty * self.area).sum() / (self.target * self.area).sum())

    def next_view(self, known_small: np.ndarray) -> View | None:
        empty = (~known_small).reshape(-1).astype(np.float32) * self.area.reshape(-1)
        empty_target = empty * self.target.reshape(-1)
        new = (self.covers @ empty) / self.cover_area
        gain = self.covers @ empty_target
        useful = gain > 1e-4
        if not useful.any():
            return None
        ok = useful & (new <= self.max_new) & (new >= self.min_new)
        if ok.any():
            # Most target filled; among near-equals, the most level view (models
            # draw level views best) and then the one nearest straight ahead.
            pitch = np.abs(np.array([p for _y, p in self.candidates]))
            yaw = np.abs(np.array([y for y, _p in self.candidates]))
            score = np.where(ok, gain * (1 - 0.15 * pitch / 90) - 1e-6 * yaw, -np.inf)
            i = int(np.argmax(score))
        else:
            # Nothing is mostly known: take the least empty view that still helps.
            i = int(np.argmin(np.where(useful, new, np.inf)))
        yaw, pitch = self.candidates[i]
        return View(yaw, pitch, float(new[i]), float(gain[i]))
