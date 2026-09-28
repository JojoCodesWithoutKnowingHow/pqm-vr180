"""How many degrees the source's long side spans: fixed, by aspect ratio, or by shot.

V.1: at 90 degrees a figure is about twice life size; the author judged 60 "much
better" and asked for it to be a setting, and suggested that the ratio may say how
far the camera was (wide images usually farther from the subject, tall ones
closer). ``shot`` goes at the same question directly: a figure the frame shows
whole is sized to a person at a comfortable distance.
"""
from __future__ import annotations

import math

import numpy as np

#: (upper bound on width/height, degrees): wide shots span more of the view,
#: tall ones less. A starting guess to be tuned in a headset, not a measurement.
RATIO_PRESETS = [(0.60, 45.0),       # 9:16 and taller
                 (0.75, 50.0),       # 2:3
                 (0.90, 55.0),       # 4:5
                 (1.15, 60.0),       # square
                 (1.45, 65.0),       # 4:3, 3:2
                 (1.90, 70.0),       # 16:9
                 (math.inf, 75.0)]   # 2.35:1 and wider

PERSON_M = 1.65      # a standing adult
DISTANCE_M = 2.0     # a comfortable distance to stand from someone
LIMITS = (40.0, 100.0)


def by_ratio(w: int, h: int) -> tuple[float, str]:
    r = w / h
    for upper, deg in RATIO_PRESETS:
        if r <= upper:
            return deg, "ratio %.2f -> %g deg" % (r, deg)
    return 60.0, "ratio %.2f" % r


def by_shot(w: int, h: int, seg: np.ndarray | None, cut: list[str]) -> tuple[float, str]:
    """Size a figure the frame shows whole to ``PERSON_M`` at ``DISTANCE_M``.

    Only a whole, upright figure says how big the frame is; a cut one (a cowboy
    shot, a close-up) or none at all falls back to the ratio preset, and says so."""
    if seg is None or not seg.any():
        deg, why = by_ratio(w, h)
        return deg, "no subject found; " + why
    if cut:
        deg, why = by_ratio(w, h)
        return deg, "subject cut at %s, its size unknown; %s" % (", ".join(cut), why)
    rows = np.flatnonzero(seg.any(1))
    frac = (rows[-1] - rows[0] + 1) / h
    target = math.degrees(2 * math.atan(PERSON_M / 2 / DISTANCE_M))   # ~44.8 deg tall
    # The figure's angular height, from the frame's vertical field of view.
    vfov = 2 * math.degrees(math.atan(math.tan(math.radians(target) / 2) / frac))
    if w >= h:
        long_side = 2 * math.degrees(math.atan(math.tan(math.radians(vfov) / 2) * w / h))
    else:
        long_side = vfov
    deg = float(np.clip(long_side, *LIMITS))
    return deg, "whole figure %.0f%% of the height -> %.1f deg" % (frac * 100, deg)
