"""How depth becomes disparity: where the range is measured, and its curve.

stereo360 normalises inverse depth to its 1st-99th percentile over the whole
panorama. After widening, the nearest 1% is the floor under the viewer, which the
equirect also oversamples, so the floor takes the depth range and the character
gets a sliver (V.1: the author found the characters flat). Here the range is
measured over the *source's region* instead, anything nearer is compressed into a
soft knee rather than clipped, and IW3's foreground-scale curve can reshape it.

Numpy only: imported by ``stereo_drive`` inside stereo360's venv, and by tests.
"""
from __future__ import annotations

import math

import numpy as np


def _softplus01(x, bias, scale):
    lo = math.log1p(math.exp((0 - bias) * scale))
    hi = math.log1p(math.exp((1 - bias) * scale))
    return (np.log1p(np.exp((x - bias) * scale)) - lo) / (hi - lo)


def _inv_softplus01(x, bias, scale):
    lo = math.log(max(math.expm1((0 - bias) * scale), 1e-6))
    hi = math.log(max(math.expm1((1 - bias) * scale), 1e-6))
    v = np.log(np.clip(np.expm1((x - bias) * scale), 1e-6, None))
    return (v - lo) / (hi - lo)


#: IW3's relative-depth mappers (nunif iw3/mapper.py, RELATIVE_MUL_MAPPER), in
#: foreground-scale order -3..3. Negative levels jump steeply at the far end and
#: then rise gently (-2: 0.49 by a tenth of the way in, slope ~0.5 after), so the
#: background separates from everything in front of it; positive levels flatten
#: the far end and steepen the near one. The author's IW3 setting is -2.
_LEVELS = [
    lambda x: _inv_softplus01(x, -0.0001, 3.4343),   # -3 inv_mul_3
    lambda x: _inv_softplus01(x, -0.0003, 6.2626),   # -2 inv_mul_2
    lambda x: _inv_softplus01(x, -0.002102, 7.8788), # -1 inv_mul_1
    lambda x: x,                                     #  0 none
    lambda x: _softplus01(x, 0.343, 12),             #  1 mul_1
    lambda x: _softplus01(x, 0.515, 12),             #  2 mul_2
    lambda x: _softplus01(x, 0.687, 12),             #  3 mul_3
]


def foreground_scale(x: np.ndarray, level: float) -> np.ndarray:
    """IW3's --foreground-scale on a 0..1 inverse-depth map (1 = nearest).
    A fractional level interpolates its two neighbours, as IW3 does."""
    level = float(np.clip(level, -3, 3))
    if level == 0:
        return x
    a = math.floor(level) if level > 0 else -math.floor(-level)
    b = math.ceil(level) if level > 0 else -math.ceil(-level)
    wa = _LEVELS[a + 3](x)
    if a == b:
        return np.clip(wa, 0, 1)
    w = abs(level - a)
    return np.clip(wa * (1 - w) + _LEVELS[b + 3](x) * w, 0, 1)


def normalise(inv_depth: np.ndarray, region: np.ndarray | None, knee: float = 0.85,
              pct: tuple = (1.0, 99.0)) -> np.ndarray:
    """0..1 inverse depth, in place where it can be.

    ``region`` (bool, same shape) is where the range is measured; None measures
    everywhere (stereo360's own behaviour, with no knee). The region's range maps
    to [0, knee]; anything nearer than the region's nearest is compressed into
    (knee, 1] instead of taking the range from it."""
    src = inv_depth if region is None or not region.any() else inv_depth[region]
    lo, hi = np.percentile(src, pct[0]), np.percentile(src, pct[1])
    if hi <= lo:
        hi = lo + 1e-6
    x = (inv_depth - lo) / (hi - lo)
    if region is None:
        return np.clip(x, 0, 1, out=x).astype(np.float32)
    top = 1.0 - knee
    over = x > 1.0
    x = x * knee
    # Past the region's nearest: 1 - top*exp(-(x-1)/top) keeps slope 1 at the knee.
    x[over] = 1.0 - top * np.exp(-(x[over] / knee - 1.0) * knee / top)
    return np.clip(x, 0, 1).astype(np.float32)


def region_from_source(source_mask: np.ndarray, grow_deg: float) -> np.ndarray:
    """The source's mask on the equirect, grown by ``grow_deg`` in every direction
    (as a box in pixels, which is close enough away from the poles)."""
    h, w = source_mask.shape[:2]
    k = max(1, int(round(grow_deg / 360.0 * w)))
    m = source_mask > 127
    rows = np.flatnonzero(m.any(1))
    cols = np.flatnonzero(m.any(0))
    out = np.zeros_like(m)
    if rows.size and cols.size:
        out[max(0, rows[0] - k): min(h, rows[-1] + k + 1),
            max(0, cols[0] - k): min(w, cols[-1] + k + 1)] = True
    return out
