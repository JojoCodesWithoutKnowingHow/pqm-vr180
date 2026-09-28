"""Equirectangular geometry.

Convention: longitude 0 is straight ahead (+z), positive to the right (+x);
latitude positive up (+y). Pixel (x, y) of a W x H (= W/2) equirect image sits at
lon = (x + .5) / W * 2pi - pi, lat = pi/2 - (y + .5) / H * pi.

A *view* is a square perspective image of field of view ``fov`` looking at
(yaw, pitch): yaw first about the vertical axis, then pitch, never roll, so a
view's horizon stays level wherever it looks.
"""
from __future__ import annotations

import functools
import math

import cv2
import numpy as np


@functools.lru_cache(maxsize=4)
def equirect_dirs(W: int) -> np.ndarray:
    """Unit direction of every pixel of a W x W/2 equirect, shape (H, W, 3).
    Cached and read-only: every view of a run asks for the same grid."""
    H = W // 2
    lon = (np.arange(W, dtype=np.float32) + 0.5) / W * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(H, dtype=np.float32) + 0.5) / H * np.pi
    lon, lat = np.meshgrid(lon, lat)
    d = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], -1)
    d.flags.writeable = False
    return d


def off_axis_deg(W: int) -> np.ndarray:
    """Angle of every pixel from straight ahead, in degrees, shape (H, W)."""
    return np.degrees(np.arccos(np.clip(equirect_dirs(W)[..., 2], -1, 1)))


def fovs(w: int, h: int, long_side_deg: float) -> tuple[float, float]:
    """Horizontal and vertical field of view of a w x h image whose long side
    spans ``long_side_deg`` as a flat (perspective) picture."""
    t = math.tan(math.radians(long_side_deg) / 2)
    if w >= h:
        return long_side_deg, math.degrees(2 * math.atan(t * h / w))
    return math.degrees(2 * math.atan(t * w / h)), long_side_deg


def place(src: np.ndarray, W: int, long_side_deg: float):
    """A flat image, straight ahead, as a W x W/2 equirect.

    Returns (pano, mask, (hfov, vfov)); ``mask`` is 255 where the source is and is
    eroded by two pixels so resampling at its rim never counts as source.
    """
    h, w = src.shape[:2]
    hfov, vfov = fovs(w, h, long_side_deg)
    d = equirect_dirs(W)
    z = np.maximum(d[..., 2], 1e-6)
    px = d[..., 0] / z / math.tan(math.radians(hfov) / 2)
    py = d[..., 1] / z / math.tan(math.radians(vfov) / 2)
    inside = (d[..., 2] > 0) & (np.abs(px) <= 1) & (np.abs(py) <= 1)
    mx = ((px + 1) / 2 * w - 0.5).astype(np.float32)
    my = ((1 - py) / 2 * h - 0.5).astype(np.float32)
    pano = cv2.remap(src, mx, my, cv2.INTER_AREA if w > W else cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REPLICATE)
    pano[~inside] = 0
    mask = cv2.erode((inside * 255).astype(np.uint8), np.ones((5, 5), np.uint8))
    return pano, mask, (hfov, vfov)


def rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """View -> world rotation: yaw about +y, then pitch up about the view's x."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])
    return (ry @ rx).astype(np.float32)


def view_dirs(yaw: float, pitch: float, fov: float, S: int) -> np.ndarray:
    """World direction of every pixel of an S x S view, shape (S, S, 3)."""
    t = math.tan(math.radians(fov) / 2)
    g = (np.arange(S, dtype=np.float32) + 0.5) / S * 2 - 1
    x, y = np.meshgrid(g * t, -g * t)
    v = np.stack([x, y, np.ones_like(x)], -1)
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    return v @ rotation(yaw, pitch).T


def view_of(pano: np.ndarray, yaw: float, pitch: float, fov: float, S: int,
            interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    """A square perspective view out of an equirect image."""
    H, W = pano.shape[:2]
    d = view_dirs(yaw, pitch, fov, S)
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    mx = ((lon + np.pi) / (2 * np.pi) * W - 0.5) % W
    my = np.clip((np.pi / 2 - lat) / np.pi * H - 0.5, 0, H - 1)
    return cv2.remap(pano, mx.astype(np.float32), my.astype(np.float32), interp,
                     borderMode=cv2.BORDER_WRAP)


def bounds(yaw: float, pitch: float, fov: float, W: int):
    """The rows (a slice) and columns (an index array, wrapping) of a W-wide
    equirect that a view can land on, with a degree to spare. A view over a pole
    reaches every column."""
    H = W // 2
    d = view_dirs(yaw, pitch, fov, 65)
    lat = np.degrees(np.arcsin(np.clip(d[..., 1], -1, 1)))
    t = math.tan(math.radians(fov) / 2)
    to_view = rotation(yaw, pitch).T

    def sees(v) -> bool:
        x, y, z = to_view @ np.array(v, np.float32)
        return bool(z > 0 and abs(x / z) <= t and abs(y / z) <= t)

    up, down = sees((0, 1, 0)), sees((0, -1, 0))
    pole = up or down
    hi = 90.0 if up else float(lat.max()) + 1
    lo = -90.0 if down else float(lat.min()) - 1
    r0 = max(0, int(math.floor((90 - hi) / 180 * H)))
    r1 = min(H, int(math.ceil((90 - lo) / 180 * H)))
    if pole:
        return slice(r0, r1), np.arange(W)
    lon = np.degrees(np.arctan2(d[..., 0], d[..., 2]))
    rel = (lon - yaw + 180) % 360 - 180
    a, b = yaw + rel.min() - 1, yaw + rel.max() + 1
    if b - a >= 358:
        return slice(r0, r1), np.arange(W)
    c0 = int(math.floor((a + 180) / 360 * W))
    c1 = int(math.ceil((b + 180) / 360 * W))
    return slice(r0, r1), np.arange(c0, c1) % W


def back_project(view: np.ndarray, yaw: float, pitch: float, fov: float, W: int,
                 inset: float = 0.98, interp: int = cv2.INTER_CUBIC, region=None):
    """Where a view lands on a W-wide equirect: (image, coverage as bool).

    ``inset`` trims the view's outermost rim, where the model has least context.
    With ``region`` (from ``bounds``) only that crop is computed and returned.
    """
    S = view.shape[0]
    d = equirect_dirs(W)
    if region is not None:
        d = d[region[0]][:, region[1]]
    d = d @ rotation(yaw, pitch)   # world -> view is R^T d, i.e. d @ R
    t = math.tan(math.radians(fov) / 2)
    z = np.maximum(d[..., 2], 1e-6)
    x = d[..., 0] / z / t
    y = d[..., 1] / z / t
    cover = (d[..., 2] > 0) & (np.abs(x) <= inset) & (np.abs(y) <= inset)
    mx = ((x + 1) / 2 * S - 0.5).astype(np.float32)
    my = ((1 - y) / 2 * S - 0.5).astype(np.float32)
    img = cv2.remap(view, mx, my, interp, borderMode=cv2.BORDER_REPLICATE)
    return img, cover
