"""Where the subject is: SkyTNT's anime-segmentation (ISNet-IS), as ONNX on the CPU.

Used to find a subject the source's frame cuts off, so the views that continue it
are prompted with the subject rather than told "no humans" (V.1: the author wants
the body continued). Model: ``skytnt/anime-seg`` ``isnetis.onnx`` (Apache-2.0),
fetched and verified by ``setup/pod_setup.sh``.
"""
from __future__ import annotations

import cv2
import numpy as np

SIZE = 1024


class Segmenter:
    def __init__(self, onnx_path: str):
        import onnxruntime as ort  # the pod's venv has it; tests use a stand-in
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def __call__(self, rgb: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """A bool mask, True on the character, the size of ``rgb``."""
        h, w = rgb.shape[:2]
        s = SIZE / max(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        canvas = np.zeros((SIZE, SIZE, 3), np.float32)
        top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
        canvas[top:top + nh, left:left + nw] = img
        x = canvas.transpose(2, 0, 1)[None]
        pred = self.session.run(None, {self.input: x})[0][0, 0]   # sigmoid is in the graph
        pred = pred[top:top + nh, left:left + nw]
        return cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR) > threshold


def touches(subject_in_view: np.ndarray, unknown: np.ndarray, reach_px: int = 48,
            min_frac: float = 0.002) -> bool:
    """Does the subject run up against what this view is about to paint?"""
    if not subject_in_view.any():
        return False
    near = cv2.dilate(subject_in_view.astype(np.uint8), np.ones((reach_px, reach_px), np.uint8)) > 0
    return float((near & unknown).mean()) >= min_frac
