"""The source's camera, estimated: field of view and the subject's distance.

Runs *inside stereo360's venv* (torch, transformers, Apple Depth Pro already
fetched by its warm-up), with this repo on the path:

    python -m vr180.estimate SRC.png OUT.json [--subject MASK.png]

Depth Pro predicts metric depth and the focal length from one image. The focal
length is the camera's field of view, which is what the source should span on the
sphere; the subject's median depth is how far away it was. The author asked for a
placement that estimates this (V.1). How well Depth Pro reads anime is untested.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vr180.estimate")
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--subject", help="subject mask PNG (255 = subject)")
    p.add_argument("--model", default="apple/DepthPro-hf")
    a = p.parse_args(argv)

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, DepthProForDepthEstimation

    img = Image.open(a.src).convert("RGB")
    w, h = img.size
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(a.model)
    model = DepthProForDepthEstimation.from_pretrained(a.model).to(dev).eval()
    inputs = proc(images=img, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inputs)
    post = proc.post_process_depth_estimation(out, target_sizes=[(h, w)])[0]
    depth = post["predicted_depth"].float().cpu().numpy()
    focal = float(post["focal_length"])
    res = {
        "focal_px": round(focal, 1),
        "hfov": round(math.degrees(2 * math.atan(w / (2 * focal))), 2),
        "vfov": round(math.degrees(2 * math.atan(h / (2 * focal))), 2),
        "median_m": round(float(np.median(depth)), 3),
    }
    if a.subject:
        m = np.array(Image.open(a.subject).convert("L").resize((w, h))) > 127
        if m.any():
            res["subject_m"] = round(float(np.median(depth[m])), 3)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
