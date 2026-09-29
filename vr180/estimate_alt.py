"""Camera estimates from other models, to compare with Depth Pro's (``estimate``).

Experimental (V.1): Depth Pro read flat, top-down and blurred anime images as
telephoto shots. MoGe-2 (normalised intrinsics + metric geometry) and GeoCalib
(single-image calibration, used by the VR-Outpaint tools) are the alternatives.
Runs in its own venv (torch, moge, geocalib), so stereo360's cannot break:

    python -m vr180.estimate_alt OUT.json IMAGE...

Writes ``{image: {"moge": {...}, "geocalib": {...}}}``; a model that fails records
its error rather than stopping the others.
"""
from __future__ import annotations

import json
import math
import sys
import traceback


def _deg(rad: float) -> float:
    return round(math.degrees(float(rad)), 2)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    only = None
    if argv and argv[0] == "--only":
        only, argv = argv[1], argv[2:]
    out_path, images = argv[0], argv[1:]
    import numpy as np
    import torch
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {p: {} for p in images}

    try:
        if only == "geocalib":
            raise ImportError("skipped (--only geocalib)")
        from moge.model.v2 import MoGeModel
        moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").to(dev).eval()
        for p in images:
            try:
                img = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255
                t = torch.tensor(img, device=dev).permute(2, 0, 1)
                with torch.no_grad():
                    o = moge.infer(t)
                k = o["intrinsics"].float().cpu().numpy()
                fx, fy = float(k[0, 0]), float(k[1, 1])      # normalised by width / height
                depth = o["depth"].float().cpu().numpy()
                mask = o["mask"].cpu().numpy() if "mask" in o else np.isfinite(depth)
                res[p]["moge"] = {"hfov": _deg(2 * math.atan(0.5 / fx)),
                                  "vfov": _deg(2 * math.atan(0.5 / fy)),
                                  "median_m": round(float(np.median(depth[mask])), 3)}
            except Exception:
                res[p]["moge"] = {"error": traceback.format_exc()[-400:]}
        del moge
    except Exception:
        for p in images:
            res[p]["moge"] = {"error": traceback.format_exc()[-400:]}

    try:
        if only == "moge":
            raise ImportError("skipped (--only moge)")
        from geocalib import GeoCalib
        gc = GeoCalib().to(dev)
        for p in images:
            try:
                r = gc.calibrate(gc.load_image(p).to(dev))
                cam = r["camera"]
                entry = {}
                for name in ("hfov", "vfov"):
                    v = getattr(cam, name, None)
                    if v is not None:
                        entry[name] = _deg(v.reshape(-1)[0])
                f = getattr(cam, "f", None)
                if f is not None:
                    entry["f_px"] = [round(float(x), 1) for x in f.reshape(-1)[:2]]
                res[p]["geocalib"] = entry or {"camera": str(cam)[:300]}
            except Exception:
                res[p]["geocalib"] = {"error": traceback.format_exc()[-400:]}
    except Exception:
        for p in images:
            res[p]["geocalib"] = {"error": traceback.format_exc()[-400:]}

    json.dump(res, open(out_path, "w"), indent=1)
    for p, v in res.items():
        print("ALT", p.split("/")[-1], json.dumps(v)[:300], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
