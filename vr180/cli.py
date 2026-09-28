"""One image in, one VR180 file out.

    python -m vr180 SRC.png -o OUT_180_LR.jpg --tags "indoors, wooden floor, couch" \\
        --checkpoint waiIllustriousSDXL_v140

Writes, beside OUT: ``<stem>.work/`` with the flat panorama, the source mask,
every generated view and ``log.json`` (timings, the views, and every choice the
program made on its own, with why).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__, forge, prompts, stereo, widen


def parse(argv=None):
    p = argparse.ArgumentParser(prog="vr180", description=__doc__.split("\n\n")[0])
    p.add_argument("src")
    p.add_argument("-o", "--out", required=True, help="the VR180 file (name it *_180_LR.jpg)")
    p.add_argument("--tags", default="", help="what surrounds the subject, comma-separated")
    p.add_argument("--tags-file", help="a file of tags; appended to --tags")
    p.add_argument("--checkpoint", required=True, help="Forge checkpoint for the fill")
    p.add_argument("--method", choices=("plain", "cn"), default="plain",
                   help="plain: the checkpoint's own inpaint (default). cn: ControlNet Union "
                        "ProMax inpaint -- NaN or noise in Forge Neo as pinned (V.1)")
    p.add_argument("--denoise", type=float, default=0.95)
    p.add_argument("--cn-model", default="", help="ControlNet name (default: the ProMax one)")
    p.add_argument("--forge", default="http://127.0.0.1:7860")
    p.add_argument("--long-side", type=float, default=90.0, help="degrees the source spans")
    p.add_argument("--width", type=int, default=4096, help="equirect width (VR180 is W/2 per eye)")
    p.add_argument("--target", type=float, default=100.0, help="fill out to this angle off-axis")
    p.add_argument("--max-new", type=float, default=0.45)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--quality", default=prompts.QUALITY,
                   help="quality words the checkpoint expects (default: anime/Illustrious)")
    p.add_argument("--negative", default=prompts.NEGATIVE)
    p.add_argument("--sampler", default="Euler a")
    p.add_argument("--scheduler", default="Automatic")
    p.add_argument("--strength", type=float, default=1.0, help="stereo360's stereo strength")
    p.add_argument("--stereo-python", default="/workspace/venvs/stereo360/bin/python")
    p.add_argument("--stereo-dir", default="/workspace/stereo360")
    p.add_argument("--pano-only", action="store_true", help="stop before stereo")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    out = Path(a.out)
    work = out.with_name(out.stem + ".work")
    work.mkdir(parents=True, exist_ok=True)
    tags = prompts.split_tags(a.tags)
    if a.tags_file:
        tags += prompts.split_tags(Path(a.tags_file).read_text(encoding="utf-8"))
    tags = list(dict.fromkeys(tags))

    f = forge.Forge(a.forge)
    s = forge.Settings(checkpoint=a.checkpoint, method=a.method, cn_model=a.cn_model,
                       steps=a.steps, cfg=a.cfg, sampler=a.sampler, scheduler=a.scheduler,
                       denoise=a.denoise)
    missing = f.resolve(s)
    if missing:
        print("not ready: " + "; ".join(missing), file=sys.stderr)
        return 2

    def inpaint(image, mask, prompt, negative, seed):
        return f.inpaint(image, mask, prompt, negative, seed, s)

    opt = widen.Options(width=a.width, long_side=a.long_side, target_deg=a.target,
                        max_new=a.max_new, seed=a.seed, quality=a.quality,
                        negative=a.negative)
    src = np.array(Image.open(a.src).convert("RGB"))
    t0 = time.time()
    r = widen.widen(src, tags, inpaint, opt, work)
    Image.fromarray(r.pano).save(work / "pano.png")
    Image.fromarray(r.source_mask).save(work / "source_mask.png")
    log = {"version": __version__, "src": a.src, "argv": sys.argv[1:] if argv is None else argv,
           "method": a.method, "cn_model": s.cn_model, "checkpoint": a.checkpoint,
           "widen": r.log}
    if not a.pano_only:
        log["stereo"] = stereo.to_vr180(work / "pano.png", out, python=a.stereo_python,
                                        checkout=a.stereo_dir, strength=a.strength)
    log["seconds"] = round(time.time() - t0, 1)
    (work / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print("done in %.0fs: %s" % (log["seconds"], work / "pano.png" if a.pano_only else out))
    return 0
