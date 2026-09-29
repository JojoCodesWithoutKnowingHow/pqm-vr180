"""One image in, one VR180 file out.

    python -m vr180 SRC.png -o OUT_180_LR.jpg --tags "indoors, wooden floor, couch" \\
        --checkpoint waiIllustriousSDXL_v140 --subject-tags "1girl, brown hair, coat"

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

from . import __version__, forge, placement, prompts, stereo, widen

SEG_MODEL = "/workspace/models/anime-seg/isnetis.onnx"
DEPTH_MODELS = {"any-b": ["--depth-backend", "depth-anything"],
                "depth-pro": ["--depth-backend", "depth-pro"],
                "da3": ["--depth-backend", "depth-anything-v3"]}


def parse(argv=None):
    p = argparse.ArgumentParser(prog="vr180", description=__doc__.split("\n\n")[0])
    p.add_argument("src")
    p.add_argument("-o", "--out", required=True, help="the VR180 file (name it *_180_LR.jpg)")
    p.add_argument("--tags", default="", help="what surrounds the subject, comma-separated")
    p.add_argument("--tags-file", help="a file of tags; appended to --tags")
    p.add_argument("--subject-tags", default="",
                   help="the subject's own tags; given, a subject the frame cuts off is "
                        "continued with them (empty: never continue a body)")
    p.add_argument("--segment-model", default=SEG_MODEL, help="anime-seg isnetis.onnx")
    p.add_argument("--checkpoint", required=True, help="Forge checkpoint for the fill")
    p.add_argument("--method", choices=("noob", "plain", "cn"), default="noob",
                   help="noob: NoobAI Inpainting ControlNet (Illustrious/NoobAI checkpoints). "
                        "plain: the checkpoint's own inpaint. cn: ControlNet Union ProMax -- "
                        "NaN or noise in Forge Neo as pinned (V.1)")
    p.add_argument("--no-reference", dest="reference", action="store_false",
                   help="do not give the source to the IP-Adapter as a reference (on by "
                        "default with --method noob; noobIPA is for Illustrious/NoobAI)")
    p.add_argument("--ipa-model", default="noobipa")
    p.add_argument("--ipa-weight", type=float, default=0.5)
    p.add_argument("--prompt-mode", choices=("tags", "minimal"), default="tags")
    p.add_argument("--no-plain-fill", action="store_true",
                   help="generate plain backgrounds too, instead of extending the colour")
    p.add_argument("--denoise", type=float, default=None,
                   help="default: 1.0 for noob, 0.95 otherwise")
    p.add_argument("--cn-model", default="", help="the inpaint ControlNet's name")
    p.add_argument("--forge", default="http://127.0.0.1:7860")
    p.add_argument("--long-side", default="60",
                   help="degrees the source's long side spans (default 60: at 90 a figure is "
                        "about twice life size and its body runs under the viewer; V.1, judged "
                        "in a headset). 'ratio': by aspect ratio; 'shot': size a whole figure to "
                        "life size at 2 m, else by ratio; 'camera': the field of view Depth Pro "
                        "estimates for the image (V.1 experiment; 60 if it fails)")
    p.add_argument("--width", type=int, default=4096, help="equirect width (VR180 is W/2 per eye)")
    p.add_argument("--target", type=float, default=100.0, help="fill out to this angle off-axis")
    p.add_argument("--max-new", type=float, default=0.45)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--taper", action="store_true",
                   help="foveated: views far from the source get fewer steps (28 within 40 deg, "
                        "easing to 12 at 100) and beyond 60 deg render at 768 px; views that "
                        "continue the subject always get full work")
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--quality", default=prompts.QUALITY,
                   help="quality words the checkpoint expects (default: anime/Illustrious)")
    p.add_argument("--negative", default=prompts.NEGATIVE)
    p.add_argument("--sampler", default="Euler a")
    p.add_argument("--scheduler", default="Automatic")
    p.add_argument("--strength", default="1.0",
                   help="stereo360's stereo strength; a comma list writes one file per value, "
                        "the first under OUT, the rest as <stem>_s<value>_180_LR.jpg. It scales "
                        "disparity on relative depth, not the eyes' separation: above 1.0 it "
                        "pinches the centre and pulls focus close without changing how big the "
                        "world feels (V.1, in a headset). Leave it at 1.0")
    # Defaults from the author's headset (V.1 tuning): region + -2 + Depth Anything V2
    # Base gave the characters the best depth; --gradient-limit 0 corrupted the
    # generated areas; --depth-tiles made no difference with Depth Pro.
    p.add_argument("--depth-norm", choices=("global", "region"), default="region",
                   help="where the depth range is measured: the source's region (default; the "
                        "floor under the viewer then stops taking it) or stereo360's whole sphere")
    p.add_argument("--fg-scale", type=float, default=-2.0,
                   help="IW3's foreground scale, -3..3 (default -2, the author's IW3 setting)")
    p.add_argument("--knee", type=float, default=0.85)
    p.add_argument("--depth-model", choices=tuple(DEPTH_MODELS), default="any-b",
                   help="stereo360's depth model: Depth Anything V2 Base (default; IW3's Any_B), "
                        "Apple Depth Pro, or Depth Anything V3")
    p.add_argument("--stereo-args", default="", help="more stereo360 arguments, passed as they are")
    p.add_argument("--stereo-python", default="/workspace/venvs/stereo360/bin/python")
    p.add_argument("--stereo-dir", default="/workspace/stereo360")
    p.add_argument("--pano-only", action="store_true", help="stop before stereo")
    return p.parse_args(argv)


def _strength_out(out: Path, value: str, first: bool) -> Path:
    if first:
        return out
    stem = out.stem[:-len("_180_LR")] if out.stem.endswith("_180_LR") else out.stem
    return out.with_name("%s_s%s_180_LR%s" % (stem, value, out.suffix))


def main(argv=None) -> int:
    a = parse(argv)
    out = Path(a.out)
    work = out.with_name(out.stem + ".work")
    work.mkdir(parents=True, exist_ok=True)
    tags = prompts.split_tags(a.tags)
    if a.tags_file:
        tags += prompts.split_tags(Path(a.tags_file).read_text(encoding="utf-8"))
    tags = list(dict.fromkeys(tags))
    subject_tags = tuple(prompts.split_tags(a.subject_tags))

    f = forge.Forge(a.forge)
    denoise = a.denoise if a.denoise is not None else (1.0 if a.method == "noob" else 0.95)
    s = forge.Settings(checkpoint=a.checkpoint, method=a.method, cn_model=a.cn_model,
                       ipa_model=a.ipa_model if a.reference and a.method == "noob" else "",
                       ipa_weight=a.ipa_weight,
                       steps=a.steps, cfg=a.cfg, sampler=a.sampler, scheduler=a.scheduler,
                       denoise=denoise)
    missing = f.resolve(s)
    segment = None
    if subject_tags or a.long_side in ("shot", "camera"):
        if not Path(a.segment_model).exists():
            missing.append("no segmentation model at %s" % a.segment_model)
        else:
            from .subject import Segmenter
            segment = Segmenter(a.segment_model)
    if missing:
        print("not ready: " + "; ".join(missing), file=sys.stderr)
        return 2

    def inpaint(image, mask, prompt, negative, seed, control=None, reference=None, steps=None):
        return f.inpaint(image, mask, prompt, negative, seed, s, control=control,
                         reference=reference, steps=steps)

    opt = widen.Options(width=a.width, target_deg=a.target,
                        max_new=a.max_new, seed=a.seed, quality=a.quality,
                        negative=a.negative, subject_tags=subject_tags,
                        plain_fill=not a.no_plain_fill, prompt_mode=a.prompt_mode,
                        reference=a.reference and a.method == "noob",
                        steps=a.steps, taper=a.taper)
    src = np.array(Image.open(a.src).convert("RGB"))
    h, w = src.shape[:2]
    if a.long_side == "ratio":
        long_side, why = placement.by_ratio(w, h)
    elif a.long_side == "shot":
        seg = segment(src) if segment else None
        long_side, why = placement.by_shot(w, h, seg, widen._cut_edges(seg) if seg is not None else [])
    elif a.long_side == "camera":
        seg = segment(src) if segment else None
        est = stereo.estimate_camera(Path(a.src), python=a.stereo_python, checkout=a.stereo_dir,
                                     subject=seg, work=work)
        long_side, why = placement.by_camera(w, h, est)
    else:
        long_side, why = float(a.long_side), "given"
    opt.long_side = long_side
    print("placement: %.1f deg (%s)" % (long_side, why))
    t0 = time.time()
    r = widen.widen(src, tags, inpaint, opt, work, segment=segment)
    Image.fromarray(r.pano).save(work / "pano.png")
    Image.fromarray(r.source_mask).save(work / "source_mask.png")
    log = {"version": __version__, "src": a.src, "argv": sys.argv[1:] if argv is None else argv,
           "method": a.method, "cn_model": s.cn_model, "ipa_model": s.ipa_model,
           "checkpoint": a.checkpoint, "denoise": denoise,
           "placement": {"long_side": long_side, "why": why}, "depth_model": a.depth_model,
           "widen": r.log}
    if not a.pano_only:
        log["stereo"] = []
        for i, value in enumerate(v.strip() for v in a.strength.split(",") if v.strip()):
            target = _strength_out(out, value, i == 0)
            depth = stereo.Depth(norm=a.depth_norm, knee=a.knee, fg_scale=a.fg_scale,
                                 strength=float(value),
                                 extra=[*DEPTH_MODELS[a.depth_model],
                                        *stereo.Depth.parse_extra(a.stereo_args)])
            info = stereo.to_vr180(work / "pano.png", target, python=a.stereo_python,
                                   checkout=a.stereo_dir, depth=depth,
                                   source_mask=work / "source_mask.png")
            log["stereo"].append(dict(info, strength=float(value), out=target.name))
    log["seconds"] = round(time.time() - t0, 1)
    (work / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print("done in %.0fs: %s" % (log["seconds"], work / "pano.png" if a.pano_only else out))
    return 0
