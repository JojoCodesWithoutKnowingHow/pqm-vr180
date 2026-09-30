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
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import (__version__, flatext, forge, grow, inlayout, layout, placement, post, prompts,
               seam as seams,
               sphere, stereo, widen)

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
    p.add_argument("--long-side", default="moge",
                   help="degrees the source's long side spans. Default 'moge': the field of view "
                        "MoGe-2 estimates for the image (the author: more realistic on every "
                        "image, V.1), or 60 when it fails or the background is plain (nothing to "
                        "measure). A number overrides it -- realistic is not always the artistic "
                        "choice. Also 'ratio', 'shot', and 'camera' (Depth Pro), all experiments")
    p.add_argument("--width", type=int, default=4096, help="equirect width (VR180 is W/2 per eye)")
    p.add_argument("--target", type=float, default=100.0, help="fill out to this angle off-axis")
    p.add_argument("--max-new", type=float, default=0.45)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--no-taper", dest="taper", action="store_false",
                   help="turn off the foveated taper (on by default: views far from the source "
                        "get fewer steps, 28 within 40 deg easing to 12 at 100, and render at "
                        "768 px beyond 60 deg; subject views always get full work). V.1: 14-30%% "
                        "faster per image, and the author could not see it in the headset")
    p.add_argument("--view-fov", type=float, default=90.0,
                   help="each view's field of view; narrower views put more pixels on each "
                        "degree (sharper fill, more views)")
    p.add_argument("--view-px", type=int, default=1024)
    # Both on by default: the author judged detail match + rim softening the best at the
    # seam in the headset (V.1, round 3). The seam was a sharpness step, not a colour one.
    p.add_argument("--no-detail-match", dest="detail_match", action="store_false",
                   help="do not sharpen the fill to the source's fine-detail level")
    p.add_argument("--soften-rim", type=int, default=12, metavar="PX",
                   help="soften the source's outermost PX pixels toward the fill (default 12; "
                        "0 turns it off; touches the source's edge, which the author allowed)")
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
    p.add_argument("--est-python", default="/workspace/venvs/est/bin/python",
                   help="the venv with MoGe-2, for --long-side moge")
    p.add_argument("--stereo-dir", default="/workspace/stereo360")
    p.add_argument("--pano-only", action="store_true", help="stop before stereo")
    # The outpaint work (round 1): each off by default until judged in the headset.
    p.add_argument("--join", choices=("blend", "hard"), default="blend",
                   help="blend: 0.4.0 (a 32 px cross-fade into earlier fill, a painted body "
                        "may be repainted). hard: a painted body is kept, scene joins blend over "
                        "8 px, the body's zone fans out, and it grows at --grow-threshold")
    p.add_argument("--zone-spread", type=float, default=30.0,
                   help="with --join hard: degrees the continuation zone widens by")
    p.add_argument("--grow-threshold", type=float, default=0.3,
                   help="with --join hard: anime-seg threshold for a continued body (0.5 finds "
                        "a torso but often not a lone limb)")
    p.add_argument("--flat-extend", type=float, default=0.0, metavar="F",
                   help="finish a body the frame cuts off on the flat picture first: grow the "
                        "canvas F times along each cut axis, symmetrically, in one inpaint "
                        "(0: off). Needs --subject-tags")
    p.add_argument("--layout", choices=("none", "fisheye"), default="none",
                   help="fisheye: lay the whole front out in one generation first, then refine "
                        "each scene view over it at --layout-denoise")
    p.add_argument("--layout-denoise", type=float, default=0.7)
    p.add_argument("--layout-deg", type=float, default=100.0,
                   help="the layout fisheye reaches this far off straight ahead")
    p.add_argument("--layout-px", type=int, default=1024)
    p.add_argument("--layout-hires", type=int, default=0, metavar="PX",
                   help="scale the layout up to PX and refine it in tiles (the hires fix), so "
                        "the scene reaches the panorama's density (0: off; 2048 ~ the panorama)")
    p.add_argument("--layout-hires-denoise", type=float, default=0.4)
    p.add_argument("--layout-strong", action="store_true",
                   help="weight the layout's fisheye words (round 1: one layout came back as "
                        "an ordinary wide-angle picture)")
    # Round 2 (the author's round-1 verdicts): off by default until judged.
    p.add_argument("--extend-side", type=float, default=0.0, metavar="G",
                   help="grow only the side(s) the frame cuts the body, each by G times the "
                        "source's size across it, keeping its optical centre (0: off). Round 1's "
                        "symmetric --flat-extend gave a cut side only +25%%")
    p.add_argument("--extend-in-layout", action="store_true",
                   help="round 6 (the author's idea): lay the scene out around the source first, "
                        "then inpaint the body's continuation inside the layout's fisheye -- the "
                        "fan where the frame cuts her the only mask, the room in view -- and "
                        "refine it at full resolution (--extend-refine). Needs --layout fisheye")
    p.add_argument("--extend-regional", action="store_true",
                   help="round 7 (the author's idea), with --extend-in-layout: generate the "
                        "layout with two regional prompts (Forge Couple) -- her tags over her "
                        "body and its continuation, the scene with no people elsewhere -- "
                        "instead of inpainting her into a finished layout")
    p.add_argument("--subject-framing", default="",
                   help="the source's pose and framing words (e.g. 'sitting, crossed legs, "
                        "cowboy shot'); with --extend-regional they go into her region's prompt "
                        "so it knows where her body ends (round 7)")
    p.add_argument("--bridge", type=int, default=0, metavar="PX",
                   help="round 23, with --extend-in-layout: before ADetailer, generate anew a "
                        "band PX px out from each cut edge (and 24 px in) so the model redraws "
                        "the join between the source's legs and the layout's (r19: a leg met "
                        "the source's ~30 px off)")
    p.add_argument("--auto-pipeline", action="store_true",
                   help="round 23 (the author's pick): grow her body stepwise first (J's "
                        "extension); if her body still reaches a grown edge after any step, "
                        "her limbs run out of frame -- use L (--extend-in-layout "
                        "--layout-full-prompt, with --adetail); if not, she is barely cut -- "
                        "keep the extension and use J (--layout-owns-scenery, no ADetailer). "
                        "r18-22: L drew a giant second Yamato whatever was tried; J did not")
    p.add_argument("--layout-omit", default="", metavar="TAGS",
                   help="round 22, with --layout-full-prompt: subject tags left out of the "
                        "layout's prompt only (ADetailer keeps them), e.g. 'breasts, nipples' "
                        "(r18-21: the layout filled the floor in front of Yamato with a giant "
                        "close-up of her chest)")
    p.add_argument("--layout-close-negative", action="store_true",
                   help="round 21, with --layout-full-prompt: add close-up / giantess / "
                        "multiple views to the layout's negative (r18-20: a giant second "
                        "Yamato over the floor in front of her, on every seed)")
    p.add_argument("--layout-her-region", type=float, default=0.0, metavar="R",
                   help="round 21, with --extend-in-layout (instead of --layout-full-prompt): "
                        "the layout draws her with her tags and pose words only within R "
                        "crossing widths of where the frame cuts her, and scenery with no "
                        "people beyond (r18: the full prompt everywhere drew a second, giant "
                        "Yamato over the floor in front of her); use with --adetail")
    p.add_argument("--layout-owns-scenery", action="store_true",
                   help="round 9 (the author's pick): with --extend-side and --layout fisheye, the "
                        "extension gives only her body; the layout is made with only the source "
                        "and her body fixed, so the floor and furniture around her legs come from "
                        "the same pass as the room; the picture keeps only the source and her "
                        "body, and her outline is seam-repainted")
    p.add_argument("--layout-full-prompt", action="store_true",
                   help="round 18 (the author's pipeline), with --extend-in-layout: the layout "
                        "is drawn round the source alone with the full prompt (her, her pose, "
                        "the scene), so it draws her body's continuation itself; no extension "
                        "step; use with --adetail")
    p.add_argument("--adetail", type=float, default=0.0, metavar="D",
                   help="round 16 (the author's idea), with --extend-in-layout: an ADetailer pass "
                        "-- her whole figure, source included, cropped and repainted once at D "
                        "with the full prompt; then the source restored (0: off)")
    p.add_argument("--source-fade", type=int, default=32, metavar="PX",
                   help="with --layout-owns-scenery: the original fades into the layout over "
                        "this many px inside its rectangle")
    p.add_argument("--keep-threshold", type=float, default=0.15,
                   help="with --layout-owns-scenery: the segmenter threshold for what counts as "
                        "her inside the extension's painted areas")
    p.add_argument("--inner-seam-outer", type=int, default=24, metavar="PX",
                   help="how far past the source's edge the inner seam repaint reaches (round 10: "
                        "24 px hid a sharpness step, not the layout's different shading)")
    p.add_argument("--inner-seam-denoise", type=float, default=0.0,
                   help="the inner seam repaint's denoise (0: --seam-repaint's)")
    p.add_argument("--extend-step", type=float, default=0.3,
                   help="grow a cut side this fraction of the source's size at a time, and stop "
                        "once the body no longer reaches the new edge (round 2: growing it all at "
                        "once painted a second body in the empty space)")
    p.add_argument("--extend-max-deg", type=float, default=70.0,
                   help="the grown edge stays within this angle of straight ahead")
    p.add_argument("--extend-refine", type=float, default=0.35,
                   help="denoise of the full-resolution pass over the grown pixels and the "
                        "source's outer 8 px (0: off)")
    p.add_argument("--order", choices=("body-first", "scene-first"), default="body-first",
                   help="scene-first (with --layout fisheye and --extend-side): lay the scene "
                        "out, then paint the body over it in a fan from the cut edge")
    p.add_argument("--compose", type=float, default=-1.0, metavar="D",
                   help="compose (G): with --layout fisheye, the laid-out scene is final; every "
                        "view is a detail pass over it at denoise D, no inpaint ControlNet, no "
                        "view with the subject's tags (0: off). Round 2: E repainted the layout "
                        "at 0.7 and inked a line at each view's edge; F grew new bodies. 0: no "
                        "detail pass, the (hires) layout as it is; below 0: off")
    p.add_argument("--seam-repaint", type=float, default=0.0, metavar="D",
                   help="repaint a narrow band across the picture's edge on the sphere at this "
                        "denoise (0: off). Use with --soften-rim 0")
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

    def inpaint(image, mask, prompt, negative, seed, control=None, reference=None, steps=None,
                denoise=None, touch_up=False, regions=None):
        return f.inpaint(image, mask, prompt, negative, seed, s, control=control,
                         reference=reference, steps=steps, denoise=denoise, touch_up=touch_up,
                         regions=regions)

    opt = widen.Options(width=a.width, target_deg=a.target,
                        max_new=a.max_new, seed=a.seed, quality=a.quality,
                        negative=a.negative, subject_tags=subject_tags,
                        plain_fill=not a.no_plain_fill, prompt_mode=a.prompt_mode,
                        reference=a.reference and a.method == "noob",
                        steps=a.steps, taper=a.taper, view_fov=a.view_fov, view_px=a.view_px,
                        join=a.join, zone_spread_deg=a.zone_spread,
                        layout_denoise=a.compose if a.compose >= 0 else a.layout_denoise,
                        compose=a.compose >= 0)
    if a.compose >= 0 and a.layout != "fisheye":
        print("--compose needs --layout fisheye: it only details the laid-out scene",
              file=sys.stderr)
        return 2
    src = np.array(Image.open(a.src).convert("RGB"))
    h, w = src.shape[:2]
    if a.long_side == "ratio":
        long_side, why = placement.by_ratio(w, h)
    elif a.long_side == "shot":
        seg = segment(src) if segment else None
        long_side, why = placement.by_shot(w, h, seg, widen._cut_edges(seg) if seg is not None else [])
    elif a.long_side == "moge" and prompts.setting(tags)[0] == "plain":
        # A plain background gives MoGe nothing to measure (V.1: 34 deg on a_plain).
        long_side, why = 60.0, "plain background, nothing for MoGe to measure; 60"
    elif a.long_side == "moge":
        est = stereo.estimate_moge(Path(a.src), python=a.est_python, work=work)
        long_side, why = placement.by_moge(w, h, est)
    elif a.long_side == "camera":
        seg = segment(src) if segment else None
        est = stereo.estimate_camera(Path(a.src), python=a.stereo_python, checkout=a.stereo_dir,
                                     subject=seg, work=work)
        long_side, why = placement.by_camera(w, h, est)
    else:
        long_side, why = float(a.long_side), "given"
    print("placement: %.1f deg (%s)" % (long_side, why))
    t0 = time.time()
    where = prompts.setting(tags)[0]
    ref = src if (a.reference and a.method == "noob") else None
    opt.long_side = long_side
    extra_log = {}
    W = a.width

    def centred(img):
        return sphere.place(img, W, long_side)

    picture, place, orig_mask, lay = src, centred, None, None
    seg_picture = segment
    grown_g, silhouette = None, None
    seg_src = segment(src) if (segment is not None and subject_tags) else None
    cut = widen._cut_edges(seg_src) if seg_src is not None else []
    lay_kw = dict(max_deg=a.layout_deg, S=a.layout_px, quality=a.quality, reference=ref,
                  steps=a.steps, work=work, strong=a.layout_strong, hires=a.layout_hires,
                  hires_denoise=a.layout_hires_denoise)

    def grown_place(g):
        """``place`` for a grown canvas that keeps its source's optical centre."""
        def place_(img):
            pano_, mask_ = sphere.place_focal(img, W, g.focal, *g.centre)
            ih, iw = img.shape[:2]
            cx, cy = g.centre
            fov = (math.degrees(math.atan(cx / g.focal) + math.atan((iw - cx) / g.focal)),
                   math.degrees(math.atan(cy / g.focal) + math.atan((ih - cy) / g.focal)))
            return pano_, mask_, fov
        return place_

    def source_in(g):
        placed_, pm_ = sphere.place_focal(g.source_mask(), W, g.focal, *g.centre)
        return ((placed_ > 127) & (pm_ > 0)).astype(np.uint8) * 255

    pre_g = None
    if a.auto_pipeline and seg_src is not None and a.layout == "fisheye" and cut:
        ext_tags = list(dict.fromkeys(list(subject_tags) + prompts.pose_words(a.subject_framing)))
        pre_g = grow.extend_side(src, cut, a.extend_side or 1.0, long_side, inpaint, ext_tags,
                                 tags, where, a.seed + 5000, quality=a.quality, reference=ref,
                                 steps=a.steps, max_deg=a.extend_max_deg,
                                 refine_denoise=a.extend_refine, step=a.extend_step,
                                 segment=segment)
        runs_out = pre_g is not None and any(p_.get("body_reaches_edge")
                                             for p_ in pre_g.log.get("passes", []))
        extra_log["auto_pipeline"] = {"chose": "L" if runs_out else "J",
                                      "reaching_steps": sum(bool(p_.get("body_reaches_edge"))
                                                            for p_ in (pre_g.log.get("passes", [])
                                                                       if pre_g else []))}
        print("auto pipeline: %s" % extra_log["auto_pipeline"])
        if runs_out:
            a.extend_in_layout, a.layout_full_prompt, a.layout_owns_scenery = True, True, False
            pre_g = None
        else:
            a.extend_in_layout, a.layout_full_prompt, a.layout_owns_scenery = False, False, True
            a.adetail = 0.0
            a.extend_side = a.extend_side or 1.0
    if a.extend_in_layout and seg_src is not None and a.layout == "fisheye":
        pano0, m0, _f = centred(src)
        region = None
        if a.extend_regional or a.layout_her_region > 0:
            her_f = inlayout.her_on_fisheye(src, seg_src, cut, long_side, W, a.layout_px,
                                            a.layout_deg, a.extend_side or 1.0,
                                            a.extend_max_deg,
                                            reach=a.layout_her_region or 1.5)
            if her_f is not None:
                # Her pose and framing, and the same projection words as the scene
                # (round 7: her region had neither, and was filled with her). Round
                # 21: the pose words only -- crop words stretched Nami in round 8.
                framing = (prompts.pose_words(a.subject_framing) if a.layout_her_region > 0
                           else prompts.split_tags(a.subject_framing))
                lens = ["(fisheye:1.3)", "fisheye lens"] if a.layout_strong else ["fisheye"]
                her_p = prompts.subject_prompt(lens + list(subject_tags) + framing, tags, where,
                                               0.0, a.quality)
                region = (her_p, her_f)
        full_p = None
        if a.layout_full_prompt:
            lens = ["(fisheye:1.3)", "fisheye lens"] if a.layout_strong else ["fisheye"]
            omit = {t.lower() for t in prompts.split_tags(a.layout_omit)}
            full_p = prompts.subject_prompt(lens + [t for t in subject_tags
                                                    if t.lower() not in omit]
                                            + prompts.pose_words(a.subject_framing),
                                            tags, where, 0.0, a.quality)
        lay, extra_log["layout"], fish = layout.make_layout(
            pano0, m0 > 0, tags, where, inpaint, a.seed + 7000, return_fisheye=True,
            region=region, full_prompt=full_p,
            full_negative=(prompts.CLOSE_NEGATIVE + ", " + prompts.SUBJECT_NEGATIVE
                           if a.layout_close_negative else None), **lay_kw)
        res = inlayout.extend(src, seg_src, cut, long_side, W, fish, a.layout_deg, inpaint,
                              subject_tags, tags, where, a.seed + 5000, quality=a.quality,
                              reference=ref, steps=a.steps, grow_frac=a.extend_side or 1.0,
                              max_side_deg=a.extend_max_deg,
                              refine_denoise=a.extend_refine or 0.5, work=work,
                              paint=region is None and full_p is None, segment=segment,
                              framing=prompts.pose_words(a.subject_framing),
                              adetail=a.adetail, bridge_px=a.bridge)
        if res is not None:
            g, fish, extra_log["extend_in_layout"] = res
            img_, cover_ = layout.from_fisheye(fish, W, a.layout_deg)
            lay = np.zeros_like(pano0)
            lay[cover_] = img_[cover_]
            lay = widen.fill_back(lay, cover_)
            picture, place, orig_mask = g.image, grown_place(g), source_in(g)
            Image.fromarray(picture).save(work / "flat_extended.png")
            print("extend in layout: cut at %s, added %s" % (", ".join(cut), g.log["added"]))
        else:
            extra_log["extend_in_layout"] = {"skipped": "the frame cuts no subject", "cut": cut}
    elif a.extend_side > 0 and seg_src is not None:
        scene_of = None
        if a.order == "scene-first" and a.layout == "fisheye":
            # The author's order: the scene first, then the body painted over it.
            pano0, m0, _f = centred(src)
            lay, extra_log["layout"] = layout.make_layout(pano0, m0 > 0, tags, where, inpaint,
                                                          a.seed + 7000, **lay_kw)
            f0 = grow.focal(w, h, long_side)

            def scene_of(cw, ch, cx, cy, lay=lay, f0=f0):
                return sphere.flat_of(lay, cw, ch, f0, cx, cy)
        ext_tags = list(dict.fromkeys(list(subject_tags) + prompts.pose_words(a.subject_framing)))
        g = pre_g if pre_g is not None else grow.extend_side(
            src, cut, a.extend_side, long_side, inpaint, ext_tags, tags, where, a.seed + 5000,
            quality=a.quality, reference=ref, steps=a.steps, max_deg=a.extend_max_deg,
            refine_denoise=a.extend_refine, step=a.extend_step, segment=segment,
            scene_of=scene_of)
        if g is not None:
            picture = g.image
            grown_g = g
            # The sphere views continue only the body we started with, never a
            # figure the extension happened to paint (round 2b).
            body_mask = g.body

            def seg_picture(img, body_mask=body_mask):
                return body_mask if img.shape[:2] == body_mask.shape else segment(img)

            place, orig_mask = grown_place(g), source_in(g)
            Image.fromarray(picture).save(work / "flat_extended.png")
            extra_log["extend_side"] = g.log
            print("extend side: cut at %s, added %s, canvas %s, %s"
                  % (", ".join(cut), g.log["added"], g.log["canvas"], g.log["order"]))
        else:
            extra_log["extend_side"] = {"skipped": "the frame cuts no subject", "cut": cut}
    elif a.flat_extend > 0 and seg_src is not None:
        ext = flatext.extend(src, cut, a.flat_extend, inpaint, subject_tags, tags, where,
                             a.seed + 5000, quality=a.quality, reference=ref, steps=a.steps)
        if ext is not None:
            picture = ext.image
            picture_long = flatext.ext_long_side(w, h, long_side, *ext.image.shape[1::-1])

            def place(img, lng=picture_long):
                return sphere.place(img, W, lng)

            placed, _m, _f = place(ext.source_mask())
            orig_mask = ((placed > 127) & (_m > 0)).astype(np.uint8) * 255
            Image.fromarray(picture).save(work / "flat_extended.png")
            extra_log["flat_extend"] = dict(ext.log, long_side=round(picture_long, 2))
            print("flat extend: cut at %s, canvas %s, long side %.1f deg"
                  % (", ".join(cut), ext.log["canvas"], picture_long))
        else:
            extra_log["flat_extend"] = {"skipped": "the frame cuts no subject"}
    elif a.flat_extend > 0 or a.extend_side > 0:
        extra_log["extend"] = {"skipped": "no subject tags or no segmenter"}
    if (a.layout == "fisheye" and lay is None and a.layout_owns_scenery and grown_g is not None
            and grown_g.body is not None):
        # J: only the source and her body are fixed; the layout paints the rest of
        # the grown canvas -- the floor under her feet, the furniture by her legs --
        # in the same pass as the room (round 8: two separately made floors met at
        # the grown picture's edge).
        gx, gy, gw, gh = grown_g.rect
        # Her tracked body, and inside the areas the extension painted with her
        # tags, whatever the segmenter sees as her at a low threshold (round 10:
        # keeping the whole fans kept the extension's own floor and furniture
        # too; keeping only the outline lost a hand and feet in round 9).
        her = grown_g.body.copy()
        if grown_g.fans is not None and grown_g.fans.any():
            her |= grown_g.fans & segment(picture, threshold=a.keep_threshold)
        keep = cv2.dilate(her.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        keep[gy:gy + gh, gx:gx + gw] = True
        pano0, m0, _f = place(picture)
        # The layout is told exactly what is kept. dev17 inset it ~10 px to hide the
        # original's rectangle lines, but those came from the extension's rim strip
        # (fixed in dev18), and the gap let the layout continue her body on its own:
        # a second pair of feet under the extension's (round 13, Fubuki's sofa).
        kp, _km, _kf = place((keep * 255).astype(np.uint8))
        known = (kp > 127) & (m0 > 0)
        lay, extra_log["layout"] = layout.make_layout(pano0, known, tags, where, inpaint,
                                                      a.seed + 7000, **lay_kw)
        ch, cw = picture.shape[:2]
        scene = sphere.flat_of(lay, cw, ch, grown_g.focal, *grown_g.centre)
        # Her body: kept, with a soft edge of a few pixels (her body grown 5 px, so
        # the fade falls on the layout's side of her outline). The original: faded
        # into the layout over ``--source-fade`` px inside its rectangle, where the
        # layout holds its own softer copy of it -- a 4 px switch at the rectangle's
        # edge read as a hard line tracing it (round 13, worst on Fubuki's bed).
        body_keep = cv2.dilate(her.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(np.float32)
        rect = np.zeros(keep.shape, np.uint8)
        rect[gy:gy + gh, gx:gx + gw] = 1
        ramp = np.clip(cv2.distanceTransform(rect, cv2.DIST_L2, 5) / max(a.source_fade, 1), 0, 1)
        wk = np.maximum(np.clip(cv2.GaussianBlur(body_keep, (0, 0), 2.0), 0, 1), ramp)[..., None]
        picture = (picture.astype(np.float32) * wk + scene.astype(np.float32) * (1 - wk)
                   ).round().astype(np.uint8)
        if a.adetail > 0:
            # The ADetailer pass over her whole figure in J's composed picture (round
            # 17, the author: round 16 put it on round 8's regional layout, which
            # stretched Nami). The source is faded back in afterwards.
            her_p = prompts.subject_prompt(ext_tags, tags, where, 0.0, a.quality)
            extra_log["adetail"] = inlayout.adetail(picture, grown_g.rect, segment, inpaint, her_p,
                                                    prompts.SUBJECT_NEGATIVE, a.seed + 900,
                                                    a.adetail, steps=a.steps)
            src_back = np.clip(cv2.distanceTransform(rect, cv2.DIST_L2, 5)
                               [gy:gy + gh, gx:gx + gw] / 32, 0, 1)[..., None]
            reg = picture[gy:gy + gh, gx:gx + gw].astype(np.float32)
            picture[gy:gy + gh, gx:gx + gw] = (src.astype(np.float32) * src_back
                                               + reg * (1 - src_back)).round().astype(np.uint8)
        Image.fromarray(picture).save(work / "flat_extended.png")
        silhouette = known
        extra_log["layout_owns_scenery"] = {"kept_frac": round(float(keep.mean()), 3)}
    elif a.layout == "fisheye" and lay is None:
        pano0, m0, _f = place(picture)
        lay, extra_log["layout"] = layout.make_layout(pano0, m0 > 0, tags, where, inpaint,
                                                      a.seed + 7000, **lay_kw)
    grow_segment = None
    if segment is not None:
        grow_segment = lambda img: segment(img, threshold=a.grow_threshold)  # noqa: E731
    r = widen.widen(picture, tags, inpaint, opt, work, segment=seg_picture,
                    grow_segment=grow_segment, layout=lay, place=place)
    src_mask = orig_mask if orig_mask is not None else r.source_mask
    seam = {}
    if a.seam_repaint > 0:
        if subject_tags:
            sp = prompts.subject_prompt(list(subject_tags), tags, where, 0.0, a.quality)
            sn = prompts.SUBJECT_NEGATIVE
        else:
            sp = prompts.view_prompt(tags, where, 0.0, a.quality)
            sn = prompts.view_negative(a.negative, where, 0.0)
        seam["repaint"] = seams.repaint(r.pano, r.source_mask > 0, inpaint, sp, sn,
                                        a.seed + 9000, denoise=a.seam_repaint, steps=a.steps)
        if orig_mask is not None:
            # Round 3: a straight line where the source met its grown side (the
            # full-resolution pass repaints only 8 px of the source's rim).
            seam["repaint_inner"] = seams.repaint(
                r.pano, orig_mask > 0, inpaint, sp, sn, a.seed + 9500,
                denoise=a.inner_seam_denoise or a.seam_repaint, outer=a.inner_seam_outer,
                steps=a.steps)
        if silhouette is not None:
            # Her outline, where her body meets the layout's scenery (J).
            seam["repaint_silhouette"] = seams.repaint(r.pano, silhouette, inpaint, sp, sn,
                                                       a.seed + 9700, denoise=0.35, inner=4,
                                                       outer=12, steps=a.steps)
    seam["ratio_before"] = round(post.detail_ratio(r.pano, src_mask > 0), 3)
    if a.detail_match:
        r.pano, seam["detail_amount"] = post.detail_match(r.pano, src_mask > 0)
    if a.soften_rim:
        r.pano = post.soften_rim(r.pano, src_mask > 0, a.soften_rim)
        seam["rim_px"] = a.soften_rim
    seam["ratio_after"] = round(post.detail_ratio(r.pano, src_mask > 0), 3)
    Image.fromarray(r.pano).save(work / "pano.png")
    Image.fromarray(src_mask).save(work / "source_mask.png")
    log = {"version": __version__, "src": a.src, "argv": sys.argv[1:] if argv is None else argv,
           "method": a.method, "cn_model": s.cn_model, "ipa_model": s.ipa_model,
           "checkpoint": a.checkpoint, "denoise": denoise,
           "placement": {"long_side": long_side, "why": why}, "depth_model": a.depth_model,
           "seam": seam, **extra_log,
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
