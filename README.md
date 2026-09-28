# pqm-vr180

The VR extension's companion program for PromptQueueManager (PQM): **one flat
image in, one VR180 file out**, run on a PQM pod beside Forge.

    python -m vr180 SRC.png -o OUT_180_LR.jpg --checkpoint waiIllustriousSDXL_v140 \
        --tags "indoors, living room, wooden floor, bookshelf" \
        --subject-tags "1girl, brown hair, white sweater, blue jeans"

Proven on single images in V.1 (2026-09-28): judged by the author in a Quest 3,
"outstanding; I can only barely see the seams when I'm looking for them".

## What it does

1. **Place** the source straight ahead on a sphere, its long side spanning
   `--long-side` degrees. **Default 60.** At 90, a figure is about twice life size
   and a continued body runs under the viewer; the author judged 60 "much better"
   and expects it to vary by image, so it is a setting.
2. **Widen** it to the front hemisphere plus a margin (`--target`, default 100°
   off-axis), one perspective view at a time. The next view is always the one
   that fills the most empty target while staying *mostly known* (`--max-new`,
   45%), so the fill grows outward from the source's edges. Each view is an
   img2img inpaint in Forge on the given checkpoint, conditioned on what is known:
   - `--method noob` (default): the **NoobAI Inpainting ControlNet**, its control
     image the view with the hole pure black, plus the source as a reference
     through the **noobIPA MARK1** IP-Adapter (weight 0.5, steps 20-80%). This is
     Krita AI Diffusion's expand recipe for Illustrious/NoobAI checkpoints, as is
     the pre-fill (Navier-Stokes from the border, then blurred).
   - `--method plain`: the checkpoint's own inpaint, for checkpoints NoobAI's
     models do not fit (V.1 used it with RealVisXL).
   - `--method cn`: ControlNet Union ProMax. **Broken in Forge Neo as PQM pins
     it** (NaN in fp16, noise in bf16), kept to retry against a later Forge.

   Views looking up or down are told so (ceiling or sky, floor or ground) and
   drop the other half's tags. **The source's pixels are never rewritten**; seams
   are blended only on the fill side.
3. **Continue a subject the frame cuts off.** SkyTNT's anime-segmentation finds
   the character; a view that runs up against it is prompted with
   `--subject-tags` and without the no-people negative, and the body it paints
   joins the subject for the next view down. Without `--subject-tags`, no body is
   continued and every view is told "no humans".
4. **Extend a plain background by colour**, with no diffusion (a checkpoint given
   an empty grey floor invents objects on it). `--no-plain-fill` turns this off.
5. **Depth, second eye and VR180** by [stereo360](https://github.com/LeonG-ZA/stereo360)
   at a pinned commit (`--output-mode vr180 --inpaint learned`). `--strength`
   stays at 1.0: it scales disparity on relative depth rather than the eyes'
   separation, and above 1.0 it pinches the centre without changing how big the
   world feels.

Beside the output, `<stem>.work/` keeps the flat panorama, every generated view
and `log.json`: timings, each view's direction, kind (scene, subject, plain),
prompt and negative, and every choice the program made on its own with the reason.

## On a pod

`setup/pod_setup.sh all` installs and verifies everything, given PQM's pod image
(Forge Neo at `:7860`) and this repo at `/workspace/pqm-vr180`: the companion's
venv (numpy, OpenCV, Pillow, onnxruntime), stereo360 at its commit (warmed, so
Depth Pro and LaMa are fetched at install), and four sha256-pinned models:
NoobAI Inpainting (2.5 GB), noobIPA MARK1 (1.4 GB), CLIP-ViT-bigG (3.7 GB) and
anime-seg (176 MB). The checkpoint is the pod's business (PQM provisions it).
Two things about that image, both measured in V.1:

- **Download into `models/` only after Forge answers.** The image's entrypoint
  replaces `models/` with a symlink partway through boot; a file written before
  that is deleted.
- **ControlNet models must be in place before Forge starts**, or Forge restarted
  after they land: Forge lists `models/ControlNet` once, at startup, and its API
  has no refresh. `pod_setup.sh models` says so by name when it happens.

Measured (V.1, RTX 5090): install about 7 minutes after Forge answers; an image
80-120 s (10-11 views at 6.5 s, about 13 s of CPU, 9 s of stereo), about $0.03.

## Tests

    py -3.13 -m venv .venv && .venv/Scripts/pip install -r requirements.txt pytest
    .venv/Scripts/python -m pytest

They run offline with a fake Forge: geometry, planning, that a widening fills the
front and leaves every source pixel exactly as placed, subject continuation,
plain fill, and the Forge payloads.
