# pqm-vr180

The VR extension's companion program for PromptQueueManager (PQM): **one flat
image in, one VR180 file out**, run on a PQM pod beside Forge.

    python -m vr180 SRC.png -o OUT_180_LR.jpg --checkpoint waiIllustriousSDXL_v140 \
        --tags "indoors, wooden floor, couch"

## What it does

1. **Place** the source straight ahead on a sphere, its long side spanning
   `--long-side` degrees (default 90).
2. **Widen** it to the front hemisphere plus a margin (`--target`, default 100°
   off-axis), one perspective view at a time. The next view is always the one
   that fills the most empty target while staying *mostly known* (`--max-new`,
   45%), so the fill grows outward from the source's edges. Each view is an
   img2img inpaint in Forge on your own checkpoint, conditioned on the known
   pixels by ControlNet Union ProMax in its Inpaint mode (`--method cn`; `plain`
   is the V.0 path without it). Views looking up or down are told so (ceiling or
   sky, floor or ground). **The source's pixels are never rewritten**; seams are
   blended only on the fill side. The back of the sphere, which VR180 never
   shows, is a cheap blur.
3. **Depth, second eye and VR180** by [stereo360](https://github.com/LeonG-ZA/stereo360)
   at a pinned commit (`--output-mode vr180 --inpaint learned`).

Beside the output, `<stem>.work/` keeps the flat panorama, every generated view
and `log.json`: timings, each view's direction and prompt, and every choice the
program made on its own with the reason.

## On a pod

`setup/pod_setup.sh all` installs and verifies everything, given PQM's pod image
(Forge Neo at `:7860`) and this repo at `/workspace/pqm-vr180`. The checkpoint is
the pod's business (PQM provisions it), not this program's.

## Tests

    py -3.13 -m venv .venv && .venv/Scripts/pip install -r requirements.txt pytest
    .venv/Scripts/python -m pytest

They run offline with a fake Forge: geometry, planning, and that a widening fills
the front and leaves every source pixel exactly as placed.
