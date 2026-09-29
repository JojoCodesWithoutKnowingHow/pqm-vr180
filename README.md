# pqm-vr180

The VR extension's companion program for PromptQueueManager (PQM): **one flat
image in, one VR180 file out**, run on a PQM pod beside Forge.

    python -m vr180 SRC.png -o OUT_180_LR.jpg --checkpoint waiIllustriousSDXL_v140 \
        --tags "indoors, living room, wooden floor, bookshelf" \
        --subject-tags "1girl, brown hair, white sweater, blue jeans"

Proven on single images in V.1 (2026-09-28/29) and tuned for anime indoor scenes
over five rounds, every default below judged by the author in a Quest 3.

## What it does, and the defaults

1. **Place** the source straight ahead on a sphere, at the field of view **MoGe-2**
   estimates for it (`--long-side moge`, the default: "more realistic on every
   image"). A plain background has nothing to measure and gets 60°. **A number
   overrides it** (`--long-side 60`), since realistic is not always the artistic
   choice. `ratio`, `shot` and `camera` (Depth Pro) are experiments that lost.
2. **Widen** it to the front hemisphere plus a margin, one 90°, 1024 px
   perspective view at a time (90° beat 72°, 105° and 110° in the headset). The
   next view is always the one that fills the most empty target while staying
   mostly known, so the fill grows outward from the source's edges. Each view is
   an img2img inpaint in Forge on the given checkpoint with the **NoobAI
   Inpainting ControlNet** (hole pure black, no preprocessor) and the source as a
   **noobIPA** reference: Krita AI Diffusion's recipe for Illustrious checkpoints.
   Views facing up or down are told so and drop the other half's tags.
   - **Foveated** (`--no-taper` to turn off): views far from the source get fewer
     steps (28 → 12) and 768 px; 14-40% faster, unseen in the headset.
   - **A cut-off subject is continued** (`--subject-tags`): anime-seg finds it; its
     tags paint only a zone extruded from the cut edge, the rest is scene with no
     people (no duplicates), and where the body reaches the zone's edge the strip
     beyond is left for the next view to carry on.
   - **A plain background** is filled in one smooth pass, with no diffusion.
   - The source's pixels are never regenerated.
3. **Hide the seam** (on by default): the seam is a *sharpness* step, not a colour
   one. `--detail-match` sharpens the fill by the measured shortfall and
   `--soften-rim 12` blends the source's outermost 12 px toward the fill (the
   author allowed touching the edge; both together looked best).
4. **Depth, second eye and VR180** by [stereo360](https://github.com/LeonG-ZA/stereo360)
   at a pinned commit, with the companion's own depth normalisation: the range is
   measured over the source's region (not the floor under the viewer), IW3's
   foreground-scale curve at -2 (the author's IW3 setting), and Depth Anything V2
   Base (IW3's Any_B). `--strength` stays 1.0 and `--gradient-limit` at its default
   (0 corrupted the generated areas).

Beside the output, `<stem>.work/` keeps the flat panorama, every generated view
and `log.json`: timings, each view's direction, kind, steps, size and prompt, the
placement and why, and the seam measurements.

## On a pod

`setup/pod_setup.sh all` installs and verifies everything, given PQM's pod image
(Forge Neo at `:7860`) and this repo at `/workspace/pqm-vr180`: the companion's
venv, stereo360 (warmed), MoGe-2 in its own venv (warmed), and four sha256-pinned
models (NoobAI Inpainting, noobIPA MARK1, CLIP-ViT-bigG, anime-seg). The
checkpoint and any style LoRA are the pod's business (PQM provisions them). Two
things about that image, measured in V.1:

- **Download into `models/` only after Forge answers**: the entrypoint replaces
  `models/` with a symlink partway through boot.
- **ControlNet models must be in place before Forge starts**, or Forge restarted
  after they land: it lists them once, at startup.

Measured on an RTX 4090: install about 5-6 minutes after Forge answers; an image
2-4 minutes with a subject to continue, under a minute for a plain background.

## Tests

    py -3.13 -m venv .venv && .venv/Scripts/pip install -r requirements.txt pytest
    .venv/Scripts/python -m pytest

55 offline tests with a fake Forge: geometry, planning, source preservation,
subject continuation, plain fill, the taper, the depth curve and normalisation,
placement, the seam remedies and the Forge payloads.
