# pqm-vr180

The VR extension's companion program for PromptQueueManager (PQM): **one flat
image in, one VR180 file out**, run on a PQM pod beside Forge.

    python -m vr180 SRC.png -o OUT_180_LR.jpg \
        --tags "indoors, living room, wooden floor, bookshelf" \
        --subject-tags "1girl, brown hair, white sweater, blue jeans" \
        --subject-framing "sitting, crossed legs, cowboy shot" --loras "<lora:her:0.8>"

Proven on single images in V.1 (2026-09-28/29) and tuned for anime indoor scenes
over five rounds, every default below judged by the author in a Quest 3.

## 0.5.0: the outpaint rework's baseline is the default

Fifty-one rounds on pods (2026-09-30 – 10-01), each judged in a Quest 3, one change
at a time against an approved baseline, settled the widening. **A run with no flags
is that baseline**; written out, it is

    --checkpoint Waifu-Inpaint-XL --method plain --denoise 1.0 --join hard --auto-pipeline
    --extend-side 1.0 --extend-refine 0.35 --layout fisheye --layout-strong
    --layout-hires 2048 --layout-hires-denoise 0.4 --compose 0.3 --seam-repaint 0.4
    --soften-rim 0 --keep-threshold 0.15 --source-fade 32 --adetail 0.27
    --layout-mask-grow 0 --layout-mask-blur 4 --touch-mask-weight-j 0 --no-detail-match

- **[Waifu-Inpaint-XL](https://huggingface.co/ShinoharaHare/Waifu-Inpaint-XL)** (at
  `a33e08f`): WAI-illustrious v14 v-pred with its input widened to 9 channels, so the
  model sees the known pixels. It ended the joins that were off by a latent cell with
  WAI + the NoobAI inpainting ControlNet. **Gated**: each user requests access with
  their own Hugging Face account.
- **`--auto-pipeline`**: her body is grown stepwise from the frame first (J); if it
  still reaches a grown edge, her limbs run out of frame and the layout is drawn with
  the full prompt and an ADetailer pass (L) instead.
- **`--loras`**: the source's LoRA calls go on every pass that draws her -- the
  extension, the layout's full prompt, ADetailer, the seam repaint across her -- and
  on no scenery-only view. A call whose file Forge does not list stops the run.
- Every flag that lost a round is still here, off; each one's help says which round.

The sections below describe V.1's pipeline, which 0.5.0 builds on; where they name a
default (the NoobAI ControlNet, `--soften-rim 12`, `--detail-match`) it is V.1's.

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

## Before widening: `vr180.analyze` (0.4.0, for PQM's VR extension)

    python -m vr180.analyze SRC.png -o analysis.json

Reports, and never decides: the [WD EVA02-Large Tagger v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3)'s
tags with scores, whether anime-seg finds a subject and which frame edges cut it,
and MoGe-2's placement **with the raw estimate before clamping**. PQM's VR
extension (V.2) reads it, sorts the tags with PQM's own classifier, and either
runs `python -m vr180` with every choice explicit or sets the image aside for the
user when it is not sure. Each part that fails is reported under `error`; the
rest still report.

Beside the output, `<stem>.work/` keeps the flat panorama, every generated view
and `log.json`: timings, each view's direction, kind, steps, size and prompt, the
placement and why, and the seam measurements.

## On a pod

`setup/pod_setup.sh all` installs and verifies everything, given PQM's pod image
(Forge Neo at `:7860`) and this repo checked out anywhere (the script finds its
own checkout; PQM's pod fetches it at a pinned commit): the companion's venv,
stereo360 (warmed), MoGe-2 in its own venv (warmed), three sha256-pinned models
(anime-seg, and the WD tagger's model and tag list), and **`checkpoint`**:
Waifu-Inpaint-XL, with `HF_TOKEN` from the environment (a 401 or 403 says to request
access). Under PQM the checkpoint and LoRAs are PQM's to fetch, with the user's own
tokens -- an install step never sees a credential -- so PQM runs every part but
`checkpoint`. **`noob`** adds V.1's NoobAI Inpainting, noobIPA MARK1 and
CLIP-ViT-bigG for `--method noob`, then **`restart`**s Forge exactly as the image
started it (`setup/restart_forge.py`), waited for until it lists the ControlNets. Two
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

62 offline tests with a fake Forge: geometry, planning, source preservation,
subject continuation, plain fill, the taper, the depth curve and normalisation,
placement, the seam remedies, the Forge payloads, the tagger, `analyze` and
the Forge restart's process matching.
