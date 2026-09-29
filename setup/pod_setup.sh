#!/usr/bin/env bash
# Install what vr180 needs on a PQM pod (Forge Neo already running at :7860).
#
#   pod_setup.sh companion|stereo360|models|moge|promax|verify|all
#
# Each part is verified before it prints "SETUP <part> OK"; a part that cannot be
# verified exits non-zero, which is what a bootstrap should read as failure.
#
# Paths match PQM's pod image (docs/04). The companion itself is expected at
# $VR180_DIR, put there at a pinned commit (V.1: uploaded as `git archive`).
#
# Two things about that image (V.1, measured): download into models/ only after
# Forge answers (the entrypoint replaces models/ partway through boot), and put
# ControlNet models in place before Forge starts, or restart it afterwards (Forge
# lists models/ControlNet once, at startup, and its API has no refresh).
set -euo pipefail

VR180_DIR=${VR180_DIR:-/workspace/pqm-vr180}
VENVS=${VENVS:-/workspace/venvs}
FORGE_DIR=${FORGE_DIR:-/workspace/forge/sd-webui-forge-neo}
FORGE_URL=${FORGE_URL:-http://127.0.0.1:7860}
SEG_DIR=${SEG_DIR:-/workspace/models/anime-seg}

STEREO360_REPO=https://github.com/LeonG-ZA/stereo360
STEREO360_REF=23d1e256a8562fe7ebaf55dc41c809a8f43e774b
STEREO360_DIR=${STEREO360_DIR:-/workspace/stereo360}
# MoGe (Microsoft, MIT), pinned at the commit that was HEAD when V.1 made it the default
# placement (V.1's tests ran main at about this point); the model is Ruicheng/moge-2-vitl.
MOGE_REF=${MOGE_REF:-74fbce054ebed49800de42d0ad0e83495065719a}

HF=https://huggingface.co
CN="$FORGE_DIR/models/ControlNet"
PRE="$FORGE_DIR/models/ControlNetPreprocessor"
# Each: destination, URL at a pinned revision, sha256.
MODELS=(
  # NoobAI Inpainting ControlNet, fp16 (Acly's copy of Wenaka_'s; fair-ai-public-license-1.0-sd)
  "$CN/noobaiInpainting_v10.fp16.safetensors|$HF/Acly/NoobAI-Inpainting/resolve/7341925b3346eb48dcac4dd4049cdb5bf2afd472/noobaiInpainting_v10.fp16.safetensors|8b3c2155ec8a49b43a8a32dac39a21b8c09ea60a7afaffe3c7a6fb51313a45d6"
  # NoobAI IP-Adapter MARK1, the reference model Krita AI Diffusion uses for Illustrious
  "$CN/noobIPAMARK1_mark1.safetensors|$HF/r3gm/noob-ipa/resolve/534fdd8fb5a221d67937d5495febd2560a1d395e/model_G/noobIPAMARK1_mark1.safetensors|5cdb6a00be1b12579745b5bed0c7b83f0869073d8a864fa8cd50a9356601919a"
  # CLIP-ViT-bigG, under the name Forge's "CLIP-ViT-bigG (IPAdapter)" preprocessor looks for
  "$PRE/CLIP-ViT-bigG.safetensors|$HF/h94/IP-Adapter/resolve/018e402774aeeddd60609b4ecdb7e298259dc729/sdxl_models/image_encoder/model.safetensors|657723e09f46a7c3957df651601029f66b1748afb12b419816330f16ed45d64d"
  # anime-segmentation ISNet-IS (SkyTNT, Apache-2.0), for continuing a cut-off subject
  "$SEG_DIR/isnetis.onnx|$HF/skytnt/anime-seg/resolve/493cb60893f47441b26ec4fb9a306bce9e342982/isnetis.onnx|f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99"
)
# ControlNet Union SDXL ProMax: NaN (fp16) or noise (bf16) in Forge Neo as pinned. Opt-in.
PROMAX="$CN/controlnet-union-sdxl-promax.safetensors|$HF/xinsir/controlnet-union-sdxl-1.0/resolve/801a4a3fa3d4c936f4feea95b98607bc6726f80c/diffusion_pytorch_model_promax.safetensors|9fae2e50cb431bfcbe05822b59ec2228df545ef27f711dea8949e9f4ed9f7cdc"

mkdir -p "$VENVS"
command -v uv >/dev/null || pip install -q uv
export UV_LINK_MODE=copy

fetch() {  # fetch "dest|url|sha256": download unless already there and verified
  IFS='|' read -r dest url sha <<<"$1"
  mkdir -p "$(dirname "$dest")"
  if echo "$sha  $dest" | sha256sum -c --quiet - >/dev/null 2>&1; then
    echo "have $(basename "$dest")"; return
  fi
  t0=$(date +%s)
  curl -sSL --fail -o "$dest.part" "$url"
  echo "$sha  $dest.part" | sha256sum -c --quiet -
  mv "$dest.part" "$dest"
  echo "$(basename "$dest") in $(( $(date +%s) - t0 ))s"
}

listed() {  # listed NAME-FRAGMENT: does Forge's ControlNet list have it?
  curl -sf "$FORGE_URL/controlnet/model_list" | grep -qi "$1"
}

companion() {
  [ -f "$VR180_DIR/vr180/__init__.py" ] || { echo "no companion at $VR180_DIR"; exit 3; }
  uv venv --allow-existing -q -p 3.12 "$VENVS/vr180"
  uv pip install -q -p "$VENVS/vr180/bin/python" -r "$VR180_DIR/requirements.txt"
  (cd "$VR180_DIR" && "$VENVS/vr180/bin/python" -m vr180 --help >/dev/null)
  "$VENVS/vr180/bin/python" -c "import sys; sys.path.insert(0, '$VR180_DIR'); import vr180, onnxruntime; print('vr180', vr180.__version__, 'onnxruntime', onnxruntime.__version__)"
}

stereo360() {
  command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg)
  if [ ! -d "$STEREO360_DIR/.git" ]; then
    git init -q "$STEREO360_DIR"
    git -C "$STEREO360_DIR" remote add origin "$STEREO360_REPO"
  fi
  git -C "$STEREO360_DIR" fetch -q --depth=1 origin "$STEREO360_REF"
  git -C "$STEREO360_DIR" checkout -q -B pinned FETCH_HEAD
  test "$(git -C "$STEREO360_DIR" rev-parse HEAD)" = "$STEREO360_REF"
  uv venv --allow-existing -q -p 3.12 "$VENVS/stereo360"
  PY="$VENVS/stereo360/bin/python"
  uv pip install -q -p "$PY" -r "$STEREO360_DIR/requirements.txt"
  # onnxruntime-gpu once hung for 16 minutes on a pod (V.1); only stereo360's onnx
  # depth backend uses it, so a time limit and the CPU build are enough.
  timeout 180 uv pip install -q -p "$PY" onnxruntime-gpu || uv pip install -q -p "$PY" onnxruntime
  # Warm up on a small synthetic pano: fetches Depth Pro and LaMa now, not mid-run.
  "$PY" - <<'EOF'
import numpy as np
from PIL import Image
y, x = np.mgrid[0:512, 0:1024]
Image.fromarray(np.dstack([(x % 256), (y % 256), ((x + y) % 256)]).astype(np.uint8)).save("/tmp/warm.png")
EOF
  (cd "$STEREO360_DIR" && "$PY" -m stereo360 /tmp/warm.png -o /tmp/warm_180_LR.jpg \
     --output-mode vr180 --inpaint learned >/tmp/warm.log 2>&1) || { tail -20 /tmp/warm.log; exit 4; }
  test -s /tmp/warm_180_LR.jpg
  echo "stereo360 $(git -C "$STEREO360_DIR" rev-parse --short HEAD) warm"
}

models() {
  for m in "${MODELS[@]}"; do fetch "$m"; done
  for name in noobaiInpainting noobIPAMARK1; do
    listed "$name" || { echo "$name is on disk but Forge started before it: restart Forge"; exit 5; }
  done
}

moge() {
  # MoGe-2 for the default placement, in its own venv (torch), warmed so the model
  # downloads at install. The author made it the default placement (V.1).
  uv venv --allow-existing -q -p 3.12 "$VENVS/est"
  timeout 900 uv pip install -q -p "$VENVS/est/bin/python" torch torchvision numpy pillow     opencv-python-headless "git+https://github.com/microsoft/MoGe.git@$MOGE_REF"
  "$VENVS/est/bin/python" - <<'EOF'
import numpy as np
from PIL import Image
y, x = np.mgrid[0:256, 0:384]
Image.fromarray(np.dstack([(x % 256), (y % 256), ((x + y) % 256)]).astype(np.uint8)).save("/tmp/moge_warm.png")
EOF
  (cd "$VR180_DIR" && PYTHONPATH="$VR180_DIR" "$VENVS/est/bin/python" -m vr180.estimate_alt --only moge      /tmp/moge_warm.json /tmp/moge_warm.png) | grep -q '"hfov"' || { echo "MoGe did not estimate"; exit 6; }
  echo "moge warm"
}

promax() {
  fetch "$PROMAX"
  listed controlnet-union-sdxl-promax \
    || { echo "promax is on disk but Forge started before it: restart Forge"; exit 5; }
}

verify() {
  "$VENVS/vr180/bin/python" -c "import sys; sys.path.insert(0, '$VR180_DIR'); import vr180, onnxruntime"
  test "$(git -C "$STEREO360_DIR" rev-parse HEAD)" = "$STEREO360_REF"
  for m in "${MODELS[@]}"; do
    IFS='|' read -r dest _url sha <<<"$m"
    echo "$sha  $dest" | sha256sum -c --quiet -
  done
  listed noobaiInpainting && listed noobIPAMARK1
  curl -sf "$FORGE_URL/controlnet/module_list" | grep -q "CLIP-ViT-bigG (IPAdapter)"
}

case "${1:-all}" in
  companion|stereo360|models|moge|promax|verify) "$1" ;;
  all) companion; stereo360; models; moge; verify ;;
  *) echo "usage: $0 companion|stereo360|models|moge|promax|verify|all"; exit 2 ;;
esac
echo "SETUP ${1:-all} OK"
