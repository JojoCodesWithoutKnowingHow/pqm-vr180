#!/usr/bin/env bash
# Install what vr180 needs on a PQM pod (Forge Neo already running at :7860).
#
#   pod_setup.sh companion|stereo360|controlnet|verify|all
#
# Each part is verified before it prints "SETUP <part> OK"; a part that cannot be
# verified exits non-zero, which is what a bootstrap should read as failure.
#
# Paths match PQM's pod image (docs/04). The companion itself is expected at
# $VR180_DIR, put there at a pinned commit (V.1: uploaded as `git archive`).
set -euo pipefail

VR180_DIR=${VR180_DIR:-/workspace/pqm-vr180}
VENVS=${VENVS:-/workspace/venvs}
FORGE_DIR=${FORGE_DIR:-/workspace/forge/sd-webui-forge-neo}
FORGE_URL=${FORGE_URL:-http://127.0.0.1:7860}

STEREO360_REPO=https://github.com/LeonG-ZA/stereo360
STEREO360_REF=23d1e256a8562fe7ebaf55dc41c809a8f43e774b
STEREO360_DIR=${STEREO360_DIR:-/workspace/stereo360}

# ControlNet Union SDXL ProMax (xinsir, Apache-2.0), at a pinned revision.
CN_URL=https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/801a4a3fa3d4c936f4feea95b98607bc6726f80c/diffusion_pytorch_model_promax.safetensors
CN_SHA256=9fae2e50cb431bfcbe05822b59ec2228df545ef27f711dea8949e9f4ed9f7cdc
CN_FILE=controlnet-union-sdxl-promax.safetensors

mkdir -p "$VENVS"
command -v uv >/dev/null || pip install -q uv
export UV_LINK_MODE=copy

companion() {
  [ -f "$VR180_DIR/vr180/__init__.py" ] || { echo "no companion at $VR180_DIR"; exit 3; }
  uv venv --allow-existing -q -p 3.12 "$VENVS/vr180"
  uv pip install -q -p "$VENVS/vr180/bin/python" -r "$VR180_DIR/requirements.txt"
  (cd "$VR180_DIR" && "$VENVS/vr180/bin/python" -m vr180 --help >/dev/null)
  "$VENVS/vr180/bin/python" -c "import sys; sys.path.insert(0, '$VR180_DIR'); import vr180; print('vr180', vr180.__version__)"
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
  uv pip install -q -p "$PY" onnxruntime-gpu || uv pip install -q -p "$PY" onnxruntime
  (cd "$STEREO360_DIR" && uv pip install -q -p "$PY" -e . || true)
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

controlnet() {
  d="$FORGE_DIR/models/ControlNet"
  mkdir -p "$d"
  f="$d/$CN_FILE"
  if ! echo "$CN_SHA256  $f" | sha256sum -c --quiet - 2>/dev/null; then
    t0=$(date +%s)
    curl -sSL --fail -o "$f.part" "$CN_URL"
    echo "$CN_SHA256  $f.part" | sha256sum -c --quiet -
    mv "$f.part" "$f"
    echo "ControlNet in $(( $(date +%s) - t0 ))s"
  fi
  # Forge reads the folder on each list call.
  curl -sf "$FORGE_URL/controlnet/model_list" | grep -q "controlnet-union-sdxl-promax" \
    || { echo "Forge does not list $CN_FILE"; exit 5; }
}

verify() {
  "$VENVS/vr180/bin/python" -c "import sys; sys.path.insert(0, '$VR180_DIR'); import vr180"
  test "$(git -C "$STEREO360_DIR" rev-parse HEAD)" = "$STEREO360_REF"
  curl -sf "$FORGE_URL/controlnet/model_list" | grep -q "controlnet-union-sdxl-promax"
  curl -sf "$FORGE_URL/controlnet/module_list" | grep -q "inpaint_only+lama"
}

case "${1:-all}" in
  companion|stereo360|controlnet|verify) "$1" ;;
  all) companion; stereo360; controlnet; verify ;;
  *) echo "usage: $0 companion|stereo360|controlnet|verify|all"; exit 2 ;;
esac
echo "SETUP ${1:-all} OK"
