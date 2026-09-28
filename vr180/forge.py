"""Forge's img2img inpaint, as the widening uses it.

Runs on the pod beside Forge and calls it at localhost, where RunPod's 100-second
proxy limit does not apply. Two methods, so they can be compared on one pod:

- ``cn`` (default): img2img inpaint with a ControlNet Union **ProMax** unit in its
  Inpaint mode (``inpaint_only+lama``). The unit is given no image, so Forge hands
  it img2img's own init image and mask: the model sees the known pixels as
  context, and ``inpaint_only`` pastes them back exactly afterwards.
- ``plain``: the V.0 path, a checkpoint in img2img with nothing but the mask.
"""
from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np
from PIL import Image

CN_MODULE = "inpaint_only+lama"
CN_TYPE = "Inpaint"   # Union type 7, which only ProMax has


def b64png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def unpng(data: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))


class ForgeError(RuntimeError):
    pass


@dataclass
class Settings:
    checkpoint: str
    method: str = "cn"
    cn_model: str = ""
    steps: int = 28
    cfg: float = 5.0
    sampler: str = "Euler a"
    scheduler: str = "Automatic"
    denoise: float = 1.0
    mask_blur: int = 8
    cn_weight: float = 1.0
    cn_end: float = 1.0


class Forge:
    def __init__(self, url: str = "http://127.0.0.1:7860", timeout: float = 900.0):
        self.url, self.timeout = url.rstrip("/"), timeout

    def _call(self, method: str, path: str, payload: dict | None = None, tries: int = 3):
        data = None if payload is None else json.dumps(payload).encode()
        for attempt in range(tries):
            req = urllib.request.Request(self.url + path, data, method=method,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")[:600]
                if exc.code < 500 or attempt == tries - 1:
                    raise ForgeError("Forge %s %s: HTTP %d %s" % (method, path, exc.code, body))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == tries - 1:
                    raise ForgeError("Forge %s %s: %s" % (method, path, exc)) from exc
            time.sleep(5 * (attempt + 1))

    def checkpoints(self) -> list[str]:
        """Every name Forge answers to for each checkpoint: its model name and its
        title without the hash (``name.safetensors``)."""
        names = []
        for m in self._call("GET", "/sdapi/v1/sd-models"):
            names.append(m.get("model_name", ""))
            names.append(m.get("title", "").split(" [")[0])
        return [n for n in names if n]

    def cn_models(self) -> list[str]:
        return list(self._call("GET", "/controlnet/model_list").get("model_list", []))

    def resolve(self, s: Settings) -> list[str]:
        """What is missing for ``s`` on this Forge, by name; empty when ready."""
        missing = []
        # Exact, never a substring: Forge silently keeps the loaded model when an
        # override names one it does not list (V.1 lost two sources that way).
        if s.checkpoint not in self.checkpoints():
            missing.append("checkpoint %r is not in Forge's list" % s.checkpoint)
        if s.method == "cn":
            names = self.cn_models()
            hit = [n for n in names if (s.cn_model or "promax").lower() in n.lower()]
            if not hit:
                missing.append("no ControlNet matching %r among %s" % (s.cn_model or "promax", names))
            else:
                s.cn_model = hit[0]
        return missing

    def inpaint(self, image: np.ndarray, mask: np.ndarray, prompt: str, negative: str,
                seed: int, s: Settings) -> np.ndarray:
        """``mask`` is uint8, 255 where to paint. Returns an image the size of ``image``."""
        h, w = image.shape[:2]
        payload = {
            "init_images": [b64png(image)], "mask": b64png(mask),
            "prompt": prompt, "negative_prompt": negative,
            "denoising_strength": s.denoise, "inpainting_fill": 1, "mask_blur": s.mask_blur,
            "inpaint_full_res": False, "inpainting_mask_invert": 0,
            "width": w, "height": h, "steps": s.steps, "cfg_scale": s.cfg,
            "sampler_name": s.sampler, "scheduler": s.scheduler, "seed": seed,
            "override_settings": {"sd_model_checkpoint": s.checkpoint},
            "override_settings_restore_afterwards": False,
            "send_images": True, "save_images": False,
        }
        if s.method == "cn":
            payload["alwayson_scripts"] = {"ControlNet": {"args": [{
                "enabled": True, "module": CN_MODULE, "model": s.cn_model,
                "type_filter": CN_TYPE, "weight": s.cn_weight, "guidance_start": 0.0,
                "guidance_end": s.cn_end, "control_mode": "Balanced", "pixel_perfect": True,
                "resize_mode": "Just Resize"}]}}
        r = self._call("POST", "/sdapi/v1/img2img", payload)
        out = unpng(r["images"][0])
        if out.shape[:2] != (h, w):
            out = np.array(Image.fromarray(out).resize((w, h), Image.LANCZOS))
        return out
