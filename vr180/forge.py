"""Forge's img2img inpaint, as the widening uses it.

Runs on the pod beside Forge and calls it at localhost, where RunPod's 100-second
proxy limit does not apply. Two methods, so they can be compared on one pod:

- ``plain`` (default): the checkpoint's own img2img inpaint with the mask. With
  views that are mostly known and told where they face, this is what V.1 kept.
- ``cn``: the same with a ControlNet Union **ProMax** unit in its Inpaint mode
  (``inpaint_only+lama``); the unit is given no image, so Forge hands it
  img2img's own init image and mask. **It does not work in Forge Neo at PQM's
  pinned commit** (V.1, measured on a 4090): the latent goes NaN in fp16 (black
  fills), and with ``--bf16-unet`` the hole comes back as pure noise, under
  ``inpaint_only+lama``, ``inpaint_only`` and ``inpaint_global_harmonious`` alike
  and at weight 0.7. Kept so it can be retried against a later Forge.
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


#: Forge's IP-Adapter preprocessor for SDXL/Illustrious (it fetches CLIP-ViT-bigG itself).
IPA_MODULE = "CLIP-ViT-bigG (IPAdapter)"
#: What each method needs from Forge's ControlNet list: (setting field, name fragment).
NEEDS = {"cn": [("cn_model", "promax")],
         "noob": [("cn_model", "noobaiinpainting")]}


@dataclass
class Settings:
    checkpoint: str
    method: str = "plain"
    cn_model: str = ""
    ipa_model: str = ""          # "" = no reference; "noobipa" or a name = use one
    ipa_weight: float = 0.5
    ipa_range: tuple = (0.2, 0.8)
    steps: int = 28
    cfg: float = 5.0
    sampler: str = "Euler a"
    scheduler: str = "Automatic"
    denoise: float = 0.95
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
        """What is missing for ``s`` on this Forge, by name; empty when ready.
        Fills ``s.cn_model`` / ``s.ipa_model`` with Forge's own names."""
        missing = []
        # Exact, never a substring: Forge silently keeps the loaded model when an
        # override names one it does not list (V.1 lost two sources that way).
        if s.checkpoint not in self.checkpoints():
            missing.append("checkpoint %r is not in Forge's list" % s.checkpoint)
        wanted = [(f, getattr(s, f) or frag) for f, frag in NEEDS.get(s.method, [])]
        if s.ipa_model:
            wanted.append(("ipa_model", s.ipa_model))
        if wanted:
            names = self.cn_models()
            for field_name, frag in wanted:
                hit = [n for n in names if frag.lower() in n.lower()]
                if hit:
                    setattr(s, field_name, hit[0])
                else:
                    missing.append("no ControlNet matching %r among %s" % (frag, names))
        return missing

    def inpaint(self, image: np.ndarray, mask: np.ndarray, prompt: str, negative: str,
                seed: int, s: Settings, control: np.ndarray | None = None,
                reference: np.ndarray | None = None, steps: int | None = None,
                denoise: float | None = None, touch_up: bool = False) -> np.ndarray:
        """``mask`` is uint8, 255 where to paint. ``control`` is the inpaint
        ControlNet's image (``noob``: the view with the hole pure black);
        ``reference`` the IP-Adapter's. ``denoise`` overrides the settings' for
        one call (a view refined over a layout). ``touch_up`` (round 2: the
        extension's full-resolution pass, the seam repaint) is a low-denoise pass
        over pixels already there, so it sends no inpaint ControlNet: that one
        exists to fill a black hole. Returns an image the size of ``image``.
        Raises ``ForgeError`` when the fill comes back black (NaN latents)."""
        h, w = image.shape[:2]
        payload = {
            "init_images": [b64png(image)], "mask": b64png(mask),
            "prompt": prompt, "negative_prompt": negative,
            "denoising_strength": s.denoise if denoise is None else denoise,
            "inpainting_fill": 1, "mask_blur": s.mask_blur,
            "inpaint_full_res": False, "inpainting_mask_invert": 0,
            "width": w, "height": h, "steps": steps or s.steps, "cfg_scale": s.cfg,
            "sampler_name": s.sampler, "scheduler": s.scheduler, "seed": seed,
            "override_settings": {"sd_model_checkpoint": s.checkpoint},
            "override_settings_restore_afterwards": False,
            "send_images": True, "save_images": False,
        }
        units = []
        if touch_up:
            pass
        elif s.method == "cn":
            units.append({"enabled": True, "module": CN_MODULE, "model": s.cn_model,
                          "type_filter": CN_TYPE, "weight": s.cn_weight, "guidance_start": 0.0,
                          "guidance_end": s.cn_end, "control_mode": "Balanced",
                          "pixel_perfect": True, "resize_mode": "Just Resize"})
        elif s.method == "noob":
            if control is None:
                raise ValueError("noob needs a control image")
            # No preprocessor: NoobAI Inpainting wants the hole pure black, and
            # Forge's inpaint preprocessors write -1 there instead.
            units.append({"enabled": True, "module": "None", "model": s.cn_model,
                          "image": b64png(control), "weight": s.cn_weight,
                          "guidance_start": 0.0, "guidance_end": s.cn_end,
                          "control_mode": "Balanced", "pixel_perfect": True,
                          "resize_mode": "Just Resize"})
        if s.ipa_model and reference is not None:
            units.append({"enabled": True, "module": IPA_MODULE, "model": s.ipa_model,
                          "image": b64png(reference), "weight": s.ipa_weight,
                          "guidance_start": s.ipa_range[0], "guidance_end": s.ipa_range[1],
                          "control_mode": "Balanced", "resize_mode": "Just Resize"})
        if units:
            payload["alwayson_scripts"] = {"ControlNet": {"args": units}}
        r = self._call("POST", "/sdapi/v1/img2img", payload)
        out = unpng(r["images"][0])
        if out.shape[:2] != (h, w):
            out = np.array(Image.fromarray(out).resize((w, h), Image.LANCZOS))
        hole = mask > 127
        if hole.any() and float((out[hole].max(1) <= 2).mean()) > 0.5:
            raise ForgeError("the fill came back black (NaN latents): %s with %s"
                             % (s.method, [u["model"] for u in units]))
        return out
