"""What is in the picture, as Danbooru tags: WD EVA02-Large Tagger v3, ONNX on the CPU.

V.0 measured that half the author's prompts name nothing about the surroundings,
while a tagger found the setting every time, so PQM's VR extension fills from the
prompt's background tags **and** a tagger's (V.2). This is the tagger. PQM's own
classifier decides which of these tags are background and which are the subject;
this module only reports what it sees and how sure it is.

Model: ``SmilingWolf/wd-eva02-large-tagger-v3`` at a pinned commit (Apache-2.0),
``model.onnx`` and ``selected_tags.csv``, fetched and verified by
``setup/pod_setup.sh models``.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

TAGGER_DIR = "/workspace/models/wd-eva02-large-tagger-v3"
#: The thresholds SmilingWolf's own demo uses: general tags are many and
#: individually unsure, a character tag is a claim about identity.
GENERAL_THRESHOLD = 0.35
CHARACTER_THRESHOLD = 0.85
#: Tags whose underscores are the tag (the demo's list), never spaces.
KAOMOJI = {"0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<",
           "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"}
_KINDS = {0: "general", 4: "character", 9: "rating"}


@dataclass(frozen=True)
class Tag:
    name: str
    score: float
    kind: str   # general, character or rating

    def as_dict(self) -> dict:
        return {"name": self.name, "score": round(float(self.score), 4), "kind": self.kind}


def read_tags(path: Path | str) -> list[tuple[str, str]]:
    """``selected_tags.csv`` as ``[(name, kind)]`` in the model's output order."""
    out = []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["name"]
            if name not in KAOMOJI:
                name = name.replace("_", " ")
            out.append((name, _KINDS.get(int(row["category"]), "general")))
    return out


def prepare(rgb: np.ndarray, size: int) -> np.ndarray:
    """The WD v3 input: padded square on white, ``size`` px, BGR, float 0-255, NHWC."""
    image = Image.fromarray(rgb).convert("RGB")
    w, h = image.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    if side != size:
        canvas = canvas.resize((size, size), Image.BICUBIC)
    arr = np.asarray(canvas, dtype=np.float32)[:, :, ::-1]
    return np.ascontiguousarray(arr[None])


def pick(scores: np.ndarray, names: list[tuple[str, str]],
         general: float = GENERAL_THRESHOLD, character: float = CHARACTER_THRESHOLD) -> list[Tag]:
    """Every general and character tag over its threshold, and the likeliest rating,
    highest score first."""
    found, ratings = [], []
    for (name, kind), score in zip(names, (float(s) for s in scores)):
        if kind == "rating":
            ratings.append(Tag(name, score, kind))
        elif score >= (character if kind == "character" else general):
            found.append(Tag(name, score, kind))
    found.sort(key=lambda t: -t.score)
    if ratings:
        found.append(max(ratings, key=lambda t: t.score))
    return found


class Tagger:
    def __init__(self, folder: str = TAGGER_DIR, session=None):
        folder = Path(folder)
        self.names = read_tags(folder / "selected_tags.csv")
        if session is None:
            import onnxruntime as ort  # the pod's venv has it; tests hand in a stand-in
            session = ort.InferenceSession(str(folder / "model.onnx"),
                                           providers=["CPUExecutionProvider"])
        self.session = session
        spec = session.get_inputs()[0]
        self.input = spec.name
        self.size = int(spec.shape[1]) if isinstance(spec.shape[1], int) else 448

    def __call__(self, rgb: np.ndarray) -> list[Tag]:
        scores = self.session.run(None, {self.input: prepare(rgb, self.size)})[0][0]
        return pick(scores, self.names)
