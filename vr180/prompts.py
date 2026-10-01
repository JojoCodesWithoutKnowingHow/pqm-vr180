"""The fill prompt for a view, from the tags that describe the surroundings.

The companion is handed those tags (PQM's classifier and a tagger choose them, as
V.0 measured); here they only get a direction. V.0's views looking down painted
another lake instead of the ground, because every view was prompted "scenery".
"""
from __future__ import annotations

import re

QUALITY = "masterpiece, best quality, amazing quality, very aesthetic, absurdres"
#: What no view wants, and what views that are not continuing the subject add.
BASE_NEGATIVE = ("text, watermark, signature, frame, border, picture frame, lowres, "
                 "worst quality, bad quality, jpeg artifacts, blurry")
NO_PEOPLE = "1girl, 1boy, multiple girls, multiple boys, person, people, human, character, face"
NEGATIVE = NO_PEOPLE + ", " + BASE_NEGATIVE
#: A view continuing the subject: one body, the one that is already there.
SUBJECT_NEGATIVE = ("multiple girls, multiple boys, extra person, extra arms, extra legs, "
                    "extra hands, bad anatomy, cropped, " + BASE_NEGATIVE)
#: Round 21: with the full prompt, the layout drew a second, giant Yamato -- a
#: headless close-up torso -- over the floor in front of her, on three seeds.
CLOSE_NEGATIVE = "close-up, giantess, multiple views, pov, head out of frame"

_INDOOR = {"indoors", "room", "bedroom", "living room", "classroom", "kitchen", "bathroom",
           "ceiling", "wooden ceiling", "wall", "shouji", "window", "curtains", "floor",
           "wooden floor", "tatami", "office", "library", "hallway", "bed", "couch"}
_OUTDOOR = {"outdoors", "sky", "night sky", "cloud", "cloudy sky", "blue sky", "city",
            "cityscape", "street", "road", "building", "skyscraper", "landscape", "mountain",
            "forest", "tree", "beach", "ocean", "lake", "field", "grass", "scenery",
            "horizon", "sunset", "nature", "river", "garden", "rooftop"}
_PLAIN = {"simple background", "white background", "grey background", "gray background",
          "black background", "gradient background", "blue background", "pink background",
          "transparent background", "two-tone background"}

#: Views steeper than this are "looking up" or "looking down". The planner's views
#: sit at 40-50 degrees, and a 90-degree view at -40 is almost all ground.
UP_DEG = 35.0
DOWN_DEG = -35.0

#: What belongs above the horizon and what below it. V.1's first landscape put
#: `sky, cloud, sunset, horizon` into the views facing the ground too, and they
#: painted a second sunset under the viewer.
SKYWARD = {"sky", "cloud", "clouds", "cloudy sky", "blue sky", "night sky", "starry sky",
           "sunset", "sunrise", "sun", "moon", "star (sky)", "horizon", "mountain",
           "mountainous horizon", "skyscraper", "cityscape", "ceiling", "wooden ceiling",
           "ceiling light", "chandelier", "twilight", "evening", "neon lights", "lamp"}
EARTHWARD = {"ground", "grass", "floor", "wooden floor", "tatami", "carpet", "rug",
             "road", "street", "path", "water", "lake", "river", "ocean", "sea", "beach",
             "reflection", "sand", "dirt", "field", "alpine lake", "couch", "chair",
             "table", "coffee table", "bench"}

#: Framing words that describe the source's crop, not her pose: given to the
#: extension they tell the model to cut her off where the frame did.
CROP = {"upper body", "cowboy shot", "portrait", "close-up", "face focus", "lower body",
        "head out of frame", "feet out of frame", "out of frame", "cropped legs",
        "cropped torso", "bust", "headshot"}


def pose_words(framing: str) -> list[str]:
    """The pose in the source's framing words, without the crop ones (round 12:
    the extension continued her body without being told she was lying on her side
    reaching towards the viewer)."""
    return [t for t in split_tags(framing) if t.lower() not in CROP]


def split_tags(text: str) -> list[str]:
    return [t.strip().replace("_", " ") for t in text.replace("\n", ",").split(",") if t.strip()]


def setting(tags: list[str]) -> tuple[str | None, str]:
    """``("indoors" | "outdoors" | "plain" | None, why)``. ``None`` is an honest
    unknown: the fill then gets no up/down words rather than a guessed ceiling."""
    low = [t.lower() for t in tags]
    if "indoors" in low:
        return "indoors", "tagged indoors"
    if "outdoors" in low:
        return "outdoors", "tagged outdoors"
    plain = [t for t in low if t in _PLAIN]
    inside = [t for t in low if t in _INDOOR]
    outside = [t for t in low if t in _OUTDOOR]
    if plain and not inside and not outside:
        return "plain", "plain background: " + ", ".join(plain)
    if inside and not outside:
        return "indoors", "indoor tags: " + ", ".join(inside)
    if outside and not inside:
        return "outdoors", "outdoor tags: " + ", ".join(outside)
    if inside and outside:
        return None, "mixed: %s / %s" % (", ".join(inside), ", ".join(outside))
    return None, "no setting in the tags"


def facing(pitch: float) -> str:
    return "up" if pitch >= UP_DEG else "down" if pitch <= DOWN_DEG else "level"


def view_prompt(tags: list[str], where: str | None, pitch: float,
                quality: str = QUALITY) -> str:
    face = facing(pitch)
    if where == "plain":
        extra = ["no humans"]
    else:
        extra = ["scenery", "no humans"]
        if face == "up":
            extra += {"indoors": ["ceiling", "from below"],
                      "outdoors": ["sky", "from below"]}.get(where, [])
        elif face == "down":
            extra += {"indoors": ["floor", "from above"],
                      "outdoors": ["ground", "from above"]}.get(where, [])
    drop = {"up": EARTHWARD, "down": SKYWARD}.get(face, set()) if where != "plain" else set()
    kept = [t for t in tags if t.lower() not in drop]
    words = list(dict.fromkeys(extra + kept))
    return ", ".join(([quality] if quality else []) + words)


def view_negative(base: str, where: str | None, pitch: float) -> str:
    """The negative for a view: the base, plus, for a view looking up or down,
    what belongs to the other half of the scene and any horizon at all."""
    face = facing(pitch)
    if where == "plain" or face == "level":
        return base
    other = {"up": "ground, floor, grass, water, horizon, landscape",
             "down": "sky, cloud, sun, sunset, horizon, mountain, ceiling"}[face]
    return base + ", " + other


#: The source's LoRA calls (``--loras``, 0.5.0), added to every prompt that draws
#: her -- :func:`subject_prompt` builds each of them (the extension, the layout's
#: full prompt, ADetailer, the seam repaint across her) -- and to no scenery-only
#: view, which a character LoRA would fill with her.
HER_LORAS: tuple = ()
_LORA_CALL = re.compile(r"<(?:lora|lyco):[^<>:]+(?::-?\d+(?:\.\d+)?){0,2}>", re.IGNORECASE)


def lora_calls(text: str) -> tuple:
    """The ``<lora:name:w>`` / ``<lyco:...>`` calls in ``text``, in order, once each.
    Anything else in it is an error: ``--loras`` carries calls, not prompt words."""
    text = (text or "").strip()
    calls = _LORA_CALL.findall(text)
    rest = _LORA_CALL.sub("", text).replace(",", " ").strip()
    if rest:
        raise ValueError("--loras takes <lora:name:weight> calls only, not %r" % rest[:80])
    return tuple(dict.fromkeys(calls))


def subject_prompt(subject_tags: list[str], fill_tags: list[str], where: str | None,
                   pitch: float, quality: str = QUALITY) -> str:
    """A view that continues a subject the frame cut off: the subject's own tags
    first, then the surroundings for this direction, never "no humans"; then her
    LoRA calls (:data:`HER_LORAS`)."""
    rest = view_prompt(fill_tags, where, pitch, quality="").split(", ")
    rest = [t for t in rest if t and t not in ("no humans", "scenery")]
    words = list(dict.fromkeys(list(subject_tags) + rest))
    return ", ".join(([quality] if quality else []) + words + list(HER_LORAS))


def minimal_prompt(where: str | None, pitch: float, quality: str = QUALITY) -> str:
    """No fill tags at all, only the direction: for use with a reference image,
    which carries the scene (Krita AI Diffusion's expand leaves the prompt empty)."""
    return view_prompt([], where, pitch, quality)
