"""The fill prompt for a view, from the tags that describe the surroundings.

The companion is handed those tags (PQM's classifier and a tagger choose them, as
V.0 measured); here they only get a direction. V.0's views looking down painted
another lake instead of the ground, because every view was prompted "scenery".
"""
from __future__ import annotations

QUALITY = "masterpiece, best quality, amazing quality, very aesthetic, absurdres"
NEGATIVE = ("1girl, 1boy, multiple girls, multiple boys, person, people, human, character, "
            "face, text, watermark, signature, frame, border, picture frame, lowres, "
            "worst quality, bad quality, jpeg artifacts, blurry")

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

UP_DEG = 45.0
DOWN_DEG = -45.0


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


def view_prompt(tags: list[str], where: str | None, pitch: float,
                quality: str = QUALITY) -> str:
    extra: list[str] = []
    if where != "plain":
        if pitch >= UP_DEG:
            extra = {"indoors": ["ceiling", "ceiling light"], "outdoors": ["sky"]}.get(where, [])
        elif pitch <= DOWN_DEG:
            extra = {"indoors": ["floor"], "outdoors": ["ground"]}.get(where, [])
        extra = ["scenery", "no humans"] + extra
    else:
        extra = ["no humans"]
    words = list(dict.fromkeys(extra + tags))
    return ", ".join(([quality] if quality else []) + words)
