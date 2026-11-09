"""Image helpers for cutting regions out of frames, de-duplicating crops and sampling text colour."""
from pathlib import Path

import numpy as np
from PIL import Image

from ..colors import nearest
from ..models import BBox


def crop_box(src: Path, box: BBox, out: Path, pad: int = 12) -> Path:
    """Crop ``box`` (expanded by ``pad`` px on every side, clamped to the image) from ``src`` and save it to ``out``."""
    with Image.open(src) as im:
        x0, y0 = max(0, box.x - pad), max(0, box.y - pad)
        x1, y1 = min(im.width, box.x + box.w + pad), min(im.height, box.y + box.h + pad)
        im.convert("RGB").crop((x0, y0, x1, y1)).save(out)
    return out


def ahash(path: Path, n: int = 8) -> int:
    """Average hash: shrink to ``n`` x ``n`` greyscale and set one bit per pixel brighter than the mean.

    Visually similar images give hashes with few differing bits. Returned as an ``n*n``-bit integer.
    """
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((n, n), Image.BILINEAR), dtype=np.float32)
    return int("".join("1" if v else "0" for v in (a > a.mean()).flatten()), 2)


def same_image(a: Path, b: Path, max_bits: int = 6) -> bool:
    """True if two images are near-duplicates: their average hashes differ in at most ``max_bits`` bits."""
    return bin(ahash(a) ^ ahash(b)).count("1") <= max_bits


def sample_text_color(frame: Path, box: BBox, palette: dict) -> tuple[str, float]:
    """Text covers fewer pixels than its background: split by brightness, take the smaller group.

    Args:
        frame: Image containing the text.
        box: Region of the text.
        palette: ``name -> (r, g, b)`` candidates (the brand's text colours).

    Returns:
        ``(palette_name, distance)``: the nearest palette colour to the text's median colour, and how far
        away the sampled colour actually was (large values mean the match is a poor one).
    """
    with Image.open(frame) as im:
        px = np.asarray(im.convert("RGB").crop(box.as_tuple())).reshape(-1, 3).astype(np.float32)
    # Perceived brightness per pixel (same weights as WCAG luminance, without gamma correction).
    lum = px @ np.array([0.2126, 0.7152, 0.0722])
    bright, dark = px[lum > lum.mean()], px[lum <= lum.mean()]
    text = bright if 0 < len(bright) < len(dark) else dark
    rgb = tuple(int(v) for v in np.median(text, axis=0))     # median is robust to anti-aliased edge pixels
    return nearest(rgb, palette)  # type: ignore[arg-type]
