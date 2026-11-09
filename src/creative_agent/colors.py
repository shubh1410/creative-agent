"""Colour parsing and comparison helpers used for brand-palette checks and contrast (WCAG) calculations."""
import math
import re

# Matches both the legacy "rgb(1, 2, 3)" / "rgba(1, 2, 3, 0.5)" syntax and the modern space-separated
# "rgb(1 2 3 / 50%)" syntax. Groups: 1=R, 2=G, 3=B, 4=optional alpha (number or percentage).
RGB_RE = re.compile(r"rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)(?:\s*[,/]\s*([\d.]+%?))?\s*\)")
RGB = tuple[int, int, int]


def hex_to_rgb(h: str) -> RGB:
    """Convert a 6-digit hex string (with or without ``#``) to an ``(r, g, b)`` tuple of 0-255 ints.

    Raises:
        ValueError: If the string is not valid hex.
    """
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb: RGB) -> str:
    """Convert an ``(r, g, b)`` tuple to an upper-case ``#RRGGBB`` string."""
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def parse_css_color(value: str) -> tuple[RGB, float] | None:
    """Parse a computed colour such as 'rgb(64, 199, 6)' or 'rgba(0, 0, 0, 0)'. Returns (rgb, alpha).

    Accepts ``#rgb``, ``#rrggbb`` and ``rgb()/rgba()`` forms. Returns ``None`` for empty / ``none`` /
    ``transparent`` values or anything unparseable (named colours, hsl(), ...).
    """
    v = (value or "").strip()
    if v in ("", "none", "transparent"):
        return None
    if v.startswith("#") and len(v) in (4, 7):
        # Expand 3-digit shorthand (#abc -> #aabbcc) before decoding.
        h = v[1:] if len(v) == 7 else "".join(c * 2 for c in v[1:])
        try:
            return hex_to_rgb(h), 1.0
        except ValueError:
            return None
    m = RGB_RE.match(v)
    if not m:
        return None
    rgb = tuple(int(float(m.group(i))) for i in (1, 2, 3))
    a = m.group(4)
    # Alpha may be omitted (opaque), a percentage ("50%") or a fraction ("0.5").
    alpha = 1.0 if a is None else (float(a[:-1]) / 100 if a.endswith("%") else float(a))
    return rgb, alpha  # type: ignore[return-value]


def colors_in(value: str) -> list[tuple[RGB, float]]:
    """Every colour inside a compound value such as a gradient or box-shadow.

    Only ``rgb()/rgba()`` occurrences are found, which is what the browser reports for computed styles.
    """
    found = (parse_css_color(m.group(0)) for m in RGB_RE.finditer(value or ""))
    return [c for c in found if c]


def distance(a: RGB, b: RGB) -> float:
    """Euclidean distance between two colours in RGB space (0 = identical, ~441 = black vs white)."""
    return math.dist(a, b)


def nearest(rgb: RGB, palette: dict[str, RGB]) -> tuple[str, float]:
    """Find the palette entry closest to ``rgb``.

    Returns:
        ``(palette_name, distance)`` of the closest colour.
    """
    name = min(palette, key=lambda n: distance(rgb, palette[n]))
    return name, distance(rgb, palette[name])


def relative_luminance(rgb: RGB) -> float:
    """WCAG 2.x relative luminance (0 = black, 1 = white), used for contrast-ratio calculations."""

    def lin(c: int) -> float:
        # sRGB gamma expansion: convert an 8-bit channel to linear light.
        s = c / 255
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in rgb)
    # Coefficients reflect the eye's differing sensitivity to red, green and blue.
    return 0.2126 * r + 0.7152 * g + 0.0722 * b
