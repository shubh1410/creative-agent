"""QC rules. measure.js collects facts in the page; every rule lives here, in Python, against the brand kit.

How it fits together:

* ``measure.js`` runs in the browser and returns a dict ``fm`` (frame measurements) with one entry per visible
  element in ``fm["elements"]``: its box, computed styles, text, ``data-*`` attributes, what covers it, ...
* Each rule below is a function ``(fm, ctx, shot) -> list[str]`` of problem descriptions, registered with
  :func:`per_frame`. ``ctx`` is the :class:`QCContext` (brand kit + plan + current frame); ``shot`` is a
  screenshot of the frame (or ``None``), used by rules that need real pixels.
* Problem strings are written for the *model*: they are fed back on retry, so they say what to change.
"""
from __future__ import annotations

import functools
import statistics
from typing import Callable

from PIL import Image

from ..colors import colors_in, nearest, parse_css_color, relative_luminance, rgb_to_hex
from ..graph.nodes.common import norm_text
from ..models import CheckResult
from ..render.finalize import corner_logo_box, end_logo_size
from .context import QCContext

# Registry filled by the @per_frame decorator; run_qc calls every entry once per frame.
PER_FRAME: list[Callable[[dict, QCContext, Image.Image | None], CheckResult]] = []
SVG_TAGS = {"svg", "path", "circle", "rect", "ellipse", "polygon", "line", "polyline"}


def per_frame(name: str, fixable: bool):
    """Register a per-frame rule. A rule returns problem strings; strings starting 'warning:' don't fail it.

    Args:
        name: Check name shown in reports (and used to merge results across frames).
        fixable: Whether the builder model can plausibly repair a failure by editing its HTML. Unfixable
            failures (e.g. wrong logo geometry, which code controls) don't trigger a retry.
    """
    def deco(fn):
        @functools.wraps(fn)
        def run(fm: dict, ctx: QCContext, shot: Image.Image | None) -> CheckResult:
            # dict.fromkeys de-duplicates while keeping order.
            problems = list(dict.fromkeys(fn(fm, ctx, shot)))
            failed = [p for p in problems if not p.startswith("warning:")]
            return CheckResult(name=name, passed=not failed, fixable=fixable, message="; ".join(problems[:6]),
                               frame=ctx.frame_index)
        PER_FRAME.append(run)
        return run
    return deco


# ---------- helpers ----------

def box(e: dict) -> tuple[float, float, float, float]:
    """Element's bounding box as corners ``(x0, y0, x1, y1)``."""
    b = e["bbox"]
    return b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]


def label(e: dict) -> str:
    """Short human name of an element for messages, e.g. ``headline (s1.headline)`` or ``div``."""
    return (e["role"] or e["tag"]) + (f" ({e['el']})" if e["el"] else "")


def intersects(a, b) -> bool:
    """True if two ``(x0, y0, x1, y1)`` boxes overlap (touching edges don't count)."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def contains(outer, inner, tol: float = 0.5) -> bool:
    """True if ``outer`` fully encloses ``inner`` (with ``tol`` px of slack for sub-pixel rounding)."""
    return (outer[0] <= inner[0] + tol and outer[1] <= inner[1] + tol
            and outer[2] >= inner[2] - tol and outer[3] >= inner[3] - tol)


def is_background(e: dict, ctx: QCContext) -> bool:
    """True for the ad root, a frame container, or anything covering >=90 % of the canvas (full-bleed backdrops).
    These legitimately have fills and sit behind everything, so several rules skip them."""
    x0, y0, x1, y1 = box(e)
    return e["id"] == "ad" or "frame" in e["cls"].split() or (x1 - x0) * (y1 - y0) >= 0.9 * ctx.W * ctx.H


def has_fill(e: dict) -> bool:
    """True if the element paints a visible background colour or a background image/gradient."""
    bg = parse_css_color(e["bg"])
    return bool(bg and bg[1] > 0) or e["bgImage"] != "none"


def text_elements(fm: dict) -> list[dict]:
    """Elements that directly contain text (not merely wrap children that do)."""
    return [e for e in fm["elements"] if e["ownText"]]


def first_family(ff: str) -> str:
    """First family of a CSS ``font-family`` list, without quotes: ``"'Figtree', sans-serif"`` -> ``Figtree``."""
    return ff.split(",")[0].strip().strip("'\"")


def logo_of(fm: dict) -> dict | None:
    """The element tagged as the logo (``data-role="logo"``), or ``None``."""
    return next((e for e in fm["elements"] if e["role"] == "logo"), None)


def ring_luminance(shot: Image.Image, b, ring: int = 4) -> float:
    """Median luminance of the pixels in a ``ring``-px band just outside box ``b`` in the screenshot.

    That is the background the element sits on. Median ignores small stray features. Returns 1.0 (light) if
    the band is empty.
    """
    x0, y0, x1, y1 = (round(v) for v in b)
    vals = [relative_luminance(shot.getpixel((x, y))[:3])
            for x in range(max(0, x0 - ring), min(shot.width, x1 + ring))
            for y in range(max(0, y0 - ring), min(shot.height, y1 + ring))
            if not (x0 <= x < x1 and y0 <= y < y1)]
    return statistics.median(vals) if vals else 1.0


# ---------- layout ----------

@per_frame("canvas_bounds", fixable=True)
def canvas_bounds(fm, ctx, shot):
    """Nothing may extend beyond the banner edges (0.5 px slack for anti-aliasing)."""
    return [f"{label(e)} extends outside the {ctx.W}x{ctx.H} canvas" for e in fm["elements"]
            if (b := box(e)) and (b[0] < -0.5 or b[1] < -0.5 or b[2] > ctx.W + 0.5 or b[3] > ctx.H + 0.5)]


OVERFLOW_TOLERANCE_PX = 4        # a line-height rounding difference, not a visible clip
OCCLUDED_SHARE = 0.1             # share of a text element's area that another element may cover
OVERLAP_SHARE = 0.25             # share of the smaller box's height that two texts may overlap by
GHOST_MIN_SIDE_PX = 12           # smaller filled shapes are decoration (dots, rules), not ghost boxes
GHOST_MIN_AREA_PX = 600
UPSCALE_FAIL, UPSCALE_WARN = 1.5, 1.15   # display/natural width ratios that fail / warn


@per_frame("text_overflow", fixable=True)
def text_overflow(fm, ctx, shot):
    """Text must fit inside its own box: scrollWidth/Height larger than clientWidth/Height means it is clipped."""
    out = []
    for e in fm["elements"]:
        if (e["role"] or e["roleCtx"]) and e["client"]["w"] > 0 and e["ownText"]:
            dx, dy = e["scroll"]["w"] - e["client"]["w"], e["scroll"]["h"] - e["client"]["h"]
            if dx > OVERFLOW_TOLERANCE_PX or dy > OVERFLOW_TOLERANCE_PX:
                out.append(f"{label(e)} overflows its box by {max(dx, dy):.0f}px")
    return out


def rect_box(r: dict) -> tuple[float, float, float, float]:
    """Convert an ``{x, y, w, h}`` rectangle to corners ``(x0, y0, x1, y1)``."""
    return r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]


def overlap_size(a, b) -> tuple[float, float]:
    """Width and height of the intersection of two corner-form boxes (negative = they don't overlap)."""
    return min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])


@per_frame("text_overlap", fixable=True)
def text_overlap(fm, ctx, shot):
    """Text drawn over other text, or over a filled box that holds other text (a CTA pill on the legal line)."""
    texts = [e for e in fm["elements"] if e["textRects"]]
    out = []
    for i, a in enumerate(texts):
        for b in texts[i + 1:]:
            if a["idx"] in b["ancestors"] or b["idx"] in a["ancestors"]:
                continue
            a_regions = [rect_box(r) for r in a["textRects"]] + ([box(a)] if has_fill(a) else [])
            b_regions = [rect_box(r) for r in b["textRects"]] + ([box(b)] if has_fill(b) else [])
            for ra in a_regions:
                for rb in b_regions:
                    w, h = overlap_size(ra, rb)
                    smaller = min(ra[3] - ra[1], rb[3] - rb[1])
                    if w > 2 and h > OVERLAP_SHARE * smaller:
                        out.append(f"{label(a)} overlaps {label(b)}")
                        break
                else:
                    continue      # inner loop finished without a hit: try the next region of a
                break             # inner loop hit: this pair is reported, stop searching it
    return out


@per_frame("text_occluded", fixable=True)
def text_occluded(fm, ctx, shot):
    """Text with an image or a filled box drawn in front of it, so part of it can't be read."""
    by_idx = {e["idx"]: e for e in fm["elements"]}
    out = []
    for e in fm["elements"]:
        for c in e["coveredBy"]:
            front = by_idx.get(c["idx"])
            if front and c["idx"] not in e["ancestors"] and c["share"] >= OCCLUDED_SHARE:
                out.append(f"{label(e)} is {c['share']:.0%} covered by {label(front)}; "
                           f"move one of them or change which is in front")
    return out


@per_frame("empty_shape", fixable=True)
def empty_shape(fm, ctx, shot):
    """A solid filled box with no text or image inside is a stray placeholder (e.g. an empty price box)."""
    out = []
    for e in fm["elements"]:
        if e["isImage"] or e["tag"] in SVG_TAGS or is_background(e, ctx) or e["hasContent"]:
            continue
        bg = parse_css_color(e["bg"])
        x0, y0, x1, y1 = box(e)
        w, h = x1 - x0, y1 - y0
        if bg and bg[1] > 0 and min(w, h) >= GHOST_MIN_SIDE_PX and w * h >= GHOST_MIN_AREA_PX:
            out.append(f"{label(e)} ({w:.0f}x{h:.0f}) is a filled box with no text or image in it; remove it")
    return out


@per_frame("image_resolution", fixable=True)
def image_resolution(fm, ctx, shot):
    """Images shown much larger than their pixel size look blurry: >1.5x fails, >1.15x warns."""
    out = []
    for e in fm["elements"]:
        nat = e["natural"]
        if not e["isImage"] or e["role"] == "logo" or not nat or not nat["w"]:
            continue
        scale = e["bbox"]["w"] / nat["w"]
        if scale > UPSCALE_FAIL:
            out.append(f"{label(e)} is shown at {scale:.1f}x its pixel size and will look blurry; make it smaller")
        elif scale > UPSCALE_WARN:
            out.append(f"warning: {label(e)} is shown at {scale:.1f}x its pixel size")
    return out


# ---------- logo ----------

@per_frame("logo_spec", fixable=False)
def logo_spec(fm, ctx, shot):
    """Exactly one logo, not stretched, at the exact position/size the brand kit prescribes (end frame:
    centred, kit-defined size). Code places the logo, so this is a not-fixable safety net against regressions."""
    logos = [e for e in fm["elements"] if e["role"] == "logo"]
    if len(logos) != 1:
        return [f"expected exactly one visible logo, found {len(logos)}"]
    e = logos[0]
    x0, y0, x1, y1 = box(e)
    w, h = x1 - x0, y1 - y0
    out = []
    nat = e["natural"]
    if nat and nat["h"] and abs((w / h) / (nat["w"] / nat["h"]) - 1) > 0.02:
        out.append("the logo is stretched")
    if ctx.is_end:
        _, ew, eh = end_logo_size(ctx.brand, ctx.size)
        if abs((x0 + x1) / 2 - ctx.W / 2) > 2 or abs((y0 + y1) / 2 - ctx.H / 2) > 2:
            out.append("the end-frame logo isn't centred")
        if abs(w - ew) > 2 or abs(h - eh) > 2:
            out.append(f"end-frame logo is {w:.0f}x{h:.0f}, expected {ew}x{eh}")
    else:
        ex, ey, ew, eh, _ = corner_logo_box(ctx.brand, ctx.size)
        if max(abs(x0 - ex), abs(y0 - ey), abs(w - ew), abs(h - eh)) > 1:
            out.append(f"corner logo is at ({x0:.0f},{y0:.0f}) {w:.0f}x{h:.0f}, expected ({ex},{ey}) {ew}x{eh}")
    return out


@per_frame("logo_variant", fixable=True)
def logo_variant(fm, ctx, shot):
    """The logo colour (green/white) must suit the actual background behind it in this frame."""
    logo = logo_of(fm)
    if not logo or shot is None:
        return []
    lum = ring_luminance(shot, box(logo))
    lg = ctx.brand.logo
    want = lg.dark_bg_variant if lum < lg.luminance_threshold else lg.light_bg_variant
    if logo["variant"] == want:
        return []
    return [f"{logo['variant']} logo on a background of luminance {lum:.2f}; keep the background behind the logo "
            f"box the same on every frame so one logo colour fits all"]


@per_frame("logo_no_container", fixable=True)
def logo_no_container(fm, ctx, shot):
    """The logo must not sit inside a box with its own fill, border or shadow."""
    logo = logo_of(fm)
    if not logo:
        return []
    lb = box(logo)
    return [f"the logo sits inside {label(e)}, which has its own fill or border (brand: don't box the logo)"
            for e in fm["elements"]
            if e is not logo and not is_background(e, ctx) and contains(box(e), lb)
            and (has_fill(e) or any(w > 0 for w in e["borderWidths"]) or e["boxShadow"] != "none")]


@per_frame("adinfo_zone", fixable=True)
def adinfo_zone(fm, ctx, shot):
    """Keep content out of the top-right corner that ad platforms cover with their info icon."""
    z = ctx.brand.logo.adinfo_zone
    zone = (ctx.W - z.width_px, 0, ctx.W, z.height_px)
    return [f"{label(e)} overlaps the top-right ad-info area" for e in fm["elements"]
            if e["role"] != "logo" and not is_background(e, ctx)
            and (e["ownText"] or e["isImage"] or has_fill(e)) and intersects(box(e), zone)]


# ---------- colour and type ----------

@per_frame("brand_colors", fixable=True)
def brand_colors(fm, ctx, shot):
    """Every colour used (text, backgrounds, borders, gradients, shadows, SVG) must be within the brand's colour
    tolerance of a palette colour; text must match the *text* palette specifically."""
    pal, text_pal, tol = ctx.brand.palette(), ctx.brand.text_palette(), ctx.brand.color_tolerance
    out = []

    def check(rgb_alpha, allowed, what, e):
        """Record a problem if this colour is farther than ``tol`` from every colour in ``allowed``."""
        if not rgb_alpha or rgb_alpha[1] == 0:        # unparseable or fully transparent: nothing is drawn
            return
        name, d = nearest(rgb_alpha[0], allowed)
        if d > tol:
            out.append(f"{what} {rgb_to_hex(rgb_alpha[0])} on {label(e)} isn't a brand colour")

    for e in fm["elements"]:
        if e["isImage"]:
            continue
        if e["ownText"]:
            check(parse_css_color(e["color"]), text_pal, "text colour", e)
        check(parse_css_color(e["bg"]), pal, "background", e)
        for c, w in zip(e["borderColors"], e["borderWidths"]):
            if w > 0:
                check(parse_css_color(c), pal, "border", e)
        for c in colors_in(e["bgImage"]) + colors_in(e["boxShadow"]):
            check(c, pal, "gradient or shadow colour", e)
        if e["tag"] in SVG_TAGS:
            check(parse_css_color(e["fill"]), pal, "SVG fill", e)
            check(parse_css_color(e["stroke"]), pal, "SVG stroke", e)
    return out


@per_frame("brand_fonts", fixable=True)
def brand_fonts(fm, ctx, shot):
    """All text must use the brand font family (the one that is actually embedded)."""
    want = ctx.brand.fonts.active_family().lower()
    return [f"{label(e)} uses '{first_family(e['fontFamily'])}'; use var(--font-brand)" for e in text_elements(fm)
            if first_family(e["fontFamily"]).lower() != want]


@per_frame("font_weight_by_role", fixable=True)
def font_weight_by_role(fm, ctx, shot):
    """Font weight must match what the brand prescribes for the element's role (headline, CTA, ...)."""
    wbr = ctx.brand.fonts.weights_by_role
    out = []
    for e in text_elements(fm):
        role = e["role"] or e["roleCtx"]
        if role in wbr and e["fontWeight"] != wbr[role]:
            out.append(f"{label(e)} has weight {e['fontWeight']}; the brand uses {wbr[role]} for {role}")
    return out


@per_frame("font_size_count", fixable=True)
def font_size_count(fm, ctx, shot):
    """Limit the number of distinct font sizes in one frame (typographic hierarchy)."""
    sizes = sorted({round(e["fontSize"]) for e in text_elements(fm)})
    limit = ctx.brand.typography.max_font_sizes_per_frame
    return [f"{len(sizes)} font sizes {sizes}; the brand allows {limit} per frame"] if len(sizes) > limit else []


@per_frame("text_color_count", fixable=True)
def text_color_count(fm, ctx, shot):
    """Limit the number of distinct text colours per frame, counted after snapping to the brand text palette."""
    pal = ctx.brand.text_palette()
    names = {nearest(p[0], pal)[0] for e in text_elements(fm) if (p := parse_css_color(e["color"]))}
    limit = ctx.brand.typography.max_text_colors_per_frame
    return [f"{len(names)} text colours {sorted(names)}; the brand allows {limit} per frame"] if len(names) > limit else []


@per_frame("headline_lines", fixable=True)
def headline_lines(fm, ctx, shot):
    """Headlines: more lines than the hard limit fails; more than the preferred limit only warns."""
    t, out = ctx.brand.typography, []
    for e in fm["elements"]:
        if e["role"] == "headline" and e["lines"]:
            if e["lines"] > t.headline_max_lines_hard:
                out.append(f"{label(e)} runs to {e['lines']} lines; the maximum is {t.headline_max_lines_hard}")
            elif e["lines"] > t.headline_max_lines:
                out.append(f"warning: {label(e)} runs to {e['lines']} lines (allowed only if necessary)")
    return out


@per_frame("min_font_size", fixable=True)
def min_font_size(fm, ctx, shot):
    """Readable text: no font below the brand minimum (a lower one applies to legal text)."""
    t, out = ctx.brand.typography, []
    for e in text_elements(fm):
        legal = (e["role"] or e["roleCtx"]) == "legal"
        limit = t.min_legal_font_px if legal else t.min_font_px
        if e["fontSize"] < limit - 0.01:
            out.append(f"{label(e)} is {e['fontSize']:.0f}px; minimum is {limit}px")
    return out


# ---------- content ----------

@per_frame("copy_fidelity", fixable=True)
def copy_fidelity(fm, ctx, shot):
    """The frame must contain exactly the planned elements with exactly the approved wording (compared after
    normalising quotes/whitespace/case), and no stray text outside any ``data-el`` element."""
    out = []
    present = {e["el"] for e in fm["elements"] if e["el"]}
    out += [f"{eid} is missing from this frame" for eid in ctx.frame.keep if eid not in present]
    for e in fm["elements"]:
        if e["el"]:
            want = ctx.content.get(e["el"])
            if want is None:
                out.append(f"unknown data-el '{e['el']}'")
            elif norm_text(e["text"]) != norm_text(want.text):
                out.append(f"{e['el']} reads '{e['text']}', expected '{want.text}'")
        elif e["ownText"] and not e["insideEl"]:
            out.append(f"text '{e['ownText'][:40]}' isn't part of any content element")
    return out


@per_frame("legal_present", fixable=True)
def legal_present(fm, ctx, shot):
    """A frame that shows products must also show legal text."""
    if ctx.frame.show_products and not any((e["role"] or e["roleCtx"]) == "legal" for e in fm["elements"]):
        return ["this frame shows products but no legal text"]
    return []


@per_frame("cta_visible", fixable=True)
def cta_visible(fm, ctx, shot):
    """If the plan puts the CTA on this frame, it must be visible and its bottom-right corner within the canvas."""
    cta_ids = {eid for eid in ctx.frame.keep if ctx.content[eid].role == "cta"}
    if not cta_ids:
        return []
    ok = any(e["el"] in cta_ids and box(e)[2] <= ctx.W + 0.5 and box(e)[3] <= ctx.H + 0.5 for e in fm["elements"])
    return [] if ok else ["the CTA isn't visible on its frame"]


@per_frame("assets_whitelist", fixable=True)
def assets_whitelist(fm, ctx, shot):
    """Only approved product images may appear, and no CSS ``url()`` backgrounds (gradients are fine)."""
    out = []
    for e in fm["elements"]:
        if e["isImage"] and e["role"] != "logo" and e["asset"] not in ctx.assets:
            out.append(f"image '{e['asset'] or 'without data-asset'}' isn't an approved asset")
        if "url(" in e["bgImage"]:
            out.append(f"{label(e)} uses a CSS background image; only gradients are allowed")
    return out
