"""Turns the model's creative HTML into the final, self-contained banner file.

The model only writes layout and styling inside ``#ad``. Everything brand- or compliance-critical is added by
code here, so the model cannot get it wrong:

1. **Contract check** (:func:`parse_model_html`): strips scripts/handlers/external images, records violations.
2. **Fonts** embedded as subset WOFF2 data URIs; **brand colour variables** and base CSS injected.
3. **Probe pass** in a browser: measures where each product image is shown, and the background behind the logo
   locations, so the right logo colour and image sizes can be chosen.
4. **Logos** inserted at exact kit-defined positions/sizes (corner logo on every frame, large one on the end frame).
5. **Images** re-encoded as WebP and sized to fit the remaining file-size budget.
6. **Assembly** of one HTML file with ``clickTag``, CSS timeline and the small JS runtime.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup, Tag
from fontTools import subset
from PIL import Image, features

from ..brand import BrandKit, LogoPlacement
from ..colors import relative_luminance
from ..models import Element, SizePlan, dims
from .timeline import RUNTIME_JS, timeline_css

DPR_STEPS = (2.0, 1.5, 1.0)         # image densities to try, sharpest first (device pixels per CSS pixel)
Q_MIN, Q_MAX = 40, 85               # WebP quality search range
SAFETY_CHARS = " .,:;!?&%$'\"-^*/()0123456789"   # always kept in font subsets, in case copy is edited later
BUDGET_MARGIN = 0.98                # use at most 98 % of the size limit, leaving headroom


@dataclass
class FinalizeInput:
    """Everything :func:`finalize_html` needs for one size."""

    model_html: str
    size: str
    brand: BrandKit
    plan: SizePlan
    content: dict[str, Element]
    assets: dict[str, str]
    click_url: str
    work_dir: Path


@dataclass
class FinalizeResult:
    """Output of :func:`finalize_html`. ``html`` is empty if the model's HTML had no ``#ad`` element."""

    html: str
    contract_errors: list[str] = field(default_factory=list)   # contract violations, reported by QC
    budget_floor_hit: bool = False            # images at the lowest quality still didn't fit


FENCE = "`" * 3                     # written this way so the source can sit inside Markdown


def extract_html(raw: str) -> str:
    """Take the HTML out of a Markdown code fence if the model added one."""
    m = re.search(FENCE + r"(?:html)?\s*(.*?)" + FENCE, raw, re.S)
    return (m.group(1) if m else raw).strip()


def b64(data: bytes) -> str:
    """Base64 text of ``data``."""
    return base64.b64encode(data).decode()


def b64_len(n: int) -> int:
    """Length of the base64 encoding of ``n`` bytes (4 chars per started 3-byte group)."""
    return 4 * ((n + 2) // 3)


# ---------- contract ----------

def parse_model_html(raw: str, plan: SizePlan, asset_ids: set[str]) -> tuple[Tag | None, str, list[str]]:
    """Parse the model's HTML and enforce the "contract" the rest of the pipeline relies on.

    Required: an element ``#ad`` whose direct ``div.frame`` children carry ``data-scene`` values equal to the
    plan's scene ids, in order. Images must be ``<img src="asset://id">`` with an approved id. Scripts, iframes,
    inline event handlers, CSS ``url()`` images and link ``href`` values are removed (clicks use ``clickTag``).

    Returns:
        ``(ad_tag, css_text, errors)``. ``ad_tag`` is ``None`` when ``#ad`` is missing. ``css_text`` is the
        concatenated ``<style>`` content (styles inside ``#ad`` are lifted out). ``errors`` lists every
        violation so the model can be told what to fix.
    """
    errors: list[str] = []
    soup = BeautifulSoup(extract_html(raw), "html.parser")
    css = "\n".join(s.get_text() for s in soup.find_all("style"))
    ad = soup.find(id="ad")
    if ad is None:
        return None, css, ["contract: no element with id='ad'"]
    for s in ad.find_all("style"):
        s.decompose()

    frames = ad.find_all("div", class_="frame", recursive=False)
    want, got = [f.scene_id for f in plan.frames], [f.get("data-scene") for f in frames]
    if got != want:
        errors.append(f"contract: #ad must contain frames {want} in this order as direct children, got {got}")
    for i, f in enumerate(frames):
        f["data-i"] = str(i)                      # index used by the generated timeline CSS selectors

    for img in ad.find_all("img"):
        src = img.get("src", "")
        aid = src.removeprefix("asset://") if src.startswith("asset://") else None
        if aid not in asset_ids:
            errors.append(f"contract: image '{src[:60]}' isn't one of the approved asset://ids {sorted(asset_ids)}")
            img.decompose()
        else:
            img["data-asset"] = aid
    if "url(" in css or any("url(" in (t.get("style") or "") for t in ad.find_all(True)):
        errors.append("contract: CSS images (url(...)) aren't allowed; use <img src=\"asset://id\">")

    for t in ad.find_all(["script", "iframe", "object", "embed"]):
        errors.append(f"contract: <{t.name}> isn't allowed")
        t.decompose()
    for t in ad.find_all(True):
        for attr in [a for a in t.attrs if a.lower().startswith("on")]:
            errors.append(f"contract: inline handler '{attr}' isn't allowed")
            del t[attr]
        if t.name == "a" and t.get("href"):
            del t["href"]                         # all clicks go through clickTag
    return ad, css, errors


# ---------- fonts and base CSS ----------

def subset_woff2(path: Path, text: str) -> bytes:
    """Shrink a font file to only the glyphs needed for ``text`` and return it as WOFF2 bytes
    (keeps kerning and ligatures). This is the biggest single saving in the file-size budget."""
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.layout_features = ["kern", "liga"]
    font = subset.load_font(str(path), opts)
    sub = subset.Subsetter(opts)
    sub.populate(text=text)
    sub.subset(font)
    buf = io.BytesIO()
    subset.save_font(font, buf, opts)
    return buf.getvalue()


def font_css(inp: FinalizeInput) -> str:
    """``@font-face`` rules with the font data embedded, one per weight actually used.

    Only weights needed by the roles present in the plan are embedded, and only glyphs for the text that
    appears (plus :data:`SAFETY_CHARS`).
    """
    fonts = inp.brand.fonts
    kept = [inp.content[e] for f in inp.plan.frames for e in f.keep if e in inp.content]
    roles = {e.role for e in kept} | ({"highlight"} if any(e.highlights for e in kept) else set())
    weights = sorted({fonts.weights_by_role.get(r, 400) for r in roles} or {400})
    text = "".join(e.text or "" for e in kept) + SAFETY_CHARS
    family, files = fonts.active_family(), fonts.active_files()
    return "".join(
        f"@font-face{{font-family:'{family}';font-weight:{w};font-style:normal;font-display:block;"
        f"src:url(data:font/woff2;base64,{b64(subset_woff2(inp.brand.path(files[w]), text))}) format('woff2')}}"
        for w in weights)


def base_css(W: int, H: int, brand: BrandKit) -> str:
    """Code-owned CSS that the model's styles cannot override.

    Defines ``--brand-*`` colour variables and ``--font-brand``, pins ``#ad`` to the exact banner size, stacks
    the frames on top of each other, and fixes how ``.logo`` images are positioned.
    """
    variables = "".join(f"--brand-{n.replace('_', '-')}:{c.hex};" for n, c in brand.colors.items())
    return (f":root{{{variables}--font-brand:'{brand.fonts.active_family()}',sans-serif}}"
            "html,body{margin:0;padding:0;background:transparent}"
            f"#ad{{position:relative!important;width:{W}px!important;height:{H}px!important;"
            "overflow:hidden!important;cursor:pointer;font-family:var(--font-brand)}"
            "#ad>.frame{position:absolute!important;inset:0}"
            # logos are placed by inline left/top only; model CSS such as `#ad img{transform:...;bottom:...}` must not move them
            "#ad .logo{position:absolute;display:block;pointer-events:none;margin:0;"
            "transform:none!important;right:auto!important;bottom:auto!important}")


# ---------- logos ----------

def corner_logo_box(brand: BrandKit, size: str) -> tuple[int, int, int, int, LogoPlacement]:
    """Where the corner logo goes: ``(x, y, width, height, placement)`` in px. It is anchored by its right and
    top margins; the width follows the logo's aspect ratio unless the kit fixes it."""
    W, _ = dims(size)
    place = brand.logo.corner_for(size)
    w = place.width_px or round(place.height_px * brand.logo_aspect(place.variant))
    return W - place.margin_right_px - w, place.margin_top_px, w, place.height_px, place


def end_logo_size(brand: BrandKit, size: str) -> tuple[str, int, int]:
    """Size of the centred end-frame logo: ``(variant, width, height)``.

    Thin strips (e.g. 728x90) use the horizontal logo at the maximum allowed height; other shapes use the
    stacked logo at a percentage of the width, shrunk if that would exceed the maximum height.
    """
    W, H = dims(size)
    rule = brand.logo.end_frame
    strip = W / H >= rule.strip_ratio
    variant = rule.strip_variant if strip else "stacked"
    aspect, max_h = brand.logo_aspect(variant), H * rule.max_height_pct / 100
    if strip:
        h = round(max_h)
        return variant, round(h * aspect), h
    w = round(W * (rule.vertical_width_pct if H > W else rule.horizontal_width_pct) / 100)
    h = round(w / aspect)
    if h > max_h:
        h = round(max_h)
        w = round(h * aspect)
    return variant, w, h


def logo_tag(brand: BrandKit, variant: str, colour: str, w: int, h: int, cls: str, style: str) -> str:
    """Build an ``<img>`` tag for a logo with the image data embedded (lossless WebP, base64).

    Args:
        brand: Brand kit that supplies the logo files and the brand name (used as ``alt`` text).
        variant: Logo layout (e.g. horizontal or stacked).
        colour: Logo colour variant to use (chosen from the background brightness).
        w: Display width in CSS px.
        h: Display height in CSS px.
        cls: Extra CSS class, e.g. ``logo-corner`` or ``logo-end``.
        style: Extra inline style, normally the ``left``/``top`` position.
    """
    with Image.open(brand.logo_asset(variant, colour)) as im:
        im = im.convert("RGBA")
        # exact 2x of the display box so the embedded aspect matches it (thumbnail() rounds to the
        # source aspect, which at small sizes drifts past the logo_spec stretch tolerance)
        im = im.resize((w * 2, h * 2), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "WEBP", lossless=True)
    return (f'<img class="logo {cls}" data-role="logo" data-variant="{colour}" alt="{brand.brand_name}" '
            f'src="data:image/webp;base64,{b64(buf.getvalue())}" style="width:{w}px;height:{h}px;{style}">')


# ---------- measuring pass ----------

# Runs in the probe page: shows all frames at once and reports, per product image id, the largest size it is
# displayed at ([width, height] in CSS px), so each image can be encoded no larger than needed.
PROBE_JS = """() => {
  const out = {};
  document.querySelectorAll('#ad>.frame').forEach(f => { f.style.display='block'; f.style.opacity='1'; f.style.animation='none'; });
  for (const img of document.querySelectorAll('#ad img[data-asset]')) {
    const r = img.getBoundingClientRect(), id = img.dataset.asset, prev = out[id] || [0, 0];
    out[id] = [Math.max(prev[0], r.width), Math.max(prev[1], r.height)];
  }
  return out;
}"""


def region_luminance(png: bytes) -> float:
    """Median relative luminance (0 dark .. 1 light) of a PNG screenshot, sampled on a 16x16 downscale."""
    with Image.open(io.BytesIO(png)) as im:
        small = im.convert("RGB").resize((16, 16))
    pixels = np.asarray(small).reshape(-1, 3)             # getdata() is deprecated in Pillow 12
    return statistics.median(relative_luminance((int(r), int(g), int(b))) for r, g, b in pixels)


async def probe(browser, page_sem, doc: str, path: Path, W: int, H: int, corner: tuple[int, int, int, int],
                n_frames: int, has_end: bool) -> tuple[dict, float, float]:
    """Render the (logo-less) banner once to learn what code needs to know before placing logos and images.

    Returns:
        ``(image_boxes, corner_luminance, end_luminance)``:

        * ``image_boxes`` – asset id -> displayed ``[w, h]`` (from :data:`PROBE_JS`).
        * ``corner_luminance`` – median background luminance under the corner-logo box across all non-end frames.
        * ``end_luminance`` – background luminance in the centre of the end frame (1.0 if there is none).
    """
    path.write_text(doc)
    x, y, w, h = corner
    async with page_sem:
        context = await browser.new_context(viewport={"width": W, "height": H})
        try:
            page = await context.new_page()
            await page.goto(path.resolve().as_uri())
            await page.evaluate("document.fonts.ready")
            lums = []
            for i in range(n_frames - (1 if has_end else 0)):
                await page.evaluate(f"window.__ad.show({i})")
                lums.append(region_luminance(await page.screenshot(clip={"x": x, "y": y, "width": w, "height": h})))
            end_lum = 1.0
            if has_end:
                await page.evaluate(f"window.__ad.show({n_frames - 1})")
                end_lum = region_luminance(await page.screenshot(clip={"x": W / 4, "y": H / 4, "width": W / 2, "height": H / 2}))
            boxes = await page.evaluate(PROBE_JS)
            return boxes, (statistics.median(lums) if lums else 1.0), end_lum
        finally:
            await context.close()


# ---------- images within budget ----------

def fit(src: Image.Image, w: float, h: float, dpr: float) -> Image.Image:
    """Scale to cover w×h at this density, without cropping and never above the source size."""
    scale = min(1.0, max(w * dpr / src.width, h * dpr / src.height))
    return src.resize((max(1, round(src.width * scale)), max(1, round(src.height * scale))), Image.LANCZOS)


def webp_bytes(img: Image.Image, q: int) -> bytes:
    """Encode ``img`` as WebP at quality ``q`` (method 6 = slowest, smallest output)."""
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=q, method=6)
    return buf.getvalue()


def best_under(img: Image.Image, limit_b64: int) -> bytes | None:
    """Highest-quality WebP of ``img`` whose base64 size is at most ``limit_b64`` characters.

    Binary search over quality between :data:`Q_MIN` and :data:`Q_MAX` (size grows with quality).
    Returns ``None`` if even :data:`Q_MIN` is too large.
    """
    lo, hi, best = Q_MIN, Q_MAX, None
    while lo <= hi:
        q = (lo + hi) // 2
        data = webp_bytes(img, q)
        if b64_len(len(data)) <= limit_b64:
            best, lo = data, q + 1
        else:
            hi = q - 1
    return best


def encode_images(boxes: dict, assets: dict[str, str], budget: int) -> tuple[dict[str, str], bool]:
    """Encode every product image as a WebP data URI so that the total fits in ``budget`` characters.

    Each image gets a share of the budget proportional to its displayed area. For each image, the sharpest
    density in :data:`DPR_STEPS` that fits its share is used.

    Args:
        boxes: Asset id -> displayed ``(w, h)`` in CSS px (from the probe pass).
        assets: Asset id -> path of the source image file.
        budget: Characters of base64 available for all images combined.

    Returns:
        ``(data_uris, floor_hit)``. ``floor_hit`` is True if some image did not fit even at 1x and the
        minimum quality (it is then encoded at that setting anyway and will be over its share).
    """
    total = sum(w * h for w, h in boxes.values()) or 1
    uris, floor_hit = {}, False
    for aid, (w, h) in boxes.items():
        share = int(budget * w * h / total)
        with Image.open(assets[aid]) as im:
            src = im.convert("RGB")
        data = next((d for dpr in DPR_STEPS if (d := best_under(fit(src, w, h, dpr), share))), None)
        if data is None:
            floor_hit, data = True, webp_bytes(fit(src, w, h, 1.0), Q_MIN)
        uris[aid] = f"data:image/webp;base64,{b64(data)}"
    return uris, floor_hit


# ---------- assembly ----------

def assemble(W: int, H: int, css: list[str], ad_html: str, click_url: str, runtime_js: str) -> str:
    """Put the pieces into one standalone HTML document: ``ad.size`` meta tag, ``clickTag`` variable, CSS,
    the ``#ad`` markup and the small runtime script that plays the timeline."""
    return ("<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
            f"<meta name=\"ad.size\" content=\"width={W},height={H}\">"
            f"<script>var clickTag={json.dumps(click_url)};</script>"
            f"<style>{''.join(css)}</style></head><body>{ad_html}<script>{runtime_js}</script></body></html>")


async def finalize_html(inp: FinalizeInput, browser, page_sem) -> FinalizeResult:
    """Run the whole finishing pipeline for one size and return the final HTML.

    Args:
        inp: The model's HTML plus the brand kit, plan, content, assets and click URL.
        browser: A running Playwright browser used for the probe pass.
        page_sem: Semaphore that limits how many pages are open at once.

    Returns:
        A :class:`FinalizeResult`. If the HTML has no ``#ad`` element, ``html`` is empty and
        ``contract_errors`` explains why.

    Raises:
        RuntimeError: If Pillow was built without WebP support.
    """
    if not features.check("webp"):
        raise RuntimeError("this Pillow build has no WebP support; reinstall Pillow with libwebp")
    W, H = dims(inp.size)
    ad, model_css, errors = parse_model_html(inp.model_html, inp.plan, set(inp.assets))
    if ad is None:
        return FinalizeResult(html="", contract_errors=errors)
    n = len(inp.plan.frames)
    has_end = inp.plan.frames[-1].scene_id == "end"
    runtime = RUNTIME_JS.replace("__HAS_END__", "true" if has_end else "false")
    css = [model_css, base_css(W, H, inp.brand), await asyncio.to_thread(font_css, inp)]

    # Pass 1: measure with images as local files and no logo yet.
    for img in ad.find_all("img", attrs={"data-asset": True}):
        img["src"] = Path(inp.assets[img["data-asset"]]).resolve().as_uri()
    cx, cy, cw, ch, place = corner_logo_box(inp.brand, inp.size)
    boxes, corner_lum, end_lum = await probe(browser, page_sem, assemble(W, H, css, str(ad), inp.click_url, runtime),
                                             inp.work_dir / f"{inp.size}.probe.html", W, H, (cx, cy, cw, ch),
                                             n, has_end)

    # Logos: green on light backgrounds, white on dark.
    lg = inp.brand.logo
    pick = lambda lum: lg.dark_bg_variant if lum < lg.luminance_threshold else lg.light_bg_variant  # noqa: E731
    ad.append(BeautifulSoup(logo_tag(inp.brand, place.variant, pick(corner_lum), cw, ch, "logo-corner",
                                     f"left:{cx}px;top:{cy}px"), "html.parser"))
    end_frame = ad.find("div", attrs={"data-scene": "end"}, recursive=False) if has_end else None
    if end_frame is not None:
        variant, ew, eh = end_logo_size(inp.brand, inp.size)
        end_frame.append(BeautifulSoup(logo_tag(inp.brand, variant, pick(end_lum), ew, eh, "logo-end",
                                                f"left:{(W - ew) / 2}px;top:{(H - eh) / 2}px"), "html.parser"))
    css.append(timeline_css([f.duration_ms for f in inp.plan.frames], hide_corner_on_last=has_end))

    # Pass 2: images get whatever the budget has left after everything else.
    for img in ad.find_all("img", attrs={"data-asset": True}):
        img["src"] = ""
    fixed = len(assemble(W, H, css, str(ad), inp.click_url, runtime).encode())
    budget = max(0, int(inp.brand.weight.max_kb * 1024 * BUDGET_MARGIN) - fixed)
    uris, floor_hit = encode_images(boxes, inp.assets, budget)
    for img in ad.find_all("img", attrs={"data-asset": True}):
        img["src"] = uris[img["data-asset"]]
    return FinalizeResult(html=assemble(W, H, css, str(ad), inp.click_url, runtime),
                          contract_errors=errors, budget_floor_hit=floor_hit)
