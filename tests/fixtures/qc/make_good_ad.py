"""Builds the hand-made 'good' 300x250 ad for the QC browser tests, from the real brand kit.

Generated from code (not stored) so it always matches the kit. Run
`uv run python tests/fixtures/qc/make_good_ad.py` to write good_300x250.html next to this file for a visual look.

The layout keeps every rule easy to break on purpose (see tests/qc_mutations.py), so the markup uses fixed
strings that the mutations replace:
  s1  headline (600) with a highlight (700), black on light grey
  s5  CTA pill (green fill, black text, 600), legal text (400, 10 px), one product image
  end centred stacked logo on white
  corner logo (stacked, green) on every frame except the end frame
"""
import base64
import io
from pathlib import Path

from PIL import Image, ImageDraw

from creative_agent.brand import BrandKit, load_brand
from creative_agent.render.finalize import corner_logo_box, end_logo_size
from creative_agent.render.timeline import RUNTIME_JS, timeline_css

SIZE = (300, 250)
CLICK_URL = "https://www.maxis.com.my/"
DURATIONS = [2500, 2500, 2000]
HEADLINE = "RM100 Million Device Sale"
CTA_TEXT = "Get yours with Maxis today"
LEGAL_TEXT = "Terms &amp; Conditions Apply"


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def png_url(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + b64(buf.getvalue())


def logo_url(brand: BrandKit, variant: str, colour: str, w: int, h: int) -> str:
    with Image.open(brand.logo_asset(variant, colour)) as im:
        im = im.convert("RGBA")
        im.thumbnail((w * 2, h * 2), Image.LANCZOS)
        return png_url(im)


def product_url() -> str:
    """A small generated 'device' picture in palette colours."""
    im = Image.new("RGB", (240, 160), (223, 226, 228))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((20, 20, 220, 140), radius=10, fill=(0, 0, 0))
    d.rectangle((30, 30, 210, 130), fill=(64, 199, 6))
    return png_url(im)


def font_faces(brand: BrandKit, weights=(400, 600, 700)) -> str:
    family, files = brand.fonts.active_family(), brand.fonts.active_files()
    return "".join(
        f"@font-face{{font-family:'{family}';font-weight:{w};font-style:normal;font-display:block;"
        f"src:url(data:font/woff2;base64,{b64(brand.path(files[w]).read_bytes())}) format('woff2')}}\n"
        for w in weights)


def build_good_ad(brand: BrandKit | None = None) -> str:
    brand = brand or load_brand("maxis")
    W, H = SIZE
    size = f"{W}x{H}"
    variables = "".join(f"--brand-{n.replace('_', '-')}:{c.hex};" for n, c in brand.colors.items())

    cx, cy, cw, ch, place = corner_logo_box(brand, size)
    variant, ew, eh = end_logo_size(brand, size)
    corner = (f'<img class="logo logo-corner" data-role="logo" data-variant="green" alt="" '
              f'src="{logo_url(brand, place.variant, "green", cw, ch)}" '
              f'style="left:{cx}px;top:{cy}px;width:{cw}px;height:{ch}px">')
    end_logo = (f'<img class="logo logo-end" data-role="logo" data-variant="green" alt="" '
                f'src="{logo_url(brand, variant, "green", ew, eh)}" '
                f'style="left:{(W - ew) / 2}px;top:{(H - eh) / 2}px;width:{ew}px;height:{eh}px">')

    frames = (
        '<div class="frame" data-i="0" data-scene="s1">'
        f'<p class="headline" data-el="s1.headline" data-role="headline">'
        f'<strong data-role="highlight">RM100 Million</strong> Device Sale</p></div>'
        '<div class="frame" data-i="1" data-scene="s5">'
        f'<img class="product" data-asset="products_1" alt="" src="{product_url()}">'
        f'<div class="cta" data-el="s5.headline" data-role="cta">{CTA_TEXT}</div>'
        f'<p class="legal" data-el="s5.legal" data-role="legal">{LEGAL_TEXT}</p></div>'
        f'<div class="frame end" data-i="2" data-scene="end">{end_logo}</div>'
    )

    css = (
        f":root{{{variables}--font-brand:'{brand.fonts.active_family()}',sans-serif}}\n"
        "html,body{margin:0;padding:0;background:transparent}\n"
        f"#ad{{position:relative!important;width:{W}px!important;height:{H}px!important;"
        "overflow:hidden!important;cursor:pointer;font-family:var(--font-brand)}\n"
        "#ad>.frame{position:absolute!important;inset:0;background:#DFE2E4}\n"
        "#ad>.frame.end{background:#FFFFFF}\n"
        "#ad .logo{position:absolute;display:block;pointer-events:none;margin:0}\n"
        ".headline{position:absolute;left:14px;top:40px;width:220px;margin:0;font-size:22px;"
        "line-height:1.15;font-weight:600;color:#000000}\n"
        ".headline strong{font-weight:700}\n"
        ".cta{position:absolute;left:14px;top:160px;padding:8px 18px;border-radius:999px;background:#40C706;"
        "color:#000000;font-size:13px;font-weight:600;white-space:nowrap}\n"
        ".legal{position:absolute;left:14px;top:225px;margin:0;font-size:10px;font-weight:400;"
        "color:#000000;white-space:nowrap}\n"
        ".product{position:absolute;left:160px;top:70px;width:120px;height:80px}\n"
    )
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f'<meta name="ad.size" content="width={W},height={H}">'
        f'<script>var clickTag="{CLICK_URL}";</script>'
        f"<style>\n{font_faces(brand)}{css}{timeline_css(DURATIONS)}</style></head>"
        f'<body><div id="ad">{frames}{corner}</div>'
        f'<script>{RUNTIME_JS.replace("__HAS_END__", "true")}</script></body></html>'
    )


if __name__ == "__main__":
    out = Path(__file__).parent / "good_300x250.html"
    out.write_text(build_good_ad())
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB)")
