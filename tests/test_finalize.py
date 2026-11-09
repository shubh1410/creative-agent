import asyncio
import re
from pathlib import Path

import numpy as np
import pytest
from bs4 import BeautifulSoup
from PIL import Image

from creative_agent.models import BBox, Element, FramePlan, SizePlan
from creative_agent.qc.browser import run_qc
from creative_agent.qc.context import QCContext
from creative_agent.render import finalize as fz
from creative_agent.render.finalize import FinalizeInput, corner_logo_box, end_logo_size, extract_html, finalize_html

pytestmark = pytest.mark.asyncio(loop_scope="session")

FIX = Path(__file__).parent / "fixtures" / "finalize"
URL = "https://www.maxis.com.my/"
BB = BBox(x=0, y=0, w=10, h=10)
CONTENT = {
    "s1.headline": Element(id="s1.headline", role="headline", text="RM100 Million Device Sale",
                           highlights=["RM100 Million"], bbox=BB),
    "s5.headline": Element(id="s5.headline", role="cta", text="Get yours with Maxis today", bbox=BB),
    "s5.legal": Element(id="s5.legal", role="legal", text="Terms & Conditions Apply", bbox=BB),
}
SIZES = ["300x250", "728x90", "320x50"]


def plan_for(size: str) -> SizePlan:
    products = size != "320x50"
    return SizePlan(size=size, backup_frame=0, frames=[
        FramePlan(scene_id="s1", keep=["s1.headline"], show_products=False, duration_ms=2500),
        FramePlan(scene_id="s5", keep=["s5.headline", "s5.legal"], show_products=products, duration_ms=2500),
        FramePlan(scene_id="end", keep=[], show_products=False, duration_ms=2000)])


@pytest.fixture(scope="module")
def product(tmp_path_factory) -> str:
    from PIL import ImageDraw
    p = tmp_path_factory.mktemp("assets") / "products_1.png"
    im = Image.new("RGB", (480, 320), (223, 226, 228))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((40, 40, 440, 280), radius=20, fill=(0, 0, 0))
    d.rectangle((60, 60, 420, 260), fill=(64, 199, 6))
    im.save(p)
    return str(p)


def make_input(brand, size, tmp_path, product, html=None, assets=None, max_kb=None) -> FinalizeInput:
    if max_kb is not None:
        brand = brand.model_copy(deep=True)
        brand.weight.max_kb = max_kb
    html = html if html is not None else (FIX / f"model_{size}.html").read_text()
    work = tmp_path / size
    work.mkdir(exist_ok=True)
    return FinalizeInput(model_html=html, size=size, brand=brand, plan=plan_for(size), content=CONTENT,
                         assets=assets if assets is not None else {"products_1": product}, click_url=URL, work_dir=work)


async def run(browser, brand, size, tmp_path, product, **kw):
    inp = make_input(brand, size, tmp_path, product, **kw)
    return inp, await finalize_html(inp, browser, asyncio.Semaphore(4))


@pytest.mark.parametrize("size", SIZES)
async def test_output_structure(browser, brand, tmp_path, product, size):
    inp, res = await run(browser, brand, size, tmp_path, product)
    html = res.html
    assert res.contract_errors == [] and not res.budget_floor_hit
    soup = BeautifulSoup(html, "html.parser")
    W, H = (int(v) for v in size.split("x"))
    assert soup.find("meta", attrs={"name": "ad.size"})["content"] == f"width={W},height={H}"
    assert re.search(r'<script>var clickTag="https://www\.maxis\.com\.my/";</script>', html)
    used_weights = {600, 700, 400}                                  # headline/cta, highlight, legal
    assert html.count("@font-face") == len(used_weights)
    assert all(f"font-weight:{w};" in html for w in used_weights)
    assert html.count("@keyframes ad-f") == 3
    assert "@keyframes ad-logo" in html
    imgs = soup.find_all("img")
    assert imgs and all(i["src"].startswith("data:image/webp") for i in imgs)
    assert "asset://" not in html
    assert len(html.encode()) <= 150 * 1024

    def style(img):
        return dict(kv.split(":") for kv in img["style"].split(";") if kv)

    cx, cy, cw, ch, _ = corner_logo_box(brand, size)
    corner = soup.select_one("#ad > img.logo-corner")
    st = style(corner)
    assert (float(st["left"][:-2]), float(st["top"][:-2])) == (cx, cy) and (st["width"], st["height"]) == (f"{cw}px", f"{ch}px")
    _, ew, eh = end_logo_size(brand, size)
    end = soup.select_one('#ad > div[data-scene="end"] > img.logo-end')
    assert end is not None and (style(end)["width"], style(end)["height"]) == (f"{ew}px", f"{eh}px")


async def test_corner_logo_variant_follows_background(browser, brand, tmp_path, product):
    dark = (FIX / "model_300x250.html").read_text().replace(
        '#ad>.frame{background:var(--brand-light-grey)}', '#ad>.frame{background:var(--brand-black)}')
    _, res = await run(browser, brand, "300x250", tmp_path, product, html=dark)
    corner = BeautifulSoup(res.html, "html.parser").select_one("#ad > img.logo-corner")
    assert corner["data-variant"] == "white"


async def test_huge_product_crop_stays_within_budget(browser, brand, tmp_path):
    rng = np.random.default_rng(0)
    big = tmp_path / "big.png"
    Image.fromarray(rng.integers(0, 255, (3000, 4000, 3), dtype=np.uint8)).save(big)
    _, res = await run(browser, brand, "300x250", tmp_path, str(big))
    assert len(res.html.encode()) <= 150 * 1024
    assert res.contract_errors == []


async def test_budget_floor_hit_when_limit_tiny(browser, brand, tmp_path, product):
    inp = make_input(brand, "300x250", tmp_path, product)
    inp.brand = inp.brand.model_copy(deep=True)
    inp.brand.weight.max_kb = 10
    res = await finalize_html(inp, browser, asyncio.Semaphore(4))
    assert res.budget_floor_hit


async def test_model_copy_keeps_brand_root(brand):
    c = brand.model_copy(deep=True)
    assert c.path("logos/stacked-green.png").exists()


CONTRACT_CASES = {
    "missing #ad": (lambda h: h.replace('<div id="ad">', '<div id="x">', 1), "no element with id='ad'"),
    "wrong frame order": (lambda h: h.replace('data-scene="s1"', 'data-scene="s9"'), "in this order as direct children"),
    "unknown asset": (lambda h: h.replace("asset://products_1", "https://evil.example/x.png"), "isn't one of the approved"),
    "script": (lambda h: h.replace("</div>\n</div>", "</div><script>alert(1)</script></div>"), "<script> isn't allowed"),
    "inline handler": (lambda h: h.replace('<p class="legal"', '<p onclick="x()" class="legal"'), "inline handler 'onclick'"),
    "css url": (lambda h: h.replace(".product{", ".bg{background:url(x.png)}.product{"), "CSS images (url(...))"),
}


@pytest.mark.parametrize("name", list(CONTRACT_CASES))
async def test_contract_violation_is_a_message_not_an_exception(browser, brand, tmp_path, product, name):
    mutate, expected = CONTRACT_CASES[name]
    html = mutate((FIX / "model_300x250.html").read_text())
    assert html != (FIX / "model_300x250.html").read_text()
    _, res = await run(browser, brand, "300x250", tmp_path, product, html=html)
    assert any(expected in e for e in res.contract_errors), res.contract_errors
    if name == "missing #ad":
        assert res.html == ""


async def test_extract_html_strips_fence():
    fence = "`" * 3
    assert extract_html(f"Here:\n{fence}html\n<div id='ad'></div>\n{fence}\nbye") == "<div id='ad'></div>"
    assert extract_html("<div id='ad'></div>") == "<div id='ad'></div>"


async def test_b64_len():
    import base64
    for n in (0, 1, 2, 3, 10, 100):
        assert fz.b64_len(n) == len(base64.b64encode(b"x" * n))


@pytest.mark.parametrize("size", SIZES)
async def test_finalized_output_passes_every_qc_check(browser, brand, tmp_path, product, size):
    inp, res = await run(browser, brand, size, tmp_path, product)
    assert res.contract_errors == []
    html = tmp_path / f"{size}.html"
    html.write_text(res.html)
    W, H = (int(v) for v in size.split("x"))
    ctx = QCContext(size=size, W=W, H=H, brand=brand, plan=inp.plan, content=CONTENT, assets={"products_1"},
                    click_url=URL, budget_floor_hit=res.budget_floor_hit)
    out = tmp_path / f"shots_{size}"
    out.mkdir()
    checks, shots = await run_qc(browser, asyncio.Semaphore(4), html, ctx, {URL: 200}, out, 1)
    failed = [(c.name, c.message) for c in checks if not c.passed]
    assert failed == []
    assert len(shots) == 3


async def test_missing_webp_support_fails_clearly(browser, brand, tmp_path, product, monkeypatch):
    monkeypatch.setattr(fz.features, "check", lambda name: False)
    with pytest.raises(RuntimeError, match="no WebP support"):
        await run(browser, brand, "300x250", tmp_path, product)


@pytest.mark.parametrize("colour", ["green", "white"])
async def test_logo_tag_embeds_exact_display_aspect(colour):
    import base64, io
    from creative_agent.brand import load_brand
    brand = load_brand("maxis")
    _, _, w, h, place = corner_logo_box(brand, "320x50")
    tag = fz.logo_tag(brand, place.variant, colour, w, h, "logo-corner", "")
    data = re.search(r"base64,([^\"]+)", tag).group(1)
    with Image.open(io.BytesIO(base64.b64decode(data))) as im:
        assert im.size == (w * 2, h * 2)


async def test_model_img_css_cannot_move_logos(browser, brand, tmp_path, product):
    hostile = (FIX / "model_300x250.html").read_text().replace(
        "</style>", "#ad img{left:50%;transform:translateX(-50%);bottom:60px}</style>", 1)
    inp, res = await run(browser, brand, "300x250", tmp_path, product, html=hostile)
    assert res.contract_errors == []
    html = tmp_path / "hostile.html"
    html.write_text(res.html)
    ctx = QCContext(size="300x250", W=300, H=250, brand=brand, plan=inp.plan, content=CONTENT, assets={"products_1"},
                    click_url=URL, budget_floor_hit=res.budget_floor_hit)
    out = tmp_path / "shots_hostile"
    out.mkdir()
    checks, _ = await run_qc(browser, asyncio.Semaphore(4), html, ctx, {URL: 200}, out, 1)
    assert [(c.name, c.message) for c in checks if c.name == "logo_spec" and not c.passed] == []
