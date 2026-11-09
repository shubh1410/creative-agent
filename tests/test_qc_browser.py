import asyncio
import time

import pytest
from fixtures.qc.make_good_ad import CLICK_URL, DURATIONS, build_good_ad
from qc_mutations import MUTATIONS

from creative_agent.models import Element, BBox, FramePlan, SizePlan
from creative_agent.qc.browser import run_qc
from creative_agent.qc.context import QCContext

pytestmark = pytest.mark.asyncio(loop_scope="session")

LANDING_CACHE = {CLICK_URL: 200}            # no network: the landing-URL check reads this cache


def make_ctx(brand) -> QCContext:
    bb = BBox(x=0, y=0, w=10, h=10)
    content = {
        "s1.headline": Element(id="s1.headline", role="headline", text="RM100 Million Device Sale", bbox=bb),
        "s5.headline": Element(id="s5.headline", role="cta", text="Get yours with Maxis today", bbox=bb),
        "s5.legal": Element(id="s5.legal", role="legal", text="Terms & Conditions Apply", bbox=bb),
    }
    plan = SizePlan(size="300x250", backup_frame=0, frames=[
        FramePlan(scene_id="s1", keep=["s1.headline"], show_products=False, duration_ms=DURATIONS[0]),
        FramePlan(scene_id="s5", keep=["s5.headline", "s5.legal"], show_products=True, duration_ms=DURATIONS[1]),
        FramePlan(scene_id="end", keep=[], show_products=False, duration_ms=DURATIONS[2]),
    ])
    return QCContext(size="300x250", W=300, H=250, brand=brand, plan=plan, content=content, assets={"products_1"},
                     click_url=CLICK_URL, budget_floor_hit=False)


async def qc(browser, brand, tmp_path, html: str):
    path = tmp_path / "ad.html"
    path.write_text(html)
    out = tmp_path / "shots"
    out.mkdir(exist_ok=True)
    checks, shots = await run_qc(browser, asyncio.Semaphore(4), path, make_ctx(brand), dict(LANDING_CACHE), out, 1)
    return checks, shots


def failed(checks) -> set[str]:
    return {c.name for c in checks if not c.passed}


async def test_good_ad_passes_every_check(browser, brand, tmp_path):
    checks, shots = await qc(browser, brand, tmp_path, build_good_ad(brand))
    assert failed(checks) == set(), [(c.name, c.message) for c in checks if not c.passed]
    assert len(shots) == 3 and all((tmp_path / "shots" / f"attempt_1_frame_{i}.png").exists() for i in (1, 2, 3))
    names = {c.name for c in checks}
    assert {"timeline", "file_size", "click_opens", "landing_url", "console_errors", "external_requests",
            "legal_once", "logo_spec", "copy_fidelity"} <= names
    assert len(checks) == len(names)                       # exactly one merged result per check


@pytest.mark.parametrize("name", list(MUTATIONS))
async def test_mutation_fails_exactly_its_check(browser, brand, tmp_path, name):
    m = MUTATIONS[name]
    checks, _ = await qc(browser, brand, tmp_path, m.fn(build_good_ad(brand)))
    assert failed(checks) == {m.expected} | set(m.extra), [(c.name, c.message) for c in checks if not c.passed]


async def test_unfixable_flags_match_the_plan(browser, brand, tmp_path):
    checks, _ = await qc(browser, brand, tmp_path, MUTATIONS["move_logo"].fn(build_good_ad(brand)))
    spec = next(c for c in checks if c.name == "logo_spec")
    assert not spec.passed and not spec.fixable
    checks, _ = await qc(browser, brand, tmp_path, MUTATIONS["looping_animation"].fn(build_good_ad(brand)))
    timeline = next(c for c in checks if c.name == "timeline")
    assert not timeline.passed and not timeline.fixable and "loop" in timeline.message


async def test_per_frame_messages_name_the_frame(browser, brand, tmp_path):
    checks, _ = await qc(browser, brand, tmp_path, MUTATIONS["off_palette_text"].fn(build_good_ad(brand)))
    colours = next(c for c in checks if c.name == "brand_colors")
    assert colours.message.startswith("frame 1:") and "#333333" in colours.message


async def test_landing_403_is_a_warning_not_a_failure(browser, brand, tmp_path):
    path = tmp_path / "ad.html"
    path.write_text(build_good_ad(brand))
    out = tmp_path / "shots"
    out.mkdir()
    checks, _ = await run_qc(browser, asyncio.Semaphore(1), path, make_ctx(brand), {CLICK_URL: 403}, out, 1)
    landing = next(c for c in checks if c.name == "landing_url")
    assert landing.passed and landing.message.startswith("warning:")


async def test_landing_unreachable_fails(browser, brand, tmp_path):
    path = tmp_path / "ad.html"
    path.write_text(build_good_ad(brand))
    out = tmp_path / "shots"
    out.mkdir()
    checks, _ = await run_qc(browser, asyncio.Semaphore(1), path, make_ctx(brand), {CLICK_URL: 0}, out, 1)
    assert failed(checks) == {"landing_url"}


async def test_qc_on_one_size_is_fast(browser, brand, tmp_path):
    await qc(browser, brand, tmp_path, build_good_ad(brand))           # warm-up: first page in a session is slower
    t0 = time.perf_counter()
    await qc(browser, brand, tmp_path, build_good_ad(brand))
    assert time.perf_counter() - t0 < 5
