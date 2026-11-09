import os

# Must run before any creative_agent import: Settings.model is required.
os.environ.setdefault("MODEL", "fake:test-model")

import pytest  # noqa: E402


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    from creative_agent.config import settings

    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setattr(settings, "data_dir", d)
    return d


@pytest.fixture
def brand():
    """The real Maxis kit, with the real logos and fonts from config/brands/maxis."""
    from creative_agent.brand import load_brand

    return load_brand("maxis")


@pytest.fixture
def qc_ctx(brand):
    """A QCContext for 300x250 with three frames: s1, s5 (CTA + legal, products) and end."""
    from creative_agent.models import BBox, Element, FramePlan, SizePlan
    from creative_agent.qc.context import QCContext

    bb = BBox(x=0, y=0, w=10, h=10)
    content = {
        "s1.headline": Element(id="s1.headline", role="headline", text="RM100 Million Device Sale",
                               highlights=["RM100 Million"], bbox=bb),
        "s5.headline": Element(id="s5.headline", role="cta", text="Get yours with Maxis today", bbox=bb),
        "s5.legal": Element(id="s5.legal", role="legal", text="Terms & Conditions Apply", bbox=bb),
    }
    plan = SizePlan(size="300x250", backup_frame=0, frames=[
        FramePlan(scene_id="s1", keep=["s1.headline"], show_products=False, duration_ms=2000),
        FramePlan(scene_id="s5", keep=["s5.headline", "s5.legal"], show_products=True, duration_ms=2500),
        FramePlan(scene_id="end", keep=[], show_products=False, duration_ms=2000),
    ])
    return QCContext(size="300x250", W=300, H=250, brand=brand, plan=plan, content=content,
                     assets={"products_1"}, click_url="https://www.maxis.com.my/", budget_floor_hit=False)


import pytest_asyncio  # noqa: E402


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def browser():
    """One headless Chromium for the whole test session. Tests that use it need loop_scope="session"."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        b = await p.chromium.launch()
        yield b
        await b.close()


@pytest.fixture
def job_ctx(tmp_path):
    """Factory for a JobContext with fakes. Call as job_ctx(llm=..., vision_llm=...)."""
    import asyncio
    from types import SimpleNamespace

    from fakes import FakeChatModel

    from creative_agent.graph.state import JobContext
    from creative_agent.trace import Tracer

    def make(llm=None, vision_llm=None, browser=None, landing=None):
        stages: list[tuple] = []
        ctx = JobContext(
            job_id="j_test", job_dir=tmp_path, tracer=Tracer(tmp_path, "j_test"),
            browser=browser, page_sem=asyncio.Semaphore(4),
            llm=llm or FakeChatModel(), vision_llm=vision_llm or FakeChatModel(),
            model_name="fake:test-model", vision_model_name="fake:test-model",
            set_stage=lambda *a, **k: stages.append((a, k)),
            landing_cache=dict(landing) if landing is not None else {"https://www.maxis.com.my/": 200})
        ctx.stages = stages                                  # type: ignore[attr-defined]
        ctx.runtime = SimpleNamespace(context=ctx)           # type: ignore[attr-defined]
        return ctx

    return make
