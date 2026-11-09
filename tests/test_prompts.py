import re
from pathlib import Path
from string import Template

import pytest
from PIL import Image

from creative_agent.brand import load_brand
from creative_agent.models import ALL_SIZES, Element, BBox, JobRequest, LayoutNotes, MasterSpec, Plan, Scene
from creative_agent.prompts import (HERE, baseline_messages, builder_messages, planner_messages,
                                    scene_summary, size_tips)
from creative_agent.render.finalize import corner_logo_box

FIX = Path(__file__).parent / "fixtures"
BB = BBox(x=0, y=0, w=10, h=10)


@pytest.fixture
def world(tmp_path):
    crop = tmp_path / "p1.png"
    Image.new("RGB", (400, 300), "white").save(crop)

    def el(i, role, text, hl=()):
        return Element(id=i, role=role, text=text, highlights=list(hl), bbox=BB)

    content = [el("s1.headline", "headline", "RM100 Million Device Sale", ["RM100 Million"]),
               el("s2.headline", "headline", "Amazing deals up for grabs"),
               el("s2.subhead", "subhead", "For a limited time only"),
               el("s2.legal", "legal", "Terms & Conditions Apply"),
               el("s5.headline", "cta", "Get yours with Maxis today", ["yours", "today"]),
               el("s5.legal", "legal", "Terms & Conditions Apply")]
    scenes = [Scene(id="s1", start_s=0.5, end_s=2.0, frame_path="x", elements=[content[0]]),
              Scene(id="s2", start_s=3.75, end_s=5.5, frame_path="x", elements=content[1:4], product_asset="products_1"),
              Scene(id="s5", start_s=13.75, end_s=16.75, frame_path="x", elements=content[4:], product_asset="products_1"),
              Scene(id="end", frame_path="x", elements=[], is_end_frame=True)]
    spec = MasterSpec(source="m.mp4", kind="video", width=1080, height=1920, detected_brand="Maxis", scenes=scenes,
                      assets={"products_1": str(crop)}, cache_key="k")
    req = JobRequest(brand_id="maxis", brief="Sell phones.", click_url="https://www.maxis.com.my/")
    return spec, content, req, str(crop)


def size_state(world, size="728x90"):
    spec, content, req, crop = world
    plan = Plan.model_validate_json((FIX / "plan" / "maxis_plan.json").read_text())
    sp = next(s for s in plan.sizes if s.size == size)
    return {"size": size, "size_plan": sp, "brand": load_brand("maxis"), "content": {e.id: e for e in content},
            "assets": {"products_1": crop}, "scene_assets": {"s2": "products_1", "s5": "products_1"},
            "layout_note": "headline left, product right"}


def test_every_template_renders():
    # Every ${VAR} in every template is supplied by the message builders (substitute raises on a miss).
    for f in HERE.glob("*.md"):
        text = f.read_text()
        assert text.strip()
        names = set(re.findall(r"\$\{?([A-Z_0-9]+)\}?", text))
        Template(text).substitute({n: "x" for n in names})


def test_planner_and_baseline_messages_render(world):
    spec, content, req, _ = world
    brand = load_brand("maxis")
    notes = [LayoutNotes(aspect="landscape", notes="copy left")]
    msgs = planner_messages(req, spec, content, notes, brand)
    assert "$" not in msgs[0].content
    assert "s1" in msgs[0].content and "30" in msgs[0].content
    assert 's2.legal [legal] "Terms & Conditions Apply"' in msgs[1].content
    assert "- landscape: copy left" in msgs[1].content
    b = baseline_messages(req, spec, content, brand)
    assert "300x250" in b[0].content and "Maxis" in b[0].content
    assert "CTA element: s5.headline" in b[1].content
    assert "asset://products_1" in b[1].content
    assert "RM100 Million" in scene_summary(spec, content)


def test_size_tips_cover_all_sizes():
    tips = size_tips(ALL_SIZES)
    assert set(tips) == set(ALL_SIZES) and all(tips.values())
    assert "Leaderboard" in tips["728x90"]


def test_builder_prompt_728x90(world):
    st = size_state(world)
    system, user = (m.content for m in builder_messages(st, None, []))
    lx, ly, lw, lh, _ = corner_logo_box(st["brand"], "728x90")
    assert f"x {lx}–{lx + lw}, y {ly}–{ly + lh}" in system
    assert "41x22" in system and "728x90" in system and "var(--brand-maxis-green)" in system
    for f in st["size_plan"].frames:
        assert f'data-scene="{f.scene_id}"' in user
        for eid in f.keep:
            assert f'data-el="{eid}"' in user and st["content"][eid].text in user
    assert "asset://products_1" in user
    assert "<previous_html>" not in user
    assert "(bold: yours, today)" in user
    assert "400x300px" in user


def test_builder_prompt_retry_has_previous_html_and_failures(world):
    st = size_state(world)
    user = builder_messages(st, "<div id='ad'>old</div>", ["brand_colors: #333"])[1].content
    assert "<previous_html>\n<div id='ad'>old</div>\n</previous_html>" in user
    assert "- brand_colors: #333" in user


def test_missing_variable_fails_loudly():
    with pytest.raises(KeyError):
        Template("hello $NAME").substitute({})
