import json
from pathlib import Path

import pytest
from fakes import FakeChatModel

from creative_agent.brand import load_brand
from creative_agent.graph.nodes.dispatch import aspect_of, dispatch_sizes
from creative_agent.graph.nodes.plan import plan_node, validate_plan
from creative_agent.models import ALL_SIZES, BBox, Element, JobRequest, LayoutNotes, MasterSpec, Plan, Scene
from PIL import Image

FIX = Path(__file__).parent / "fixtures"
BB = BBox(x=0, y=0, w=10, h=10)
SIZES = list(ALL_SIZES)


@pytest.fixture
def world(tmp_path):
    crops = {}
    for a in ("products_1", "products_2", "products_3"):
        crops[a] = str(tmp_path / f"{a}.png")
        Image.new("RGB", (300, 200), "white").save(crops[a])

    def el(i, role, text):
        return Element(id=i, role=role, text=text, bbox=BB)

    legal = lambda s: el(f"{s}.legal", "legal", "Terms & Conditions Apply")  # noqa: E731
    sc = {
        "s1": [el("s1.headline", "headline", "RM100 Million Device Sale")],
        "s2": [el("s2.headline", "headline", "Amazing deals up for grabs"), el("s2.subhead", "subhead", "For a limited time only"), legal("s2")],
        "s3": [el("s3.headline", "headline", "Choose from a wide range of 5G phones"), legal("s3")],
        "s4": [el("s4.headline", "headline", "Complete your experience with Home Fibre"), legal("s4")],
        "s5": [el("s5.headline", "cta", "Get yours with Maxis today"), legal("s5")],
        "end": [],
    }
    prod = {"s2": "products_1", "s3": "products_2", "s4": "products_3", "s5": "products_1"}
    spec = MasterSpec(source="m.mp4", kind="video", width=1080, height=1920, detected_brand="Maxis",
                      scenes=[Scene(id=k, frame_path="x", elements=v, product_asset=prod.get(k), is_end_frame=k == "end")
                              for k, v in sc.items()], assets=crops, cache_key="k")
    content = [e for v in sc.values() for e in v]
    return spec, content


def plan_json():
    return json.loads((FIX / "plan" / "maxis_plan.json").read_text())


def check(world, mutate=None, sizes=SIZES):
    spec, content = world
    d = plan_json()
    if mutate:
        mutate(d)
    return validate_plan(Plan.model_validate(d), spec, content, load_brand("maxis"), sizes)


def size_of(d, s):
    return next(x for x in d["sizes"] if x["size"] == s)


def test_valid_plan_passes(world):
    assert check(world) == []


def drop_size(d):
    d["sizes"] = [s for s in d["sizes"] if s["size"] != "160x600"]


def no_offer(d):
    size_of(d, "728x90")["frames"].pop(0)


def end_not_last(d):
    f = size_of(d, "300x250")["frames"]
    f[-1], f[-2] = f[-2], f[-1]


def cta_wrong_frame(d):
    f = size_of(d, "300x250")["frames"]
    f[3]["keep"].append("s5.headline")


def foreign_element(d):
    size_of(d, "300x250")["frames"][0]["keep"].append("s2.headline")


def products_without_legal(d):
    size_of(d, "300x250")["frames"][1]["keep"] = ["s2.headline"]


def short_frame(d):
    size_of(d, "300x250")["frames"][0]["duration_ms"] = 900


def too_long(d):
    for f in size_of(d, "300x250")["frames"]:
        f["duration_ms"] = 6000


def backup_on_end(d):
    s = size_of(d, "300x250")
    s["backup_frame"] = len(s["frames"]) - 1


@pytest.mark.parametrize("mutate,message", [
    (drop_size, "the plan must cover exactly"),
    (no_offer, "the offer scene s1 must be included"),
    (end_not_last, "300x250: the end frame must be last"),
    (cta_wrong_frame, "300x250: the CTA must appear once"),
    (foreign_element, "'s2.headline' doesn't belong to scene s1"),
    (products_without_legal, "300x250: frame 1 shows products, so it must keep the legal text"),
    (short_frame, "300x250: frame 0 is shorter than 1000 ms"),
    (too_long, "300x250: total duration 36000 ms exceeds 30.0 s"),
    (backup_on_end, "300x250: backup_frame must point to a frame other than the end frame"),
])
def test_each_rule_is_caught(world, mutate, message):
    errors = check(world, mutate)
    assert any(message in e for e in errors), errors


def test_unknown_element_and_no_legal(world):
    def m(d):
        f = size_of(d, "320x50")["frames"]
        f[1]["keep"] = ["s5.headline", "s5.nope"]
    errors = check(world, m)
    assert any("unknown element 's5.nope'" in e for e in errors)
    assert any("320x50: legal text must appear at least once" in e for e in errors)


async def test_plan_node_repairs_invalid_plan(job_ctx, world, tmp_path):
    spec, content = world
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "master_spec.json").write_text(spec.model_dump_json())
    bad = plan_json()
    no_offer(bad)
    llm = FakeChatModel(structured=[bad, plan_json()])
    ctx = job_ctx(llm=llm)
    state = {"request": JobRequest(brand_id="maxis", brief="b", click_url="https://www.maxis.com.my/"),
             "master_spec_path": str(tmp_path / "work" / "master_spec.json"), "brand": load_brand("maxis"),
             "content": content, "layout_notes": []}
    out = await plan_node(state, ctx.runtime, None)
    assert len(llm.calls) == 2
    assert "Your previous answer had these problems" in llm.calls[1].messages[-1].content
    assert (tmp_path / "work" / "plan.json").exists()
    assert len(out["plan"].sizes) == 5


def test_aspect_of():
    assert [aspect_of(s) for s in ("728x90", "970x250", "320x50", "300x250", "160x600")] == \
        ["landscape", "landscape", "landscape", "square", "portrait"]


def test_dispatch_sends(world, tmp_path):
    spec, content = world
    (tmp_path / "ms.json").write_text(spec.model_dump_json())
    state = {"job_id": "j", "request": JobRequest(brand_id="maxis", brief="b", click_url="https://www.maxis.com.my/"),
             "master_spec_path": str(tmp_path / "ms.json"), "brand": load_brand("maxis"), "content": content,
             "plan": Plan.model_validate(plan_json()),
             "layout_notes": [LayoutNotes(aspect="landscape", notes="LAND"), LayoutNotes(aspect="square", notes="SQ")]}
    sends = {s.arg["size"]: s.arg for s in dispatch_sizes(state)}
    assert len(sends) == 5 and all(s for s in sends)
    assert dispatch_sizes(state)[0].node == "size_worker"
    assert set(sends["320x50"]["assets"]) == set()                       # no product frames
    assert set(sends["728x90"]["assets"]) == {"products_1"}
    assert set(sends["300x250"]["assets"]) == {"products_1", "products_2", "products_3"}
    assert sends["728x90"]["layout_note"] == sends["970x250"]["layout_note"] == sends["320x50"]["layout_note"] == "LAND"
    assert sends["300x250"]["layout_note"] == "SQ"
    assert "master's own composition" in sends["160x600"]["layout_note"]
    assert sends["300x250"]["attempt"] == 0 and sends["300x250"]["click_url"] == "https://www.maxis.com.my/"
    assert sends["300x250"]["scene_assets"]["s5"] == "products_1"
