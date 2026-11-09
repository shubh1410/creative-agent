import json
import shutil
from pathlib import Path

import pytest
from fakes import FakeChatModel

from creative_agent.errors import JobInputError
from creative_agent.graph.nodes.check_inputs import check_inputs_node
from creative_agent.graph.nodes.ingest_master import ingest_master_node
from creative_agent.graph.state import initial_state
from creative_agent.ingest.vision import ExtractionDraft, FrameBoxes, LayoutDraft
from creative_agent.models import JobRequest, LayoutNotes, MasterSpec

ROOT = Path(__file__).parent.parent
FIX = Path(__file__).parent / "fixtures" / "ingest"
MAXIS = ROOT / "inputs" / "maxis-merdeka"
INCOME = ROOT / "inputs" / "income-storyboard"

pytestmark = pytest.mark.skipif(not (MAXIS / "master.mp4").exists(), reason="master video not present")


def maxis_draft() -> ExtractionDraft:
    return ExtractionDraft.model_validate_json((FIX / "maxis_extraction.json").read_text())


def vision_for_maxis() -> FakeChatModel:
    notes = LayoutDraft(layouts=[LayoutNotes(aspect="square", notes="headline top left, product right")])
    return FakeChatModel(structured=[maxis_draft(), notes])


def stage_inputs(ctx, src: Path, master_name: str, refs: list[str]):
    inp = ctx.job_dir / "inputs"
    inp.mkdir(exist_ok=True)
    for name in ["job.json", master_name, *refs]:
        shutil.copy(src / name, inp / name)
    req = JobRequest(**json.loads((src / "job.json").read_text()), brief=(src / "brief.md").read_text())
    return initial_state(ctx.job_id, req, inp / master_name, [inp / r for r in refs])


async def ingest_maxis(ctx):
    state = stage_inputs(ctx, MAXIS, "master.mp4", ["reference-static.png"])
    out = await ingest_master_node(state, ctx.runtime, None)
    return {**state, **out}


async def ingest_income(ctx):
    state = stage_inputs(ctx, INCOME, "master.png", [])
    out = await ingest_master_node(state, ctx.runtime, None)
    return {**state, **out}


async def test_ingest_maxis_video(job_ctx, data_dir):
    ctx = job_ctx(vision_llm=vision_for_maxis())
    state = await ingest_maxis(ctx)
    spec = MasterSpec.model_validate_json(Path(state["master_spec_path"]).read_text())
    assert [s.id for s in spec.scenes] == ["s1", "s2", "s3", "s4", "s5", "end"]
    assert spec.scene("s2").product_asset == spec.scene("s5").product_asset is not None
    assert len({s.product_asset for s in spec.scenes if s.product_asset}) == 3
    assert spec.scene("end").is_end_frame and spec.scene("end").product_asset is None
    assert (ctx.job_dir / "work" / "master_spec.json").exists()
    assert state["layout_notes"][0].source == "reference-static.png"
    assert spec.scene("s1").start_s == pytest.approx(0.5, abs=0.3)


async def test_ingest_cache_hit(job_ctx, data_dir):
    llm = vision_for_maxis()
    ctx = job_ctx(vision_llm=llm)
    await ingest_maxis(ctx)
    n = len(llm.calls)
    assert n == 2 and not ctx.tracer.events("ingest_cache_hit")
    await ingest_maxis(ctx)
    assert len(llm.calls) == n
    assert len(ctx.tracer.events("ingest_cache_hit")) == 1


async def test_boxes_override_changes_crop_and_key(job_ctx, data_dir):
    from PIL import Image
    ctx = job_ctx(vision_llm=vision_for_maxis())
    base = await ingest_maxis(ctx)
    base_spec = MasterSpec.model_validate_json(Path(base["master_spec_path"]).read_text())

    ctx2 = job_ctx(vision_llm=vision_for_maxis())
    (ctx2.job_dir / "inputs").mkdir(exist_ok=True)
    (ctx2.job_dir / "inputs" / "boxes.json").write_text(json.dumps({"s3": {"x": 100, "y": 1200, "w": 400, "h": 300}}))
    over = await ingest_maxis(ctx2)
    over_spec = MasterSpec.model_validate_json(Path(over["master_spec_path"]).read_text())

    assert over_spec.cache_key != base_spec.cache_key
    a, b = base_spec.scene("s3").product_asset, over_spec.scene("s3").product_asset
    with Image.open(base_spec.assets[a]) as x, Image.open(over_spec.assets[b]) as y:
        assert x.size != y.size
        assert y.size == (424, 324)


async def test_income_brand_mismatch(job_ctx, data_dir):
    vision = FakeChatModel(structured=[
        FrameBoxes.model_validate_json((FIX / "income_frames.json").read_text()),
        ExtractionDraft.model_validate_json((FIX / "income_extraction.json").read_text())])
    ctx = job_ctx(vision_llm=vision)
    state = await ingest_income(ctx)
    spec = MasterSpec.model_validate_json(Path(state["master_spec_path"]).read_text())
    assert spec.detected_brand == "Income" and len(spec.scenes) == 5
    with pytest.raises(JobInputError) as e:
        await check_inputs_node(state, ctx.runtime, None)
    assert e.value.code == "BRAND_MISMATCH"
    assert not [c for c in ctx.tracer.events("llm_call") if c.get("node") in ("plan", "build")]
    assert not ctx.llm.calls


async def test_check_inputs_maxis_issues(job_ctx, data_dir):
    ctx = job_ctx(vision_llm=vision_for_maxis())
    state = await ingest_maxis(ctx)
    out = await check_inputs_node(state, ctx.runtime, None)
    codes = {i.code for i in out["issues"]}
    assert {"CTA_ADDED", "BRAND_FONT_SUBSTITUTED", "LOGO_FROM_CROP", "LOGO_VARIANT_SWITCHED",
            "LOGO_SPEC_EXTRAPOLATED", "MASTER_RULE_DEVIATION"} <= codes
    assert {i["code"] for i in ctx.tracer.issues()} >= codes
    by_id = {e.id: e for e in out["content"]}
    assert by_id["s5.headline"].role == "cta"
    assert all(e.role != "logo" for e in out["content"])
    assert by_id["s5.headline"].text_color == "white"
    assert (ctx.job_dir / "work" / "content.json").exists()
    assert out["brand"].brand_id == "maxis"


async def test_copy_override(job_ctx, data_dir):
    ctx = job_ctx(vision_llm=vision_for_maxis())
    state = await ingest_maxis(ctx)
    (ctx.job_dir / "inputs" / "copy.json").write_text(json.dumps({"s1.headline": "RM100 Million Mega Sale"}))
    out = await check_inputs_node(state, ctx.runtime, None)
    e = next(e for e in out["content"] if e.id == "s1.headline")
    assert e.text == "RM100 Million Mega Sale" and e.highlights == ["RM100 Million"]
    assert "COPY_OVERRIDDEN" in {i.code for i in out["issues"]}


async def test_copy_override_unknown_id(job_ctx, data_dir):
    ctx = job_ctx(vision_llm=vision_for_maxis())
    state = await ingest_maxis(ctx)
    (ctx.job_dir / "inputs" / "copy.json").write_text(json.dumps({"s9.headline": "x"}))
    with pytest.raises(JobInputError) as e:
        await check_inputs_node(state, ctx.runtime, None)
    assert e.value.code == "COPY_OVERRIDE_UNKNOWN_ID"


async def test_landing_page_403_warns_404_raises(job_ctx, data_dir):
    url = "https://www.maxis.com.my/"
    ctx = job_ctx(vision_llm=vision_for_maxis(), landing={url: 403})
    state = await ingest_maxis(ctx)
    out = await check_inputs_node(state, ctx.runtime, None)
    assert "CLICK_URL_BLOCKS_BOTS" in {i.code for i in out["issues"]}

    ctx.landing_cache[url] = 404
    with pytest.raises(JobInputError) as e:
        await check_inputs_node(state, ctx.runtime, None)
    assert e.value.code == "CLICK_URL_UNREACHABLE"
