import json
from pathlib import Path

import pytest
from fakes import FakeChatModel
from PIL import Image

from creative_agent.brand import load_brand
from creative_agent.graph import graph as graph_mod
from creative_agent.graph.graph import JOB_GRAPH, SIZE_GRAPH
from creative_agent.graph.state import initial_state
from creative_agent.models import (BBox, CheckResult, Element, JobRequest, MasterSpec, Plan, Scene)
from creative_agent.render.finalize import FinalizeResult

FIX = Path(__file__).parent / "fixtures"
URL = "https://www.maxis.com.my/"
BB = BBox(x=0, y=0, w=10, h=10)
SIZES = ["300x250", "728x90", "320x50", "160x600", "970x250"]


def plan() -> Plan:
    return Plan.model_validate_json((FIX / "plan" / "maxis_plan.json").read_text())


def ok(name="brand_colors") -> CheckResult:
    return CheckResult(name=name, passed=True, fixable=True)


def bad(name="brand_colors", fixable=True, msg="#333333 isn't a brand colour") -> CheckResult:
    return CheckResult(name=name, passed=False, fixable=fixable, message=msg, frame=0 if fixable else None)


class ScriptedQC:
    """Stands in for run_qc: a list of check lists per size (the last one repeats), plus tiny screenshots."""

    def __init__(self, tmp_path: Path, script: dict[str, list[list[CheckResult]]]):
        self.script, self.calls, self.tmp = script, {}, tmp_path

    async def __call__(self, browser, page_sem, html_path, ctx, landing_cache, out_dir, attempt):
        n = self.calls[ctx.size] = self.calls.get(ctx.size, 0) + 1
        seq = self.script.get(ctx.size) or self.script["*"]
        checks = seq[min(n, len(seq)) - 1]
        shots = []
        for i in range(len(ctx.plan.frames)):
            p = self.tmp / f"shot_{ctx.size}_{n}_{i}.png"
            Image.new("RGB", (ctx.W, ctx.H), "white").save(p)
            shots.append(str(p))
        return list(checks), shots


@pytest.fixture
def patched(monkeypatch, tmp_path):
    builds: list[str] = []

    async def fake_finalize(inp, browser, page_sem):
        builds.append(inp.size)
        return FinalizeResult(html=f"<html><body>{inp.size}</body></html>")

    monkeypatch.setattr("creative_agent.graph.nodes.build.finalize_html", fake_finalize)

    def install(script):
        qc = ScriptedQC(tmp_path, script)
        monkeypatch.setattr("creative_agent.graph.nodes.qc.run_qc", qc)
        return qc

    return builds, install


def spec_fixture(tmp_path: Path) -> Path:
    frame = tmp_path / "frame.png"
    Image.new("RGB", (100, 100), "white").save(frame)
    crops = {}
    for a in ("products_1", "products_2", "products_3"):
        crops[a] = str(tmp_path / f"{a}.png")
        Image.new("RGB", (300, 200), "white").save(crops[a])

    def el(i, role, text, hl=()):
        return Element(id=i, role=role, text=text, highlights=list(hl), bbox=BB)

    legal = lambda s: el(f"{s}.legal", "legal", "Terms & Conditions Apply")  # noqa: E731
    sc = {"s1": [el("s1.headline", "headline", "RM100 Million Device Sale", ["RM100 Million"])],
          "s2": [el("s2.headline", "headline", "Amazing deals up for grabs"),
                 el("s2.subhead", "subhead", "For a limited time only"), legal("s2")],
          "s3": [el("s3.headline", "headline", "Choose from a wide range of 5G phones"), legal("s3")],
          "s4": [el("s4.headline", "headline", "Complete your experience with Home Fibre"), legal("s4")],
          "s5": [el("s5.headline", "cta", "Get yours with Maxis today"), legal("s5")], "end": []}
    prod = {"s2": "products_1", "s3": "products_2", "s4": "products_3", "s5": "products_1"}
    spec = MasterSpec(source="m.mp4", kind="video", width=1080, height=1920, detected_brand="Maxis", assets=crops,
                      scenes=[Scene(id=k, frame_path=str(frame), elements=v, product_asset=prod.get(k),
                                    is_end_frame=k == "end") for k, v in sc.items()], cache_key="k")
    out = tmp_path / "work"
    out.mkdir(exist_ok=True)
    (out / "master_spec.json").write_text(spec.model_dump_json())
    return out / "master_spec.json"


def size_state(tmp_path: Path, size="300x250") -> dict:
    spec = MasterSpec.model_validate_json(spec_fixture(tmp_path).read_text())
    content = {e.id: e for sc in spec.scenes for e in sc.elements}
    sp = next(s for s in plan().sizes if s.size == size)
    used = {spec.scene(f.scene_id).product_asset for f in sp.frames if f.show_products} - {None}
    return {"job_id": "j_test", "size": size, "size_plan": sp, "brand": load_brand("maxis"), "content": content,
            "assets": {a: spec.assets[a] for a in used},
            "scene_assets": {s.id: s.product_asset for s in spec.scenes if s.product_asset},
            "layout_note": "note", "click_url": URL, "attempt": 0}


def run_size(ctx, st):
    return SIZE_GRAPH.ainvoke(st, context=ctx)


def html_text(n: int = 8) -> list[str]:
    return ["<style></style><div id='ad'></div>"] * n


async def test_fixable_failure_then_pass(job_ctx, patched, data_dir, tmp_path):
    builds, install = patched
    qc = install({"*": [[bad()], [ok()]]})
    ctx = job_ctx(llm=FakeChatModel(texts=html_text()))
    final = await run_size(ctx, size_state(tmp_path))
    assert len(ctx.llm.calls) == 2 and builds == ["300x250"] * 2
    assert final["result"].status == "approved" and final["result"].attempts == 2
    assert qc.calls["300x250"] == 2


async def test_always_failing_fixable_stops_at_three(job_ctx, patched, data_dir, tmp_path):
    builds, install = patched
    install({"*": [[bad()]]})
    ctx = job_ctx(llm=FakeChatModel(texts=html_text()))
    final = await run_size(ctx, size_state(tmp_path))
    assert len(ctx.llm.calls) == 3 and len(builds) == 3
    r = final["result"]
    assert r.status == "failed" and r.attempts == 3 and r.failed_checks == ["brand_colors"]


async def test_unfixable_failure_stops_at_once(job_ctx, patched, data_dir, tmp_path):
    builds, install = patched
    install({"*": [[ok(), bad("logo_spec", fixable=False, msg="logo moved")]]})
    ctx = job_ctx(llm=FakeChatModel(texts=html_text()))
    final = await run_size(ctx, size_state(tmp_path))
    assert len(ctx.llm.calls) == 1 and final["result"].status == "failed"
    assert final["result"].failed_checks == ["logo_spec"]


async def test_finish_writes_files(job_ctx, patched, data_dir, tmp_path):
    _, install = patched
    install({"*": [[ok()]]})
    ctx = job_ctx(llm=FakeChatModel(texts=html_text()))
    final = await run_size(ctx, size_state(tmp_path))
    d = ctx.job_dir / "sizes" / "300x250"
    r = final["result"]
    assert (d / "index.html").exists() and (d / "300x250.zip").exists() and (d / "backup.jpg").exists()
    assert len(r.screenshots) == 6 and r.backup_path == str(d / "backup.jpg")
    with Image.open(d / "backup.jpg") as im:
        assert im.size == (300, 250)
    assert (d / "qc_attempt_1.json").exists()
    assert {e["name"] for e in ctx.tracer.events("qc_check")} == {"contract", "brand_colors"}


async def test_retry_prompt_has_previous_html_and_only_fixable_failures(job_ctx, patched, data_dir, tmp_path):
    _, install = patched
    install({"*": [[bad("brand_colors", msg="#333333 isn't a brand colour"),
                    bad("logo_spec", fixable=False, msg="UNFIXABLE-LOGO-MESSAGE")], [ok()]]})
    first = "<style>/*first*/</style><div id='ad'><div class='frame' data-scene='s1'>MODEL-OUTPUT-ONE</div></div>"
    ctx = job_ctx(llm=FakeChatModel(texts=[first, "<div id='ad'></div>"]))
    await run_size(ctx, size_state(tmp_path))
    p1 = ctx.llm.calls[0].messages[1].content
    p2 = ctx.llm.calls[1].messages[1].content
    assert "<previous_html>" not in p1
    assert "<previous_html>" in p2 and "MODEL-OUTPUT-ONE" in p2
    assert "brand_colors (frame 1): #333333 isn't a brand colour" in p2
    assert "UNFIXABLE-LOGO-MESSAGE" not in p2
    assert ctx.llm.calls[1].config["metadata"]["attempt"] == 2


async def test_contract_errors_become_a_failing_fixable_check(job_ctx, monkeypatch, patched, data_dir, tmp_path):
    _, install = patched

    async def contract_fail(inp, browser, page_sem):
        return FinalizeResult(html="", contract_errors=["contract: no element with id='ad'"])

    monkeypatch.setattr("creative_agent.graph.nodes.build.finalize_html", contract_fail)
    qc = install({"*": [[ok()]]})
    ctx = job_ctx(llm=FakeChatModel(texts=html_text()))
    final = await run_size(ctx, size_state(tmp_path))
    assert qc.calls == {}                                           # empty html is never opened in a browser
    assert len(ctx.llm.calls) == 3 and final["result"].failed_checks == ["contract"]


async def test_five_sizes_through_job_graph(job_ctx, patched, monkeypatch, data_dir, tmp_path):
    builds, install = patched
    spec_path = spec_fixture(tmp_path)

    async def fake_ingest(state, runtime, config):
        return {"master_spec_path": str(spec_path), "layout_notes": [], "issues": []}

    monkeypatch.setattr(graph_mod, "ingest_master_node", fake_ingest)
    job_graph = graph_mod.build_job_graph()                      # rebuilt so the patched node is wired in
    install({"*": [[ok()]], "160x600": [[ok(), bad("logo_spec", fixable=False, msg="logo moved")]]})
    llm = FakeChatModel(structured=[plan()], texts=html_text(6))
    ctx = job_ctx(llm=llm)
    req = JobRequest(brand_id="maxis", brief="Sell phones.", click_url=URL)
    out = await job_graph.ainvoke(initial_state("j_test", req, tmp_path / "m.mp4", []), context=ctx)

    assert sorted(out["results"]) == sorted(SIZES)
    assert {s: r.status for s, r in out["results"].items()} == {**{s: "approved" for s in SIZES}, "160x600": "failed"}
    assert out["job_status"] == "completed_with_failures"
    assert sorted(builds) == sorted(SIZES)
    result = json.loads((ctx.job_dir / "result.json").read_text())
    assert result["status"] == "completed_with_failures" and [s["size"] for s in result["sizes"]] == SIZES
    assert (ctx.job_dir / "contact_sheet.png").exists() and result["contact_sheet"] == "contact_sheet.png"
    assert result["usage"]["llm_calls"] == 0                      # fakes bypass the trace callback
    assert (ctx.job_dir / "work" / "plan.json").exists()
    codes = {i["code"] for i in result["issues"]}
    assert {"CTA_ADDED"} & codes == set() and "BRAND_FONT_SUBSTITUTED" in codes   # s5.headline is already a CTA
    assert any(s[0][0].startswith("building") for s in ctx.stages)


async def test_all_failed_status(job_ctx, patched, monkeypatch, data_dir, tmp_path):
    _, install = patched
    spec_path = spec_fixture(tmp_path)

    async def fake_ingest(state, runtime, config):
        return {"master_spec_path": str(spec_path), "layout_notes": [], "issues": []}

    monkeypatch.setattr(graph_mod, "ingest_master_node", fake_ingest)
    install({"*": [[bad("logo_spec", fixable=False)]]})
    ctx = job_ctx(llm=FakeChatModel(structured=[plan()], texts=html_text(6)))
    req = JobRequest(brand_id="maxis", brief="b", click_url=URL)
    out = await graph_mod.build_job_graph().ainvoke(initial_state("j_test", req, tmp_path / "m.mp4", []), context=ctx)
    assert out["job_status"] == "failed"


def test_mermaid_has_every_node():
    m = JOB_GRAPH.get_graph().draw_mermaid()
    for n in ("ingest_master", "check_inputs", "plan", "size_worker", "collect"):
        assert n in m
    s = SIZE_GRAPH.get_graph().draw_mermaid()
    for n in ("build", "qc", "finish"):
        assert n in s
